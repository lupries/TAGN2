import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
import numpy as np

from ..convgru import ConvGRU
from ..attention import SelfAttention, InterAttention, GAP

class AGNN(MessagePassing):
    """
    Graph Neural Network with Attention Modules for Message Passing and Convolutional GRU for Node Update

    as described in Zero-Shot Video Object Segmentation via Attentive Graph Neural Networks

    """

    def __init__(self, loops, channels, num_nodes, edge_index=None):
        super(AGNN, self).__init__(aggr='add')
        self.loops = loops
        if edge_index is None:
          edge_index = create_fully_connected(num_nodes)
          if torch.cuda.is_available():
            edge_index = edge_index.cuda()
        self.edge_index = edge_index
        # Attention Modules
        self.intraAttention = SelfAttention(channels)
        self.interAttention = InterAttention(channels, channels)
        self.gate           = GAP(channels, channels)
        # Convolutional Gated Recurrent Unit
        self.convGRU        = ConvGRU(channels, channels, 3, 1)
        self.hidden         = None

    def forward(self, x):
        # x has shape [N, W, H, C]

        # Propagate messages
        for itr in range(self.loops):
            x = self.propagate(edge_index=self.edge_index, x=x)
        return x

    def message(self, x_i, x_j, edge_index):
        # x_j and x_j   have shape  [E, C, W, H]
        # edge_index    has shape   [2, E]

        mask_selfAtt = edge_index[0] == edge_index[1]
        x_i_selfAtt, x_j_selfAtt    = x_i[mask_selfAtt], x_j[mask_selfAtt]
        x_i_interAtt, x_j_interAtt  = x_i[mask_selfAtt==False], x_j[mask_selfAtt==False]
        assert (x_i_selfAtt == x_j_selfAtt).all()
        
        msg = torch.zeros_like(x_i)
        # Intra-Attention messages
        msg[mask_selfAtt]         = self.intraAttention.forward(x_i_selfAtt)
        # Inter-Attention messages
        msg[mask_selfAtt==False]  = self.interAttention.forward(x_i_interAtt, x_j_interAtt)
        # Gate
        gate_multiplier = self.gate.forward(msg)[:,:,0,0]
        msg = (gate_multiplier.T * msg.T).T

        return msg

    def update(self, aggr_out):
        # aggr_out has shape [N, C, H, W]
        self.hidden = self.convGRU.forward(aggr_out, self.hidden)
        # Return new node embeddings.
        return self.hidden[0]

    
def create_fully_connected(num_nodes=3):

  """
  Returns assignement matrix for fully-connected graph including self-loops

  """

  first_row = torch.tensor([])
  second_row = torch.tensor([])
  for num in range(num_nodes):
    first_row =  torch.cat([first_row,  torch.ones(1,num_nodes)*num],1)
    second_row = torch.cat([second_row, torch.tensor(np.arange(num_nodes)).float()])
  edge_index = torch.cat([first_row, second_row.view(-1,second_row.size()[0])])

  return edge_index.long()