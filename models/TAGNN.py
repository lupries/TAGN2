import torch.nn as nn
from torch.nn import functional as F
from torchvision import models
import torch

from .graphnet import AGNN
from .graphnet import create_fully_connected

class TAGNN(nn.Module):

    def __init__(self, loops, num_nodes):
        super(TAGNN, self).__init__()

        deeplab = models.segmentation.deeplabv3_resnet50(pretrained=False)
        self.backbone = deeplab.backbone
        self.graph = AGNN(loops=loops, channels=2048, num_nodes=num_nodes)
        self.readout = models.segmentation.deeplabv3.DeepLabHead(2048, num_classes=1)

    def forward(self, x):

        # reset hidden state for Gated Recurrent Units in Graph Update function
        if self.graph.hidden is not None:
            self.graph.hidden = None

        input_shape = x.shape[-2:]
        features = self.backbone(x)
        x = features['out']
        x = self.graph(x)
        x = self.readout(x)

        return F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)


class TAGNN_batch(nn.Module):

    def __init__(self, loops, frames, batch_size, backbone=None):
        super(TAGNN_batch, self).__init__()

        edge_index = create_fully_connected(frames)
        new_edge_index = edge_index
        for i in range(1,batch_size):
          new_edge_index = torch.cat((new_edge_index,edge_index + torch.ones_like(edge_index) * i * frames),dim=1)
        
        if backbone is not None:
            deeplab = backbone
        else:
            deeplab = models.segmentation.deeplabv3_resnet50(pretrained=False)
            deeplab.classifier = models.segmentation.deeplabv3.DeepLabHead(2048, num_classes=1)
        
        self.backbone = deeplab.backbone
        self.deeplabhead = deeplab.classifier
        self.ASPP     = nn.Sequential(
            self.deeplabhead[0],
            self.deeplabhead[1],
            self.deeplabhead[2],
            self.deeplabhead[3]
        )
        self.graph = AGNN(loops=loops, channels=256, num_nodes=frames, edge_index=new_edge_index.cuda())
        self.classifier = nn.Sequential(
            nn.Conv2d(256*2, 256, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            self.deeplabhead[4]
        )
        self.node_states = []

    def forward(self, x):

        input_shape = x.shape[-2:]
        frames = x.shape[1]
        
        # backbone (feature extraction)
        features = self.encode(x,frames)
        
        # flatten batches for graph
        batch, frames, channel, height, width = features.shape
        x = features.view(-1, channel, height, width)
        
        # set node states
        self.graph.hidden = x
        
        # graphnet (attention mechanism)
        x = self.graph(x)

        # list intermediate node states and pass through readout
        self.node_states = []
        for state in self.graph.hidden_states:
            self.node_states.append(self.readout(state.view(features.shape),features,input_shape,frames))
        # reshape (unflatten batches)
        x = x.view(features.shape)
        
        # readout (pixelwise classification)
        out = self.readout(x, features, input_shape, frames)

        return out

    def encode(self, x, frames):
        
        features = torch.Tensor().cuda()
        for frame in range(frames):
          frame_feature = self.backbone(x[:,frame])['out']
          frame_feature = self.ASPP(frame_feature)
          frame_feature = frame_feature.unsqueeze(1)
          features = torch.cat((features, frame_feature),dim=1)
        return features
        
    def readout(self, x, res_x, input_shape, frames):

        x = torch.cat((x,res_x),dim=2)
        out = torch.Tensor().cuda()
        for frame in range(frames):
          frame_out = self.classifier(x[:,frame])
          frame_out = F.interpolate(frame_out, size=input_shape, mode='bilinear', align_corners=False)
          out = torch.cat((out,frame_out),dim=1)
        return out

