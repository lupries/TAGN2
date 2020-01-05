import torch
import torch.nn as nn


class SelfAttention(nn.Module):

    """
    Self Attention Module as described in the AGNN Framework of

    Zero-Shot Video Object Segmentation via Attentive Graph Neural Networks

    """

    def __init__(self, input_channels):
        super(SelfAttention, self).__init__()

        self.W_f = nn.Conv2d(input_channels, input_channels//4, kernel_size=1, stride=1)
        self.W_h = nn.Conv2d(input_channels, input_channels//4, kernel_size=1, stride=1)
        self.activation = nn.Softmax(dim=-1)
        self.W_l = nn.Conv2d(input_channels, input_channels, kernel_size=1, stride=1)
        self.alpha = nn.Parameter(torch.rand(1))

    def forward(self, x):

        """
        calculates e_ii = alpha*(softmax((W_f conv x)(W_h conv x).T)(W_l conv x) + x
        returns message m_ii for node h_i

        :param x: node h_i (feature embedding)
        :return: message m_ii = e_ii for that node (tensor of form W x H x C)
        """
        batch, channels, height, width = x.size()

        x1 = self.W_f(x).view(batch, -1, width*height).permute(0, 2, 1)
        x2 = self.W_h(x).view(batch, -1, width*height)
        t = torch.bmm(x1, x2)
        t = self.activation(t)
        x3 = self.W_l(x).view(batch, -1, width*height)
        t = torch.bmm(x3, t.permute(0, 2, 1))
        t = t.view(batch, channels, height, width)
        t = t * self.alpha.expand_as(t) + x

        return t


class InterAttention(nn.Module):

    """
    Inter Attention Module as described in the AGNN Framework of

    Zero-Shot Video Object Segmentation via Attentive Graph Neural Networks

    """

    def __init__(self, input_features, output_features):
        super(InterAttention, self).__init__()

        self.W_c = nn.Linear(input_features, output_features, bias=False)
        self.activation = nn.Softmax(dim=-1)

    def forward(self, node1, node2):

        """
        calculates e_ij = h_i * W_c * h_j.T
        and m_ij = softmax(e_ij) * h_j

        :param node1: node h_i (flattened to matrix form for multiplication)
        :param node2: node h_j (flattened to matrix form for multiplication)
        :return: message m_ij of form (WH) x C (needs to be reshaped to tensor W x H x C in main class)
        """
        batch, _, height, width = node1.size()

        x = self.W_c(node1.view(batch, -1, width*height))
        x = torch.bmm(x.permute(0, 2, 1), node2.view(batch, -1, width*height))
        x = torch.bmm(node2.view(batch, -1, width*height), self.activation(x))

        return x


class GAP(nn.Module):

    def __init__(self, input_channels, output_channels, pooling_kernel):
        super(GAP, self).__init__()

        self.W_g = nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=1, bias=True)
        self.pooling = nn.AvgPool2d(pooling_kernel)
        self.activation = nn.Sigmoid()

    def forward(self, x):

        """
        calculates sigmoid(AvgPool(W_g conv x + b_g))

        :param x: input message m_ji of size W x H x C
        :return: confidence g_ji with channel wise responses [0,1]^C
        """

        x = self.W_g(x)
        x = self.pooling(x)
        x = self.activation(x)

        return x