import torch.nn as nn
from torch.nn import functional as F
from torchvision import models

from .graphnet import AGNN

class TAGNN(nn.Module):

    def __init__(self, loops):
        super(TAGNN, self).__init__()

        deeplab = models.segmentation.deeplabv3_resnet50(pretrained=False)
        self.backbone = deeplab.backbone
        self.graph = AGNN(loops=loops, channels=2048)
        self.readout = models.segmentation.deeplabv3.DeepLabHead(2048, num_classes=1)

    def forward(self, x):

        input_shape = x.shape[-2:]
        features = self.backbone(x)
        x = features['out']
        x = self.graph(x)
        x = self.readout(x)

        return F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)

