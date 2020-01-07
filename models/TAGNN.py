import torch.nn as nn

from .graphnet import AGNN
from torchvision import models

class TAGNN(nn.Module):

    def __init__(self, loops):
        super(TAGNN, self).__init__()

        self.deeplab = models.segmentation.deeplabv3_resnet50(pretrained=False)
        self.backbone = self.deeplab.backbone
        self.graph = AGNN(loops=loops, channels=2048)
        self.readout = models.segmentation.deeplabv3.DeepLabHead(2048, num_classes=1)

    def forward(self, x):

        features = self.backbone(x)
        x = features['out']
        x = self.graph(x)
        x = self.readout(x)

        return x

