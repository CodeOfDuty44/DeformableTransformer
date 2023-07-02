import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath, to_2tuple

from modules import Stage
from dat import DAT
import torchvision

class DATDetectorBackbone(DAT):
    
    def __init__(self, img_size=448, num_classes=1000):
        super().__init__()
        self.channel_adj = nn.Conv2d(768,256,kernel_size=1)

    def forward(self, x):
        x = self.patch_embed1(x).permute((0,2,3,1)) # 1,96,56,56 -> 1,56,56,96
        x = self.stage1(x).permute((0,3,1,2))
        x= self.patch_embed2(x).permute((0,2,3,1))

        x = self.stage2(x).permute((0,3,1,2))
        x= self.patch_embed3(x).permute((0,2,3,1))

        x = self.stage3(x).permute((0,3,1,2))
        x= self.patch_embed4(x).permute((0,2,3,1))

        x = self.stage4(x).permute((0,3,1,2))
        x = self.channel_adj(x)

        return [x]
    
class DATDetector(nn.Module):

    def __init__(self, img_size=448, num_classes=1000):
        super().__init__()
        self.backbone = DATDetectorBackbone(224,1000)
        retina = torchvision.models.detection.retinanet_resnet50_fpn()
        self.head = retina.head
    def forward(self, x):
        features = self.backbone(x)
        out = self.head(features)

        return out
    
def main():
    # detector = DATDetectorBackbone()
    # x = torch.zeros(5,3,224,224)
    
    # features = detector(x)

    # retina = torchvision.models.detection.retinanet_resnet50_fpn().eval()
    # retina_head = retina.head

    # ccc = retina(x).shape
    # retina.backbone = detector
    # ccc = retina(x).shape
    detector = DATDetector(224,1000)
    x = torch.zeros(5,3,224,224)
    x = detector(x)

    44

if __name__ == "__main__":
    main()