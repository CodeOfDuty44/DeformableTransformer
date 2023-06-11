import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath, to_2tuple

from models.modules import Stage

class DAT(nn.Module):

    def __init__(self, img_size=224, num_classes=1000):
        super().__init__()

        self.patch_embed1 = nn.Conv2d(3, 96, 4, 4, 0), #kernel=4 stride=4
    
        self.stage1 = Stage(dim = 96, window_size=7, spatial_dim=56,
                N = 2, block_types=["Local", "Shift"],
                n_head=3)
        
        self.patch_embed2 = nn.Conv2d(96,192,2,2,0,bias=False)

        self.stage2 = Stage(dim = 192, window_size=7, spatial_dim=28,
                N = 2, block_types=["Local", "Shift"],
                n_head=6)
        
        self.stage3 = Stage(dim = 384, window_size=7, spatial_dim=14,
                N = 6, block_types=["Local", "Deformable","Local", "Deformable","Local", "Deformable"],
                n_head=12)
        
        self.stage4 = Stage(dim = 768, window_size=7, spatial_dim=7,
                N = 2, block_types=["Local", "Deformable"],
                n_head=12)
        
        
        self.patch_embed3 = nn.Conv2d(192,384,2,2,0,bias=False)
        self.patch_embed4 = nn.Conv2d(384,768,2,2,0,bias=False)
        self.linear = nn.Linear(768, num_classes) # 1000 classes
        
        
    
    
    
    def forward(self, x):
        x = self.patch_embed1(x) # 1,96,56,56
        x = self.stage1(x)
        x= self.patch_embed2(x)

        x = self.stage2(x)
        x= self.patch_embed3(x)

        x = self.stage3(x)
        x= self.patch_embed4(x)

        x = self.stage4(x)
        x = self.linear(x)
        return x


        # #i=0 x: 1,96,56,56 -> 1,96,56,56 -> 1,192,28,28 
        # #i=1 x: 1,192,28,28 -> 1,192,28,28 -> 1,384,14,14
        # #i=2 x: 1,384,14,14 -> 1,384,14,14 -> 1,768,7,7
        # #i=3 1,768,7,7 -> 1,768,7,7
        # for i in range(4):
        #     x, pos, ref = self.stages[i](x)
        #     if i < 3:
        #         x = self.down_projs[i](x)
        #     positions.append(pos)
        #     references.append(ref)
        # x = self.cls_norm(x) # 1,768,1,1
        # x = F.adaptive_avg_pool2d(x, 1) #1,768
        # x = torch.flatten(x, 1)
        # x = self.cls_head(x) #1,1000
        
        # return x, positions, references




def build_model(config):
    model = DAT(**config.MODEL.DAT)
    return model
