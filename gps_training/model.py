import torch
import torch.nn as nn
import torchvision.models as models

class SiameseTracker(nn.Module):
    def __init__(self):
        super(SiameseTracker, self).__init__()
        
        # Backbone
        resnet = models.resnet18(pretrained=True)
        # Remove FC
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # Feature concat dimension: 512 + 512 = 1024
        
        self.regressor = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 3) # dx, dy, dz
        )
        
    def forward(self, img1, img2):
        # Extract features
        # (B, 512, 1, 1)
        f1 = self.backbone(img1).flatten(1)
        f2 = self.backbone(img2).flatten(1)
        
        # Concatenate
        combined = torch.cat((f1, f2), dim=1)
        
        # Regress
        out = self.regressor(combined)
        
        return out
