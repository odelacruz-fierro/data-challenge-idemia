import torch.nn as nn
import torchvision.models as models
import torch

from src import config
from src import model as mtl


# ==============================================================================
# MobileNet V3 Small
# ==============================================================================

class MultiTaskMobileNet(nn.Module):

    def __init__(self):
        super(MultiTaskMobileNet, self).__init__()

        # Load backbone architecture + weights (no modification)
        self.backbone = models.mobilenet_v3_small(weights='IMAGENET1K_V1')
        
        # Size of the features entering the final layer
        # Here in_features=1024
        in_features = self.backbone.classifier[3].in_features
        
        # Remove final classification layer
        # nn.Identity() => A placeholder identity operator that is argument-insensitive.
        # It forwards the input given to it (no operation)
        self.backbone.classifier[3] = nn.Identity()
        
        # --- Define occlusion head ---
        self.occlusion_head = nn.Sequential(
            nn.Linear(in_features, 1),
            # Constrains output between 0.0 and 1.0
            nn.Sigmoid()
        )

        # --- Define gender head ---        
        self.gender_head = nn.Sequential(
            # Outputs 2 raw values (logits)
            nn.Linear(in_features, 1)
        )
        
    def forward(self, x):

        # Backbone features
        features = self.backbone(x)
        
        # Occlusion prediction
        occlusion_pred = self.occlusion_head(features)
        
        # Gender prediction
        gender_pred = self.gender_head(features)
        
        return occlusion_pred, gender_pred


# ==============================================================================
# ResNet-50
# ==============================================================================
class MultiTaskResNet50(nn.Module):
    def __init__(self):
        super(MultiTaskResNet50, self).__init__()

        # Load backbone architecture + weights (ImageNet)
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        # Size of the features entering the final layer (in_features=2048)
        in_features = self.backbone.fc.in_features
        
        # Remove final classification layer by replacing it with Identity
        self.backbone.fc = nn.Identity()
        
        # --- Define occlusion head ---
        self.occlusion_head = nn.Sequential(
            nn.Linear(in_features, 1),
            nn.Sigmoid()
        )

        # --- Define gender head ---        
        self.gender_head = nn.Sequential(
            nn.Linear(in_features, 1)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        occlusion_pred = self.occlusion_head(features)
        gender_pred = self.gender_head(features)
        return occlusion_pred, gender_pred


if __name__ == "__main__":

    print("*****************************************************")
    print("model.py")
    print("*****************************************************")

    print("\nBuilding Model (ResNet-50 Multi-Task) ...")

    if config.USE_CUDA:
        torch.backends.cudnn.benchmark = True
    
    model = mtl.MultiTaskResNet50().to(config.DEVICE)

    #print(model)