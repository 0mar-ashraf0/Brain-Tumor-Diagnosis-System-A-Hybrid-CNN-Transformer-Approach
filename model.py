import torch
import torch.nn as nn
import torchvision.models as models

class SOTA_HybridModel(nn.Module):
    def __init__(self, num_classes=4):
        super(SOTA_HybridModel, self).__init__()

        # Branch 1: EfficientNet (CNN)
        self.effnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.effnet.classifier = nn.Identity() # Out: 1280

        # Branch 2: Swin Transformer
        self.swin = models.swin_t(weights=models.Swin_T_Weights.DEFAULT)
        self.swin.head = nn.Identity() # Out: 768

        # Fusion Classifier
        self.classifier = nn.Sequential(
            nn.Linear(1280 + 768, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.6),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        eff_feat = self.effnet(x)
        swin_feat = self.swin(x)
        combined = torch.cat((eff_feat, swin_feat), dim=1)
        return self.classifier(combined)
