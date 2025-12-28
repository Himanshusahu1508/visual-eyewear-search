# models/vit_shape_classifier.py

import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights


class ViTShapeClassifier(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

        weights = ViT_B_16_Weights.IMAGENET1K_V1
        self.vit = vit_b_16(weights=weights)

        # Freeze ViT backbone
        for param in self.vit.parameters():
            param.requires_grad = False

        in_features = self.vit.heads.head.in_features

        # Replace classifier head
        self.vit.heads.head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.vit(x)
