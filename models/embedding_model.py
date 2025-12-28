import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models import ViT_B_16_Weights


def load_vit(device):
    weights = ViT_B_16_Weights.DEFAULT
    model = models.vit_b_16(weights=weights)

    # Remove classification head
    model.heads = torch.nn.Identity()

    model.eval()
    model.to(device)
    return model


def get_vit_transform():
    weights = ViT_B_16_Weights.DEFAULT
    return weights.transforms()
