import os
import sys
import pandas as pd
from PIL import Image
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from models.vit_shape_classifier import ViTShapeClassifier


# ================= PATHS =================
DATASET_ROOT = os.path.join("data", "dataset_v1")
IMAGES_ROOT = os.path.join(DATASET_ROOT, "raw_images")
METADATA_PATH = os.path.join(DATASET_ROOT, "metadata.csv")
MODEL_SAVE_PATH = os.path.join("artifacts", "shape_classifier.pth")


# ================= DATASET =================
class ShapeDataset(Dataset):
    def __init__(self, df, transform, shape_to_idx):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.shape_to_idx = shape_to_idx

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = os.path.join(IMAGES_ROOT, row["image_path"])
        image = Image.open(img_path).convert("RGB")

        label = self.shape_to_idx[row["shape"]]
        image = self.transform(image)

        return image, label


# ================= TRANSFORMS =================
def get_train_transform():
    return transforms.Compose([
        # ---- Geometry (PIL) ----
        transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
        transforms.RandomResizedCrop(
            224,
            scale=(0.85, 1.0),
            ratio=(0.95, 1.05)
        ),
        transforms.RandomRotation(7),
        transforms.RandomHorizontalFlip(p=0.5),

        # ---- Color (PIL) ----
        transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.10
        ),

        # ---- Convert ----
        transforms.ToTensor(),

        # ---- Tensor-only regularization ----
        transforms.RandomErasing(
            p=0.15,
            scale=(0.02, 0.08),
            ratio=(0.3, 3.3),
            value="random"
        ),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])



def get_val_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


# ================= TRAINING =================
def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = pd.read_csv(METADATA_PATH)

    shapes = sorted(df["shape"].unique().tolist())
    shape_to_idx = {s: i for i, s in enumerate(shapes)}

    # -------- Dataset --------
    train_dataset = ShapeDataset(
        df,
        transform=get_train_transform(),
        shape_to_idx=shape_to_idx
    )

    # -------- Class imbalance (CORRECT WAY) --------
    shape_counts = Counter(df["shape"])
    sample_weights = df["shape"].map(
        lambda s: 1.0 / shape_counts[s]
    ).values

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )

    # -------- Model --------
    model = ViTShapeClassifier(num_classes=len(shapes)).to(device)

    # Lighter class weighting + label smoothing
    class_weights = torch.tensor(
        [1.0 / shape_counts[s] for s in shapes],
        dtype=torch.float,
        device=device
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=0.05
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=1e-4
    )

    epochs = 12
    model.train()

    # -------- Training loop --------
    for epoch in range(epochs):
        total_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {total_loss:.4f}")

    # -------- Save --------
    os.makedirs("artifacts", exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "shape_to_idx": shape_to_idx,
        },
        MODEL_SAVE_PATH,
    )

    print(f"Shape classifier saved at {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    train()
