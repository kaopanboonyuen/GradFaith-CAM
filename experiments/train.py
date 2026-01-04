"""
============================================================
GradFaith-CAM
Seeing Isn’t Always Believing:
Faithfulness Analysis of Grad-CAM in Lung Cancer CT Classification

Author:
  Teerapong Panboonyuen
  Chulalongkorn University
  MARSAIL – Motor AI Recognition Solution Artificial Intelligence Laboratory

Supports:
- ResNet / DenseNet / EfficientNet / ViT
- Binary or multi-class classification
- Reproducible training (seeded)
============================================================
"""

import os
import random
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch import nn, optim


# -----------------------------------------------------------
# Utilities
# -----------------------------------------------------------

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(name, num_classes):
    if name == "resnet50":
        model = models.resnet50(weights="IMAGENET1K_V2")
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        target_layer = model.layer4[-1]
    elif name == "densenet121":
        model = models.densenet121(weights="IMAGENET1K_V1")
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        target_layer = model.features[-1]
    else:
        raise ValueError("Unsupported model")

    return model, target_layer


# -----------------------------------------------------------
# Training
# -----------------------------------------------------------

def train(args):
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    train_set = datasets.ImageFolder(args.train_dir, transform)
    val_set = datasets.ImageFolder(args.val_dir, transform)

    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch)

    model, _ = build_model(args.model, len(train_set.classes))
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    best_acc = 0.0
    os.makedirs(args.out, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=loss.item())

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)

        acc = correct / total
        print(f"Validation Accuracy: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), f"{args.out}/best_model.pth")

    print("Training completed ✔")


# -----------------------------------------------------------
# CLI
# -----------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", required=True)
    parser.add_argument("--val_dir", required=True)
    parser.add_argument("--model", default="resnet50")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="checkpoints")
    args = parser.parse_args()

    train(args)