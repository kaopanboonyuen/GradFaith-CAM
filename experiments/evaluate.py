"""
============================================================
GradFaith-CAM
Seeing Isn’t Always Believing:
Faithfulness Analysis of Grad-CAM in Lung Cancer CT Classification

Author:
  Teerapong Panboonyuen
  Chulalongkorn University
  MARSAIL – Motor AI Recognition Solution Artificial Intelligence Laboratory
============================================================
"""

import torch
import argparse
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from gradcam.gradcam import GradCAM
from gradcam.faithfulness import perturbation_faithfulness


def evaluate(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = datasets.ImageFolder(args.data_dir, transform)
    loader = DataLoader(dataset, batch_size=1)

    model = torch.load(args.model, map_location=device)
    model.eval().to(device)

    gradcam = GradCAM(model, args.target_layer)

    y_true, y_pred, faith_scores = [], [], []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        pred = logits.argmax(1)

        cam = gradcam(x, pred.item())
        faith = perturbation_faithfulness(model, x, cam, pred.item())

        y_true.append(y.item())
        y_pred.append(pred.item())
        faith_scores.append(faith)

    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Avg Faithfulness:", sum(faith_scores) / len(faith_scores))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--target_layer", required=True)
    args = parser.parse_args()

    evaluate(args)
