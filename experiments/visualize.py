"""
============================================================
GradFaith-CAM
Seeing Isn’t Always Believing:
Faithfulness Analysis of Grad-CAM in Lung Cancer CT Classification

Author:
  Teerapong Panboonyuen
  Chulalongkorn University
  MARSAIL – Motor AI Recognition Solution Artificial Intelligence Laboratory
  
GradFaith-CAM – Visualization Script
Produces publication-quality Grad-CAM figures
============================================================
"""

import os
import cv2
import torch
import argparse
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from gradcam.gradcam import GradCAM


def overlay_cam(img, cam):
    heatmap = cv2.applyColorMap(
        (cam.cpu().numpy() * 255).astype("uint8"),
        cv2.COLORMAP_JET
    )
    heatmap = heatmap / 255.0
    return (0.5 * img + 0.5 * heatmap).clip(0, 1)


def visualize(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = torch.load(args.model, map_location=device)
    model.eval()

    gradcam = GradCAM(model, args.target_layer)

    img = Image.open(args.image).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    x = transform(img).unsqueeze(0).to(device)
    pred = model(x).argmax(1).item()

    cam = gradcam(x, pred)
    cam = cam.squeeze()

    img_np = x.squeeze().permute(1, 2, 0).cpu().numpy()
    vis = overlay_cam(img_np, cam)

    plt.figure(figsize=(5, 5))
    plt.imshow(vis)
    plt.axis("off")
    os.makedirs(args.out, exist_ok=True)
    plt.savefig(f"{args.out}/gradcam.png", dpi=300, bbox_inches="tight")
    print("Saved Grad-CAM visualization ✔")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--target_layer", required=True)
    parser.add_argument("--out", default="results/figures")
    args = parser.parse_args()

    visualize(args)