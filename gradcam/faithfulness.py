"""
============================================================
GradFaith-CAM
Faithfulness & Localization Metrics for Grad-CAM

Author:
  Teerapong Panboonyuen
  Chulalongkorn University
  MARSAIL

Paper:
  Seeing Isn’t Always Believing: Analysis of Grad-CAM Faithfulness
  and Localization Reliability in Lung Cancer CT Classification (KST 2026)

Implements:
  - Localization Accuracy (Eq. 4)
  - Perturbation-based Faithfulness (Eq. 5)
  - Explanation Consistency (Eq. 6)

All metrics are model-agnostic and dataset-independent.
============================================================
"""

import torch
import numpy as np


# -----------------------------------------------------------
# Localization Accuracy
# -----------------------------------------------------------

def localization_accuracy(cam: torch.Tensor,
                          mask: torch.Tensor,
                          threshold: float = 0.5) -> float:
    """
    Eq. (4): Localization Accuracy

    LocAcc = |L_Grad-CAM ∩ M| / |M|

    Args:
        cam: Grad-CAM heatmap [H, W] ∈ [0,1]
        mask: Ground-truth tumor mask [H, W] ∈ {0,1}
        threshold: Binarization threshold

    Returns:
        localization accuracy score
    """
    cam_bin = (cam >= threshold).float()
    mask = mask.float()

    intersection = (cam_bin * mask).sum()
    union = mask.sum() + 1e-8

    return (intersection / union).item()


# -----------------------------------------------------------
# Perturbation-based Faithfulness
# -----------------------------------------------------------

def perturbation_faithfulness(model: torch.nn.Module,
                              x: torch.Tensor,
                              cam: torch.Tensor,
                              label: int) -> float:
    """
    Eq. (5): Perturbation-based Faithfulness

    Faith(x) = f(x)_y − f(x ⊙ (1 − L_Grad-CAM))_y

    Args:
        model: Trained classifier
        x: Input image [1, C, H, W]
        cam: Grad-CAM heatmap [H, W]
        label: Ground-truth class

    Returns:
        faithfulness score
    """
    model.eval()

    cam = cam.unsqueeze(0).unsqueeze(0)
    cam = cam.to(x.device)

    masked_x = x * (1 - cam)

    with torch.no_grad():
        original_score = model(x)[0, label]
        perturbed_score = model(masked_x)[0, label]

    return (original_score - perturbed_score).item()


# -----------------------------------------------------------
# Explanation Consistency
# -----------------------------------------------------------

def iou(cam1: torch.Tensor,
        cam2: torch.Tensor,
        threshold: float = 0.5) -> float:
    """
    Intersection-over-Union for Grad-CAM maps
    """
    cam1 = (cam1 >= threshold).float()
    cam2 = (cam2 >= threshold).float()

    intersection = (cam1 * cam2).sum()
    union = cam1.sum() + cam2.sum() - intersection + 1e-8

    return (intersection / union).item()


def explanation_consistency(cam_list: list,
                            reference_cam: torch.Tensor) -> float:
    """
    Eq. (6): Explanation Consistency

    Consist = (1/R) Σ IoU(L_r, L_ref)

    Args:
        cam_list: List of Grad-CAM maps from different seeds
        reference_cam: Reference Grad-CAM

    Returns:
        consistency score
    """
    scores = []
    for cam in cam_list:
        scores.append(iou(cam, reference_cam))

    return float(np.mean(scores))