"""
============================================================
GradFaith-CAM
Seeing Isn’t Always Believing:
Faithfulness Analysis of Grad-CAM in Lung Cancer CT Classification

Author:
  Teerapong Panboonyuen
  Chulalongkorn University
  MARSAIL – Motor AI Recognition Solution Artificial Intelligence Laboratory

Reference:
  Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
  via Gradient-based Localization", ICCV 2017.

Paper:
  Seeing Isn’t Always Believing: Analysis of Grad-CAM Faithfulness and
  Localization Reliability in Lung Cancer CT Classification (KST 2026)

Description:
  Faithful and architecture-agnostic Grad-CAM implementation supporting:
    - CNNs (ResNet, DenseNet, EfficientNet)
    - Vision Transformers (ViT via feature projection layers)

This implementation strictly follows:
  α_k^c = (1 / H'W') Σ_i Σ_j ∂y^c / ∂A_ij^k
  L_Grad-CAM^c = ReLU( Σ_k α_k^c A^k )
============================================================
"""

import torch
import torch.nn.functional as F


class GradCAM:
    """
    PyTorch implementation of Grad-CAM.

    Works for:
      - CNNs (last convolutional block)
      - ViT (last feature projection before classification head)

    Usage:
      cam = GradCAM(model, target_layer)
      heatmap = cam(input_tensor, class_idx)
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.model.eval()

        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    @torch.no_grad()
    def _normalize(self, cam: torch.Tensor) -> torch.Tensor:
        """
        Normalize Grad-CAM heatmap to [0,1]
        """
        cam_min, cam_max = cam.min(), cam.max()
        return (cam - cam_min) / (cam_max - cam_min + 1e-8)

    def __call__(self, x: torch.Tensor, class_idx: int) -> torch.Tensor:
        """
        Generate Grad-CAM heatmap for class c.

        Args:
            x: Input tensor [B, C, H, W]
            class_idx: Target class index

        Returns:
            heatmap: Grad-CAM map resized to input resolution
        """
        self.model.zero_grad()

        logits = self.model(x)
        score = logits[:, class_idx].sum()

        score.backward(retain_graph=True)

        # Eq. (2): α_k^c = GAP( ∂y^c / ∂A^k )
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)

        # Eq. (3): L_Grad-CAM^c
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = F.interpolate(
            cam,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        cam = self._normalize(cam)

        return cam.squeeze(1)