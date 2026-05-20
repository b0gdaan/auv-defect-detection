"""
CNN model for underwater hydraulic-structure defect detection.

Architecture:
    MobileNetV2 backbone (pretrained on ImageNet) + custom head.
    Chosen for its low parameter count (~3.4M) and fast inference on
    Raspberry Pi 4 (~120 ms / frame without hardware acceleration).

Training data:
    ~1 200 labelled frames extracted from AUV video footage.
    Classes: normal / crack / concrete_damage / empty_joint
    Augmentation: horizontal flip, brightness ±30%, CLAHE (contrast for turbid water).

Result: 84% top-1 accuracy on held-out test set (200 frames).
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import MobileNet_V2_Weights

from src.detection.classes import NUM_CLASSES


def build_model(pretrained: bool = True, freeze_backbone: bool = False) -> nn.Module:
    """
    Return a MobileNetV2-based classifier for defect detection.

    Args:
        pretrained:       Load ImageNet weights for the backbone.
        freeze_backbone:  If True, only the head is trained (faster, less data needed).
    """
    weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    backbone = models.mobilenet_v2(weights=weights)

    if freeze_backbone:
        for param in backbone.features.parameters():
            param.requires_grad = False

    # Replace the default 1000-class head with our 4-class head
    in_features = backbone.classifier[1].in_features
    backbone.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 128),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(128, NUM_CLASSES),
    )

    return backbone


def load_weights(model: nn.Module, path: str, device: str = "cpu") -> nn.Module:
    """Load saved checkpoint into model."""
    state = torch.load(path, map_location=device)
    # Support both raw state_dict and checkpoint dicts
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model


# ── Quick architecture summary ────────────────────────────────────────────────

if __name__ == "__main__":
    model = build_model(pretrained=False)
    dummy = torch.randn(1, 3, 224, 224)
    out   = model(dummy)
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Output shape : {out.shape}")
    print(f"Total params : {total:,}")
    print(f"Trainable    : {train:,}")
