"""
DenseNet121 classifier, ImageNet-pretrained, with most layers frozen so
fine-tuning is realistic on CPU. Only the last dense block + classifier head
are trainable.
"""

import torch
import torch.nn as nn
from torchvision.models import densenet121, DenseNet121_Weights


def build_model(num_classes, freeze_until="denseblock3"):
    weights = DenseNet121_Weights.IMAGENET1K_V1
    model = densenet121(weights=weights)

    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze from `freeze_until` onward (default: last dense block + norm + head)
    unfreeze = False
    for name, module in model.features.named_children():
        if name == freeze_until:
            unfreeze = True
        if unfreeze:
            for param in module.parameters():
                param.requires_grad = True

    # Replace classifier head (always trainable)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)

    return model


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    m = build_model(num_classes=5)
    print(f"Trainable params: {count_trainable_params(m):,}")