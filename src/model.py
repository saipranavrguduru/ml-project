import torch
import torch.nn as nn
from torchvision import models


SUPPORTED_ARCHITECTURES = ("resnet18", "densenet121", "vitb32", "efficientnet_b0")


def create_multilabel_model(architecture="resnet18", num_labels=12, pretrained=False):
    """Create a supported torchvision classifier with a multilabel output head."""

    if architecture == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_labels)
        return model

    if architecture == "densenet121":
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        model = models.densenet121(weights=weights)
        model.classifier = nn.Linear(model.classifier.in_features, num_labels)
        return model

    if architecture == "vitb32":
        weights = models.ViT_B_32_Weights.DEFAULT if pretrained else None
        model = models.vit_b_32(weights=weights)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_labels)
        return model

    if architecture == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_labels)
        return model

    raise ValueError(
        f"Unsupported architecture: {architecture}. "
        f"Supported values: {', '.join(SUPPORTED_ARCHITECTURES)}"
    )


def freeze_backbone(model, architecture):
    """Freeze all backbone parameters while leaving the classifier head trainable."""

    for param in model.parameters():
        param.requires_grad = False

    if architecture == "resnet18":
        for param in model.fc.parameters():
            param.requires_grad = True
        return model

    if architecture == "densenet121":
        for param in model.classifier.parameters():
            param.requires_grad = True
        return model

    if architecture == "vitb32":
        for param in model.heads.head.parameters():
            param.requires_grad = True
        return model

    if architecture == "efficientnet_b0":
        for param in model.classifier.parameters():
            param.requires_grad = True
        return model

    raise ValueError(
        f"Unsupported architecture: {architecture}. "
        f"Supported values: {', '.join(SUPPORTED_ARCHITECTURES)}"
    )
