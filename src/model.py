import torch.nn as nn
from torchvision import models


SUPPORTED_ARCHITECTURES = ("resnet18", "densenet121")


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

    raise ValueError(
        f"Unsupported architecture: {architecture}. "
        f"Supported values: {', '.join(SUPPORTED_ARCHITECTURES)}"
    )
