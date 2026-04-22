import torch.nn as nn
from torchvision import models


def create_resnet18_multilabel(num_labels=12, pretrained=False):
    """Create a ResNet-18 with a 12-logit multilabel classification head."""

    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_labels)
    return model
