import torch
import torch.nn as nn
from torchvision import models

def get_model(num_classes, pretrained=True):
    # Use EfficientNet-B0 from torchvision or EfficientNet-PyTorch
    model = models.efficientnet_b0(pretrained=pretrained)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, num_classes)
    )
    return model
