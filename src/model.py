"""
ResNet50 Feature Extraction Model

Removes the final fully-connected layer and outputs the 2048-dim
feature vector after global average pooling.
"""
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights

class ResNet50FeatureExtractor(nn.Module):
    """
    ResNet50 feature extractor.

    Input:  image tensor of shape (B, 3, 224, 224)
    Output: feature vectors of shape (B, 2048) (after global average pooling)
    """

    def __init__(self, weight_option):
        super(ResNet50FeatureExtractor, self).__init__()


        if weight_option == "DEFAULT":
            # Load ImageNet-1K pretrained weights (official 1000-class weights)
            resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        else:
            # weight_option is None, initialize model with random weights
            resnet = models.resnet50(weights=None)

        # Keep all layers except the final fully-connected layer.
        # Includes: conv1, bn1, relu, maxpool, layer1~4, avgpool
        self.features = nn.Sequential(*list(resnet.children())[:-1])

        # Feature dimension
        self.feature_dim = 2048

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: input image tensor (B, 3, 224, 224)
        Returns:
            features: feature vectors (B, 2048)
        """
        x = self.features(x)          # (B, 2048, 1, 1)
        x = x.view(x.size(0), -1)    # (B, 2048)
        return x


def get_feature_extractor(cfg, device):
    """Create and return the feature extraction model."""
    weight_opt = cfg['model']['pretrained']
    model = ResNet50FeatureExtractor(
        weight_option=weight_opt
    )
    model = model.to(device)
    model.eval()  # Eval mode: disable dropout and BN updates
    return model
