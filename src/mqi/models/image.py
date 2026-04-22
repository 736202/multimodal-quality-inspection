from __future__ import annotations

import warnings

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


class ImageBackbone(nn.Module):
    """ResNet18 feature extractor with the classification head removed.

    The final fully-connected layer is replaced by an identity mapping so the
    module outputs a 512-dimensional embedding vector per image.  When used as
    part of the multimodal classifier, this embedding is concatenated with the
    sensor embedding before the fusion head.

    Parameters
    ----------
    pretrained:
        If ``True``, load ImageNet-pretrained weights.  Falls back to random
        initialisation if the weights cannot be downloaded.
    """

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        try:
            backbone = resnet18(weights=weights)
        except Exception as exc:
            warnings.warn(
                f"Falling back to randomly initialized ResNet18 because pretrained weights are unavailable: {exc}",
                stacklevel=2,
            )
            backbone = resnet18(weights=None)
        num_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.output_dim = num_features

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Extract a 512-dim feature vector from a batch of images.

        Parameters
        ----------
        image:
            Float tensor of shape ``(B, 3, H, W)``.

        Returns
        -------
        torch.Tensor
            Shape ``(B, 512)``.
        """
        return self.backbone(image)


class ImageClassifier(nn.Module):
    """Binary classifier built on top of :class:`ImageBackbone`.

    Appends a single linear layer to map 512-dim features to a scalar logit
    used with ``BCEWithLogitsLoss``.

    Parameters
    ----------
    pretrained:
        Forwarded to :class:`ImageBackbone`.
    """

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        self.encoder = ImageBackbone(pretrained=pretrained)
        self.head = nn.Linear(self.encoder.output_dim, 1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Return an unbounded scalar logit per sample.

        Parameters
        ----------
        image:
            Float tensor of shape ``(B, 3, H, W)``.

        Returns
        -------
        torch.Tensor
            Shape ``(B,)``.
        """
        features = self.encoder(image)
        return self.head(features).squeeze(1)
