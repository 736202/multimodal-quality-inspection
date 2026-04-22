from __future__ import annotations

import torch
from torch import nn

from mqi.models.image import ImageBackbone
from mqi.models.sensors import SensorEncoder


class MultimodalClassifier(nn.Module):
    """Late-fusion multimodal classifier combining image and sensor modalities.

    The architecture follows a **late-fusion** strategy: both modalities are
    encoded independently, their embeddings are concatenated, and a shared MLP
    head produces the final binary prediction.

    Fusion equation::

        z = [z_image ; z_sensor]        (concatenation along feature dim)
        logit = MLP_head(z)

    Parameters
    ----------
    pretrained:
        Whether to initialise the image backbone with ImageNet weights.
    sensor_dropout:
        Dropout applied inside :class:`~mqi.models.sensors.SensorEncoder`.
    fusion_dropout_1:
        Dropout after the first fusion layer (512+64 → 256).
    fusion_dropout_2:
        Dropout after the second fusion layer (256 → 128).
    """

    def __init__(
        self,
        pretrained: bool = True,
        sensor_dropout: float = 0.3,
        fusion_dropout_1: float = 0.4,
        fusion_dropout_2: float = 0.3,
    ) -> None:
        super().__init__()
        self.image_encoder = ImageBackbone(pretrained=pretrained)
        self.sensor_encoder = SensorEncoder(dropout=sensor_dropout)

        fusion_input_dim = self.image_encoder.output_dim + self.sensor_encoder.output_dim
        self.classifier = nn.Sequential(
            nn.Linear(fusion_input_dim, 256),
            nn.ReLU(),
            nn.Dropout(fusion_dropout_1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(fusion_dropout_2),
            nn.Linear(128, 1),
        )

    def forward(self, image: torch.Tensor, sensors: torch.Tensor) -> torch.Tensor:
        """Produce a scalar logit from the fused multimodal representation.

        Parameters
        ----------
        image:
            Float tensor of shape ``(B, 3, H, W)``.
        sensors:
            Float32 tensor of shape ``(B, 6)``.

        Returns
        -------
        torch.Tensor
            Shape ``(B,)`` — unbounded logits for ``BCEWithLogitsLoss``.
        """
        image_features = self.image_encoder(image)
        sensor_features = self.sensor_encoder(sensors)
        fused = torch.cat([image_features, sensor_features], dim=1)
        return self.classifier(fused).squeeze(1)
