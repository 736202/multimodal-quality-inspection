from __future__ import annotations

import torch
from torch import nn


class SensorEncoder(nn.Module):
    """Multi-layer perceptron that maps a raw sensor vector to an embedding.

    Architecture: Linear → ReLU → BatchNorm → Dropout → Linear → ReLU → BatchNorm.
    BatchNorm stabilises training on small tabular inputs; Dropout regularises
    the hidden representation.

    Parameters
    ----------
    input_dim:
        Number of input sensor features (default: 6).
    hidden_dim:
        Width of the first hidden layer.
    output_dim:
        Embedding dimension exposed to downstream heads.
    dropout:
        Dropout probability applied after the first hidden layer.
    """

    def __init__(
        self,
        input_dim: int = 6,
        hidden_dim: int = 128,
        output_dim: int = 64,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.BatchNorm1d(output_dim),
        )
        self.output_dim = output_dim

    def forward(self, sensors: torch.Tensor) -> torch.Tensor:
        """Map a batch of sensor vectors to embeddings.

        Parameters
        ----------
        sensors:
            Float32 tensor of shape ``(B, input_dim)``.

        Returns
        -------
        torch.Tensor
            Shape ``(B, output_dim)``.
        """
        return self.network(sensors)


class SensorClassifier(nn.Module):
    """Binary classifier built on top of :class:`SensorEncoder`.

    Appends a linear head that maps the 64-dim embedding to a scalar logit.

    Parameters
    ----------
    dropout:
        Forwarded to :class:`SensorEncoder`.
    """

    def __init__(self, dropout: float = 0.3) -> None:
        super().__init__()
        self.encoder = SensorEncoder(dropout=dropout)
        self.head = nn.Linear(self.encoder.output_dim, 1)

    def forward(self, sensors: torch.Tensor) -> torch.Tensor:
        """Return an unbounded scalar logit per sample.

        Parameters
        ----------
        sensors:
            Float32 tensor of shape ``(B, 6)``.

        Returns
        -------
        torch.Tensor
            Shape ``(B,)``.
        """
        features = self.encoder(sensors)
        return self.head(features).squeeze(1)
