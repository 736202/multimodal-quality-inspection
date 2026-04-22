from __future__ import annotations

import torch
import pytest

from mqi.models.image import ImageBackbone, ImageClassifier
from mqi.models.sensors import SensorEncoder, SensorClassifier
from mqi.models.multimodal import MultimodalClassifier


BATCH = 4
H = W = 64  # small spatial size for fast tests


# ── Image models ─────────────────────────────────────────────────────────────

def test_image_backbone_output_shape():
    model = ImageBackbone(pretrained=False)
    x = torch.randn(BATCH, 3, H, W)
    out = model(x)
    assert out.shape == (BATCH, model.output_dim)


def test_image_classifier_output_shape():
    model = ImageClassifier(pretrained=False)
    x = torch.randn(BATCH, 3, H, W)
    out = model(x)
    assert out.shape == (BATCH,)


def test_image_classifier_output_is_unbounded():
    model = ImageClassifier(pretrained=False)
    x = torch.randn(BATCH, 3, H, W)
    out = model(x)
    # Logits should not be squashed to [0, 1]
    assert out.dtype == torch.float32


# ── Sensor models ─────────────────────────────────────────────────────────────

def test_sensor_encoder_output_shape():
    model = SensorEncoder()
    x = torch.randn(BATCH, 6)
    out = model(x)
    assert out.shape == (BATCH, model.output_dim)


def test_sensor_classifier_output_shape():
    model = SensorClassifier()
    x = torch.randn(BATCH, 6)
    out = model(x)
    assert out.shape == (BATCH,)


def test_sensor_encoder_single_sample_batch_norm():
    """BatchNorm should not crash with batch_size=1 in eval mode."""
    model = SensorClassifier()
    model.eval()
    x = torch.randn(1, 6)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1,)


# ── Multimodal model ──────────────────────────────────────────────────────────

def test_multimodal_classifier_output_shape():
    model = MultimodalClassifier(pretrained=False)
    images = torch.randn(BATCH, 3, H, W)
    sensors = torch.randn(BATCH, 6)
    out = model(images, sensors)
    assert out.shape == (BATCH,)


def test_multimodal_fusion_dim():
    model = MultimodalClassifier(pretrained=False)
    expected = model.image_encoder.output_dim + model.sensor_encoder.output_dim
    assert expected == 512 + 64


def test_multimodal_gradient_flows():
    model = MultimodalClassifier(pretrained=False)
    images = torch.randn(BATCH, 3, H, W, requires_grad=False)
    sensors = torch.randn(BATCH, 6, requires_grad=False)
    logits = model(images, sensors)
    loss = logits.sum()
    loss.backward()
    # At least one parameter should have a gradient
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0
