from __future__ import annotations

import random

import torch
import torch.nn.functional as F


# ── Corruption levels ──────────────────────────────────────────────────────────
# Each level is a dict of parameters forwarded to corrupt_image().
# "none"     : identity — standard evaluation baseline
# "moderate" : realistic camera/lighting variation
# "severe"   : harsh conditions (dust, vibration blur, strong under-exposure)

DEGRADATION_LEVELS = ["none", "moderate", "severe"]


def corrupt_image(
    tensor: torch.Tensor,
    level: str,
    seed: int | None = None,
) -> torch.Tensor:
    """Apply reproducible image corruption to a normalised CHW or BCHW tensor.

    All operations are performed in-place on a cloned tensor so the original
    is never modified.  The function is deterministic when `seed` is provided.

    Parameters
    ----------
    tensor:
        Float32 tensor of shape ``(3, H, W)`` or ``(B, 3, H, W)``, already
        normalised with ImageNet statistics (mean/std applied).
    level:
        One of ``"none"``, ``"moderate"``, ``"severe"``.
    seed:
        Optional integer seed for reproducibility.

    Returns
    -------
    torch.Tensor
        Corrupted tensor with the same shape and dtype as the input.
    """
    if level == "none":
        return tensor

    rng = random.Random(seed)
    out = tensor.clone()
    batched = out.ndim == 4

    if not batched:
        out = out.unsqueeze(0)   # (1, C, H, W)

    B, C, H, W = out.shape

    if level == "moderate":
        # ── Gaussian noise (σ ≈ 0.05 in normalised space) ───────────────────
        noise = torch.randn_like(out) * 0.05
        out = out + noise

        # ── Brightness / contrast jitter ─────────────────────────────────────
        for b in range(B):
            alpha = rng.uniform(0.75, 1.25)   # contrast
            beta  = rng.uniform(-0.15, 0.15)  # brightness
            out[b] = out[b] * alpha + beta

    elif level == "severe":
        # ── Strong Gaussian noise (σ ≈ 0.12) ────────────────────────────────
        noise = torch.randn_like(out) * 0.12
        out = out + noise

        # ── Gaussian blur (simulates vibration / defocus) ────────────────────
        kernel_size = 5
        sigma = 1.5
        # Build 2-D Gaussian kernel
        ax = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        gauss1d = torch.exp(-ax ** 2 / (2 * sigma ** 2))
        gauss1d /= gauss1d.sum()
        kernel = gauss1d.outer(gauss1d)
        kernel = kernel.expand(C, 1, kernel_size, kernel_size)
        out = F.conv2d(
            out,
            kernel.to(out.device),
            padding=kernel_size // 2,
            groups=C,
        )

        # ── Random rectangular occlusion (simulates dust / obstruction) ──────
        occ_h = H // 5   # ~20 % of height
        occ_w = W // 5
        for b in range(B):
            y0 = rng.randint(0, H - occ_h)
            x0 = rng.randint(0, W - occ_w)
            out[b, :, y0:y0 + occ_h, x0:x0 + occ_w] = 0.0

        # ── Strong brightness / contrast jitter ──────────────────────────────
        for b in range(B):
            alpha = rng.uniform(0.5, 1.5)
            beta  = rng.uniform(-0.3, 0.3)
            out[b] = out[b] * alpha + beta

    if not batched:
        out = out.squeeze(0)

    return out


def mask_sensors(
    sensors: torch.Tensor,
    missing_rate: float,
    seed: int | None = None,
) -> torch.Tensor:
    """Randomly zero-out a fraction of sensor features to simulate missing data.

    Each feature of each sample is independently zeroed with probability
    `missing_rate`.  This models sensor dropouts (cable fault, calibration
    timeout) common in industrial PLC environments.

    Parameters
    ----------
    sensors:
        Float32 tensor of shape ``(B, n_features)``.
    missing_rate:
        Probability in ``[0, 1)`` that any given feature is zeroed.
    seed:
        Optional seed for reproducibility.

    Returns
    -------
    torch.Tensor
        Copy of `sensors` with some entries set to 0.
    """
    if missing_rate <= 0.0:
        return sensors
    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(seed)
    mask = torch.bernoulli(
        torch.full(sensors.shape, 1.0 - missing_rate), generator=gen
    )
    return sensors * mask
