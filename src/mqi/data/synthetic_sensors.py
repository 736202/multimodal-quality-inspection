from __future__ import annotations

import math
from dataclasses import dataclass
import hashlib
import random

from mqi.data.catalog import SampleRecord


SENSOR_COLUMNS = [
    "mold_temperature_c",
    "injection_pressure_bar",
    "cycle_time_s",
    "vibration_mm_s",
    "humidity_pct",
    "rotation_speed_rpm",
]


@dataclass(slots=True)
class SensorSample:
    """Container for the synthetic sensor readings of a single sample.

    Attributes
    ----------
    sample_id:
        Identifier matching the corresponding :class:`~mqi.data.catalog.SampleRecord`.
    values:
        Mapping from sensor name to its raw (unscaled) float value.
    """

    sample_id: str
    values: dict[str, float]


def _seed_from_sample(sample_id: str, base_seed: int) -> int:
    """Derive a deterministic integer seed from a sample ID and a base seed.

    Uses SHA-256 to guarantee that each sample always receives the same sensor
    values regardless of iteration order, while remaining independent across
    different base seeds.
    """
    digest = hashlib.sha256(f"{sample_id}-{base_seed}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _draw_normal(rng: random.Random, mean: float, std: float, lower: float, upper: float) -> float:
    """Draw a truncated-Gaussian sample clipped to [lower, upper]."""
    return _clip(rng.gauss(mean, std), lower, upper)


def _draw_laplace(rng: random.Random, mean: float, scale: float, lower: float, upper: float) -> float:
    """Sample from a Laplace distribution via inverse CDF, clipped to [lower, upper].

    Heavier tails than Gaussian; used for humidity to reflect atmospheric
    measurement variability.
    """
    u = rng.random() - 0.5
    if u == 0.0:
        return _clip(mean, lower, upper)
    sign = 1.0 if u > 0 else -1.0
    value = mean - sign * scale * math.log(1.0 - 2.0 * abs(u))
    return _clip(value, lower, upper)


def generate_sensor_features(record: SampleRecord, base_seed: int) -> SensorSample:
    """Generate realistic synthetic sensor readings for one casting sample.

    Three regimes are modelled to reflect real industrial variability:

    * **Process anomaly** (~82 % of defective parts): all six variables deviate
      from nominal with moderate overlap of the OK distribution.  The
      inter-class mean separation is intentionally small (~1 sigma per variable)
      so that sensor-only classification is non-trivial.
    * **Hidden defect** (~18 % of defective parts): the defect originates from a
      material or geometric flaw not reflected in the process variables; sensor
      readings are statistically indistinguishable from conforming parts.  These
      samples can only be caught by the image branch.
    * **Conforming part**: nominal distributions with controlled noise.

    Physical couplings are introduced through inter-sensor correlations:

    * Injection pressure is positively correlated with mould temperature
      (coefficient ~0.35).
    * Rotation speed is positively correlated with vibration amplitude.

    Humidity follows a Laplace distribution (heavier tails than Gaussian) to
    reflect atmospheric measurement variability.  A 5 % probability of a
    transient measurement spike is applied to mould temperature and injection
    pressure to simulate sensor faults.

    Vibration for process-anomaly parts is bimodal: 28 % of such parts show a
    high-amplitude burst caused by worn tooling.

    The generation is fully deterministic: the same (sample_id, base_seed) pair
    always produces identical values regardless of iteration order.

    Parameters
    ----------
    record:
        The sample for which sensor features are generated.
    base_seed:
        Pipeline-level seed ensuring global reproducibility across runs.

    Returns
    -------
    SensorSample
        Six sensor values keyed by the names in :data:`SENSOR_COLUMNS`.
    """
    rng = random.Random(_seed_from_sample(record.sample_id, base_seed))
    is_defect = record.label == 1

    # 18% of defective parts are "hidden defects": the manufacturing process ran
    # normally, but the casting has an internal flaw.  Process sensors give no
    # indication — only the image branch can detect these cases.
    is_process_anomaly = is_defect and (rng.random() >= 0.18)

    # ── Mould temperature (°C) ──────────────────────────────────────────────
    # Inter-class delta reduced to 12 °C (vs 16 °C in the naive model);
    # standard deviation increased to create genuine class overlap.
    T_mean = 210.0 if is_process_anomaly else 198.0
    T_std  = 12.0  if is_process_anomaly else  9.5
    mold_temperature = _draw_normal(rng, T_mean, T_std, 168.0, 242.0)
    # 5% transient spike: simulates a brief sensor fault
    if rng.random() < 0.05:
        mold_temperature = _clip(
            mold_temperature + rng.uniform(-20.0, 20.0), 168.0, 242.0
        )

    # ── Injection pressure (bar) — positively correlated with temperature ───
    # Physical coupling: hotter moulds require higher injection pressure.
    # Pearson correlation coefficient ≈ 0.35.
    P_base   = 79.0 if is_process_anomaly else 67.0
    P_std    = 10.5 if is_process_anomaly else  8.5
    T_ref    = 210.0 if is_process_anomaly else 198.0
    pressure_corr = 0.35 * (mold_temperature - T_ref)
    injection_pressure = _draw_normal(
        rng, P_base + pressure_corr, P_std, 38.0, 102.0
    )
    # 5% transient spike
    if rng.random() < 0.05:
        injection_pressure = _clip(
            injection_pressure + rng.uniform(-15.0, 15.0), 38.0, 102.0
        )

    # ── Cycle time (s) ──────────────────────────────────────────────────────
    # Reduced delta: 3.5 s (was 5 s); higher noise to increase overlap.
    cycle_time = _draw_normal(
        rng,
        mean=33.0 if is_process_anomaly else 29.5,
        std=5.0   if is_process_anomaly else  4.2,
        lower=16.0,
        upper=45.0,
    )

    # ── Vibration (mm/s) — bimodal for process-anomaly defects ─────────────
    # 28% of process-anomaly parts show a high-amplitude vibration burst
    # (worn cutting tool or loose fixture).  This bimodal distribution is more
    # realistic than a single elevated Gaussian.
    if is_process_anomaly and rng.random() < 0.28:
        vibration = _draw_normal(rng, 0.66, 0.09, 0.05, 0.92)
    else:
        vibration = _draw_normal(
            rng,
            mean=0.43 if is_process_anomaly else 0.22,
            std=0.15  if is_process_anomaly else  0.11,
            lower=0.05,
            upper=0.92,
        )

    # ── Humidity (%) — Laplace distribution ─────────────────────────────────
    # Atmospheric humidity follows a heavier-tailed distribution than Gaussian,
    # leading to more frequent extreme values near the bounds.
    humidity = _draw_laplace(
        rng,
        mean=56.0  if is_process_anomaly else 47.0,
        scale=7.5  if is_process_anomaly else  6.5,
        lower=18.0,
        upper=82.0,
    )

    # ── Rotation speed (rpm) — positively correlated with vibration ─────────
    # Higher vibration (worn tooling) is typically associated with higher
    # effective rotation speed due to chatter.
    R_mean   = 1150.0 if is_process_anomaly else 1010.0
    R_std    =  105.0 if is_process_anomaly else   90.0
    V_ref    =   0.43 if is_process_anomaly else   0.22
    rot_corr = 60.0 * (vibration - V_ref)
    rotation_speed = _clip(
        _draw_normal(rng, R_mean + rot_corr, R_std, 680.0, 1400.0),
        680.0, 1400.0,
    )

    return SensorSample(
        sample_id=record.sample_id,
        values={
            "mold_temperature_c":     mold_temperature,
            "injection_pressure_bar": injection_pressure,
            "cycle_time_s":           cycle_time,
            "vibration_mm_s":         vibration,
            "humidity_pct":           humidity,
            "rotation_speed_rpm":     rotation_speed,
        },
    )


def generate_sensor_table(records: list[SampleRecord], base_seed: int) -> dict[str, dict[str, float]]:
    """Generate sensor features for an entire catalogue in one call.

    Parameters
    ----------
    records:
        Full list of split records (all partitions).
    base_seed:
        Seed forwarded to :func:`generate_sensor_features`.

    Returns
    -------
    dict
        ``{sample_id: {sensor_name: value}}`` mapping ready for dataset classes.
    """
    table: dict[str, dict[str, float]] = {}
    for record in records:
        sample = generate_sensor_features(record, base_seed)
        table[sample.sample_id] = sample.values
    return table
