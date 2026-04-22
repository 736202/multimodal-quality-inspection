from __future__ import annotations

from pathlib import Path

from mqi.data.catalog import SampleRecord
from mqi.data.synthetic_sensors import (
    SENSOR_COLUMNS,
    generate_sensor_features,
    generate_sensor_table,
)


def _record(label: int, idx: int = 0) -> SampleRecord:
    return SampleRecord(
        sample_id=f"sample_{idx:04d}",
        image_path=Path(f"/fake/{idx}.jpeg"),
        label=label,
        split="train",
    )


def test_generate_sensor_features_returns_all_columns():
    sample = generate_sensor_features(_record(0), base_seed=42)
    assert set(sample.values.keys()) == set(SENSOR_COLUMNS)


def test_generate_sensor_features_deterministic():
    record = _record(1, idx=7)
    a = generate_sensor_features(record, base_seed=42)
    b = generate_sensor_features(record, base_seed=42)
    assert a.values == b.values


def test_generate_sensor_features_differs_by_base_seed():
    record = _record(0, idx=3)
    a = generate_sensor_features(record, base_seed=0)
    b = generate_sensor_features(record, base_seed=1)
    assert a.values != b.values


def test_defect_has_higher_mean_temperature_on_average():
    """Process-anomaly defects dominate (~82 %), so the mean must be higher."""
    n = 400
    ok_temps  = [generate_sensor_features(_record(0, idx=i), base_seed=0).values["mold_temperature_c"] for i in range(n)]
    def_temps = [generate_sensor_features(_record(1, idx=i), base_seed=0).values["mold_temperature_c"] for i in range(n)]
    assert sum(def_temps) / n > sum(ok_temps) / n


def test_distributions_are_not_trivially_separated():
    """The inter-class SNR per variable must be < 2 (non-trivial separation).

    A SNR >= 2 on six independent variables yields Mahalanobis distance >= 4.4,
    making sensor-only classification near-perfect — an unrealistic setting.
    """
    n = 400
    ok_vals  = [generate_sensor_features(_record(0, idx=i), base_seed=0).values["mold_temperature_c"] for i in range(n)]
    def_vals = [generate_sensor_features(_record(1, idx=i), base_seed=0).values["mold_temperature_c"] for i in range(n)]
    ok_mean  = sum(ok_vals)  / n
    def_mean = sum(def_vals) / n
    ok_std   = (sum((v - ok_mean) ** 2 for v in ok_vals) / n) ** 0.5
    snr = abs(def_mean - ok_mean) / ok_std
    assert snr < 2.0, f"SNR={snr:.2f} is too high — distributions are trivially separated"


def test_pressure_positively_correlated_with_temperature():
    """Injection pressure must be positively correlated with mould temperature."""
    n = 300
    pairs = [
        (
            generate_sensor_features(_record(0, idx=i), base_seed=0).values["mold_temperature_c"],
            generate_sensor_features(_record(0, idx=i), base_seed=0).values["injection_pressure_bar"],
        )
        for i in range(n)
    ]
    t_vals = [p[0] for p in pairs]
    p_vals = [p[1] for p in pairs]
    t_mean = sum(t_vals) / n
    p_mean = sum(p_vals) / n
    cov = sum((t - t_mean) * (p - p_mean) for t, p in zip(t_vals, p_vals)) / n
    assert cov > 0, "Pressure should be positively correlated with temperature"


def test_generate_sensor_table_keys_match_records():
    records = [_record(label=i % 2, idx=i) for i in range(10)]
    table = generate_sensor_table(records, base_seed=42)
    assert set(table.keys()) == {r.sample_id for r in records}


def test_sensor_values_within_physical_bounds():
    """All generated values must stay within physically plausible bounds."""
    records = [_record(label=i % 2, idx=i) for i in range(100)]
    table = generate_sensor_table(records, base_seed=42)
    for values in table.values():
        assert 168.0 <= values["mold_temperature_c"]    <= 242.0
        assert  38.0 <= values["injection_pressure_bar"] <= 102.0
        assert  16.0 <= values["cycle_time_s"]           <=  45.0
        assert   0.05 <= values["vibration_mm_s"]         <=  0.92
        assert  18.0 <= values["humidity_pct"]            <=  82.0
        assert 680.0 <= values["rotation_speed_rpm"]     <= 1400.0


def test_class_distributions_overlap():
    """Defect and OK distributions must overlap sufficiently (non-trivial separation)."""
    n = 300
    ok_temps  = sorted(generate_sensor_features(_record(0, idx=i), base_seed=1).values["mold_temperature_c"] for i in range(n))
    def_temps = sorted(generate_sensor_features(_record(1, idx=i), base_seed=1).values["mold_temperature_c"] for i in range(n))
    # The 75th percentile of OK must exceed the 25th percentile of defects
    ok_q75  = ok_temps[int(0.75 * n)]
    def_q25 = def_temps[int(0.25 * n)]
    assert ok_q75 > def_q25, "Distributions should overlap — problem is too easy otherwise"
