# Multimodal Quality Inspection — Industrial Casting Defect Detection

> Late-fusion multimodal deep learning system combining **computer vision** (ResNet18) and **process sensors** (MLP) for automated defect detection in industrial casting production.

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4+-ee4c2c.svg)](https://pytorch.org/)
[![Tests](https://github.com/736202/Projet1/actions/workflows/test.yml/badge.svg)](https://github.com/736202/Projet1/actions)

---

## Quick Start

```bash
# 1. Install
pip install -r requirements.txt && pip install -e .

# 2. Download dataset → extract to casting_512x512/  (see data/README.md)

# 3. Run full pipeline (train → evaluate → robustness → stats)
python scripts/run_pipeline.py --epochs 20
```

**Results** are written to `outputs/final_report/` — 14 figures (PNG) + `robustness_results.json` + `statistical_validation.json`.  
**Report PDF**: [`report/multimodal_quality_inspection_report.pdf`](report/multimodal_quality_inspection_report.pdf)

---

## Industrial Problem

Surface defects in die-cast metal parts (porosity, cracks, shrinkage) cause downstream failures and costly recalls. Manual visual inspection is slow, operator-dependent, and misses internal or process-level anomalies. This project builds a production-ready automated inspection pipeline that fuses two complementary data sources:

- **Camera images** (512×512 px) — detect surface defects invisible to sensors
- **6 process sensors** — temperature, pressure, cycle time, vibration, humidity, rotation speed — detect process anomalies invisible to the camera

The key industrial claim: **neither modality alone is sufficient**. Fusion makes the system robust to partial failures of either source.

---

## Architecture

```
Image 512×512 ──► ResNet18 (fine-tuned) ──► z_img ∈ ℝ⁵¹²  ─┐
                                                               ├─► [concat] ─► MLP Head ──► P(defect)
6 Sensors ───────► MLP Encoder ──────────► z_sen ∈ ℝ⁶⁴    ─┘
```

**Late fusion**: both modalities are encoded independently, concatenated (ℝ⁵⁷⁶), then classified by a 3-layer MLP head. The image backbone is pre-trained on ImageNet with 3 frozen epochs followed by full fine-tuning.

Sensor data is **entirely synthetic** with industrial realism: hidden defects (~18 % of defective parts with normal process signatures), inter-sensor correlations (pressure ↔ temperature ρ≈0.35), bimodal vibration, Laplace-distributed humidity, and 5 % transient measurement spikes.

---

## Results

### Baseline (clean test set, n=195)

| Model | Accuracy | F1 | Recall | AUC-ROC |
|---|---|---|---|---|
| Image only | 1.000 | 1.000 | 1.000 | 1.000 |
| Sensors only | 0.887 | 0.852 | 0.829 | 0.883 |
| **Multimodal fusion** | **1.000** | **1.000** | **1.000** | **1.000** |

*Sensors alone cannot exceed ~82 % recall due to hidden defects (~18 %) with normal process signatures — a theoretical ceiling independent of architecture.*

### Robustness under image degradation

The fusion value becomes quantifiable when images are corrupted (noise, blur, occlusions):

| Degradation level | Image only (F1) | Fusion (F1) | Gain |
|---|---|---|---|
| None | 1.000 | 1.000 | — |
| Moderate (noise σ=0.05, brightness ±25 %) | 0.987 | 0.996 | +0.9 pts |
| **Severe (noise σ=0.12, blur, occlusion)** | **0.883** | **0.974** | **+9.1 pts** |

### Robustness to missing sensors

| Missing rate | Sensors only (F1) | Fusion (F1) |
|---|---|---|
| 0 % | 0.852 | 1.000 |
| 20 % | 0.820 | 1.000 |
| **40 %** | **0.792** | **1.000** |

Fusion maintains F1 = 1.000 at 40 % sensor dropout — the image branch compensates fully.

### Statistical validation

Bootstrap 95 % CI (B=1 000 resamples) — sensor model: F1 = 0.851 [0.796 – 0.899].  
McNemar test image vs sensors: χ²=29.03, **p < 0.001** — error profiles are significantly different, confirming true complementarity.

---

## Business Impact

| Criterion | Sensors only | Image only | Fusion |
|---|---|---|---|
| Min. cost (C_FN = 10×C_FP) | **74** (irreducible) | 0 | **0** |
| Robust to sensor dropout | No | Yes | **Yes** |
| Robust to image degradation | Yes | No | **Yes** |
| Inference latency (CPU) | 0.003 ms/sample | 15.5 ms | **16.8 ms (+8 %)** |

The multimodal overhead is negligible (+8 % latency) while eliminating the irreducible false-negative cost of sensor-only classification.

---

## Interpretability

- **SHAP (sensors)**: KernelExplainer with 50 k-means background clusters. Vibration (ΔAUC=−0.014) and mold temperature (ΔAUC=−0.007) are the most important features.
- **Grad-CAM (image)**: Activations consistently focus on surface defect regions (porosity, cracks, shrinkage), validating that the model detects actual defects rather than background artefacts.

---

## Repository Structure

```
├── src/mqi/                        # Core library (installable package)
│   ├── data/                       #   catalog, datasets, synthetic sensors, degradation
│   ├── models/                     #   ImageBackbone, SensorEncoder, MultimodalClassifier
│   ├── training/                   #   training loop, early stopping, metrics
│   └── utils/                      #   reproducibility, logging
│
├── scripts/
│   ├── train.py                    # Train one mode: image | sensor | multimodal
│   ├── evaluate.py                 # Evaluate + Grad-CAM visualisation
│   ├── robustness_eval.py          # Degradation, missing sensors, ablation, cost
│   ├── statistical_validation.py   # Bootstrap CI + McNemar tests
│   ├── build_report_assets.py      # Generate all 14 report figures
│   ├── generate_latex_report.py    # Assemble report from template + results
│   └── run_pipeline.py             # Full pipeline: train → eval → robustness → report
│
├── tests/                          # Unit tests (pytest)
│
├── outputs/
│   ├── {image,sensor,multimodal}/  # config.json + test_metrics.json per model
│   └── final_report/               # All figures + summary JSONs
│
├── report/
│   └── multimodal_quality_inspection_report.pdf   # Full technical report
│
├── data/README.md                  # Dataset download instructions
└── requirements.txt
```

---

## Installation

```bash
git clone https://github.com/736202/Projet1.git
cd Projet1
python -m venv .venv
# Linux/Mac:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

pip install -r requirements.txt
pip install -e .
```

**Dataset**: download [Casting Product Image Data](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product) from Kaggle and extract to `casting_512x512/`. See `data/README.md` for details.

> **Model weights** (`outputs/*/best_model.pt`) are excluded from git due to size (~44 MB each). Retrain with `scripts/run_pipeline.py` or contact the author.

---

## Usage

### Full pipeline (train → evaluate → robustness → report)
```bash
python scripts/run_pipeline.py --epochs 20
```

### Train a single model
```bash
python scripts/train.py --mode image       # or sensor / multimodal
```

### Evaluate + Grad-CAM
```bash
python scripts/evaluate.py --run-dir outputs/image --gradcam
```

### Robustness experiments + statistical tests
```bash
python scripts/robustness_eval.py
python scripts/statistical_validation.py
```

### Regenerate report figures and PDF
```bash
python scripts/build_report_assets.py
python scripts/generate_latex_report.py
cd report
pdflatex multimodal_quality_inspection_report.tex
pdflatex multimodal_quality_inspection_report.tex
```

### Run tests
```bash
pytest tests/ -v
```

---

## Report

The full technical report (methods, experiments, interpretability, statistical validation) is available as a PDF:

**[`report/multimodal_quality_inspection_report.pdf`](report/multimodal_quality_inspection_report.pdf)**

---

## Tech Stack

| Component | Library |
|---|---|
| Deep learning | PyTorch 2.4, torchvision |
| Data / metrics | scikit-learn, pandas, numpy |
| Visualisation | matplotlib, seaborn |
| Interpretability | SHAP 0.51, Grad-CAM (custom) |
| Tests | pytest |

---

*Moad Hamidoune — 2026*
