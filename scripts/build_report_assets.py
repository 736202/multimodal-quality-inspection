from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns


MODES = ["image", "sensor", "multimodal"]
METRIC_COLUMNS = ["accuracy", "precision", "recall", "f1", "auc_roc"]

# ── Palette professionnelle cohérente ─────────────────────────────────────────
PALETTE = {
    "image":      "#2C6FAC",   # bleu académique
    "sensor":     "#2E8B57",   # vert forêt
    "multimodal": "#C0392B",   # rouge sombre
}
MODE_LABELS = {
    "image":      "Image seule",
    "sensor":     "Capteurs seuls",
    "multimodal": "Fusion multimodale",
}
CLASS_COLORS = {0: "#3498DB", 1: "#E74C3C"}   # ok=bleu, défaut=rouge

SENSOR_LABELS = {
    "mold_temperature_c":    "Température moule (°C)",
    "injection_pressure_bar":"Pression injection (bar)",
    "cycle_time_s":          "Durée de cycle (s)",
    "vibration_mm_s":        "Vibration (mm/s)",
    "humidity_pct":          "Humidité (%)",
    "rotation_speed_rpm":    "Vitesse rotation (tr/min)",
}


# ── Style global ──────────────────────────────────────────────────────────────

def apply_style() -> None:
    """Configure matplotlib/seaborn for a clean academic appearance."""
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    matplotlib.rcParams.update({
        "figure.dpi":         150,
        "savefig.dpi":        220,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.08,
        "font.family":        "serif",
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.grid":          True,
        "axes.grid.axis":     "y",
        "grid.alpha":         0.35,
        "grid.linestyle":     "--",
        "legend.framealpha":  0.9,
        "legend.edgecolor":   "#cccccc",
        "xtick.direction":    "out",
        "ytick.direction":    "out",
    })


def save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path)
    plt.close(fig)
    print(f"  -> {path.name}")


# ── I/O helpers ───────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build consolidated figures and summary files.")
    parser.add_argument("--outputs-dir", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument("--report-dir",  type=Path, default=PROJECT_ROOT / "outputs" / "final_report")
    return parser.parse_args()


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as h:
        return json.load(h)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_mode_artifacts(outputs_dir: Path) -> dict[str, dict]:
    return {
        mode: {
            "run_dir":    outputs_dir / mode,
            "config":     read_json(outputs_dir / mode / "config.json"),
            "training":   read_json(outputs_dir / mode / "training_summary.json"),
            "validation": read_json(outputs_dir / mode / "val_metrics.json"),
            "test":       read_json(outputs_dir / mode / "test_metrics.json"),
        }
        for mode in MODES
    }


def load_manifest(outputs_dir: Path) -> pd.DataFrame:
    for mode in MODES:
        p = outputs_dir / mode / "dataset_manifest.csv"
        if p.exists():
            return pd.read_csv(p)
    raise FileNotFoundError("dataset_manifest.csv not found in outputs")


def build_summary_rows(artifacts: dict[str, dict]) -> list[dict]:
    rows = []
    for mode, payload in artifacts.items():
        m = payload["test"]["metrics"]
        rows.append({
            "mode":             mode,
            "accuracy":         m["accuracy"],
            "precision":        m["precision"],
            "recall":           m["recall"],
            "f1":               m["f1"],
            "auc_roc":          m["auc_roc"],
            "threshold":        payload["test"]["threshold_selection"]["selected_threshold"],
            "epochs_ran":       payload["training"]["epochs_ran"],
            "duration_seconds": payload["training"]["duration_seconds"],
        })
    return rows


def export_summary(rows: list[dict], report_dir: Path) -> None:
    ensure_dir(report_dir)
    (report_dir / "comparison_summary.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    with (report_dir / "comparison_metrics.csv").open("w", encoding="utf-8", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


# ── Figure 1 : Distribution des classes ──────────────────────────────────────

def plot_class_distribution(manifest: pd.DataFrame, report_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle("Distribution des données", fontsize=13, fontweight="bold", y=1.01)

    # Gauche : global
    counts = manifest["label"].value_counts().sort_index()
    bars = axes[0].bar(
        ["Conforme (OK)", "Défectueux"],
        counts.values,
        color=[CLASS_COLORS[0], CLASS_COLORS[1]],
        edgecolor="white", linewidth=0.8, width=0.5,
    )
    for bar, count in zip(bars, counts.values):
        pct = 100 * count / counts.sum()
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + counts.max() * 0.01,
            f"{count:,}\n({pct:.1f}\u202f%)",
            ha="center", va="bottom", fontsize=10,
        )
    axes[0].set_title("Distribution globale", fontweight="bold")
    axes[0].set_ylabel("Nombre d'images")
    axes[0].set_ylim(0, counts.max() * 1.18)
    axes[0].set_axisbelow(True)

    # Droite : par partition
    split_order = ["train", "val", "test"]
    split_labels = {"train": "Entraînement", "val": "Validation", "test": "Test"}
    x = np.arange(len(split_order))
    width = 0.35
    for i, (label, color, name) in enumerate(
        [(0, CLASS_COLORS[0], "Conforme"), (1, CLASS_COLORS[1], "Défectueux")]
    ):
        vals = [manifest[(manifest["split"] == s) & (manifest["label"] == label)].shape[0] for s in split_order]
        rects = axes[1].bar(x + (i - 0.5) * width, vals, width,
                            label=name, color=color, edgecolor="white", linewidth=0.8)
        for rect, v in zip(rects, vals):
            axes[1].text(
                rect.get_x() + rect.get_width() / 2,
                rect.get_height() + 4,
                str(v), ha="center", va="bottom", fontsize=9,
            )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([split_labels[s] for s in split_order])
    axes[1].set_title("Répartition par partition", fontweight="bold")
    axes[1].set_ylabel("Nombre d'images")
    axes[1].legend(frameon=True)
    axes[1].set_axisbelow(True)

    fig.tight_layout()
    save(fig, report_dir / "class_distribution.png")


# ── Figure 2 : Distributions des capteurs (grille 2×3) ───────────────────────

def plot_sensor_distributions(manifest: pd.DataFrame, report_dir: Path) -> None:
    sensor_cols = list(SENSOR_LABELS.keys())
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    fig.suptitle("Distributions des capteurs synthétiques par classe", fontsize=13, fontweight="bold", y=1.01)

    for ax, col in zip(axes.flatten(), sensor_cols):
        ok_vals  = manifest.loc[manifest["label"] == 0, col]
        def_vals = manifest.loc[manifest["label"] == 1, col]

        bp = ax.boxplot(
            [ok_vals, def_vals],
            patch_artist=True,
            widths=0.45,
            medianprops=dict(color="white", linewidth=2),
            whiskerprops=dict(linewidth=1.2),
            capprops=dict(linewidth=1.2),
            flierprops=dict(marker="o", markersize=2, alpha=0.4),
        )
        for patch, color in zip(bp["boxes"], [CLASS_COLORS[0], CLASS_COLORS[1]]):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)

        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Conforme", "Défectueux"], fontsize=9)
        ax.set_title(SENSOR_LABELS[col], fontsize=9, fontweight="bold")
        ax.set_axisbelow(True)

        # Annoter les médianes
        for i, vals in enumerate([ok_vals, def_vals], start=1):
            ax.text(i, vals.median(), f" {vals.median():.1f}",
                    va="center", fontsize=7.5, color="black")

    legend_handles = [
        mpatches.Patch(color=CLASS_COLORS[0], label="Conforme (OK)"),
        mpatches.Patch(color=CLASS_COLORS[1], label="Défectueux"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.02), frameon=True)
    fig.tight_layout()
    save(fig, report_dir / "sensor_distributions.png")


# ── Figure 3 : Comparaison des métriques ─────────────────────────────────────

def plot_metrics_comparison(summary_df: pd.DataFrame, report_dir: Path) -> None:
    metric_labels = {
        "accuracy":  "Accuracy",
        "precision": "Précision",
        "recall":    "Rappel",
        "f1":        "F1-score",
        "auc_roc":   "AUC-ROC",
    }

    fig, ax = plt.subplots(figsize=(11, 5))
    n_metrics = len(METRIC_COLUMNS)
    n_models  = len(MODES)
    x = np.arange(n_metrics)
    total_width = 0.7
    width = total_width / n_models

    for i, mode in enumerate(MODES):
        row   = summary_df[summary_df["mode"] == mode].iloc[0]
        vals  = [row[m] for m in METRIC_COLUMNS]
        offset = (i - (n_models - 1) / 2) * width
        bars  = ax.bar(
            x + offset, vals, width,
            label=MODE_LABELS[mode],
            color=PALETTE[mode],
            edgecolor="white", linewidth=0.6,
        )
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.004,
                f"{v:.3f}",
                ha="center", va="bottom", fontsize=7.5, rotation=90,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([metric_labels[m] for m in METRIC_COLUMNS], fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_ylim(0, 1.13)
    ax.set_title("Comparaison des performances — jeu de test", fontsize=13, fontweight="bold")
    ax.legend(frameon=True, loc="lower right")
    ax.set_axisbelow(True)

    fig.tight_layout()
    save(fig, report_dir / "metrics_comparison.png")


# ── Figure 4 : Courbes ROC ────────────────────────────────────────────────────

def plot_roc_comparison(artifacts: dict[str, dict], report_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    for mode in MODES:
        curve = artifacts[mode]["test"]["curves"]["roc_curve"]
        auc   = artifacts[mode]["test"]["metrics"]["auc_roc"]
        ax.plot(
            curve["fpr"], curve["tpr"],
            color=PALETTE[mode],
            linewidth=2.2,
            label=f"{MODE_LABELS[mode]}  (AUC = {auc:.4f})",
        )

    ax.plot([0, 1], [0, 1], linestyle="--", color="#999999", linewidth=1.2, label="Aléatoire (AUC = 0.5)")
    ax.fill_between([0, 1], [0, 1], alpha=0.04, color="gray")

    ax.set_xlabel("Taux de faux positifs (FPR)", fontsize=11)
    ax.set_ylabel("Taux de vrais positifs (TPR)", fontsize=11)
    ax.set_title("Courbes ROC — comparaison des modèles", fontsize=12, fontweight="bold")
    ax.legend(frameon=True, loc="lower right", fontsize=9)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.04)
    ax.set_axisbelow(True)

    fig.tight_layout()
    save(fig, report_dir / "roc_comparison.png")


# ── Figure 5 : Seuils de décision ────────────────────────────────────────────

def plot_threshold_comparison(summary_df: pd.DataFrame, report_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))

    colors = [PALETTE[m] for m in MODES]
    labels = [MODE_LABELS[m] for m in MODES]
    vals   = [summary_df[summary_df["mode"] == m]["threshold"].values[0] for m in MODES]

    bars = ax.bar(labels, vals, color=colors, edgecolor="white", linewidth=0.8, width=0.45)
    for bar, v in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.015,
            f"{v:.3f}",
            ha="center", va="bottom", fontsize=11, fontweight="bold",
        )

    ax.axhline(0.5, linestyle="--", color="#888888", linewidth=1.2, label="Seuil par défaut (0.5)")
    ax.set_ylim(0, 0.85)
    ax.set_ylabel("Seuil de décision $\\tau^*$", fontsize=11)
    ax.set_title("Seuil de décision calibré par modèle", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, frameon=True)
    ax.set_axisbelow(True)

    fig.tight_layout()
    save(fig, report_dir / "threshold_comparison.png")


# ── Figure 6 : Distributions de probabilités ─────────────────────────────────

def plot_probability_distributions(artifacts: dict[str, dict], report_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=False)
    fig.suptitle("Distributions des probabilités prédites par modèle", fontsize=13, fontweight="bold", y=1.01)

    for ax, mode in zip(axes, MODES):
        payload = artifacts[mode]["test"]
        labels  = np.array(payload["labels"])
        probs   = np.array(payload["probabilities"])
        thr     = float(payload["metrics"]["threshold"])

        for lbl, color, name in [(0, CLASS_COLORS[0], "Conforme"), (1, CLASS_COLORS[1], "Défectueux")]:
            mask = labels == lbl
            if mask.sum() == 0:
                continue
            sns.kdeplot(
                probs[mask], ax=ax,
                color=color, linewidth=2,
                fill=True, alpha=0.25,
                label=name,
            )

        ax.axvline(thr, color="#333333", linestyle="--", linewidth=1.5,
                   label=f"$\\tau^*$ = {thr:.3f}")
        ax.set_title(MODE_LABELS[mode], fontweight="bold", fontsize=11)
        ax.set_xlabel("Probabilité prédite $\\hat{p}$", fontsize=10)
        ax.set_ylabel("Densité", fontsize=10)
        ax.set_xlim(-0.02, 1.02)
        ax.legend(fontsize=8.5, frameon=True)
        ax.set_axisbelow(True)

    fig.tight_layout()
    save(fig, report_dir / "probability_distributions.png")


# ── Figure 7 : Courbes d'apprentissage comparées ─────────────────────────────

def plot_training_curves_comparison(artifacts: dict[str, dict], report_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Courbes d'apprentissage par modèle", fontsize=13, fontweight="bold", y=1.01)

    for mode in MODES:
        history = artifacts[mode]["training"]["history"]
        epochs     = [e["epoch"]          for e in history]
        train_loss = [e["train"]["loss"]   for e in history]
        val_loss   = [e["val"]["loss"]     for e in history]
        val_f1     = [e["val"]["f1"]       for e in history]

        axes[0].plot(epochs, train_loss, color=PALETTE[mode],
                     linewidth=1.8, linestyle="--", alpha=0.6)
        axes[0].plot(epochs, val_loss,   color=PALETTE[mode],
                     linewidth=2.2, label=MODE_LABELS[mode])
        axes[1].plot(epochs, val_f1,     color=PALETTE[mode],
                     linewidth=2.2, label=MODE_LABELS[mode])

    for ax, ylabel, title in zip(
        axes,
        ["BCE Loss", "F1-score (validation)"],
        ["Perte BCE (trait plein = val, pointillé = train)", "F1-score sur validation"],
    ):
        ax.set_xlabel("Époque", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontweight="bold")
        ax.legend(frameon=True, fontsize=9)
        ax.set_axisbelow(True)

    fig.tight_layout()
    save(fig, report_dir / "training_curves_comparison.png")


# ── Figure 8 : Radar chart (diagramme en araignée) ───────────────────────────

def plot_radar_chart(summary_df: pd.DataFrame, report_dir: Path) -> None:
    metrics = ["accuracy", "precision", "recall", "f1", "auc_roc"]
    labels  = ["Accuracy", "Précision", "Rappel", "F1-score", "AUC-ROC"]
    n = len(metrics)

    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]  # fermer le polygone

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    for mode in MODES:
        row  = summary_df[summary_df["mode"] == mode].iloc[0]
        vals = [row[m] for m in metrics]
        vals += vals[:1]
        ax.plot(angles, vals, color=PALETTE[mode], linewidth=2, label=MODE_LABELS[mode])
        ax.fill(angles, vals, color=PALETTE[mode], alpha=0.12)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=7.5)
    ax.grid(color="gray", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.set_title("Profil de performance multi-métriques", fontsize=12,
                 fontweight="bold", pad=18)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.12), frameon=True, fontsize=9)

    fig.tight_layout()
    save(fig, report_dir / "radar_comparison.png")


# ── Figure 9 : Matrices de confusion côte à côte ─────────────────────────────

def plot_confusion_matrices(artifacts: dict[str, dict], report_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle("Matrices de confusion — jeu de test", fontsize=13, fontweight="bold", y=1.01)

    for ax, mode in zip(axes, MODES):
        cm = np.array(artifacts[mode]["test"]["metrics"]["confusion_matrix"])
        total = cm.sum()

        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max() * 1.2)

        for i in range(2):
            for j in range(2):
                pct = 100 * cm[i, j] / total
                color = "white" if cm[i, j] > cm.max() * 0.5 else "#222222"
                ax.text(j, i, f"{cm[i, j]}\n({pct:.1f}\u202f%)",
                        ha="center", va="center", fontsize=11,
                        fontweight="bold", color=color)

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Prédit OK", "Prédit Défaut"], fontsize=9)
        ax.set_yticklabels(["Réel OK", "Réel Défaut"], fontsize=9)
        ax.set_title(MODE_LABELS[mode], fontweight="bold", fontsize=11)

    fig.tight_layout()
    save(fig, report_dir / "confusion_matrices.png")


# ── Figure 10 : Analyse SHAP — branche capteurs ───────────────────────────────

def plot_shap_sensor(outputs_dir: Path, report_dir: Path) -> None:
    """Compute and plot SHAP values for the sensor-only MLP.

    Uses KernelExplainer (model-agnostic) with k-means background sampling.
    Produces two subplots: beeswarm (left) and mean |SHAP| bar chart (right).
    Silently skips if the sensor model or shap library is unavailable.
    """
    try:
        import shap as shap_lib
    except ImportError:
        print("  -> shap non installe, analyse ignoree")
        return

    import torch

    sensor_dir  = outputs_dir / "sensor"
    config_path = sensor_dir / "config.json"
    model_path  = sensor_dir / "best_model.pt"
    if not config_path.exists() or not model_path.exists():
        print("  -> modele capteurs introuvable, SHAP ignore")
        return

    from mqi.data.catalog import build_catalog, stratified_split
    from mqi.data.datasets import split_records, fit_sensor_scaler
    from mqi.data.synthetic_sensors import generate_sensor_table, SENSOR_COLUMNS
    from mqi.models.sensors import SensorClassifier
    from mqi.utils.repro import seed_everything

    with config_path.open() as f:
        cfg = json.load(f)

    seed_everything(cfg["seed"])
    records      = build_catalog(Path(cfg["dataset_dir"]))
    records      = stratified_split(records, val_ratio=cfg["val_ratio"],
                                    test_ratio=cfg["test_ratio"], seed=cfg["seed"])
    sensor_table = generate_sensor_table(records, base_seed=cfg["seed"])
    train_recs, _, test_recs = split_records(records)
    scaler       = fit_sensor_scaler(train_recs, sensor_table)

    def to_matrix(recs):
        return np.array(
            [scaler.transform(sensor_table[r.sample_id]) for r in recs],
            dtype=np.float32,
        )

    X_train = to_matrix(train_recs)
    X_test  = to_matrix(test_recs)

    device = torch.device("cpu")
    model  = SensorClassifier().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    def predict_proba(x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            logits = model(torch.tensor(x, dtype=torch.float32).to(device))
            return torch.sigmoid(logits).cpu().numpy().flatten()

    # 50 k-means background clusters — fast and representative
    background  = shap_lib.kmeans(X_train, 50)
    explainer   = shap_lib.KernelExplainer(predict_proba, background)

    rng_idx     = np.random.default_rng(cfg["seed"])
    n_explain   = min(150, len(X_test))
    idx         = rng_idx.choice(len(X_test), n_explain, replace=False)
    shap_vals   = explainer.shap_values(X_test[idx], nsamples=128, silent=True)

    feat_labels = [SENSOR_LABELS[c] for c in SENSOR_COLUMNS]
    mean_abs    = np.abs(shap_vals).mean(axis=0)
    order       = np.argsort(mean_abs)            # ascending for barh

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Analyse SHAP — branche capteurs (MLP)", fontsize=13,
                 fontweight="bold", y=1.02)

    # ── Panneau gauche : beeswarm ──
    ax_b = axes[0]
    x_vals  = shap_vals               # (n, 6)
    raw_std = X_test[idx]             # standardised feature values for colouring

    cmap = plt.cm.RdBu_r
    for fi, fname in enumerate(feat_labels):
        y_jitter = fi + np.random.default_rng(fi).uniform(-0.3, 0.3, n_explain)
        col_range = raw_std[:, fi].max() - raw_std[:, fi].min()
        norm_col  = (raw_std[:, fi] - raw_std[:, fi].min()) / (col_range + 1e-8)
        colors    = cmap(norm_col)
        ax_b.scatter(x_vals[:, fi], y_jitter, c=colors, alpha=0.55,
                     s=14, linewidths=0, zorder=2)

    ax_b.axvline(0, color="#444444", linewidth=0.8, linestyle="--", zorder=1)
    ax_b.set_yticks(range(len(feat_labels)))
    ax_b.set_yticklabels(feat_labels, fontsize=9)
    ax_b.set_xlabel("Valeur SHAP  (impact sur $P(\\text{défaut})$)", fontsize=9)
    ax_b.set_title("Distribution des valeurs SHAP", fontweight="bold", fontsize=10)
    ax_b.grid(axis="x", alpha=0.3, linestyle="--")
    ax_b.set_axisbelow(True)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax_b, orientation="vertical", fraction=0.03, pad=0.02)
    cb.set_label("Valeur capteur\n(normalisée)", fontsize=8)
    cb.set_ticks([0, 1])
    cb.set_ticklabels(["Basse", "Haute"])

    # ── Panneau droit : importance bar ──
    ax_i = axes[1]
    bars = ax_i.barh(
        [feat_labels[i] for i in order],
        mean_abs[order],
        color=PALETTE["sensor"],
        edgecolor="white",
        height=0.6,
    )
    for bar, val in zip(bars, mean_abs[order]):
        ax_i.text(val + mean_abs.max() * 0.015,
                  bar.get_y() + bar.get_height() / 2,
                  f"{val:.3f}", va="center", fontsize=9)
    ax_i.set_xlabel("Importance moyenne  $\\overline{|\\phi_j|}$", fontsize=9)
    ax_i.set_title("Importance des capteurs (SHAP)", fontweight="bold", fontsize=10)
    ax_i.set_xlim(0, mean_abs.max() * 1.30)
    ax_i.set_axisbelow(True)

    fig.tight_layout()
    save(fig, report_dir / "shap_sensor.png")


def _load_robustness(outputs_dir: Path) -> dict | None:
    p = outputs_dir / "final_report" / "robustness_results.json"
    if not p.exists():
        return None
    with p.open() as f:
        return json.load(f)


def plot_robustness_degradation(outputs_dir: Path, report_dir: Path) -> None:
    """F1 and recall vs image degradation level for image-only and fusion."""
    rob = _load_robustness(outputs_dir)
    if rob is None:
        print("  -> robustness_results.json absent, figure dégradation ignorée")
        return

    data = rob["degradation"]
    levels     = ["none", "moderate", "severe"]
    level_labels = ["Aucune", "Modérée\n(bruit σ=0.05, luminosité ±25 %)",
                    "Sévère\n(bruit σ=0.12, occlusion, flou)"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle("Robustesse à la dégradation d'image", fontsize=13,
                 fontweight="bold", y=1.02)

    for ax, metric, ylabel in zip(
        axes,
        ["f1", "recall"],
        ["F1-score", "Rappel (Recall)"],
    ):
        for mode, label, color, marker in [
            ("image",      "Image seule",       PALETTE["image"],      "o"),
            ("multimodal", "Fusion multimodale", PALETTE["multimodal"], "s"),
        ]:
            vals = [data[lv][mode][metric] for lv in levels]
            ax.plot(level_labels, vals, marker=marker, color=color,
                    linewidth=2, markersize=8, label=label)
            for x, y in zip(level_labels, vals):
                ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                            xytext=(0, 8), ha="center", fontsize=8.5, color=color)

        ax.set_ylim(0.82, 1.025)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xlabel("Niveau de dégradation", fontsize=10)
        ax.legend(fontsize=9)
        ax.tick_params(axis="x", labelsize=8)
        ax.axhline(1.0, color="gray", linewidth=0.7, linestyle="--", alpha=0.5)

    fig.tight_layout()
    save(fig, report_dir / "robustness_degradation.png")


def plot_sensor_missing(outputs_dir: Path, report_dir: Path) -> None:
    """F1 vs sensor missing rate for sensor-only and fusion."""
    rob = _load_robustness(outputs_dir)
    if rob is None:
        print("  -> robustness_results.json absent, figure capteurs manquants ignorée")
        return

    data  = rob["sensor_missing"]
    keys  = sorted(data.keys(), key=lambda k: data[k]["missing_rate"])
    rates = [data[k]["missing_rate"] * 100 for k in keys]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.set_title("Robustesse aux capteurs manquants", fontsize=13, fontweight="bold")

    for mode, label, color, marker in [
        ("sensor",     "Capteurs seuls",     PALETTE["sensor"],     "^"),
        ("multimodal", "Fusion multimodale",  PALETTE["multimodal"], "s"),
    ]:
        vals = [data[k][mode]["f1"] for k in keys]
        ax.plot(rates, vals, marker=marker, color=color,
                linewidth=2, markersize=8, label=label)
        for x, y in zip(rates, vals):
            ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=8.5, color=color)

    ax.set_xlabel("Taux de capteurs manquants (%)", fontsize=10)
    ax.set_ylabel("F1-score", fontsize=10)
    ax.set_ylim(0.70, 1.05)
    ax.set_xticks(rates)
    ax.legend(fontsize=9)
    ax.axhline(1.0, color="gray", linewidth=0.7, linestyle="--", alpha=0.5)

    fig.tight_layout()
    save(fig, report_dir / "sensor_missing.png")


def plot_sensor_ablation(outputs_dir: Path, report_dir: Path) -> None:
    """AUC drop when each sensor feature is zeroed (inference-time ablation)."""
    rob = _load_robustness(outputs_dir)
    if rob is None:
        print("  -> robustness_results.json absent, figure ablation ignorée")
        return

    data     = rob["sensor_ablation"]
    baseline = data["baseline"]["auc_roc"]

    from mqi.data.synthetic_sensors import SENSOR_COLUMNS
    sensor_labels = {
        "mold_temperature_c":    "Température\nmoule (°C)",
        "injection_pressure_bar":"Pression\ninjection (bar)",
        "cycle_time_s":          "Durée de\ncycle (s)",
        "vibration_mm_s":        "Vibration\n(mm/s)",
        "humidity_pct":          "Humidité\n(%)",
        "rotation_speed_rpm":    "Vitesse\nrotation (tr/min)",
    }

    cols  = SENSOR_COLUMNS
    drops = [data[c]["auc_drop"] for c in cols]
    labels_plot = [sensor_labels[c] for c in cols]

    order = np.argsort(drops)[::-1]   # most important first
    drops_sorted  = [drops[i]       for i in order]
    labels_sorted = [labels_plot[i] for i in order]

    colors = [PALETTE["sensor"] if d >= 0 else "#aaaaaa" for d in drops_sorted]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.set_title(
        f"Ablation par capteur — chute d'AUC (référence = {baseline:.4f})",
        fontsize=12, fontweight="bold",
    )
    bars = ax.barh(labels_sorted, drops_sorted, color=colors, edgecolor="white", height=0.55)
    for bar, d in zip(bars, drops_sorted):
        sign = "+" if d >= 0 else ""
        ax.text(
            max(d, 0) + 0.0005, bar.get_y() + bar.get_height() / 2,
            f"{sign}{d:.4f}", va="center", fontsize=9,
        )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("ΔAUC (AUC référence − AUC avec capteur zeroed)", fontsize=9)
    ax.invert_yaxis()
    fig.tight_layout()
    save(fig, report_dir / "sensor_ablation.png")


def plot_cost_analysis(outputs_dir: Path, report_dir: Path) -> None:
    """Total asymmetric cost vs threshold for all three models (C_FN=10, C_FP=1)."""
    rob = _load_robustness(outputs_dir)
    if rob is None:
        print("  -> robustness_results.json absent, figure coût ignorée")
        return

    data = rob["cost_analysis"]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.set_title(
        "Coût total asymétrique vs seuil de décision\n"
        r"$\mathcal{C} = 10 \times \mathrm{FN} + 1 \times \mathrm{FP}$",
        fontsize=11, fontweight="bold",
    )

    for mode, label, color, ls in [
        ("image",      "Image seule",        PALETTE["image"],      "-"),
        ("sensor",     "Capteurs seuls",      PALETTE["sensor"],     "--"),
        ("multimodal", "Fusion multimodale",  PALETTE["multimodal"], "-."),
    ]:
        thrs   = data[mode]["curve_thresholds"]
        costs  = data[mode]["curve_costs"]
        opt_t  = data[mode]["optimal_threshold"]
        opt_c  = data[mode]["optimal_cost"]
        ax.plot(thrs, costs, color=color, linewidth=2, linestyle=ls, label=label)
        ax.axvline(opt_t, color=color, linewidth=1.2, linestyle=":", alpha=0.7)
        ax.scatter([opt_t], [opt_c], color=color, zorder=5, s=60)

    ax.set_xlabel("Seuil de décision τ", fontsize=10)
    ax.set_ylabel("Coût total", fontsize=10)
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1)

    fig.tight_layout()
    save(fig, report_dir / "cost_analysis.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    apply_style()
    report_dir = ensure_dir(args.report_dir)

    print("Chargement des artefacts...")
    artifacts   = load_mode_artifacts(args.outputs_dir)
    manifest    = load_manifest(args.outputs_dir)
    summary_rows = build_summary_rows(artifacts)
    summary_df  = pd.DataFrame(summary_rows)
    export_summary(summary_rows, report_dir)

    print("Génération des figures...")
    plot_class_distribution(manifest, report_dir)
    plot_sensor_distributions(manifest, report_dir)
    plot_metrics_comparison(summary_df, report_dir)
    plot_roc_comparison(artifacts, report_dir)
    plot_threshold_comparison(summary_df, report_dir)
    plot_probability_distributions(artifacts, report_dir)
    plot_training_curves_comparison(artifacts, report_dir)
    plot_radar_chart(summary_df, report_dir)
    plot_confusion_matrices(artifacts, report_dir)
    plot_shap_sensor(args.outputs_dir, report_dir)
    plot_robustness_degradation(args.outputs_dir, report_dir)
    plot_sensor_missing(args.outputs_dir, report_dir)
    plot_sensor_ablation(args.outputs_dir, report_dir)
    plot_cost_analysis(args.outputs_dir, report_dir)

    # ── Copy Grad-CAM grids if they exist ────────────────────────────────────
    import shutil as _shutil
    for mode, dest_name in [("image", "gradcam_image.png"), ("multimodal", "gradcam_multimodal.png")]:
        src = args.outputs_dir / mode / "gradcam_grid.png"
        if src.exists():
            _shutil.copy2(src, report_dir / dest_name)
            print(f"  -> {dest_name}")

    print(f"OK - figures sauvegardees dans {report_dir}")


if __name__ == "__main__":
    main()
