from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full MQI pipeline end-to-end.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size-image", type=int, default=24)
    parser.add_argument("--batch-size-sensor", type=int, default=64)
    parser.add_argument("--batch-size-multimodal", type=int, default=24)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def run_command(command: list[str]) -> None:
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def train_mode(mode: str, epochs: int, batch_size: int, cpu: bool, seed: int, no_pretrained: bool) -> None:
    command = [
        sys.executable,
        "scripts/train.py",
        "--mode",
        mode,
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--seed",
        str(seed),
    ]
    if cpu:
        command.append("--cpu")
    if no_pretrained:
        command.append("--no-pretrained")
    run_command(command)
    eval_cmd = [sys.executable, "scripts/evaluate.py", "--run-dir", str(PROJECT_ROOT / "outputs" / mode)]
    if mode in ("image", "multimodal"):
        eval_cmd += ["--gradcam", "--gradcam-samples", "12"]
    run_command(eval_cmd)


def main() -> None:
    args = parse_args()

    train_mode("image", args.epochs, args.batch_size_image, args.cpu, args.seed, args.no_pretrained)
    train_mode("sensor", args.epochs, args.batch_size_sensor, args.cpu, args.seed, True)
    train_mode("multimodal", args.epochs, args.batch_size_multimodal, args.cpu, args.seed, args.no_pretrained)

    run_command([sys.executable, "scripts/robustness_eval.py"])
    run_command([sys.executable, "scripts/statistical_validation.py"])
    run_command([sys.executable, "scripts/build_report_assets.py"])


if __name__ == "__main__":
    main()
