# -*- coding: utf-8 -*-
"""Extract lm loss values from the SFT log and render loss_curve.png."""

import os
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


WORKDIR = Path(os.environ.get("WORKDIR", Path(__file__).resolve().parents[1]))
LOSS_LOG = WORKDIR / "logs" / "tune_mcore_pangu_1b_full_ptd.log"
OUTPUT_PNG = WORKDIR / "loss_curve.png"
ITERATION_PATTERN = re.compile(r"iteration\s+(\d+)")
LOSS_PATTERN = re.compile(r"\|\s*lm loss:\s*([0-9.eE+-]+)")


def main() -> None:
    iterations: list[int] = []
    losses: list[float] = []

    with LOSS_LOG.open("r", encoding="utf-8", errors="ignore") as log:
        for line in log:
            loss_match = LOSS_PATTERN.search(line)
            if not loss_match:
                continue
            iteration_match = ITERATION_PATTERN.search(line)
            iterations.append(int(iteration_match.group(1)) if iteration_match else len(losses) + 1)
            losses.append(float(loss_match.group(1)))

    if not losses:
        raise RuntimeError(f"No lm loss values found in {LOSS_LOG}")

    plt.figure(figsize=(10, 6))
    plt.plot(iterations, losses, linewidth=1.5, label="Training Loss")
    plt.xlabel("Iteration")
    plt.ylabel("LM Loss")
    plt.title("SFT Training Loss Curve")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=200)

    print(f"[OK] Loss curve saved: {OUTPUT_PNG}")
    print(f"[INFO] Loss points: {len(losses)}")
    print(f"[INFO] First loss: {losses[0]}")
    print(f"[INFO] Last loss: {losses[-1]}")


if __name__ == "__main__":
    main()
