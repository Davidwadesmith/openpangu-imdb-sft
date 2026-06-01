# -*- coding: utf-8 -*-
"""Evaluate strict positive/negative sentiment predictions."""

import json
import os
from pathlib import Path


WORKDIR = Path(os.environ.get("WORKDIR", Path(__file__).resolve().parents[1]))
RESULT_PATH = WORKDIR / "results" / "test_imdb_results.jsonl"


def main() -> None:
    total = 0
    correct = 0

    with RESULT_PATH.open("r", encoding="utf-8") as results:
        for index, line in enumerate(results, start=1):
            if not line.strip():
                continue
            entry = json.loads(line)
            label = int(entry["label"] if "label" in entry else entry["gt"])
            response = entry["response"].strip()
            if response not in ("负面", "正面"):
                print(f"[WARN] Invalid response at line {index}: {response}")
                continue
            prediction = 0 if response == "负面" else 1
            total += 1
            if prediction == label:
                correct += 1

    accuracy = correct / total if total else 0.0
    print("=" * 40)
    print(f"Valid samples: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Accuracy: {accuracy:.4f}")
    print("=" * 40)


if __name__ == "__main__":
    main()
