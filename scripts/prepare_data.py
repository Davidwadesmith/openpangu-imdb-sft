# -*- coding: utf-8 -*-
"""Download IMDB through the configured mirror and create MindSpeed SFT JSONL."""

import json
import os
from pathlib import Path

import pandas as pd

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from datasets import load_dataset


WORKDIR = Path(os.environ.get("WORKDIR", Path(__file__).resolve().parents[1]))
DOWNLOAD_DIR = WORKDIR / "downloads"
DATA_DIR = WORKDIR / "data"
TRAIN_PARQUET = DOWNLOAD_DIR / "train-00000-of-00001.parquet"
TEST_PARQUET = DOWNLOAD_DIR / "test-00000-of-00001.parquet"
TRAIN_JSONL = DATA_DIR / "train_imdb.jsonl"


def load_imdb() -> tuple[pd.DataFrame, pd.DataFrame]:
    if TRAIN_PARQUET.exists() and TEST_PARQUET.exists():
        print("[OK] Local IMDB parquet files found")
        return pd.read_parquet(TRAIN_PARQUET), pd.read_parquet(TEST_PARQUET)

    print("[INFO] Downloading IMDB from the Hugging Face mirror")
    train = load_dataset("stanfordnlp/imdb", split="train").to_pandas()
    test = load_dataset("stanfordnlp/imdb", split="test").to_pandas()
    train.to_parquet(TRAIN_PARQUET)
    test.to_parquet(TEST_PARQUET)
    print(f"[OK] Saved parquet files under {DOWNLOAD_DIR}")
    return train, test


def write_training_jsonl(train: pd.DataFrame) -> None:
    with TRAIN_JSONL.open("w", encoding="utf-8") as output:
        for _, row in train.iterrows():
            record = {
                "instruction": "请判断以下文本的情感类别是正面还是负面。",
                "input": row["text"],
                "output": "正面" if int(row["label"]) == 1 else "负面",
            }
            output.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train, test = load_imdb()
    write_training_jsonl(train)
    print(f"[OK] Training JSONL created: {TRAIN_JSONL}")
    print(f"[INFO] Training samples: {len(train)}")
    print(f"[INFO] Test samples: {len(test)}")


if __name__ == "__main__":
    main()
