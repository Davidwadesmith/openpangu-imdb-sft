# -*- coding: utf-8 -*-
"""Run deterministic sentiment inference over the IMDB test set."""

import json
import os
from pathlib import Path

import pandas as pd
import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")

try:
    import torch_npu  # noqa: F401
except Exception:
    pass

from transformers import AutoModelForCausalLM, AutoTokenizer


WORKDIR = Path(os.environ.get("WORKDIR", Path(__file__).resolve().parents[1]))
MODEL_PATH = WORKDIR / "ckpt" / "mg2hf" / "mg2hf"
TEST_PARQUET = WORKDIR / "downloads" / "test-00000-of-00001.parquet"
OUTPUT_PATH = WORKDIR / "results" / "test_imdb_results.jsonl"


def main() -> None:
    max_samples = int(os.environ.get("MAX_SAMPLES", "0"))
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Model path: {MODEL_PATH}")
    print(f"[INFO] Test parquet: {TEST_PARQUET}")
    print(f"[INFO] MAX_SAMPLES={max_samples or 'all'}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        use_fast=False,
        trust_remote_code=True,
        local_files_only=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    model = model.to("npu")
    model.eval()

    test = pd.read_parquet(TEST_PARQUET)
    if max_samples > 0:
        test = test.head(max_samples)

    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is None:
        eos_token_id = 45892

    with OUTPUT_PATH.open("w", encoding="utf-8") as output:
        for index, (_, row) in enumerate(test.iterrows(), start=1):
            messages = [
                {"role": "system", "content": ""},
                {
                    "role": "user",
                    "content": "请判断以下英文电影评论的情感类别。只回答：正面 或 负面。\n" + row["text"],
                },
            ]
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=8,
                    do_sample=False,
                    eos_token_id=eos_token_id,
                )
            response = tokenizer.decode(
                generated[0][inputs.input_ids.shape[1] :],
                skip_special_tokens=True,
            ).strip()
            output.write(
                json.dumps(
                    {
                        "prompt": row["text"],
                        "gt": int(row["label"]),
                        "response": response,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

            if index % 100 == 0:
                print(f"[INFO] Inferred {index} samples")

    print(f"[OK] Inference completed: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
