# -*- coding: utf-8 -*-
"""IMDB 测试集推理"""
import os, json, pandas as pd, torch

# 防止 huggingface_hub 对本地路径做 repo_id 校验
os.environ.setdefault("HF_HUB_OFFLINE", "1")

try:
    import torch_npu  # noqa
except Exception:
    pass
from transformers import AutoModelForCausalLM, AutoTokenizer

work_dir = os.environ.get("WORKDIR", os.path.dirname(os.path.dirname(__file__)))

# 真正转换后的 HF 模型在 mg2hf/mg2hf 子目录
model_path = f"{work_dir}/ckpt/mg2hf/mg2hf"
test_path = f"{work_dir}/downloads/test-00000-of-00001.parquet"
out_dir = f"{work_dir}/results"
out_path = f"{out_dir}/test_imdb_results.jsonl"

max_samples = int(os.environ.get("MAX_SAMPLES", "0"))
os.makedirs(out_dir, exist_ok=True)

print(f"[INFO] 模型路径: {model_path}")
print(f"[INFO] 测试集: {test_path}")

tokenizer = AutoTokenizer.from_pretrained(
    model_path, use_fast=False, trust_remote_code=True, local_files_only=True
)
model = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True, torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True, local_files_only=True
)
model = model.to("npu")
model.eval()

df = pd.read_parquet(test_path)
if max_samples > 0:
    df = df.head(max_samples)

eos_token_id = getattr(tokenizer, "eos_token_id", None)
if eos_token_id is None:
    eos_token_id = 45892

with open(out_path, "w", encoding="utf-8") as fout:
    for i, row in df.iterrows():
        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": "请判断以下英文电影评论的情感类别。只回答：正面 或 负面。\n" + row["text"]},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer([text], return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=8,
                do_sample=False,
                eos_token_id=eos_token_id,
            )
        response = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        ).strip()

        rec = {
            "prompt": row["text"],
            "gt": int(row["label"]),
            "response": response,
        }
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

        if (i + 1) % 100 == 0:
            print(f"[INFO] 已推理 {i + 1} 条")

print(f"[OK] 推理完成：{out_path}")
