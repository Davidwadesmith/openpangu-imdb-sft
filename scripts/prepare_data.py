# -*- coding: utf-8 -*-
"""IMDB 数据下载与格式转换 → train_imdb.jsonl"""
import os, json, pandas as pd

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from datasets import load_dataset

work_dir = os.environ.get("WORKDIR", os.path.dirname(os.path.dirname(__file__)))
download_dir = f"{work_dir}/downloads"
data_dir = f"{work_dir}/data"
os.makedirs(download_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)

train_parquet = f"{download_dir}/train-00000-of-00001.parquet"
test_parquet = f"{download_dir}/test-00000-of-00001.parquet"
out_jsonl = f"{data_dir}/train_imdb.jsonl"

if os.path.exists(train_parquet) and os.path.exists(test_parquet):
    print("[OK] 发现本地 parquet，直接读取。")
    df_train = pd.read_parquet(train_parquet)
    df_test = pd.read_parquet(test_parquet)
else:
    print("[INFO] 未发现本地 parquet，从 HuggingFace 镜像下载 IMDB。")
    ds_train = load_dataset("stanfordnlp/imdb", split="train")
    ds_test = load_dataset("stanfordnlp/imdb", split="test")
    df_train = ds_train.to_pandas()
    df_test = ds_test.to_pandas()
    df_train.to_parquet(train_parquet)
    df_test.to_parquet(test_parquet)
    print("[OK] 已保存 train/test parquet 到 downloads/。")

with open(out_jsonl, "w", encoding="utf-8") as f:
    for _, row in df_train.iterrows():
        label_str = "正面" if int(row["label"]) == 1 else "负面"
        rec = {
            "instruction": "请判断以下文本的情感类别是正面还是负面。",
            "input": row["text"],
            "output": label_str,
        }
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

print(f"[OK] 训练 JSONL 已生成：{out_jsonl}")
print(f"[OK] 训练样本数：{len(df_train)}")
print(f"[OK] 测试样本数：{len(df_test)}")
