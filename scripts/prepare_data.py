# -*- coding: utf-8 -*-
"""
IMDB Parquet → JSONL 格式转换
输出格式适配 MindSpeed-LLM 的 PanguInstructionHandler
"""
import pandas as pd
import json
import os
import sys
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="训练集 parquet 路径")
    parser.add_argument("--output", required=True, help="输出 JSONL 路径")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[ERROR] 输入文件不存在: {args.input}")
        sys.exit(1)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    df = pd.read_parquet(args.input)
    with open(args.output, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            rec = {
                "meta_prompt": [],
                "data": [
                    {"role": "user", "content": "请判断以下文本的情感类别：" + row["text"]},
                    {"role": "assistant", "content": "正面" if row["label"] else "负面"},
                ],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[DONE] 共 {len(df)} 条数据 → {args.output}")


if __name__ == "__main__":
    main()
