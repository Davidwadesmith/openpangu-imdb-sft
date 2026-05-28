# -*- coding: utf-8 -*-
"""
IMDB 测试集推理脚本
加载微调后的 HF 模型，对每条测试样本进行情感分类推理
"""
import torch
import pandas as pd
import json
import os
import argparse
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--test-parquet", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=0,
                        help="限制推理样本数（0=全部）")
    args = parser.parse_args()

    print(f"[inference] 加载模型: {args.model_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, use_fast=False, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="npu",
    )
    model.eval()

    print(f"[inference] 加载测试集: {args.test_parquet}")
    df = pd.read_parquet(args.test_parquet)
    if args.limit > 0:
        df = df.head(args.limit)
    total = len(df)
    print(f"[inference] 共 {total} 条样本")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as fout:
        for idx, (_, row) in enumerate(tqdm(df.iterrows(), total=total)):
            text = row["text"]
            label = int(row["label"])

            messages = [
                {"role": "system", "content": ""},
                {"role": "user", "content": "请判断以下文本的情感类别：" + text},
            ]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(prompt, return_tensors="pt").to("npu")
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            # 只取新生成的部分
            generated = outputs[0][inputs.input_ids.shape[1]:]
            response = tokenizer.decode(generated, skip_special_tokens=True).strip()

            rec = {
                "prompt": text,
                "gt": label,
                "response": response,
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

            if (idx + 1) % 1000 == 0:
                print(f"[inference] {idx + 1}/{total} 完成")

    print(f"[inference] 完成: {args.output}")


if __name__ == "__main__":
    main()
