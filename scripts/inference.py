# -*- coding: utf-8 -*-
"""
IMDB 测试集推理脚本
加载微调后的 HF 模型，对每条测试样本进行情感分类推理

不依赖 wyc 中的代码。
"""
import torch
import pandas as pd
import json
import os
import argparse
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def format_prompt(text, tokenizer):
    """构建与训练时一致的 prompt（仅 user 部分，等待模型生成 assistant 回复）"""
    messages = [
        {"role": "user", "content": "请判断以下文本的情感类别：" + text},
    ]

    # 尝试使用 tokenizer 的 chat_template
    if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            pass

    # 手动拼接 fallback（与 preprocess.py 一致）
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "user":
            parts.append(f"User: {content}")
        elif role == "system" and content:
            parts.append(f"System: {content}")
    text = "\n".join(parts) + "\nAssistant: "
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, help="微调后的 HF 模型目录")
    parser.add_argument("--test-parquet", required=True, help="测试集 parquet 路径")
    parser.add_argument("--output", required=True, help="输出 JSONL 路径")
    parser.add_argument("--limit", type=int, default=0, help="限制样本数（0=全部）")
    parser.add_argument("--max-new-tokens", type=int, default=10,
                        help="生成最大 token 数")
    args = parser.parse_args()

    print(f"[inference] 加载模型: {args.model_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, use_fast=False, trust_remote_code=True
    )
    # 确保 pad_token 存在
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id or 0

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

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    correct = 0
    with open(args.output, "w", encoding="utf-8") as fout:
        for idx, (_, row) in enumerate(tqdm(df.iterrows(), total=total)):
            text = row["text"]
            label = int(row["label"])

            prompt = format_prompt(text, tokenizer)
            inputs = tokenizer(prompt, return_tensors="pt").to("npu")

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

            # 只取新生成的部分
            generated_ids = outputs[0][inputs.input_ids.shape[1]:]
            response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

            rec = {
                "prompt": text,
                "label": label,
                "response": response,
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

            # 进度统计
            if response in ("正面", "负面"):
                pred = 0 if response == "负面" else 1
                if pred == label:
                    correct += 1

            if (idx + 1) % 1000 == 0:
                acc = correct / (idx + 1) * 100 if (idx + 1) > 0 else 0
                print(f"[inference] {idx + 1}/{total} | 当前准确率: {acc:.1f}%")

    final_acc = correct / total * 100 if total > 0 else 0
    print(f"[inference] 完成: {args.output}")
    print(f"[inference] 推理阶段准确率: {final_acc:.1f}% ({correct}/{total})")


if __name__ == "__main__":
    main()
