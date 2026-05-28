# -*- coding: utf-8 -*-
"""
IMDB 情感分类推理结果评测脚本
读取 JSONL 推理结果，计算分类准确率 (Accuracy)
"""
import json
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="推理结果 JSONL 路径")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]

    total = 0
    correct = 0

    for entry in lines:
        label = int(entry["gt"])
        response = entry["response"].strip()

        if response not in ("负面", "正面"):
            print(f"[WARN] 第 {total + 1} 条结果无效: {response}")
            continue

        pred = 0 if response == "负面" else 1
        total += 1
        if pred == label:
            correct += 1

    accuracy = correct / total if total > 0 else 0.0

    print("=" * 40)
    print(f"  样本总数 : {total}")
    print(f"  预测正确 : {correct}")
    print(f"  准确率   : {accuracy * 100:.2f}%")
    print("=" * 40)


if __name__ == "__main__":
    main()
