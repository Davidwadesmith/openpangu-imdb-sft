# -*- coding: utf-8 -*-
"""
IMDB 情感分类推理结果评测脚本
读取推理结果 JSONL，按 response 是否严格等于"负面"或"正面"计算准确率。
"""
import json, os

work_dir = os.environ.get("WORKDIR", os.path.dirname(os.path.dirname(__file__)))
result_path = f"{work_dir}/results/test_imdb_results.jsonl"

with open(result_path, "r", encoding="utf-8") as f:
    lines = [json.loads(line) for line in f if line.strip()]

total = 0
correct = 0

for idx, entry in enumerate(lines, start=1):
    if "label" in entry:
        label = int(entry["label"])
    else:
        label = int(entry["gt"])
    response = entry["response"].strip()
    if response not in ("负面", "正面"):
        print(f"[警告] 第 {idx} 条结果无效：{response}")
        continue
    pred = 0 if response == "负面" else 1
    total += 1
    if pred == label:
        correct += 1

accuracy = correct / total if total > 0 else 0.0
print("=" * 40)
print(f"  样本总数: {total}")
print(f"  预测正确: {correct}")
print(f"  准确率: {accuracy:.4f}")
print("=" * 40)
