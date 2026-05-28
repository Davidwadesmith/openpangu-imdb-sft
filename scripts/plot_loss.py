# -*- coding: utf-8 -*-
"""
绘制 SFT 训练 Loss 曲线
从训练日志中提取 lm loss 值，生成 loss_curve.png
"""
import os
import argparse

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-file", required=True, help="训练日志路径")
    parser.add_argument("--output", required=True, help="输出图片路径")
    args = parser.parse_args()

    if not os.path.exists(args.log_file):
        print(f"[ERROR] 日志文件不存在: {args.log_file}")
        return

    loss_list = []
    with open(args.log_file, "r", encoding="utf-8") as f:
        for line in f:
            if "| lm loss:" in line:
                try:
                    loss_value = float(line.split("| lm loss:")[-1].split("|")[0].strip())
                    loss_list.append(loss_value)
                except ValueError:
                    continue

    if not loss_list:
        print("[WARN] 未从日志中提取到 loss 值")
        return

    steps = list(range(1, len(loss_list) + 1))

    plt.figure(figsize=(10, 6))
    plt.plot(steps, loss_list, "b-", linewidth=1.5, label="Training Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("SFT Training Loss Curve")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()

    if len(steps) > 20:
        plt.xticks(np.arange(min(steps), max(steps) + 1, max(1, len(steps) // 10)))

    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"[plot_loss] 图片已保存: {args.output}")
    print(f"[plot_loss] 共 {len(loss_list)} 个 loss 点，"
          f"初始 loss={loss_list[0]:.4f}，最终 loss={loss_list[-1]:.4f}")


if __name__ == "__main__":
    main()
