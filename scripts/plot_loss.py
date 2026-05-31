# -*- coding: utf-8 -*-
"""从训练日志提取 lm loss 并绘制曲线"""
import os, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

work_dir = os.environ.get("WORKDIR", os.path.dirname(os.path.dirname(__file__)))
loss_log = f"{work_dir}/logs/tune_mcore_pangu_1b_full_ptd.log"
out_png = f"{work_dir}/loss_curve.png"

loss_list = []
iter_list = []

pattern_iter = re.compile(r"iteration\s+(\d+)")
pattern_loss = re.compile(r"\|\s*lm loss:\s*([0-9.eE+-]+)")

if not os.path.exists(loss_log):
    print(f"[WARN] 日志不存在: {loss_log}")
    exit(0)

with open(loss_log, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        m_loss = pattern_loss.search(line)
        if not m_loss:
            continue
        loss = float(m_loss.group(1))
        m_iter = pattern_iter.search(line)
        step = int(m_iter.group(1)) if m_iter else len(loss_list) + 1
        iter_list.append(step)
        loss_list.append(loss)

if not loss_list:
    raise RuntimeError("没有从日志中解析到 lm loss。")

plt.figure(figsize=(10, 6))
plt.plot(iter_list, loss_list, linewidth=1.5, label="Training Loss")
plt.xlabel("Iteration")
plt.ylabel("LM Loss")
plt.title("SFT Training Loss Curve")
plt.grid(True, linestyle="--", alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig(out_png, dpi=200)
print(f"[OK] loss 曲线已保存：{out_png}")
print(f"[INFO] loss 点数：{len(loss_list)}")
print(f"[INFO] first loss：{loss_list[0]}")
print(f"[INFO] last loss：{loss_list[-1]}")
