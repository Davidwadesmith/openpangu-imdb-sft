# OpenPangu-1B IMDB SFT

本仓库按照实验三 PDF 整理。服务器克隆后执行一个脚本，即可完成环境搭建、数据准备、权重转换、SFT 训练、loss 绘图、全量推理和准确率评测。

## 一键运行

```bash
git clone https://github.com/Davidwadesmith/openpangu-imdb-sft.git /mnt/workspace/hcc/openpangu-imdb-sft
cd /mnt/workspace/hcc/openpangu-imdb-sft
bash run_all.sh
```

脚本默认自动识别仓库目录。也可以显式指定：

```bash
WORKDIR=/mnt/workspace/hcc/openpangu-imdb-sft bash run_all.sh
```

## 自动下载

- Python 依赖：清华 PyPI 镜像。
- MindSpeed-LLM、Megatron-LM、MindSpeed：Gitee。
- openPangu-Embedded-1B-V1.1：GitCode。
- IMDB：Hugging Face 镜像。

## 默认实验参数

- `SEQ_LENGTH=4096`
- `GLOBAL_BATCH_SIZE=4`
- `TRAIN_ITERS=300`
- `LR=1e-5`
- 推理：完整处理 25,000 条 IMDB 测试数据。

## 输出文件

- `logs/tune_mcore_pangu_1b_full_ptd.log`
- `loss_curve.png`
- `ckpt/mg2hf/mg2hf/`
- `results/test_imdb_results.jsonl`

脚本可以在中断后重新执行。已经完成的下载、预处理缓存、初始 mcore checkpoint 和 SFT checkpoint 会被复用。Hugging Face 导出目录和推理结果会重新生成，避免混入旧文件。

## 可选小样本推理

一键脚本默认执行全量推理。如需单独检查 100 条样本：

```bash
source env.sh
MAX_SAMPLES=100 python scripts/inference.py
```

## 本地静态校验

没有昇腾 NPU 的开发机可以运行：

```bash
python -m compileall -q scripts
bash -n env.sh run_all.sh scripts/*.sh
```
