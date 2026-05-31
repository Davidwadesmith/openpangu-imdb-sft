#!/bin/bash
# ============================================================
#  openPangu-1B IMDB SFT — 环境配置文件
#  每次新终端必须先执行: source env.sh
# ============================================================

export WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 虚拟环境（如果存在）
if [ -f "$WORKDIR/.venv/bin/activate" ]; then
    source "$WORKDIR/.venv/bin/activate"
fi

# CANN 环境（set -e 下可能被误杀，临时关闭）
__OLD_ERREXIT_STATE=$(set +o | grep errexit)
set +e
if [ -f /usr/local/Ascend/cann-8.5.2/set_env.sh ]; then
    source /usr/local/Ascend/cann-8.5.2/set_env.sh
else
    echo "[WARN] 未找到 CANN set_env.sh"
fi
eval "$__OLD_ERREXIT_STATE"
unset __OLD_ERREXIT_STATE

export PATH="$WORKDIR/.venv/bin:$PATH"

# Megatron 源码嵌入 MindSpeed-LLM
if [ -d "$WORKDIR/MindSpeed-LLM/megatron" ]; then
    export PYTHONPATH="$WORKDIR/MindSpeed-LLM:$WORKDIR/MindSpeed:$PYTHONPATH"
else
    export PYTHONPATH="$WORKDIR/MindSpeed:$WORKDIR/MindSpeed-LLM:$PYTHONPATH"
fi

# HuggingFace 镜像
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME="$WORKDIR/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCH_EXTENSIONS_DIR="$WORKDIR/.cache/torch_extensions"
export TMPDIR="$WORKDIR/tmp"

mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" \
         "$TORCH_EXTENSIONS_DIR" "$TMPDIR"

echo "[OK] 环境已加载：$WORKDIR"
