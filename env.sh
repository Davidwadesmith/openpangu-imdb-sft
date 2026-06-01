#!/bin/bash
# openPangu-1B IMDB SFT runtime environment.

export WORKDIR="${WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

if [ -f "$WORKDIR/.venv/bin/activate" ]; then
    source "$WORKDIR/.venv/bin/activate"
fi

# CANN set_env.sh may fail under errexit even when the environment is usable.
case "$-" in
    *e*) __ERREXIT_WAS_SET=1 ;;
    *) __ERREXIT_WAS_SET=0 ;;
esac
set +e
if [ -f /usr/local/Ascend/cann-8.5.2/set_env.sh ]; then
    source /usr/local/Ascend/cann-8.5.2/set_env.sh
else
    echo "[WARN] CANN set_env.sh not found"
fi
if [ "$__ERREXIT_WAS_SET" -eq 1 ]; then
    set -e
else
    set +e
fi
unset __ERREXIT_WAS_SET
set -o pipefail 2>/dev/null || true

export PATH="$WORKDIR/.venv/bin:$PATH"
export PYTHONPATH="$WORKDIR/MindSpeed:$WORKDIR/MindSpeed-LLM:${PYTHONPATH:-}"

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME="$WORKDIR/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCH_EXTENSIONS_DIR="$WORKDIR/.cache/torch_extensions"
export TMPDIR="$WORKDIR/tmp"
# MindSpeed checkpoints contain argparse.Namespace metadata and are generated locally.
# PyTorch 2.6+ otherwise defaults torch.load() to weights_only=True.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" \
         "$TORCH_EXTENSIONS_DIR" "$TMPDIR"

echo "[OK] Environment loaded: $WORKDIR"
