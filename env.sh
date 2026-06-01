#!/bin/bash
# openPangu-1B IMDB SFT runtime environment.

export WORKDIR="${WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

if [ -f "$WORKDIR/.venv/bin/activate" ]; then
    source "$WORKDIR/.venv/bin/activate"
fi

# CANN set_env.sh may fail under errexit even when the environment is usable.
__OLD_ERREXIT_STATE="$(set +o | grep errexit)"
set +e
if [ -f /usr/local/Ascend/cann-8.5.2/set_env.sh ]; then
    source /usr/local/Ascend/cann-8.5.2/set_env.sh
else
    echo "[WARN] CANN set_env.sh not found"
fi
eval "$__OLD_ERREXIT_STATE"
unset __OLD_ERREXIT_STATE
set -o pipefail 2>/dev/null || true

export PATH="$WORKDIR/.venv/bin:$PATH"
export PYTHONPATH="$WORKDIR/MindSpeed:$WORKDIR/MindSpeed-LLM:${PYTHONPATH:-}"

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME="$WORKDIR/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCH_EXTENSIONS_DIR="$WORKDIR/.cache/torch_extensions"
export TMPDIR="$WORKDIR/tmp"

mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" \
         "$TORCH_EXTENSIONS_DIR" "$TMPDIR"

echo "[OK] Environment loaded: $WORKDIR"
