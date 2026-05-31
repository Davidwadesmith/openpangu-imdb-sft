#!/bin/bash
set -euo pipefail
source "${WORKDIR:-$(dirname "$0")/..}/env.sh"
export PATH=/home/service/.local/bin:$PATH

echo "[INFO] convert_hf2mcore.sh started"

MODEL_DIR="$WORKDIR/openPangu-Embedded-1B-V1.1"
OUT_DIR="$WORKDIR/ckpt/mcore"
mkdir -p "$OUT_DIR"

echo "[INFO] 模型目录:"
ls -ld "$MODEL_DIR"
find "$MODEL_DIR" -maxdepth 2 -type f \( -name "*.safetensors" -o -name "*.bin" \) -exec ls -lh {} \;

cd "$WORKDIR/MindSpeed-LLM"
python convert_ckpt.py \
    --model-type GPT \
    --load-model-type hf \
    --save-model-type mg \
    --load-dir "$MODEL_DIR" \
    --save-dir "$OUT_DIR" \
    --tokenizer-model "$MODEL_DIR" \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1 \
    --params-dtype bf16 \
    --use-mcore-models

echo "[OK] HF -> mcore 转换完成：$OUT_DIR"
find "$OUT_DIR" -maxdepth 3 -type f | head -n 20
