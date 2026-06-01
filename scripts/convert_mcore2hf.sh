#!/bin/bash
set -euo pipefail
shopt -s nullglob

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../env.sh"
export PATH=/home/service/.local/bin:$PATH
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MAX_JOBS=1

MODEL_DIR="$WORKDIR/openPangu-Embedded-1B-V1.1"
SFT_DIR="$WORKDIR/sft_output"
HF_OUT="$WORKDIR/ckpt/mg2hf"
INFERENCE_DIR="$HF_OUT/mg2hf"

copy_model_shell() {
    local destination="$1"
    local file
    for file in \
        "$MODEL_DIR"/*.json \
        "$MODEL_DIR"/*.model \
        "$MODEL_DIR"/*.py \
        "$MODEL_DIR"/*.tiktoken \
        "$MODEL_DIR"/*.safetensors \
        "$MODEL_DIR"/*.bin \
        "$MODEL_DIR"/*.index.json; do
        cp -f "$file" "$destination/"
    done
}

copy_metadata() {
    local destination="$1"
    local file
    for file in \
        "$MODEL_DIR"/*.json \
        "$MODEL_DIR"/*.model \
        "$MODEL_DIR"/*.py \
        "$MODEL_DIR"/*.tiktoken; do
        cp -f "$file" "$destination/"
    done
}

echo "[INFO] Recreating Hugging Face export directory: $HF_OUT"
rm -rf "$HF_OUT"
mkdir -p "$HF_OUT"

echo "[INFO] Copying the base Hugging Face model shell"
copy_model_shell "$HF_OUT"

echo "[INFO] SFT checkpoint files:"
find "$SFT_DIR" -maxdepth 5 -type f -exec ls -lh {} \;

cd "$WORKDIR/MindSpeed-LLM"
python convert_ckpt.py \
    --model-type GPT \
    --load-model-type mg \
    --save-model-type hf \
    --load-dir "$SFT_DIR" \
    --save-dir "$HF_OUT" \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1 \
    --use-mcore-models

echo "[INFO] Copying tokenizer/config/code files into the inference directory"
mkdir -p "$INFERENCE_DIR"
copy_metadata "$HF_OUT"
copy_metadata "$INFERENCE_DIR"

echo "[INFO] Inference model files:"
find "$INFERENCE_DIR" -maxdepth 1 -type f -exec ls -lh {} \;

if ! find "$INFERENCE_DIR" -maxdepth 1 -type f \( -name "*.bin" -o -name "*.safetensors" \) | grep -q .; then
    echo "[ERROR] Converted inference weights not found under $INFERENCE_DIR"
    exit 1
fi

echo "[OK] mcore -> HF conversion completed: $INFERENCE_DIR"
