#!/bin/bash
set -euo pipefail

# ============================================================
#  模型权重转换 HF → mcore
#  用法：bash convert_hf2mcore.sh <MS_LLM_DIR> <MODEL_HF_DIR> <CKPT_MCORE_DIR>
# ============================================================

MS_LLM_DIR="$1"
MODEL_HF_DIR="$2"
CKPT_MCORE_DIR="$3"

echo "[convert:hf2mcore] 开始权重转换 HF → mcore..."

mkdir -p "${CKPT_MCORE_DIR}"

cd "${MS_LLM_DIR}"
python convert_ckpt.py \
    --model-type GPT \
    --load-model-type hf \
    --save-model-type mg \
    --load-dir "${MODEL_HF_DIR}" \
    --save-dir "${CKPT_MCORE_DIR}" \
    --tokenizer-model "${MODEL_HF_DIR}" \
    --add-qkv-bias \
    --add-dense-bias \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1 \
    --params-dtype bf16 \
    --use-mcore-models

echo "[convert:hf2mcore] 完成: ${CKPT_MCORE_DIR}"
