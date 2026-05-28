#!/bin/bash
set -euo pipefail

# ============================================================
#  数据预处理 — 生成 .bin/.idx 缓存
#  用法：bash preprocess.sh <MS_LLM_DIR> <DATA_DIR> <MODEL_HF_DIR> <CACHE_DIR> <SEQ_LENGTH>
# ============================================================

MS_LLM_DIR="$1"
DATA_DIR="$2"
MODEL_HF_DIR="$3"
CACHE_DIR="$4"
SEQ_LENGTH="$5"

echo "[preprocess] 开始数据预处理..."
echo "  MindSpeed-LLM: ${MS_LLM_DIR}"
echo "  数据目录:      ${DATA_DIR}"
echo "  模型目录:      ${MODEL_HF_DIR}"
echo "  缓存目录:      ${CACHE_DIR}"
echo "  序列长度:      ${SEQ_LENGTH}"

mkdir -p "${CACHE_DIR}"

cd "${MS_LLM_DIR}"
python preprocess_data.py \
    --input "${DATA_DIR}" \
    --tokenizer-name-or-path "${MODEL_HF_DIR}" \
    --output-prefix "${CACHE_DIR}/sft" \
    --workers 4 \
    --tokenizer-type PretrainedFromHF \
    --handler-name PanguInstructionHandler \
    --seq-length "${SEQ_LENGTH}" \
    --pack

echo "[preprocess] 完成: ${CACHE_DIR}/sft_text_document.bin + .idx"
