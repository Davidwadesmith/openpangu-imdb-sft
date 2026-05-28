#!/bin/bash
set -euo pipefail

# ============================================================
#  模型权重转换 mcore → HF（训练后用于推理）
#  用法：bash convert_mcore2hf.sh <MS_LLM_DIR> <SFT_OUTPUT_DIR> <CKPT_MG2HF_DIR> <MODEL_HF_DIR>
# ============================================================

MS_LLM_DIR="$1"
SFT_OUTPUT_DIR="$2"
CKPT_MG2HF_DIR="$3"
MODEL_HF_DIR="$4"

echo "[convert:mcore2hf] 开始权重转换 mcore → HF..."

# 找到最新的 checkpoint
LATEST_CKPT=$(ls -d "${SFT_OUTPUT_DIR}"/iter_* 2>/dev/null | sort -V | tail -1)
if [ -z "${LATEST_CKPT}" ]; then
    echo "[ERROR] 未找到 SFT checkpoint"
    exit 1
fi
echo "  最新 checkpoint: ${LATEST_CKPT}"

mkdir -p "${CKPT_MG2HF_DIR}"

cd "${MS_LLM_DIR}"
python convert_ckpt.py \
    --model-type GPT \
    --load-model-type mg \
    --save-model-type hf \
    --load-dir "${LATEST_CKPT}" \
    --save-dir "${CKPT_MG2HF_DIR}" \
    --tokenizer-model "${MODEL_HF_DIR}" \
    --add-qkv-bias \
    --add-dense-bias \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1 \
    --params-dtype bf16 \
    --use-mcore-models

# 拷贝 tokenizer 和模型配置文件
echo "  拷贝 tokenizer/配置文件..."
cp "${MODEL_HF_DIR}/tokenizer.model" "${CKPT_MG2HF_DIR}/" 2>/dev/null || true
cp "${MODEL_HF_DIR}/tokenizer_config.json" "${CKPT_MG2HF_DIR}/" 2>/dev/null || true
cp "${MODEL_HF_DIR}/tokenization_openpangu.py" "${CKPT_MG2HF_DIR}/" 2>/dev/null || true
cp "${MODEL_HF_DIR}/special_tokens_map.json" "${CKPT_MG2HF_DIR}/" 2>/dev/null || true
cp "${MODEL_HF_DIR}/generation_config.json" "${CKPT_MG2HF_DIR}/" 2>/dev/null || true
cp "${MODEL_HF_DIR}/configuration_openpangu_dense.py" "${CKPT_MG2HF_DIR}/" 2>/dev/null || true
cp "${MODEL_HF_DIR}/modeling_openpangu_dense.py" "${CKPT_MG2HF_DIR}/" 2>/dev/null || true

echo "[convert:mcore2hf] 完成: ${CKPT_MG2HF_DIR}"
