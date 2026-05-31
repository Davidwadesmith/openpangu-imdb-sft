#!/bin/bash
set -euo pipefail
source "${WORKDIR:-$(dirname "$0")/..}/env.sh"
export PATH=/home/service/.local/bin:$PATH
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MAX_JOBS=1

MODEL_DIR="$WORKDIR/openPangu-Embedded-1B-V1.1"
SFT_DIR="$WORKDIR/sft_output"
HF_OUT="$WORKDIR/ckpt/mg2hf"

echo "[INFO] 清空旧 mg2hf"
rm -rf "$HF_OUT"
mkdir -p "$HF_OUT"

echo "[INFO] 转换前复制完整 HF 模型文件（含 model.safetensors）"
cp -f "$MODEL_DIR"/*.json "$HF_OUT"/ 2>/dev/null || true
cp -f "$MODEL_DIR"/*.model "$HF_OUT"/ 2>/dev/null || true
cp -f "$MODEL_DIR"/*.py "$HF_OUT"/ 2>/dev/null || true
cp -f "$MODEL_DIR"/*.tiktoken "$HF_OUT"/ 2>/dev/null || true
cp -f "$MODEL_DIR"/*.safetensors "$HF_OUT"/ 2>/dev/null || true
cp -f "$MODEL_DIR"/*.bin "$HF_OUT"/ 2>/dev/null || true
cp -f "$MODEL_DIR"/*.index.json "$HF_OUT"/ 2>/dev/null || true

echo "[INFO] SFT checkpoint 内容:"
find "$SFT_DIR" -maxdepth 5 -type f -exec ls -lh {} \; | head -n 80

cd "$WORKDIR/MindSpeed-LLM"
echo "[INFO] 开始 mcore -> HF 转换"
python convert_ckpt.py \
    --model-type GPT \
    --load-model-type mg \
    --save-model-type hf \
    --load-dir "$SFT_DIR" \
    --save-dir "$HF_OUT" \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1 \
    --use-mcore-models

echo "[INFO] 转换后补齐 tokenizer/config/code 文件"
cp -f "$MODEL_DIR"/*.json "$HF_OUT"/ 2>/dev/null || true
cp -f "$MODEL_DIR"/*.model "$HF_OUT"/ 2>/dev/null || true
cp -f "$MODEL_DIR"/*.py "$HF_OUT"/ 2>/dev/null || true
cp -f "$MODEL_DIR"/*.tiktoken "$HF_OUT"/ 2>/dev/null || true

echo "[INFO] mg2hf 文件列表:"
find "$HF_OUT" -maxdepth 2 -type f -exec ls -lh {} \;

if ! find "$HF_OUT" -maxdepth 2 -type f \( -name "*.bin" -o -name "*.safetensors" \) | grep -q .; then
    echo "[ERROR] 转换结束但没有权重文件。"
    exit 1
fi

# 真正用于推理的模型在 mg2hf/mg2hf 子目录
INFERENCE_DIR="$HF_OUT/mg2hf"
echo "[INFO] 补齐推理目录 tokenizer: $INFERENCE_DIR"
cp -f "$HF_OUT"/tokenizer.model "$INFERENCE_DIR/" 2>/dev/null || true
cp -f "$HF_OUT"/tokenizer_config.json "$INFERENCE_DIR/" 2>/dev/null || true
cp -f "$HF_OUT"/special_tokens_map.json "$INFERENCE_DIR/" 2>/dev/null || true
cp -f "$HF_OUT"/tokenization_openpangu.py "$INFERENCE_DIR/" 2>/dev/null || true

echo "[OK] mcore -> HF 转换完成。推理模型: $INFERENCE_DIR"
