#!/bin/bash
set -euo pipefail

# ============================================================
#  openPangu-1B SFT 全量微调训练脚本
#  参数：LR GBS MBS SEQ_LENGTH TRAIN_ITERS
# ============================================================

MS_LLM_DIR="$1"
CACHE_DIR="$2"
CKPT_MCORE_DIR="$3"
SFT_OUTPUT_DIR="$4"
MODEL_HF_DIR="$5"
LOGS_DIR="$6"
LR="$7"
GBS="$8"
MBS="$9"
SEQ_LENGTH="${10}"
TRAIN_ITERS="${11}"

echo "[train] =========================================="
echo "[train]  openPangu-1B SFT 训练"
echo "[train]  LR=${LR}  GBS=${GBS}  MBS=${MBS}  Iters=${TRAIN_ITERS}  SeqLen=${SEQ_LENGTH}"
echo "[train] =========================================="

TRAIN_LOG="${LOGS_DIR}/tune_mcore_pangu_1b_full_ptd.log"
mkdir -p "${SFT_OUTPUT_DIR}" "${LOGS_DIR}"

cd "${MS_LLM_DIR}"

torchrun --nproc_per_node=1 \
    posttrain_gpt.py \
    --stage sft \
    --finetune \
    --is-instruction-dataset \
    --use-mcore-models \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --sequence-parallel \
    --num-layers 26 \
    --hidden-size 1536 \
    --ffn-hidden-size 6144 \
    --num-attention-heads 12 \
    --group-query-attention \
    --num-query-groups 6 \
    --make-vocab-size-divisible-by 1 \
    --padded-vocab-size 153376 \
    --vocab-size 153376 \
    --max-position-embeddings "${SEQ_LENGTH}" \
    --seq-length "${SEQ_LENGTH}" \
    --micro-batch-size "${MBS}" \
    --global-batch-size "${GBS}" \
    --lr "${LR}" \
    --train-iters "${TRAIN_ITERS}" \
    --lr-decay-style cosine \
    --min-lr 1e-6 \
    --lr-warmup-iters 100 \
    --weight-decay 0.1 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.999 \
    --initial-loss-scale 4096 \
    --init-method-std 0.02 \
    --bf16 \
    --seed 42 \
    --data-path "${CACHE_DIR}/sft_text_document" \
    --split 100,0,0 \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path "${MODEL_HF_DIR}" \
    --tokenizer-not-use-fast \
    --save-interval 500 \
    --save "${SFT_OUTPUT_DIR}" \
    --load "${CKPT_MCORE_DIR}" \
    --log-interval 1 \
    --eval-interval 999999 \
    --eval-iters 0 \
    --no-load-optim \
    --no-load-rng \
    --no-gradient-accumulation-fusion \
    --add-qkv-bias \
    --add-dense-bias \
    --no-bias-swiglu-fusion \
    --normalization RMSNorm \
    --norm-epsilon 1e-5 \
    --swiglu \
    --use-rotary-position-embeddings \
    --rotary-percent 1.0 \
    --rotary-base 4000000 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --use-flash-attn \
    --variable-seq-lengths \
    --distributed-backend hccl \
    2>&1 | tee "${TRAIN_LOG}"

echo "[train] 完成: checkpoint → ${SFT_OUTPUT_DIR}"
echo "[train]       日志    → ${TRAIN_LOG}"
