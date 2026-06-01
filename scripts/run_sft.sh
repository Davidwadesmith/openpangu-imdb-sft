#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../env.sh"
export PATH=/home/service/.local/bin:$PATH

SEQ_LENGTH="${SEQ_LENGTH:-4096}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-4}"
TRAIN_ITERS="${TRAIN_ITERS:-300}"
LR="${LR:-1e-5}"

MODEL_DIR="$WORKDIR/openPangu-Embedded-1B-V1.1"
CKPT_LOAD_DIR="$WORKDIR/ckpt/mcore"
CKPT_SAVE_DIR="$WORKDIR/sft_output"
LOG_DIR="$WORKDIR/logs"
LOG_FILE="$LOG_DIR/tune_mcore_pangu_1b_full_ptd.log"

mkdir -p "$CKPT_SAVE_DIR" "$LOG_DIR"

if [ -f "$WORKDIR/cache/sft_packed_attention_mask_document.bin" ]; then
    DATA_PREFIX="$WORKDIR/cache/sft_packed_attention_mask_document"
elif [ -f "$WORKDIR/cache/sft_packed_text_document.bin" ]; then
    DATA_PREFIX="$WORKDIR/cache/sft_packed_text_document"
elif [ -f "$WORKDIR/cache/sft_text_document.bin" ]; then
    DATA_PREFIX="$WORKDIR/cache/sft_text_document"
else
    DATA_PREFIX="$(find "$WORKDIR/cache" -maxdepth 1 -type f -name "*.bin" -print -quit | sed 's/\.bin$//')"
fi

if [ -z "$DATA_PREFIX" ] || [ ! -f "${DATA_PREFIX}.bin" ] || [ ! -f "${DATA_PREFIX}.idx" ]; then
    echo "[ERROR] No usable .bin/.idx cache found under $WORKDIR/cache"
    find "$WORKDIR/cache" -maxdepth 1 -type f -exec ls -lh {} \;
    exit 1
fi

if [ ! -d "$CKPT_LOAD_DIR" ]; then
    echo "[ERROR] Initial mcore checkpoint not found: $CKPT_LOAD_DIR"
    exit 1
fi

if [ -f "$LOG_FILE" ]; then
    cp "$LOG_FILE" "$LOG_FILE.before_$(date '+%Y%m%d_%H%M%S')"
fi

echo "[INFO] run_sft.sh started"
echo "[INFO] DATA_PREFIX=$DATA_PREFIX"
echo "[INFO] CKPT_LOAD_DIR=$CKPT_LOAD_DIR"
echo "[INFO] CKPT_SAVE_DIR=$CKPT_SAVE_DIR"
echo "[INFO] SEQ_LENGTH=$SEQ_LENGTH GLOBAL_BATCH_SIZE=$GLOBAL_BATCH_SIZE TRAIN_ITERS=$TRAIN_ITERS LR=$LR"
echo "[INFO] START_TIME=$(date '+%F %T')" | tee "$LOG_FILE"

export CUDA_DEVICE_MAX_CONNECTIONS=1

cd "$WORKDIR/MindSpeed-LLM"
torchrun --nproc_per_node 1 pretrain_gpt.py \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --num-layers 26 \
    --hidden-size 1536 \
    --ffn-hidden-size 6144 \
    --num-attention-heads 12 \
    --group-query-attention \
    --num-query-groups 6 \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path "$MODEL_DIR" \
    --seq-length "$SEQ_LENGTH" \
    --max-position-embeddings 32768 \
    --micro-batch-size 1 \
    --global-batch-size "$GLOBAL_BATCH_SIZE" \
    --make-vocab-size-divisible-by 1 \
    --padded-vocab-size 153376 \
    --lr "$LR" \
    --train-iters "$TRAIN_ITERS" \
    --lr-decay-style cosine \
    --disable-bias-linear \
    --attention-dropout 0.0 \
    --init-method-std 0.01 \
    --hidden-dropout 0.0 \
    --position-embedding-type rope \
    --normalization RMSNorm \
    --swiglu \
    --use-flash-attn \
    --use-fused-rmsnorm \
    --use-fused-swiglu \
    --use-fused-rotary-pos-emb \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --min-lr 1e-6 \
    --weight-decay 1e-1 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --initial-loss-scale 4096 \
    --no-load-optim \
    --no-load-rng \
    --bf16 \
    --load "$CKPT_LOAD_DIR" \
    --save "$CKPT_SAVE_DIR" \
    --data-path "$DATA_PREFIX" \
    --split 80,10,10 \
    --log-interval 10 \
    --save-interval "$TRAIN_ITERS" \
    --eval-interval 1000 \
    --eval-iters 10 \
    --finetune \
    --use-mcore-models \
    --recompute-granularity full \
    --recompute-method block \
    --recompute-num-layers 26 \
    2>&1 | tee -a "$LOG_FILE"

if [ ! -f "$CKPT_SAVE_DIR/latest_checkpointed_iteration.txt" ]; then
    echo "[ERROR] SFT finished without a saved checkpoint: $CKPT_SAVE_DIR" | tee -a "$LOG_FILE"
    exit 1
fi

if ! grep -q "| lm loss:" "$LOG_FILE"; then
    echo "[ERROR] SFT finished without training loss entries: $LOG_FILE" | tee -a "$LOG_FILE"
    exit 1
fi

echo "[INFO] END_TIME=$(date '+%F %T')" | tee -a "$LOG_FILE"
echo "[OK] SFT completed: $LOG_FILE"
