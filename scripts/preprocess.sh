#!/bin/bash
set -euo pipefail
source "${WORKDIR:-$(dirname "$0")/..}/env.sh"
export PATH=/home/service/.local/bin:$PATH

echo "[INFO] preprocess.sh started"
mkdir -p "$WORKDIR/cache"
cd "$WORKDIR/MindSpeed-LLM"

SEQ_LENGTH="${SEQ_LENGTH:-4096}"
MODEL_DIR="$WORKDIR/openPangu-Embedded-1B-V1.1"

echo "[INFO] 输入文件:"
ls -lh "$WORKDIR/data/train_imdb.jsonl"
ls -ld "$MODEL_DIR"
ls -lh "$WORKDIR/MindSpeed-LLM/preprocess_data.py"

python preprocess_data.py \
    --input "$WORKDIR/data/train_imdb.jsonl" \
    --tokenizer-name-or-path "$MODEL_DIR" \
    --output-prefix "$WORKDIR/cache/sft" \
    --workers 4 \
    --tokenizer-type PretrainedFromHF \
    --handler-name GeneralInstructionHandler \
    --seq-length "$SEQ_LENGTH" \
    --pack

echo "[OK] 预处理完成。"
find "$WORKDIR/cache" -maxdepth 1 -type f -exec ls -lh {} \;
