#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../env.sh"
export PATH=/home/service/.local/bin:$PATH

SEQ_LENGTH="${SEQ_LENGTH:-4096}"
MODEL_DIR="$WORKDIR/openPangu-Embedded-1B-V1.1"

echo "[INFO] preprocess.sh started"
mkdir -p "$WORKDIR/cache"

ls -lh "$WORKDIR/data/train_imdb.jsonl"
ls -ld "$MODEL_DIR"
ls -lh "$WORKDIR/MindSpeed-LLM/preprocess_data.py"

cd "$WORKDIR/MindSpeed-LLM"
python preprocess_data.py \
    --input "$WORKDIR/data/train_imdb.jsonl" \
    --tokenizer-name-or-path "$MODEL_DIR" \
    --output-prefix "$WORKDIR/cache/sft" \
    --workers 4 \
    --tokenizer-type PretrainedFromHF \
    --handler-name GeneralInstructionHandler \
    --seq-length "$SEQ_LENGTH" \
    --pack

echo "[OK] Preprocessing completed"
find "$WORKDIR/cache" -maxdepth 1 -type f -exec ls -lh {} \;
