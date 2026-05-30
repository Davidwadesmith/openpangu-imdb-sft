#!/bin/bash
set -euo pipefail

# ============================================================
#  openPangu-1B IMDB 情感分类微调 — 一键运行脚本
#  使用方法：bash run_all.sh
#  服务器：昇腾 910B NPU
# ============================================================

# ===== 颜色输出 =====
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC}  $(date '+%H:%M:%S') $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $(date '+%H:%M:%S') $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $(date '+%H:%M:%S') $*"; }

# ===== 路径配置 =====
WYC_DIR="$(cd "$(dirname "$0")/../../wyc" && pwd)"

# 验证共享目录是否存在
if [ ! -d "${WYC_DIR}" ]; then
    log_error "共享目录不存在: ${WYC_DIR}"
    log_error "请确认 wyc 目录在正确的位置（/mnt/workspace/wyc/）"
    exit 1
fi
MS_LLM_DIR="${WYC_DIR}/MindSpeed-LLM"
MINDSPEED_DIR="${WYC_DIR}/MindSpeed"
MODEL_HF_DIR="${WYC_DIR}/openPangu-Embedded-1B-V1.1"
DATA_DIR="${WYC_DIR}/data"
DOWNLOADS_DIR="${WYC_DIR}/downloads"

WORK_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPTS_DIR="${WORK_DIR}/scripts"
CACHE_DIR="${WORK_DIR}/cache"
CKPT_MCORE_DIR="${WORK_DIR}/ckpt/mcore"
SFT_OUTPUT_DIR="${WORK_DIR}/sft_output"
CKPT_MG2HF_DIR="${WORK_DIR}/ckpt/mg2hf"
RESULTS_DIR="${WORK_DIR}/results"
LOGS_DIR="${WORK_DIR}/logs"

# ===== 训练参数 =====
LR="2e-5"
GBS=16
MBS=1
SEQ_LENGTH=8192
TRAIN_ITERS=1563
PRECISION="bf16"

# ===== 目录初始化 =====
mkdir -p "${CACHE_DIR}" "${CKPT_MCORE_DIR}" "${SFT_OUTPUT_DIR}" \
         "${CKPT_MG2HF_DIR}" "${RESULTS_DIR}" "${LOGS_DIR}"

log_info "工作目录: ${WORK_DIR}"
log_info "共享目录: ${WYC_DIR}"
log_info "所有步骤将依次执行，出错即停..."

# ============================================================
#  Step 0: 环境初始化（安装 MindSpeed / MindSpeed-LLM）
# ============================================================
env_setup() {
    log_info "Step 0/8: 检查 Python 环境..."

    # 检查当前安装的 mindspeed 是否指向 wyc 目录
    CURRENT_MS=$(python -c "import mindspeed; print(mindspeed.__file__)" 2>/dev/null || echo "")
    if echo "${CURRENT_MS}" | grep -q "${MINDSPEED_DIR}"; then
        log_warn "mindspeed 已指向 wyc 版本，跳过环境初始化"
        return
    fi

    log_info "  强制安装 wyc 版本的 MindSpeed..."
    pip install -e "${MINDSPEED_DIR}" --quiet
    log_info "  强制安装 wyc 版本的 MindSpeed-LLM..."
    pip install -e "${MS_LLM_DIR}" --quiet
    log_info "Step 0 完成: 环境就绪"
}

# ============================================================
#  Step 1: 数据格式转换（Parquet → JSONL）
# ============================================================
step1_prepare_data() {
    log_info "Step 1/8: 数据格式转换..."

    if [ -f "${DATA_DIR}/train_imdb.jsonl" ]; then
        log_warn "train_imdb.jsonl 已存在，跳过数据转换"
        return
    fi

    python "${SCRIPTS_DIR}/prepare_data.py"
    log_info "Step 1 完成: ${DATA_DIR}/train_imdb.jsonl"
}

# ============================================================
#  Step 2: 数据预处理（生成 .bin / .idx 缓存）
# ============================================================
step2_preprocess() {
    log_info "Step 2/8: 数据预处理..."

    if [ -f "${CACHE_DIR}/sft_text_document.bin" ] && [ -f "${CACHE_DIR}/sft_text_document.idx" ]; then
        log_warn "缓存文件已存在，跳过预处理"
        return
    fi

    cd "${MS_LLM_DIR}"
    python preprocess_data.py \
        --input "${DATA_DIR}" \
        --tokenizer-name-or-path "${MODEL_HF_DIR}" \
        --output-prefix "${CACHE_DIR}/sft" \
        --workers 4 \
        --tokenizer-type PretrainedFromHF \
        --handler-name PanguInstructionHandler \
        --seq-length "${SEQ_LENGTH}"
    cd "${WORK_DIR}"

    log_info "Step 2 完成: ${CACHE_DIR}/sft_text_document.bin + .idx"
}

# ============================================================
#  Step 3: 模型权重转换 HF → mcore
# ============================================================
step3_convert_hf2mcore() {
    log_info "Step 3/8: 权重转换 HF → mcore..."

    if [ -f "${CKPT_MCORE_DIR}/latest_checkpointed_iteration.txt" ]; then
        log_warn "mcore 权重已存在，跳过转换"
        return
    fi

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
        --params-dtype "${PRECISION}" \
        --use-mcore-models
    cd "${WORK_DIR}"

    log_info "Step 3 完成: ${CKPT_MCORE_DIR}"
}

# ============================================================
#  Step 4: SFT 训练
# ============================================================
step4_train() {
    log_info "Step 4/8: 启动 SFT 训练..."
    log_info "  LR=${LR}  GBS=${GBS}  MBS=${MBS}  Iters=${TRAIN_ITERS}  SeqLen=${SEQ_LENGTH}"

    # 设置日志输出路径
    TRAIN_LOG="${LOGS_DIR}/tune_mcore_pangu_1b_full_ptd.log"

    cd "${MS_LLM_DIR}"

    # 基于 MindSpeed-LLM llama3 模板，适配 openPangu-1B 架构参数
    # 模型架构参数来自 openPangu-Embedded-1B-V1.1 的 config.json
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

    cd "${WORK_DIR}"

    log_info "Step 4 完成: 训练日志 → ${TRAIN_LOG}"
    log_info "  Checkpoint → ${SFT_OUTPUT_DIR}"
}

# ============================================================
#  Step 5: 权重转换 mcore → HF（用于推理）
# ============================================================
step5_convert_mcore2hf() {
    log_info "Step 5/8: 权重转换 mcore → HF..."

    # 找到最新的 checkpoint
    LATEST_CKPT=$(ls -d "${SFT_OUTPUT_DIR}"/iter_* 2>/dev/null | sort -V | tail -1)
    if [ -z "${LATEST_CKPT}" ]; then
        log_error "未找到 SFT checkpoint，请检查训练是否成功"
        exit 1
    fi
    log_info "  使用 checkpoint: ${LATEST_CKPT}"

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
        --params-dtype "${PRECISION}" \
        --use-mcore-models
    cd "${WORK_DIR}"

    # 拷贝 tokenizer 和模型配置文件到 mg2hf 目录
    log_info "  拷贝 tokenizer/配置文件..."
    cp "${MODEL_HF_DIR}/tokenizer.model" "${CKPT_MG2HF_DIR}/" 2>/dev/null || true
    cp "${MODEL_HF_DIR}/tokenizer_config.json" "${CKPT_MG2HF_DIR}/" 2>/dev/null || true
    cp "${MODEL_HF_DIR}/tokenization_openpangu.py" "${CKPT_MG2HF_DIR}/" 2>/dev/null || true
    cp "${MODEL_HF_DIR}/special_tokens_map.json" "${CKPT_MG2HF_DIR}/" 2>/dev/null || true
    cp "${MODEL_HF_DIR}/generation_config.json" "${CKPT_MG2HF_DIR}/" 2>/dev/null || true
    cp "${MODEL_HF_DIR}/configuration_openpangu_dense.py" "${CKPT_MG2HF_DIR}/" 2>/dev/null || true
    cp "${MODEL_HF_DIR}/modeling_openpangu_dense.py" "${CKPT_MG2HF_DIR}/" 2>/dev/null || true

    log_info "Step 5 完成: ${CKPT_MG2HF_DIR}"
}

# ============================================================
#  Step 6: 推理
# ============================================================
step6_inference() {
    log_info "Step 6/8: 模型推理（在测试集上）..."

    if [ ! -f "${DOWNLOADS_DIR}/test-00000-of-00001.parquet" ]; then
        log_error "测试集文件不存在: ${DOWNLOADS_DIR}/test-00000-of-00001.parquet"
        exit 1
    fi

    python "${SCRIPTS_DIR}/inference.py" \
        --model-path "${CKPT_MG2HF_DIR}" \
        --test-parquet "${DOWNLOADS_DIR}/test-00000-of-00001.parquet" \
        --output "${RESULTS_DIR}/test_imdb_results.jsonl"

    log_info "Step 6 完成: ${RESULTS_DIR}/test_imdb_results.jsonl"
}

# ============================================================
#  Step 7: 评测准确率
# ============================================================
step7_evaluate() {
    log_info "Step 7/8: 评测准确率..."

    python "${SCRIPTS_DIR}/evaluate.py" \
        --input "${RESULTS_DIR}/test_imdb_results.jsonl"

    log_info "Step 7 完成"
}

# ============================================================
#  Step 8: 绘制 loss 曲线
# ============================================================
step8_plot_loss() {
    log_info "Step 8/8: 绘制 loss 曲线..."

    python "${SCRIPTS_DIR}/plot_loss.py" \
        --log-file "${LOGS_DIR}/tune_mcore_pangu_1b_full_ptd.log" \
        --output "${WORK_DIR}/loss_curve.png"

    log_info "Step 8 完成: ${WORK_DIR}/loss_curve.png"
}

# ============================================================
#  主流程
# ============================================================
main() {
    echo ""
    echo "============================================"
    echo "  openPangu-1B IMDB 情感分类微调"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================"
    echo ""

    START_TIME=$(date +%s)

    env_setup
    step1_prepare_data
    step2_preprocess
    step3_convert_hf2mcore
    step4_train
    step5_convert_mcore2hf
    step6_inference
    step7_evaluate
    step8_plot_loss

    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))

    echo ""
    echo "============================================"
    echo -e "  ${GREEN}全部步骤完成!${NC}"
    echo "  总耗时: $((DURATION / 60)) 分 $((DURATION % 60)) 秒"
    echo "============================================"
    echo "  输出文件:"
    echo "    cache/          → 预处理缓存"
    echo "    ckpt/mcore/     → mcore 格式权重"
    echo "    sft_output/     → SFT 训练 checkpoint"
    echo "    ckpt/mg2hf/     → HF 格式微调模型（推理用）"
    echo "    results/        → 推理结果 JSONL"
    echo "    logs/           → 训练日志"
    echo "    loss_curve.png  → Loss 曲线图"
    echo "============================================"
}

main "$@"
