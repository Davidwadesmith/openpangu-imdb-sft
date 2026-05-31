#!/bin/bash
set -euo pipefail

# ============================================================
#  openPangu-1B IMDB 情感分类微调 — 一键运行脚本（完全自包含）
#  自动下载：MindSpeed/MindSpeed-LLM、模型权重、IMDB 数据集
#  使用方法：bash run_all.sh
# ============================================================

# ===== 颜色输出 =====
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'
log_info()  { echo -e "${GREEN}[INFO]${NC}  $(date '+%H:%M:%S') $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $(date '+%H:%M:%S') $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $(date '+%H:%M:%S') $*"; }

# ===== 路径配置（全部在项目目录内）=====
WORK_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPTS_DIR="${WORK_DIR}/scripts"

MODEL_DIR="${WORK_DIR}/model"                  # HuggingFace 模型
DOWNLOADS_DIR="${WORK_DIR}/downloads"          # IMDB parquet 文件
DATA_DIR="${WORK_DIR}/data"                    # train_imdb.jsonl
CACHE_DIR="${WORK_DIR}/cache"                  # 预处理 .bin/.idx
CKPT_MCORE_DIR="${WORK_DIR}/ckpt/mcore"        # mcore 格式权重
SFT_OUTPUT_DIR="${WORK_DIR}/sft_output"         # 训练 checkpoint
CKPT_MG2HF_DIR="${WORK_DIR}/ckpt/mg2hf"         # HF 格式微调模型
RESULTS_DIR="${WORK_DIR}/results"               # 推理结果
LOGS_DIR="${WORK_DIR}/logs"                     # 训练日志

MS_LLM_DIR=""  # env_setup 会设置
export PYTHONPATH=""        # env_setup 会追加 MindSpeed/MindSpeed-LLM

# HuggingFace 镜像
HF_MIRROR="https://hf-mirror.com"
export HF_ENDPOINT="${HF_MIRROR}"

# 模型标识
MODEL_HF_ID="FreedomIntelligence/openPangu-Embedded-1B-V1.1"

# ===== 训练参数 =====
LR="${LR:-2e-5}"
GBS="${GBS:-16}"
MBS="${MBS:-1}"
SEQ_LENGTH="${SEQ_LENGTH:-8192}"
TRAIN_ITERS="${TRAIN_ITERS:-1563}"

# ===== 目录初始化 =====
mkdir -p "${DOWNLOADS_DIR}" "${DATA_DIR}" "${CACHE_DIR}" \
         "${CKPT_MCORE_DIR}" "${SFT_OUTPUT_DIR}" "${CKPT_MG2HF_DIR}" \
         "${RESULTS_DIR}" "${LOGS_DIR}"

log_info "工作目录: ${WORK_DIR}"
log_info "HF 镜像:   ${HF_MIRROR}"

# ============================================================
#  Step 0: 环境搭建（下载框架、模型、数据集）
# ============================================================
env_setup() {
    log_info "============================================="
    log_info "Step 0: 环境搭建"
    log_info "============================================="

    # ---- 0a. 获取 MindSpeed / MindSpeed-LLM / Megatron-LM ----
    log_info "--- 0a. 获取 MindSpeed / MindSpeed-LLM / Megatron-LM ---"

    # 清除之前失败残留的本地烂仓库
    rm -rf "${WORK_DIR}/MindSpeed" "${WORK_DIR}/MindSpeed-LLM" "${WORK_DIR}/Megatron-LM"

    # 1) Megatron-LM（MindSpeed 的底层依赖，必须）
    log_info "克隆 Megatron-LM..."
    git clone --depth 1 --branch core_v0.12.0 \
        https://github.com/NVIDIA/Megatron-LM.git "${WORK_DIR}/Megatron-LM" 2>&1 | tail -1 || {
        log_info "github 不通或分支不存在，尝试 gitee master..."
        git clone --depth 1 \
            https://gitee.com/ascend/Megatron-LM.git "${WORK_DIR}/Megatron-LM" 2>&1 | tail -1
    }

    # 2) MindSpeed（用主分支，不 pin 特定 tag）
    log_info "克隆 MindSpeed..."
    git clone --depth 1 \
        https://gitee.com/ascend/MindSpeed.git "${WORK_DIR}/MindSpeed" 2>&1 | tail -1

    # 3) MindSpeed-LLM
    log_info "克隆 MindSpeed-LLM..."
    git clone --depth 1 --branch v1.0.0 \
        https://gitee.com/ascend/MindSpeed-LLM.git "${WORK_DIR}/MindSpeed-LLM" 2>&1 | tail -1
    MS_LLM_DIR="${WORK_DIR}/MindSpeed-LLM"

    # 用 PYTHONPATH 加载（不 pip install，C 扩展编译会失败）
    export PYTHONPATH="${WORK_DIR}/Megatron-LM:${MS_LLM_DIR}:${WORK_DIR}/MindSpeed:${PYTHONPATH:-}"
    log_info "PYTHONPATH 已设置（Megatron-LM + MindSpeed + MindSpeed-LLM）"

    # 验证 import
    if python3 -c "import megatron.core; print('  megatron.core OK')" 2>/dev/null; then
        log_info "megatron.core 就绪"
    else
        log_warn "megatron.core 导入失败，将在 Step 2 实际运行时再检查"
    fi

    # ---- 0b. 下载 openPangu-1B 模型 ----
    log_info "--- 0b. 下载模型: ${MODEL_HF_ID} ---"
    if [ -f "${MODEL_DIR}/config.json" ] && [ -f "${MODEL_DIR}/model.safetensors" ]; then
        log_warn "模型已存在，跳过下载"
    else
        log_info "从 ${HF_MIRROR} 下载（约 2GB）..."
        python "${SCRIPTS_DIR}/download_model.py" \
            --model-id "${MODEL_HF_ID}" \
            --save-dir "${MODEL_DIR}"
        log_info "模型下载完成: ${MODEL_DIR}"
    fi

    # ---- 0c. 下载 IMDB 数据集 ----
    log_info "--- 0c. 下载 IMDB 数据集 ---"
    TRAIN_PQ="${DOWNLOADS_DIR}/train-00000-of-00001.parquet"
    TEST_PQ="${DOWNLOADS_DIR}/test-00000-of-00001.parquet"

    if [ -f "${TRAIN_PQ}" ] && [ -f "${TEST_PQ}" ]; then
        # 还要检查文件大小（之前可能下载了无效的 15 字节文件）
        TRAIN_SIZE=$(stat -c%s "${TRAIN_PQ}" 2>/dev/null || stat -f%z "${TRAIN_PQ}" 2>/dev/null || echo 0)
        TEST_SIZE=$(stat -c%s "${TEST_PQ}" 2>/dev/null || stat -f%z "${TEST_PQ}" 2>/dev/null || echo 0)
        if [ "${TRAIN_SIZE}" -gt 10000 ] && [ "${TEST_SIZE}" -gt 10000 ]; then
            log_warn "IMDB 数据集已存在，跳过下载"
            return
        else
            log_warn "已有文件无效（train=${TRAIN_SIZE}B, test=${TEST_SIZE}B），重新下载"
        fi
    fi

    log_info "下载 IMDB 数据集（使用 Python huggingface_hub API）..."
    python "${SCRIPTS_DIR}/download_data.py" --save-dir "${DOWNLOADS_DIR}" --split train
    python "${SCRIPTS_DIR}/download_data.py" --save-dir "${DOWNLOADS_DIR}" --split test
    log_info "数据集下载完成: ${DOWNLOADS_DIR}"

    log_info "Step 0 完成: 环境就绪"
}

# ============================================================
#  Step 1: 数据格式转换（Parquet → JSONL）
# ============================================================
step1_prepare_data() {
    log_info "Step 1/8: 数据格式转换 (Parquet → JSONL)..."

    if [ -f "${DATA_DIR}/train_imdb.jsonl" ]; then
        log_warn "train_imdb.jsonl 已存在，跳过"
        return
    fi

    python "${SCRIPTS_DIR}/prepare_data.py" \
        --input "${DOWNLOADS_DIR}/train-00000-of-00001.parquet" \
        --output "${DATA_DIR}/train_imdb.jsonl"

    log_info "Step 1 完成: ${DATA_DIR}/train_imdb.jsonl"
}

# ============================================================
#  Step 2: 数据预处理（tokenize → .bin / .idx）
# ============================================================
step2_preprocess() {
    log_info "Step 2/8: 数据预处理..."

    if [ -f "${CACHE_DIR}/sft_text_document.bin" ] && \
       [ -f "${CACHE_DIR}/sft_text_document.idx" ]; then
        log_warn "缓存已存在，跳过"
        return
    fi

    python "${SCRIPTS_DIR}/preprocess.py" \
        --input "${DATA_DIR}/train_imdb.jsonl" \
        --tokenizer-path "${MODEL_DIR}" \
        --output-prefix "${CACHE_DIR}/sft" \
        --seq-length "${SEQ_LENGTH}"

    log_info "Step 2 完成"
}

# ============================================================
#  Step 3: 模型权重转换 HF → mcore
# ============================================================
step3_convert_hf2mcore() {
    log_info "Step 3/8: HF → mcore..."

    if [ -f "${CKPT_MCORE_DIR}/latest_checkpointed_iteration.txt" ]; then
        log_warn "mcore 权重已存在，跳过"
        return
    fi

    python "${MS_LLM_DIR}/convert_ckpt.py" \
        --model-type GPT \
        --load-model-type hf \
        --save-model-type mg \
        --load-dir "${MODEL_DIR}" \
        --save-dir "${CKPT_MCORE_DIR}" \
        --tokenizer-model "${MODEL_DIR}" \
        --add-qkv-bias \
        --add-dense-bias \
        --target-tensor-parallel-size 1 \
        --target-pipeline-parallel-size 1 \
        --params-dtype bf16 \
        --use-mcore-models

    log_info "Step 3 完成"
}

# ============================================================
#  Step 4: SFT 训练
# ============================================================
step4_train() {
    log_info "Step 4/8: SFT 训练"
    log_info "  LR=${LR} GBS=${GBS} MBS=${MBS} Iters=${TRAIN_ITERS}"

    TRAIN_LOG="${LOGS_DIR}/train.log"

    torchrun --nproc_per_node=1 \
        "${MS_LLM_DIR}/posttrain_gpt.py" \
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
        --tokenizer-name-or-path "${MODEL_DIR}" \
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

    log_info "Step 4 完成: ${TRAIN_LOG}"
}

# ============================================================
#  Step 5: 权重转换 mcore → HF
# ============================================================
step5_convert_mcore2hf() {
    log_info "Step 5/8: mcore → HF..."

    LATEST_CKPT=$(ls -d "${SFT_OUTPUT_DIR}"/iter_* 2>/dev/null | sort -V | tail -1)
    if [ -z "${LATEST_CKPT}" ]; then
        log_error "未找到 SFT checkpoint"
        exit 1
    fi
    log_info "  checkpoint: ${LATEST_CKPT}"

    python "${MS_LLM_DIR}/convert_ckpt.py" \
        --model-type GPT \
        --load-model-type mg \
        --save-model-type hf \
        --load-dir "${LATEST_CKPT}" \
        --save-dir "${CKPT_MG2HF_DIR}" \
        --tokenizer-model "${MODEL_DIR}" \
        --add-qkv-bias \
        --add-dense-bias \
        --target-tensor-parallel-size 1 \
        --target-pipeline-parallel-size 1 \
        --params-dtype bf16 \
        --use-mcore-models

    log_info "拷贝 tokenizer/model 文件..."
    for f in tokenizer.model tokenizer_config.json tokenization_openpangu.py \
             special_tokens_map.json generation_config.json \
             configuration_openpangu_dense.py modeling_openpangu_dense.py; do
        cp "${MODEL_DIR}/${f}" "${CKPT_MG2HF_DIR}/" 2>/dev/null || true
    done

    log_info "Step 5 完成"
}

# ============================================================
#  Step 6: 推理
# ============================================================
step6_inference() {
    log_info "Step 6/8: 推理..."

    python "${SCRIPTS_DIR}/inference.py" \
        --model-path "${CKPT_MG2HF_DIR}" \
        --test-parquet "${DOWNLOADS_DIR}/test-00000-of-00001.parquet" \
        --output "${RESULTS_DIR}/test_imdb_results.jsonl"

    log_info "Step 6 完成"
}

# ============================================================
#  Step 7: 评测
# ============================================================
step7_evaluate() {
    log_info "Step 7/8: 评测..."

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
        --log-file "${LOGS_DIR}/train.log" \
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
    echo -e "  ${GREEN}全部完成!${NC}"
    echo "  总耗时: $((DURATION / 60)) 分 $((DURATION % 60)) 秒"
    echo "============================================"
}

main "$@"
