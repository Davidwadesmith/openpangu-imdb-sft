#!/bin/bash
set -euo pipefail

# ============================================================
#  openPangu-1B IMDB 情感分类 SFT — 一键运行脚本
#  用法: bash run_all.sh
#  首次运行会自动搭建虚拟环境、下载所有依赖
#  后续运行会跳过已完成步骤
# ============================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'
log_info()  { echo -e "${GREEN}[INFO]${NC}  $(date '+%H:%M:%S') $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $(date '+%H:%M:%S') $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $(date '+%H:%M:%S') $*"; }

WORKDIR="$(cd "$(dirname "$0")" && pwd)"
export WORKDIR
SCRIPTS_DIR="$WORKDIR/scripts"
mkdir -p "$WORKDIR/cache" "$WORKDIR/ckpt" "$WORKDIR/sft_output" \
         "$WORKDIR/logs" "$WORKDIR/results" "$WORKDIR/downloads" "$WORKDIR/data"

# ===== 环境加载 =====
if [ -f "$WORKDIR/env.sh" ]; then
    source "$WORKDIR/env.sh"
fi

# ============================================================
#  Step 0: 环境搭建（首次运行）
# ============================================================
step0_setup() {
    log_info "============================================="
    log_info "Step 0: 环境搭建"
    log_info "============================================="

    # --- 0a. 虚拟环境 ---
    if [ ! -f "$WORKDIR/.venv/bin/activate" ]; then
        log_info "创建虚拟环境..."
        python3.10 -m venv --system-site-packages "$WORKDIR/.venv"
        source "$WORKDIR/.venv/bin/activate"
        pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple -q
        pip install ninja pybind11 transformers==4.53.2 datasets pandas \
            pyarrow matplotlib tqdm sentencepiece safetensors \
            -i https://pypi.tuna.tsinghua.edu.cn/simple -q
    else
        source "$WORKDIR/.venv/bin/activate"
    fi

    # --- 0b. MindSpeed-LLM 1.0.0 ---
    if [ ! -f "$WORKDIR/MindSpeed-LLM/convert_ckpt.py" ]; then
        log_info "克隆 MindSpeed-LLM 1.0.0..."
        git clone https://gitee.com/ascend/MindSpeed-LLM.git "$WORKDIR/MindSpeed-LLM"
        cd "$WORKDIR/MindSpeed-LLM"
        git checkout 1.0.0
        cd "$WORKDIR"
    fi

    # --- 0c. Megatron-LM core_r0.6.0 + 嵌入 ---
    MEG_DIR="$WORKDIR/MindSpeed-LLM/megatron"
    if [ ! -d "$MEG_DIR/core" ]; then
        log_info "克隆 Megatron-LM core_r0.6.0..."
        git clone https://gitee.com/mirrors/Megatron-LM.git "$WORKDIR/Megatron-LM"
        cd "$WORKDIR/Megatron-LM"
        git checkout core_r0.6.0
        cd "$WORKDIR"
        rm -rf "$WORKDIR/MindSpeed-LLM/megatron"
        cp -r "$WORKDIR/Megatron-LM/megatron" "$WORKDIR/MindSpeed-LLM/"
        log_info "Megatron 已嵌入 MindSpeed-LLM"
    fi

    # --- 0d. MindSpeed 稳定 commit ---
    if [ ! -f "$WORKDIR/MindSpeed/setup.py" ]; then
        log_info "克隆 MindSpeed..."
        git clone https://gitee.com/ascend/MindSpeed.git "$WORKDIR/MindSpeed"
        cd "$WORKDIR/MindSpeed"
        git checkout 969686ff
        cd "$WORKDIR"
    fi

    # --- 0e. openPangu 模型 ---
    MODEL_DIR="$WORKDIR/openPangu-Embedded-1B-V1.1"
    if [ ! -f "$MODEL_DIR/model.safetensors" ]; then
        log_info "下载 openPangu-1B 模型..."
        git clone https://gitcode.com/ascend-tribe/openPangu-Embedded-1B-V1.1.git "$MODEL_DIR"
    fi
    MODEL_SIZE=$(stat -c%s "$MODEL_DIR/model.safetensors" 2>/dev/null || echo 0)
    log_info "模型权重: $MODEL_DIR/model.safetensors ($(( MODEL_SIZE / 1024 / 1024 / 1024 ))GB)"

    # --- 0f. 编译 Megatron helpers ---
    HELPERS_SO="$WORKDIR/MindSpeed-LLM/megatron/core/datasets/helpers.cpython-310-aarch64-linux-gnu.so"
    if [ ! -f "$HELPERS_SO" ]; then
        log_info "编译 Megatron helpers..."
        pip install ninja pybind11 -i https://pypi.tuna.tsinghua.edu.cn/simple -q
        cd "$WORKDIR/MindSpeed-LLM"
        cat > build_megatron_helpers.py <<'PYEOF'
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension
import os, sys
# 确保 megatron 在 sys.path
sys.path.insert(0, ".")
setup(
    name="megatron_core_datasets_helpers",
    ext_modules=[
        CppExtension(
            name="megatron.core.datasets.helpers",
            sources=["megatron/core/datasets/helpers.cpp"],
            extra_compile_args=["-O3", "-std=c++17"],
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
)
PYEOF
        python build_megatron_helpers.py build_ext --inplace
        cd "$WORKDIR"
    fi

    source "$WORKDIR/env.sh"
    log_info "Step 0 完成"
}

# ============================================================
#  Step 1: 数据下载与转换
# ============================================================
step1_prepare_data() {
    log_info "Step 1/8: 数据下载与格式转换..."
    if [ -f "$WORKDIR/data/train_imdb.jsonl" ]; then
        log_warn "train_imdb.jsonl 已存在，跳过"
        return
    fi
    source "$WORKDIR/env.sh"
    python "$SCRIPTS_DIR/prepare_data.py"
}

# ============================================================
#  Step 2: 数据预处理
# ============================================================
step2_preprocess() {
    log_info "Step 2/8: 数据预处理..."
    local cache="$WORKDIR/cache"
    if ls "$cache"/*.bin "$cache"/*.idx 2>/dev/null | head -1 | grep -q .; then
        log_warn "缓存已存在，跳过"
        return
    fi
    source "$WORKDIR/env.sh"
    bash "$SCRIPTS_DIR/preprocess.sh"
}

# ============================================================
#  Step 3: HF → mcore 转换
# ============================================================
step3_convert_hf2mcore() {
    log_info "Step 3/8: HF → mcore 权重转换..."
    if [ -f "$WORKDIR/ckpt/mcore/latest_checkpointed_iteration.txt" ]; then
        log_warn "mcore 权重已存在，跳过"
        return
    fi
    source "$WORKDIR/env.sh"
    bash "$SCRIPTS_DIR/convert_hf2mcore.sh"
}

# ============================================================
#  Step 4: SFT 训练
# ============================================================
step4_train() {
    log_info "Step 4/8: SFT 训练..."
    log_info "  SEQ=4096 GBS=4 ITERS=300 LR=1e-5"
    source "$WORKDIR/env.sh"
    bash "$SCRIPTS_DIR/run_sft.sh"
}

# ============================================================
#  Step 5: mcore → HF 转换
# ============================================================
step5_convert_mcore2hf() {
    log_info "Step 5/8: mcore → HF 转换..."
    source "$WORKDIR/env.sh"
    bash "$SCRIPTS_DIR/convert_mcore2hf.sh"
}

# ============================================================
#  Step 6: 推理
# ============================================================
step6_inference() {
    log_info "Step 6/8: 推理..."
    source "$WORKDIR/env.sh"
    python "$SCRIPTS_DIR/inference.py"
}

# ============================================================
#  Step 7: 评测
# ============================================================
step7_evaluate() {
    log_info "Step 7/8: 评测准确率..."
    source "$WORKDIR/env.sh"
    python "$SCRIPTS_DIR/evaluate.py"
}

# ============================================================
#  Step 8: 绘制 loss 曲线
# ============================================================
step8_plot_loss() {
    log_info "Step 8/8: 绘制 loss 曲线..."
    source "$WORKDIR/env.sh"
    python "$SCRIPTS_DIR/plot_loss.py"
}

# ============================================================
#  主流程
# ============================================================
main() {
    echo ""
    echo "============================================"
    echo "  openPangu-1B IMDB SFT"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================"
    echo ""

    START_TIME=$(date +%s)

    step0_setup
    step1_prepare_data
    step2_preprocess
    step3_convert_hf2mcore
    step4_train
    step5_convert_mcore2hf
    step6_inference
    step7_evaluate
    step8_plot_loss

    END_TIME=$(date +%s)
    log_info "总耗时: $(( (END_TIME - START_TIME) / 60 )) 分 $(( (END_TIME - START_TIME) % 60 )) 秒"
    log_info "全部完成!"
}

main "$@"
