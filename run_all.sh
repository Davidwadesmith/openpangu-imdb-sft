#!/bin/bash
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC}  $(date '+%H:%M:%S') $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $(date '+%H:%M:%S') $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $(date '+%H:%M:%S') $*"; }

WORKDIR="${WORKDIR:-$(cd "$(dirname "$0")" && pwd)}"
export WORKDIR
SCRIPTS_DIR="$WORKDIR/scripts"

mkdir -p "$WORKDIR/cache" "$WORKDIR/ckpt" "$WORKDIR/sft_output" \
         "$WORKDIR/logs" "$WORKDIR/results" "$WORKDIR/downloads" \
         "$WORKDIR/data" "$WORKDIR/tmp"

source "$WORKDIR/env.sh"

ensure_git_checkout() {
    local name="$1"
    local url="$2"
    local revision="$3"
    local target="$WORKDIR/$name"

    if [ ! -d "$target/.git" ]; then
        if [ -e "$target" ]; then
            log_error "$target exists but is not a Git checkout"
            exit 1
        fi
        log_info "Cloning $name..."
        git clone "$url" "$target"
    fi

    log_info "Checking out $name revision $revision"
    git -C "$target" checkout "$revision"
}

embed_megatron() {
    local marker="$WORKDIR/.megatron_core_r0.6.0_embedded"

    if [ -f "$marker" ] && [ -d "$WORKDIR/MindSpeed-LLM/megatron/core" ]; then
        log_warn "Megatron core_r0.6.0 already embedded"
        return
    fi

    log_info "Embedding Megatron-LM core_r0.6.0 into MindSpeed-LLM"
    rm -rf "$WORKDIR/MindSpeed-LLM/megatron"
    cp -r "$WORKDIR/Megatron-LM/megatron" "$WORKDIR/MindSpeed-LLM/"
    touch "$marker"
}

patch_megatron_compatibility() {
    local marker="$WORKDIR/.runtime_compatibility_v3_patched"
    local megatron_source="$WORKDIR/Megatron-LM/megatron/core"
    local embedded_core="$WORKDIR/MindSpeed-LLM/megatron/core"
    local relative_path

    if [ -f "$marker" ]; then
        log_warn "Megatron compatibility patches already applied"
        return
    fi

    log_info "Restoring compatibility patch targets from clean checkouts"
    for relative_path in \
        optimizer/__init__.py \
        optimizer/distrib_optimizer.py \
        tensor_parallel/random.py \
        extensions/transformer_engine.py; do
        if [ -f "$megatron_source/$relative_path" ]; then
            cp -f "$megatron_source/$relative_path" "$embedded_core/$relative_path"
        fi
    done
    git -C "$WORKDIR/MindSpeed-LLM" checkout -- mindspeed_llm/core/optimizer/__init__.py

    log_info "Applying Megatron apex/transformer_engine compatibility patches"
    python - <<'PY'
import os
from pathlib import Path


def patch_file(relative_path: str, old: str, new: str, patched_marker: str) -> None:
    path = Path(os.environ["WORKDIR"]) / "MindSpeed-LLM" / relative_path
    if not path.is_file():
        print(f"  skipped missing {relative_path}")
        return
    text = path.read_text(encoding="utf-8")
    if patched_marker in text:
        print(f"  already patched {relative_path}")
        return
    if old in text:
        patched = text.replace(old, new)
        compile(patched, str(path), "exec")
        path.write_text(patched, encoding="utf-8")
        print(f"  patched {relative_path}")
        return
    print(f"  skipped unchanged {relative_path}")


patch_file(
    "megatron/core/optimizer/__init__.py",
    "from apex.optimizers import FusedSGD as SGD",
    """try:
    from apex.optimizers import FusedSGD as SGD
except (ImportError, AttributeError):
    class SGD:
        def __init__(self, *args, **kwargs):
            pass""",
    "except (ImportError, AttributeError):",
)
patch_file(
    "megatron/core/optimizer/distrib_optimizer.py",
    "HAVE_APEX_OR_TE = True",
    "HAVE_APEX_OR_TE = False",
    "HAVE_APEX_OR_TE = False",
)
patch_file(
    "megatron/core/tensor_parallel/random.py",
    "from transformer_engine.pytorch.distributed import activation_recompute_forward",
    """try:
    from transformer_engine.pytorch.distributed import activation_recompute_forward
except ImportError:
    def activation_recompute_forward(*args, **kwargs):
        raise NotImplementedError""",
    "def activation_recompute_forward(*args, **kwargs):",
)
patch_file(
    "megatron/core/extensions/transformer_engine.py",
    "from transformer_engine.pytorch.distributed import activation_recompute_forward",
    """try:
    from transformer_engine.pytorch.distributed import activation_recompute_forward
except ImportError:
    def activation_recompute_forward(*args, **kwargs):
        raise NotImplementedError""",
    "def activation_recompute_forward(*args, **kwargs):",
)
patch_file(
    "mindspeed_llm/core/optimizer/__init__.py",
    "from apex.optimizers import FusedSGD as SGD",
    """try:
    from apex.optimizers import FusedSGD as SGD
except (ImportError, AttributeError):
    class SGD:
        def __init__(self, *args, **kwargs):
            pass""",
    "except (ImportError, AttributeError):",
)
PY
    touch "$marker"
}

step0_setup() {
    log_info "============================================="
    log_info "Step 0/9: Environment and repositories"
    log_info "============================================="
    log_info "Working directory: $WORKDIR"

    if [ ! -f "$WORKDIR/.venv/bin/activate" ]; then
        log_info "Creating Python 3.10 virtual environment"
        python3.10 -m venv --system-site-packages "$WORKDIR/.venv"
    else
        log_warn "Python virtual environment already exists"
    fi

    source "$WORKDIR/.venv/bin/activate"
    python -m pip install --upgrade pip \
        -i https://pypi.tuna.tsinghua.edu.cn/simple
    python -m pip install \
        ninja \
        pybind11 \
        transformers==4.53.2 \
        datasets \
        pandas \
        pyarrow \
        matplotlib \
        tqdm \
        sentencepiece \
        safetensors \
        -i https://pypi.tuna.tsinghua.edu.cn/simple

    ensure_git_checkout \
        MindSpeed-LLM \
        https://gitee.com/ascend/MindSpeed-LLM.git \
        1.0.0
    ensure_git_checkout \
        Megatron-LM \
        https://gitee.com/mirrors/Megatron-LM.git \
        core_r0.6.0
    ensure_git_checkout \
        MindSpeed \
        https://gitee.com/ascend/MindSpeed.git \
        969686ff

    embed_megatron
    patch_megatron_compatibility

    rm -f "$(python -c 'import site; print(site.getusersitepackages())')/fix_apex.pth"

    local model_dir="$WORKDIR/openPangu-Embedded-1B-V1.1"
    local model_weight="$model_dir/model.safetensors"
    local model_size=0

    if [ ! -d "$model_dir/.git" ]; then
        if [ -e "$model_dir" ]; then
            log_error "$model_dir exists but is not a Git checkout"
            exit 1
        fi
        log_info "Cloning openPangu-Embedded-1B-V1.1 from GitCode"
        git clone \
            https://gitcode.com/ascend-tribe/openPangu-Embedded-1B-V1.1.git \
            "$model_dir"
    else
        log_warn "openPangu model checkout already exists"
    fi

    if [ -f "$model_weight" ]; then
        model_size="$(stat -c%s "$model_weight" 2>/dev/null || echo 0)"
    fi
    if [ "$model_size" -lt 1000000000 ]; then
        log_warn "Model weight is missing or too small; attempting git lfs pull"
        git -C "$model_dir" lfs pull || log_warn "git lfs pull failed; checking weight size before stopping"
        if [ -f "$model_weight" ]; then
            model_size="$(stat -c%s "$model_weight" 2>/dev/null || echo 0)"
        fi
    fi
    if [ "$model_size" -lt 1000000000 ]; then
        log_error "Model weight is incomplete: $model_weight ($model_size bytes)"
        exit 1
    fi

    log_info "Model weight:"
    ls -lh "$model_weight"
    source "$WORKDIR/env.sh"
}

step1_prepare_data() {
    log_info "Step 1/9: Download and convert IMDB"

    if [ -f "$WORKDIR/downloads/train-00000-of-00001.parquet" ] && \
       [ -f "$WORKDIR/downloads/test-00000-of-00001.parquet" ] && \
       [ -f "$WORKDIR/data/train_imdb.jsonl" ]; then
        log_warn "IMDB parquet and training JSONL already exist"
        return
    fi

    source "$WORKDIR/env.sh"
    python "$SCRIPTS_DIR/prepare_data.py"
}

step2_preprocess() {
    log_info "Step 2/9: Preprocess training data"

    local bin_file
    while IFS= read -r bin_file; do
        if [ -f "${bin_file%.bin}.idx" ]; then
            log_warn "Preprocessed .bin/.idx cache already exists"
            return
        fi
    done < <(find "$WORKDIR/cache" -maxdepth 1 -type f -name "*.bin")

    source "$WORKDIR/env.sh"
    bash "$SCRIPTS_DIR/preprocess.sh"
}

step3_compile_helpers() {
    log_info "Step 3/9: Compile Megatron dataset helpers"
    source "$WORKDIR/env.sh"

    if (
        cd "$WORKDIR/MindSpeed-LLM"
        python - <<'PY'
from megatron.core.datasets import helpers
print(f"[OK] Megatron helpers importable: {helpers}")
PY
    ); then
        log_warn "Megatron helpers already compiled"
        return
    fi

    cd "$WORKDIR/MindSpeed-LLM"
    cat > build_megatron_helpers.py <<'PY'
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension


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
PY
    python build_megatron_helpers.py build_ext --inplace
    python - <<'PY'
from megatron.core.datasets import helpers
print(f"[OK] Megatron helpers importable: {helpers}")
PY
    cd "$WORKDIR"
}

step4_convert_hf2mcore() {
    log_info "Step 4/9: Convert Hugging Face checkpoint to mcore"

    if [ -f "$WORKDIR/ckpt/mcore/latest_checkpointed_iteration.txt" ]; then
        log_warn "Initial mcore checkpoint already exists"
        return
    fi

    source "$WORKDIR/env.sh"
    bash "$SCRIPTS_DIR/convert_hf2mcore.sh"
}

step5_train() {
    log_info "Step 5/9: Run SFT training"

    if [ -f "$WORKDIR/sft_output/latest_checkpointed_iteration.txt" ]; then
        log_warn "SFT checkpoint already exists"
        return
    fi

    source "$WORKDIR/env.sh"
    bash "$SCRIPTS_DIR/run_sft.sh"
}

step6_plot_loss() {
    log_info "Step 6/9: Plot training loss"
    source "$WORKDIR/env.sh"
    python "$SCRIPTS_DIR/plot_loss.py"
}

step7_convert_mcore2hf() {
    log_info "Step 7/9: Convert trained mcore checkpoint to Hugging Face"
    source "$WORKDIR/env.sh"
    bash "$SCRIPTS_DIR/convert_mcore2hf.sh"
}

step8_inference() {
    log_info "Step 8/9: Run full IMDB test-set inference"
    source "$WORKDIR/env.sh"
    python "$SCRIPTS_DIR/inference.py"
}

step9_evaluate() {
    log_info "Step 9/9: Evaluate sentiment predictions"
    source "$WORKDIR/env.sh"
    python "$SCRIPTS_DIR/evaluate.py"
}

main() {
    local start_time
    local end_time

    echo ""
    echo "============================================"
    echo "  openPangu-1B IMDB SFT"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================"
    echo ""

    start_time="$(date +%s)"

    step0_setup
    step1_prepare_data
    step2_preprocess
    step3_compile_helpers
    step4_convert_hf2mcore
    step5_train
    step6_plot_loss
    step7_convert_mcore2hf
    step8_inference
    step9_evaluate

    end_time="$(date +%s)"
    log_info "Total duration: $(( (end_time - start_time) / 60 ))m $(( (end_time - start_time) % 60 ))s"
    log_info "All steps completed"
}

main "$@"
