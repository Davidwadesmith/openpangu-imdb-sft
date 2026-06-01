# OpenPangu IMDB SFT One-Command Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the repository so a fresh server clone can reproduce the PDF's complete OpenPangu IMDB SFT workflow by running `bash run_all.sh`.

**Architecture:** Keep the deployment surface intentionally small: `run_all.sh` orchestrates setup and the PDF steps, `env.sh` owns portable runtime environment configuration, and focused scripts under `scripts/` own each data, conversion, training, inference, evaluation, and plotting operation. Add local static contract tests because the Windows development machine cannot execute the Ascend NPU workflow.

**Tech Stack:** Bash, Python 3.10, `unittest`, MindSpeed-LLM 1.0.0, Megatron-LM `core_r0.6.0`, MindSpeed commit `969686ff`, Hugging Face Transformers 4.53.2, Ascend CANN 8.5.2, torch-npu.

---

## File Structure

| File | Responsibility |
| --- | --- |
| `run_all.sh` | Clone-time entrypoint, dependency setup, repository/model download, compatibility patches, helper compilation, and ordered orchestration. |
| `env.sh` | Portable `WORKDIR` resolution, CANN activation, Python path, cache paths, and mirror environment variables. |
| `scripts/prepare_data.py` | Download IMDB through the Hugging Face mirror when needed and build SFT JSONL. |
| `scripts/preprocess.sh` | Generate MindSpeed `.bin/.idx` data cache. |
| `scripts/convert_hf2mcore.sh` | Convert the base Hugging Face model into the initial mcore checkpoint. |
| `scripts/run_sft.sh` | Run the PDF-specified 300-iteration SFT training command and log output. |
| `scripts/plot_loss.py` | Extract `lm loss` values and render `loss_curve.png`. |
| `scripts/convert_mcore2hf.sh` | Convert the trained checkpoint back to Hugging Face format and assemble the inference directory. |
| `scripts/inference.py` | Run full IMDB test-set inference by default and overwrite the result JSONL. |
| `scripts/evaluate.py` | Compute strict `正面` / `负面` accuracy. |
| `tests/test_runtime_contract.py` | Verify portable paths, PDF revisions, mirrors, defaults, ordering, and runtime file layout without an NPU. |
| `README.md` | Document fresh-clone usage, mirror-backed downloads, output artifacts, resume behavior, and server verification. |

### Task 1: Add Local Runtime Contract Tests

**Files:**
- Create: `tests/test_runtime_contract.py`

- [ ] **Step 1: Write the failing contract tests**

Create `tests/test_runtime_contract.py`:

```python
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_FILES = [
    ROOT / "run_all.sh",
    ROOT / "env.sh",
    *sorted((ROOT / "scripts").glob("*.sh")),
    *sorted((ROOT / "scripts").glob("*.py")),
]


class RuntimeContractTests(unittest.TestCase):
    def read(self, relative_path: str) -> str:
        return (ROOT / relative_path).read_text(encoding="utf-8")

    def test_runtime_layout_is_complete(self):
        expected = {
            "run_all.sh",
            "env.sh",
            "scripts/prepare_data.py",
            "scripts/preprocess.sh",
            "scripts/convert_hf2mcore.sh",
            "scripts/run_sft.sh",
            "scripts/plot_loss.py",
            "scripts/convert_mcore2hf.sh",
            "scripts/inference.py",
            "scripts/evaluate.py",
            "README.md",
        }
        missing = sorted(path for path in expected if not (ROOT / path).is_file())
        self.assertEqual([], missing)

    def test_runtime_files_do_not_hard_code_server_workdir(self):
        for path in RUNTIME_FILES:
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("/mnt/workspace/LXZ", text, path)
            self.assertNotIn("/mnt/workspace/hcc/openpangu-imdb-sft", text, path)

    def test_environment_is_portable_and_uses_mirrors(self):
        env = self.read("env.sh")
        self.assertIn('export WORKDIR="${WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"', env)
        self.assertIn("https://hf-mirror.com", env)
        self.assertIn("${PYTHONPATH:-}", env)

    def test_setup_uses_pdf_sources_revisions_and_pypi_mirror(self):
        run_all = self.read("run_all.sh")
        for expected in (
            "https://pypi.tuna.tsinghua.edu.cn/simple",
            "https://gitee.com/ascend/MindSpeed-LLM.git",
            "git checkout 1.0.0",
            "https://gitee.com/mirrors/Megatron-LM.git",
            "git checkout core_r0.6.0",
            "https://gitee.com/ascend/MindSpeed.git",
            "git checkout 969686ff",
            "https://gitcode.com/ascend-tribe/openPangu-Embedded-1B-V1.1.git",
        ):
            self.assertIn(expected, run_all)

    def test_training_defaults_match_pdf(self):
        train = self.read("scripts/run_sft.sh")
        for expected in (
            'SEQ_LENGTH="${SEQ_LENGTH:-4096}"',
            'GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-4}"',
            'TRAIN_ITERS="${TRAIN_ITERS:-300}"',
            'LR="${LR:-1e-5}"',
        ):
            self.assertIn(expected, train)

    def test_inference_defaults_to_full_test_set(self):
        inference = self.read("scripts/inference.py")
        self.assertIn('os.environ.get("MAX_SAMPLES", "0")', inference)
        run_all = self.read("run_all.sh")
        self.assertNotIn("MAX_SAMPLES=100", run_all)

    def test_orchestration_follows_pdf_order(self):
        run_all = self.read("run_all.sh")
        calls = [
            "step0_setup",
            "step1_prepare_data",
            "step2_preprocess",
            "step3_convert_hf2mcore",
            "step4_train",
            "step5_plot_loss",
            "step6_convert_mcore2hf",
            "step7_inference",
            "step8_evaluate",
        ]
        main_body = run_all.split("main() {", 1)[1]
        positions = [main_body.index(call) for call in calls]
        self.assertEqual(sorted(positions), positions)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the contract tests to verify they fail**

Run:

```powershell
python -m unittest discover -s tests -v
```

Expected: FAIL because `README.md` does not exist yet, `env.sh` does not use the agreed `WORKDIR` expression, and `run_all.sh` still invokes plotting after evaluation.

- [ ] **Step 3: Commit the failing tests**

```bash
git add tests/test_runtime_contract.py
git commit -m "test: add openpangu runtime contract checks"
```

### Task 2: Make Environment Resolution Portable

**Files:**
- Modify: `env.sh`

- [ ] **Step 1: Replace the environment bootstrap with the portable contract**

Update `env.sh` so it:

```bash
#!/bin/bash
# openPangu-1B IMDB SFT runtime environment.

export WORKDIR="${WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

if [ -f "$WORKDIR/.venv/bin/activate" ]; then
    source "$WORKDIR/.venv/bin/activate"
fi

__OLD_ERREXIT_STATE=$(set +o | grep errexit)
set +e
if [ -f /usr/local/Ascend/cann-8.5.2/set_env.sh ]; then
    source /usr/local/Ascend/cann-8.5.2/set_env.sh
else
    echo "[WARN] 未找到 CANN set_env.sh"
fi
eval "$__OLD_ERREXIT_STATE"
unset __OLD_ERREXIT_STATE
set -o pipefail 2>/dev/null || true

export PATH="$WORKDIR/.venv/bin:$PATH"
export PYTHONPATH="$WORKDIR/MindSpeed:$WORKDIR/MindSpeed-LLM:${PYTHONPATH:-}"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME="$WORKDIR/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCH_EXTENSIONS_DIR="$WORKDIR/.cache/torch_extensions"
export TMPDIR="$WORKDIR/tmp"

mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" \
         "$TORCH_EXTENSIONS_DIR" "$TMPDIR"

echo "[OK] 环境已加载：$WORKDIR"
```

- [ ] **Step 2: Run the focused environment contract test**

Run:

```powershell
python -m unittest tests.test_runtime_contract.RuntimeContractTests.test_environment_is_portable_and_uses_mirrors -v
```

Expected: PASS.

- [ ] **Step 3: Commit the environment change**

```bash
git add env.sh
git commit -m "refactor: make runtime environment path portable"
```

### Task 3: Align The One-Command Orchestrator With The PDF

**Files:**
- Modify: `run_all.sh`

- [ ] **Step 1: Preserve setup behavior while making the entrypoint explicit**

Update `run_all.sh` so:

- `WORKDIR` respects an existing override:

```bash
WORKDIR="${WORKDIR:-$(cd "$(dirname "$0")" && pwd)}"
export WORKDIR
```

- Runtime directories are created before setup:

```bash
mkdir -p "$WORKDIR/cache" "$WORKDIR/ckpt" "$WORKDIR/sft_output" \
         "$WORKDIR/logs" "$WORKDIR/results" "$WORKDIR/downloads" \
         "$WORKDIR/data" "$WORKDIR/tmp"
```

- The existing mirror-backed `pip install`, PDF repository clones, exact checkouts, Megatron embedding, compatibility patches, model download, and helper compilation remain in `step0_setup`.
- The completion log explicitly reports the resolved path:

```bash
log_info "工作目录: $WORKDIR"
```

- [ ] **Step 2: Reorder the workflow to match the PDF**

Rename the final functions and `main` calls so the order is:

```bash
step0_setup
step1_prepare_data
step2_preprocess
step3_convert_hf2mcore
step4_train
step5_plot_loss
step6_convert_mcore2hf
step7_inference
step8_evaluate
```

Keep `step7_inference` as:

```bash
python "$SCRIPTS_DIR/inference.py"
```

Do not set `MAX_SAMPLES=100` in `run_all.sh`.

- [ ] **Step 3: Run the orchestrator contract tests**

Run:

```powershell
python -m unittest tests.test_runtime_contract.RuntimeContractTests.test_setup_uses_pdf_sources_revisions_and_pypi_mirror tests.test_runtime_contract.RuntimeContractTests.test_inference_defaults_to_full_test_set tests.test_runtime_contract.RuntimeContractTests.test_orchestration_follows_pdf_order -v
```

Expected: PASS.

- [ ] **Step 4: Commit the orchestrator change**

```bash
git add run_all.sh
git commit -m "refactor: align one-command workflow with pdf"
```

### Task 4: Verify And Tighten Focused Runtime Scripts

**Files:**
- Modify if required: `scripts/prepare_data.py`
- Modify if required: `scripts/preprocess.sh`
- Modify if required: `scripts/convert_hf2mcore.sh`
- Modify if required: `scripts/run_sft.sh`
- Modify if required: `scripts/plot_loss.py`
- Modify if required: `scripts/convert_mcore2hf.sh`
- Modify if required: `scripts/inference.py`
- Modify if required: `scripts/evaluate.py`

- [ ] **Step 1: Compare each focused script against the PDF contract**

Confirm each script derives paths from `WORKDIR` and preserves these behaviors:

```text
prepare_data.py       downloads IMDB via HF_ENDPOINT and writes data/train_imdb.jsonl
preprocess.sh         writes cache/sft*.bin and cache/sft*.idx with --pack
convert_hf2mcore.sh   writes ckpt/mcore using --params-dtype bf16 --use-mcore-models
run_sft.sh            uses the PDF architecture and optimization arguments
plot_loss.py          writes loss_curve.png from the training log
convert_mcore2hf.sh   writes ckpt/mg2hf and assembles ckpt/mg2hf/mg2hf
inference.py          defaults MAX_SAMPLES to 0 and opens results with mode "w"
evaluate.py           strictly counts only 正面 and 负面 responses
```

- [ ] **Step 2: Apply only required corrections**

Do not introduce new configuration files or a framework. Keep the PDF defaults visible in the existing focused scripts. If no correction is required for a file, leave it untouched.

- [ ] **Step 3: Run Python compilation and static contracts**

Run:

```powershell
python -m compileall -q scripts tests
python -m unittest discover -s tests -v
```

Expected: Python compilation succeeds. The contract suite may still fail only because `README.md` is not created yet.

- [ ] **Step 4: Commit any focused-script corrections**

If files changed:

```bash
git add scripts
git commit -m "refactor: tighten pdf runtime scripts"
```

If no files changed, skip this commit.

### Task 5: Document Fresh-Clone Usage

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write the server runbook**

Create `README.md` with:

```markdown
# OpenPangu-1B IMDB SFT

本仓库按照实验三 PDF 整理。服务器克隆后执行一个脚本即可完成环境搭建、数据准备、权重转换、SFT 训练、loss 绘图、全量推理和准确率评测。

## 一键运行

```bash
git clone https://github.com/Davidwadesmith/openpangu-imdb-sft.git /mnt/workspace/hcc/openpangu-imdb-sft
cd /mnt/workspace/hcc/openpangu-imdb-sft
bash run_all.sh
```

脚本默认自动识别仓库目录。也可以显式指定：

```bash
WORKDIR=/mnt/workspace/hcc/openpangu-imdb-sft bash run_all.sh
```

## 自动下载

- Python 依赖：清华 PyPI 镜像
- MindSpeed-LLM、Megatron-LM、MindSpeed：Gitee
- openPangu-Embedded-1B-V1.1：GitCode
- IMDB：Hugging Face 镜像

## 默认实验参数

- `SEQ_LENGTH=4096`
- `GLOBAL_BATCH_SIZE=4`
- `TRAIN_ITERS=300`
- `LR=1e-5`
- 推理：默认处理完整的 25,000 条 IMDB 测试数据

## 输出文件

- `logs/tune_mcore_pangu_1b_full_ptd.log`
- `loss_curve.png`
- `ckpt/mg2hf/mg2hf/`
- `results/test_imdb_results.jsonl`

脚本可以在中断后重新执行。已经完成的下载、缓存和初始 checkpoint 会被复用；推理结果会重新写入，避免混入旧结果。

## 可选小样本推理

一键脚本默认执行全量推理。如需单独检查 100 条样本：

```bash
source env.sh
MAX_SAMPLES=100 python scripts/inference.py
```

## 本地静态校验

没有昇腾 NPU 的开发机可以运行：

```bash
python -m compileall -q scripts tests
python -m unittest discover -s tests -v
```
```

- [ ] **Step 2: Run the full local contract suite**

Run:

```powershell
python -m unittest discover -s tests -v
```

Expected: PASS.

- [ ] **Step 3: Commit the README**

```bash
git add README.md
git commit -m "docs: add one-command server runbook"
```

### Task 6: Run Final Local Verification

**Files:**
- Verify only: `run_all.sh`
- Verify only: `env.sh`
- Verify only: `scripts/*.sh`
- Verify only: `scripts/*.py`
- Verify only: `tests/test_runtime_contract.py`
- Verify only: `README.md`

- [ ] **Step 1: Compile Python files**

Run:

```powershell
python -m compileall -q scripts tests
```

Expected: exit code `0`.

- [ ] **Step 2: Run static runtime contracts**

Run:

```powershell
python -m unittest discover -s tests -v
```

Expected: all tests PASS.

- [ ] **Step 3: Run shell syntax checks in WSL when available**

Run:

```powershell
wsl bash -lc "cd /mnt/e/Projects/AI-Lab && bash -n env.sh run_all.sh scripts/*.sh"
```

Expected: exit code `0`. If WSL is unavailable, report that shell syntax verification is blocked locally and run it on the Ascend server before the full workflow.

- [ ] **Step 4: Check Git status**

Run:

```powershell
git status --short
```

Expected: only pre-existing untracked user directories such as `.claude/` and `exp3_openpangu_imdb/` remain.

## Server Verification Handoff

The NPU-backed workflow cannot be completed on the Windows development machine. After pushing the refactor, verify from a fresh Ascend server clone:

```bash
git clone https://github.com/Davidwadesmith/openpangu-imdb-sft.git /mnt/workspace/hcc/openpangu-imdb-sft
cd /mnt/workspace/hcc/openpangu-imdb-sft
bash run_all.sh
```

Expected artifacts:

```text
cache/*.bin
cache/*.idx
ckpt/mcore/latest_checkpointed_iteration.txt
sft_output/
logs/tune_mcore_pangu_1b_full_ptd.log
loss_curve.png
ckpt/mg2hf/mg2hf/
results/test_imdb_results.jsonl
```

Expected final inference record count:

```bash
wc -l results/test_imdb_results.jsonl
# 25000 results/test_imdb_results.jsonl
```
