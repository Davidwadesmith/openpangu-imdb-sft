# OpenPangu IMDB SFT One-Command Refactor Design

## Goal

Refactor the repository so a user can clone it onto the Ascend server and run the complete IMDB sentiment-classification SFT experiment with one command:

```bash
git clone <repository-url> /mnt/workspace/hcc/openpangu-imdb-sft
cd /mnt/workspace/hcc/openpangu-imdb-sft
bash run_all.sh
```

The implementation must follow the provided PDF workflow while remaining portable across clone locations. The repository directory is detected automatically and can be overridden with `WORKDIR`.

## Scope

The final tracked project uses a single flat deployment layout:

```text
openpangu-imdb-sft/
├── run_all.sh
├── env.sh
├── README.md
├── scripts/
│   ├── prepare_data.py
│   ├── preprocess.sh
│   ├── convert_hf2mcore.sh
│   ├── run_sft.sh
│   ├── convert_mcore2hf.sh
│   ├── inference.py
│   ├── evaluate.py
│   └── plot_loss.py
└── docs/
```

The existing untracked `exp3_openpangu_imdb/` directory is outside the final deployment layout. It is not deleted automatically because it may contain user files.

## Runtime Contract

`bash run_all.sh` is the only required entrypoint. It runs the entire workflow without interactive pauses:

1. Detect `WORKDIR` from the repository directory unless the caller has already provided it.
2. Create the runtime directories used for downloads, caches, checkpoints, logs, results, and temporary files.
3. Create a Python 3.10 virtual environment with system site packages.
4. Install required Python packages through the Tsinghua PyPI mirror.
5. Clone the PDF-specified repositories and revisions:
   - `MindSpeed-LLM` tag `1.0.0` from Gitee.
   - `Megatron-LM` tag `core_r0.6.0` from the Gitee mirror.
   - `MindSpeed` commit `969686ff` from Gitee.
   - `openPangu-Embedded-1B-V1.1` from GitCode.
6. Embed the checked-out Megatron source tree into `MindSpeed-LLM`.
7. Apply the required compatibility patches for the PDF runtime environment.
8. Download IMDB through the configured Hugging Face mirror and create `data/train_imdb.jsonl`.
9. Run MindSpeed-LLM preprocessing to generate `.bin/.idx` cache files.
10. Compile the Megatron dataset helpers extension.
11. Convert the Hugging Face checkpoint to mcore format.
12. Train SFT with the PDF parameters: sequence length `4096`, global batch size `4`, `300` iterations, and learning rate `1e-5`.
13. Plot the training loss curve.
14. Convert the trained mcore checkpoint back to Hugging Face format and copy tokenizer/configuration files into the inference directory.
15. Run inference over all `25,000` IMDB test samples by default.
16. Evaluate strict `正面` / `负面` predictions and print accuracy.

The PDF's optional `MAX_SAMPLES=100` smoke run remains available when invoking `scripts/inference.py` manually, but `run_all.sh` does not run it automatically.

## Path Handling

No tracked file hard-codes `/mnt/workspace/LXZ` or `/mnt/workspace/hcc/openpangu-imdb-sft`.

`env.sh` resolves the repository directory from its own location:

```bash
export WORKDIR="${WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
```

Every shell and Python script derives its files from `WORKDIR`. This supports both the expected server path and arbitrary clone locations.

## Mirror Strategy

The script performs all setup downloads automatically and uses the mirrors required by the experiment environment:

- Python packages: `https://pypi.tuna.tsinghua.edu.cn/simple`
- Hugging Face model and dataset access: `https://hf-mirror.com`
- MindSpeed repositories: Gitee URLs from the PDF
- openPangu model: GitCode URL from the PDF

## Idempotency And Recovery

The one-command entrypoint is restartable:

- Existing virtual environments and cloned repositories are reused.
- Existing downloaded parquet files, preprocessed `.bin/.idx` files, and initial mcore checkpoints are reused when complete.
- Compatibility patches use marker files and must be safe to re-run.
- Trained checkpoints are preserved.
- Hugging Face export may replace its own generated output directory to prevent stale converted weights.
- Inference rewrites `results/test_imdb_results.jsonl` to avoid mixing old and new predictions.

The script prints clear step names, resolved paths, and completion markers so failures can be diagnosed from server logs.

## Validation

Local validation covers behavior that does not require an Ascend NPU:

- Shell syntax checks for all `.sh` files.
- Python compilation checks for all Python scripts.
- Static checks that tracked runtime scripts do not contain the old `/mnt/workspace/LXZ` path.
- Static checks for the expected PDF revisions, mirror URLs, default training values, and full-inference default.

Server validation covers the actual experiment:

- `bash run_all.sh` completes after a fresh clone.
- Preprocessing creates `.bin/.idx` files under `cache/`.
- HF-to-mcore conversion creates `ckpt/mcore/latest_checkpointed_iteration.txt`.
- Training creates the SFT checkpoint and `logs/tune_mcore_pangu_1b_full_ptd.log`.
- Plotting creates `loss_curve.png`.
- mcore-to-HF conversion creates a usable model under `ckpt/mg2hf/mg2hf`.
- Inference writes `25,000` JSONL records to `results/test_imdb_results.jsonl`.
- Evaluation prints the valid-sample count, correct-prediction count, and accuracy.

