# -*- coding: utf-8 -*-
"""
从 HuggingFace 镜像下载 openPangu-1B 模型
"""
import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True,
                        help="HF model ID, e.g. FreedomIntelligence/openPangu-Embedded-1B-V1.1")
    parser.add_argument("--save-dir", required=True,
                        help="本地保存目录")
    args = parser.parse_args()

    print(f"[download] 模型: {args.model_id}")
    print(f"[download] 目标: {args.save_dir}")
    print(f"[download] HF_ENDPOINT: {os.environ.get('HF_ENDPOINT', '(default)')}")

    try:
        from huggingface_hub import snapshot_download

        # 只下载必需文件，跳过不需要的大文件
        ignore_patterns = [
            "*.bin",           # 跳过 pytorch .bin 权重（用 safetensors）
            "*.h5",
            "*.msgpack",
            "*.ot",
            "*.pdf",
            "pytorch_model*",
            "tf_model*",
            "flax_model*",
            "onnx/*",
        ]

        snapshot_download(
            repo_id=args.model_id,
            local_dir=args.save_dir,
            ignore_patterns=ignore_patterns,
            resume_download=True,
            local_files_only=False,
        )
        print("[download] 模型下载完成")

    except ImportError:
        print("[download] huggingface_hub 未安装，尝试用 git clone...")
        # Fallback: 用 git clone
        import subprocess
        mirror = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
        url = f"{mirror}/{args.model_id}"
        cmd = ["git", "clone", "--depth", "1", url, args.save_dir]
        subprocess.run(cmd, check=True)
        print("[download] git clone 完成")

    # 验证
    required = ["config.json", "tokenizer.model", "tokenizer_config.json"]
    for f in required:
        path = os.path.join(args.save_dir, f)
        if not os.path.exists(path):
            print(f"[WARN] 缺少文件: {f}")
        else:
            print(f"  ✓ {f}")


if __name__ == "__main__":
    main()
