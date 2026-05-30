# -*- coding: utf-8 -*-
"""
从 HuggingFace 下载 IMDB 数据集 parquet 文件
使用 datasets 库 API（支持镜像、断点续传）
"""
import argparse
import os
import sys
import shutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", required=True, help="保存目录")
    parser.add_argument("--split", default="train", choices=["train", "test"])
    args = parser.parse_args()

    filename = f"{args.split}-00000-of-00001.parquet"
    save_path = os.path.join(args.save_dir, filename)
    os.makedirs(args.save_dir, exist_ok=True)

    if os.path.exists(save_path) and os.path.getsize(save_path) > 1000:
        print(f"[download] {filename} 已存在，跳过")
        return

    print(f"[download] 下载 IMDB {args.split} 集...")

    try:
        # 方法1: 用 datasets 库（最可靠，自动处理镜像）
        from datasets import load_dataset
        dataset = load_dataset(
            "stanfordnlp/imdb",
            split=args.split,
            trust_remote_code=False,
            cache_dir=os.path.join(args.save_dir, "_hf_cache"),
        )
        # datasets 库下载后是 arrow 格式，转存为 parquet
        dataset.to_parquet(save_path)
        print(f"[download] {args.split} 集已保存: {save_path}")

    except Exception as e1:
        print(f"[download] datasets 方式失败: {e1}")
        print("[download] 尝试直接下载 parquet...")

        try:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(
                repo_id="stanfordnlp/imdb",
                filename=f"data/{filename}",
                repo_type="dataset",
                local_dir=args.save_dir,
                local_dir_use_symlinks=False,
            )
            # 移动到正确位置
            final = os.path.join(args.save_dir, filename)
            if path != final:
                shutil.move(path, final)
            print(f"[download] {args.split} 集已保存: {final}")

        except Exception as e2:
            print(f"[download] huggingface_hub 方式也失败: {e2}")
            print("[download] 请手动下载:")
            print(f"  https://hf-mirror.com/datasets/stanfordnlp/imdb")
            sys.exit(1)

    size_mb = os.path.getsize(save_path) / 1024 / 1024
    print(f"[download] 完成: {save_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
