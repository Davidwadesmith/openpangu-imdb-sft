# -*- coding: utf-8 -*-
"""
IMDB 数据预处理脚本
将 JSONL 格式的 instruction 数据 tokenize 并保存为 Megatron 兼容的 .bin/.idx 缓存

不依赖 MindSpeed-LLM 的 preprocess_data.py，自己处理所有逻辑。
"""
import argparse
import json
import os
import sys
import numpy as np

from transformers import AutoTokenizer


def format_chat(messages, tokenizer):
    """
    将对话消息转换为文本字符串。
    优先使用 tokenizer 的 chat_template，否则手动拼接。
    """
    # 尝试使用 tokenizer 内置的 chat_template
    if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        except Exception:
            pass

    # 手动拼接 fallback
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "user":
            parts.append(f"User: {content}")
        elif role == "assistant":
            parts.append(f"Assistant: {content}")
        elif role == "system":
            parts.append(f"System: {content}")
    text = "\n".join(parts)
    # 末尾加 eos
    if tokenizer.eos_token:
        text += tokenizer.eos_token
    elif tokenizer.eos_token_id is not None:
        text += tokenizer.decode([tokenizer.eos_token_id])
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="JSONL 文件路径")
    parser.add_argument("--tokenizer-path", required=True, help="HuggingFace 模型目录")
    parser.add_argument("--output-prefix", required=True, help="输出 .bin/.idx 文件前缀")
    parser.add_argument("--seq-length", type=int, default=8192)
    args = parser.parse_args()

    print(f"[preprocess] 加载 tokenizer: {args.tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path, trust_remote_code=True, use_fast=False
    )
    print(f"[preprocess] 词表大小: {tokenizer.vocab_size}")
    print(f"[preprocess] eos_token_id: {tokenizer.eos_token_id}")
    print(f"[preprocess] pad_token_id: {tokenizer.pad_token_id}")

    # 确保有 pad_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id or 0

    print(f"[preprocess] 读取数据: {args.input}")
    all_sequences = []
    all_labels = []  # loss mask

    with open(args.input, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            rec = json.loads(line)
            messages = rec.get("data", [rec])

            # === 只对 prompt（user 部分）和 response（assistant 部分）分别处理 ===
            # 构建整体文本用于 tokenization
            full_text = format_chat(messages, tokenizer)

            tokens = tokenizer.encode(full_text, add_special_tokens=True)
            token_ids = np.array(tokens, dtype=np.int32)

            # 截断到 seq_length
            if len(token_ids) > args.seq_length:
                token_ids = token_ids[:args.seq_length]

            all_sequences.append(token_ids)

            if line_num % 5000 == 0:
                print(f"  已处理 {line_num} 条, 当前序列长度: {len(token_ids)}")

    print(f"[preprocess] 共 {len(all_sequences)} 条序列")

    # ===== 写 Megatron 兼容的 .bin / .idx =====
    bin_path = args.output_prefix + "_text_document.bin"
    idx_path = args.output_prefix + "_text_document.idx"
    os.makedirs(os.path.dirname(args.output_prefix) or ".", exist_ok=True)

    # .bin: 所有序列的 token IDs 拼接（int32）
    offsets = [0]
    with open(bin_path, "wb") as fbin:
        for seq in all_sequences:
            fbin.write(seq.tobytes())
            offsets.append(offsets[-1] + len(seq))

    # .idx: header [version, dtype_code, num_sequences] + sequence_offsets
    # dtype_code: 0=uint8, 1=int8, 2=uint16, 3=int16, 4=uint32, 5=int32, ...
    # Megatron 格式: int64 header + int64 offsets
    num_seqs = len(all_sequences)
    with open(idx_path, "wb") as fidx:
        # header: version=1, dtype=5 (int32), num_sequences
        header = np.array([1, 5, num_seqs], dtype=np.int64)
        fidx.write(header.tobytes())
        # 每个 offset 为 int64
        offsets_arr = np.array(offsets, dtype=np.int64)
        fidx.write(offsets_arr.tobytes())

    total_tokens = offsets[-1]
    print(f"[preprocess] 完成:")
    print(f"  bin: {bin_path} ({total_tokens} tokens, {os.path.getsize(bin_path)/1024/1024:.1f} MB)")
    print(f"  idx: {idx_path}")
    print(f"  avg seq len: {total_tokens / num_seqs:.0f}")
    print(f"  max seq len: {max(len(s) for s in all_sequences)}")


if __name__ == "__main__":
    main()
