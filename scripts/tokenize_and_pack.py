import json
import os
from pathlib import Path
from tqdm import tqdm

import torch
from transformers import AutoTokenizer

# ========= 配置区（只改这里） =========
MODEL_NAME = "Qwen/Qwen3-0.6B"
INPUT_PATH = "data/cleaned/cleaned.jsonl"
OUTPUT_PATH = "data/tokenized/train.pt"
MAX_SEQ_LEN = 2048
TEXT_KEY = "text"
# =====================================


def main():
    # 1. 加载 tokenizer（必须与模型一致）
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True,
        trust_remote_code=True
    )

    assert tokenizer.eos_token_id is not None, "Tokenizer must have eos_token_id"

    input_ids_list = []

    # 2. 逐行读取清洗后的文本
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Tokenizing"):
            obj = json.loads(line)
            text = obj.get(TEXT_KEY, "").strip()
            if not text:
                continue

            # 3. tokenize（CLM：text -> tokens）
            tokens = tokenizer.encode(
                text,
                add_special_tokens=False
            )

            # 4. 加 EOS
            tokens.append(tokenizer.eos_token_id)

            # 5. 截断
            if len(tokens) > MAX_SEQ_LEN:
                tokens = tokens[:MAX_SEQ_LEN]

            input_ids_list.append(tokens)

    # 6. 转 tensor（不 pad，交给 DataLoader 处理）
    dataset = {
        "input_ids": input_ids_list,
        "max_seq_len": MAX_SEQ_LEN,
        "tokenizer": MODEL_NAME,
    }

    # 7. 保存
    os.makedirs(Path(OUTPUT_PATH).parent, exist_ok=True)
    torch.save(dataset, OUTPUT_PATH)

    print(f"Saved {len(input_ids_list)} samples to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

