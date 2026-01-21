import json
import os
import re
import hashlib
from pathlib import Path
from typing import Iterator

# ======================
# 基本配置
# ======================
RAW_DIR = Path("data/raw")
OUT_FILE = Path("data/cleaned/cleaned.jsonl")

MIN_CHARS = 200      # 太短没价值
MAX_CHARS = 8000     # 太长显存不友好

# ======================
# 文本清洗规则
# ======================

def normalize_text(text: str) -> str:
    """基础归一化"""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def is_valid_text(text: str) -> bool:
    """规则过滤"""
    if len(text) < MIN_CHARS:
        return False
    if len(text) > MAX_CHARS:
        return False
    # 非自然语言比例过高（简单 heuristic）
    if text.count("{") + text.count("}") > 50:
        return False
    return True


def text_hash(text: str) -> str:
    """用于去重"""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


# ======================
# 数据读取
# ======================

def read_jsonl(path: Path) -> Iterator[str]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                if "text" in obj:
                    yield obj["text"]
            except json.JSONDecodeError:
                continue


# ======================
# 主流程
# ======================

def main():
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    seen_hashes = set()
    total, kept = 0, 0

    with OUT_FILE.open("w", encoding="utf-8") as out_f:
        for file in RAW_DIR.glob("*.jsonl"):
            for raw_text in read_jsonl(file):
                total += 1

                text = normalize_text(raw_text)
                if not is_valid_text(text):
                    continue

                h = text_hash(text)
                if h in seen_hashes:
                    continue

                seen_hashes.add(h)

                out_f.write(
                    json.dumps({"text": text}, ensure_ascii=False) + "\n"
                )
                kept += 1

                if kept % 1000 == 0:
                    print(f"kept {kept} / processed {total}")

    print(f"Done. kept {kept} / total {total}")


if __name__ == "__main__":
    main()

