import json
from transformers import AutoTokenizer
import numpy as np
from tqdm import tqdm

tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen3-0.6B",
    use_fast=True
)

token_lengths = []

with open("data/cleaned/cleaned.jsonl") as f:
    for line in tqdm(f):
        text = json.loads(line)["text"]
        tokens = tokenizer(
            text,
            add_special_tokens=False
        )["input_ids"]
        token_lengths.append(len(tokens))

token_lengths = np.array(token_lengths)

print("mean:", token_lengths.mean())
print("p50:", np.percentile(token_lengths, 50))
print("p90:", np.percentile(token_lengths, 90))
print("p99:", np.percentile(token_lengths, 99))
print("max:", token_lengths.max())
print("total tokens:", token_lengths.sum())

