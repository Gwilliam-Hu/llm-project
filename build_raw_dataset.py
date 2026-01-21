import json
from datasets import load_dataset

dataset = load_dataset(
    "HuggingFaceFW/fineweb",
    name="sample-10BT",
    split="train",
    streaming=True
)

MAX_SAMPLES = 100_000

with open("data/raw_fineweb.jsonl", "w") as f:
    for i, sample in enumerate(dataset):
        if i >= MAX_SAMPLES:
            break
        text = sample["text"]
        f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")

