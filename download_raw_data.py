from datasets import load_dataset

dataset = load_dataset(
    "HuggingFaceFW/fineweb",
    name="sample-10BT",
    split="train",
    streaming=True   # 🔥 关键
)

for i, sample in enumerate(dataset):
    text = sample["text"]
    print(text[:200])
    if i > 5:
        break

