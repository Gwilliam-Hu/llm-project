import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen3-0.6B"
DATA_PATH = "data/tokenized/train.pt"
BATCH_SIZE = 1
LR = 5e-5
NUM_STEPS = 100


class PretrainDataset(Dataset):
    def __init__(self, data):
        self.samples = data["input_ids"]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ids = torch.tensor(self.samples[idx], dtype=torch.long)
        return {"input_ids": ids, "labels": ids.clone()}


def collate_fn(batch):
    # 不 pad，直接返回（batch_size=1）
    return batch[0]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True
    ).to(device)
    model.train()

    data = torch.load(DATA_PATH)
    dataset = PretrainDataset(data)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    step = 0
    for batch in loader:
        if step >= NUM_STEPS:
            break

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = torch.ones_like(input_ids)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step % 10 == 0:
            print(f"step {step} | loss {loss.item():.4f}")

        step += 1

    print("Pretrain dry run finished.")


if __name__ == "__main__":
    main()

