import json
import numpy as np

lengths = []

with open("data/cleaned/cleaned.jsonl") as f:
    for line in f:
        text = json.loads(line)["text"]
        lengths.append(len(text.split()))

print("mean:", np.mean(lengths))
print("p50:", np.percentile(lengths, 50))
print("p90:", np.percentile(lengths, 90))
print("p99:", np.percentile(lengths, 99))

