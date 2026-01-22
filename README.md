## Project Overview

This project implements an end-to-end LLM training pipeline,
from data cleaning and pretraining to SFT, DPO, and RAG deployment,
under single-GPU constraints.

python3 build_raw_dataset.py<br>
ll data/raw/raw_fineweb.jsonl<br>
vi scripts/clean_text.py<br>
python3 scripts/clean_text.py<br>
ll data/cleaned/<br>
head -n 5  data/cleaned/cleaned.jsonl<br>
vi scripts/stats_length.py<br>
python3 scripts/stats_length.py<br>


"""
mean: 480.35147845887025<br> 
p50: 365.0 <br>
p90: 1031.0 <br>
p99: 1641.0 <br>
max: 3168 <br>
total tokens: 45144873<br>
"""


Pretrain max_seq_len = 2048
