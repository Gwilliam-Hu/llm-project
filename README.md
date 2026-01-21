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
