## Project Overview

This project implements an end-to-end LLM training pipeline,
from data cleaning and pretraining to SFT, DPO, and RAG deployment,
under single-GPU constraints.

python3 build_raw_dataset.py
ll data/raw/raw_fineweb.jsonl
vi scripts/clean_text.py
python3 scripts/clean_text.py
ll data/cleaned/
head -n 5  data/cleaned/cleaned.jsonl
vi scripts/stats_length.py
python3 scripts/stats_length.py
