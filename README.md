# LLM-Judge

**LLM-Judge** is a research framework for evaluating and training Large Language Models (LLMs) as structured automated judges.  
It supports prompt-based evaluation, supervised fine-tuning, reward modeling with feedback, and reliability analysis.

## Overview

This project investigates whether LLMs can reliably score generated responses across multiple evaluation axes, particularly in safety-critical domains (e.g., clinical AI). The framework enables:

- Prompt-based evaluation of foundation models  
- Supervised fine-tuning (SFT) for judge behavior  
- Preference and reward modeling with or without feedback  
- Multi-sampling experiments (temperature & aggregation)  
- Inter-rater reliability analysis (e.g., ICC)  

## Project Structure
```text
llm-judge/
├── prompting-pipelines/        # Prompt-based judge evaluation
├── finetuning/       # SFT and reward modeling
├── evaluation/       # Metrics, ICC, jury aggregation
├── utils/            # Various useful methods
├── data/             # Datasets (JSONL)
└── results/          # Model outputs and analyses
```

## Installation

```bash
git clone https://github.com/Khadija21102/llm-judge.git
cd llm-judge
pip install -r requirements.txt
```


## Citation
```text

@software{llmjudge2026,
  author = {Tagemouati, Khadija},
  title = {LLM-Judge: A Framework for Training and Evaluating LLM Judges},
  year = {2026},
  url = {https://github.com/Khadija21102/llm-judge}
}
```
