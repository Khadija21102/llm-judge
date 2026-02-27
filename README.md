# LLM-Judge

**LLM-Judge** is a research framework for evaluating and training Large Language Models (LLMs) as structured automated judges.  
It supports prompt-based evaluation, supervised fine-tuning, reward modeling with feedback, and reliability analysis.

## Overview

This project investigates whether LLMs can reliably score generated responses across multiple evaluation axes, particularly in safety-critical domains (e.g., clinical AI). The framework enables:

- Prompt-based evaluation of foundation models  :  in this folder, you can find the code to send requests to each propietary models but also to run inference pipeline on open models. You should first download the model weights. You have first to process the dataset and then send batch api requests. 
- Supervised fine-tuning (SFT) for judge behavior  : in this folder you will find all the necessary scripts to train your fine-tuned models, there are various possible models for score-based and preference based judge. You can choose which type of models you want to run (classification, regression, generation). You can add arguments to each script to define your hyperparameters.
- Evaluating a judge : in this folder, similarly to the finetuning folder, you will have a script to evaluate your fine-tuned model or the result of the prompting pipeline of state-of-the-art models. You will also find a script for the evaluation of the agreement between clinicians
- Utils contains some utilitaries that are used in other folder. You can also find the notebook that contains the preprocessing of the data
- The data folder is private, you should ask access by email to khadija2112002@gmail.com and giorgia.carra@chuv.ch
## Project Structure
```text
llm-judge/
├── prompting-pipelines/        # Prompt-based judge evaluation
├── finetuning/       # SFT and reward modeling
├── evaluation/       # Metrics, ICC, jury aggregation
├── utils/            # Various useful methods
├── data/             # Datasets (JSONL)
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
  title = {Large Language Models as Clinical Answer Judges: Developing an LLM-Based Evaluation Framework for Clinical Question Answering },
  year = {2026},
  url = {https://github.com/Khadija21102/llm-judge}
}
```
