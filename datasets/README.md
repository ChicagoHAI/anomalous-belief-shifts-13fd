# Downloaded Datasets

This directory stores datasets for belief-shift experiments. Raw data is large and ignored by git; see instructions below.

## Dataset: TruthfulQA (generation)
- **Source**: HuggingFace `truthful_qa` (generation config)
- **Size**: 817 validation prompts
- **Format**: HuggingFace Dataset (arrow), with fields `question`, `best_answer`, `correct_answers`, `incorrect_answers`, `category`
- **Task**: Generation / truthfulness assessment
- **License**: MIT License (per dataset card)
- **Location**: `datasets/truthful_qa/`
- **Samples**: `datasets/truthful_qa/samples.json`

### Download Instructions
Using Python/HuggingFace:
```python
from datasets import load_dataset

dataset = load_dataset('truthful_qa', 'generation')
dataset.save_to_disk('datasets/truthful_qa')
```

## Dataset: ANLI (Adversarial NLI)
- **Source**: HuggingFace `anli`
- **Size**: R1 16,946 train / 1k dev / 1k test; R2 45,460 train; R3 100,459 train with dev/test
- **Format**: HuggingFace Dataset (arrow), fields include `premise`, `hypothesis`, `label`, `uid`
- **Task**: Adversarial natural language inference (stress-test for belief shifts and robustness)
- **License**: MIT License (dataset card)
- **Location**: `datasets/anli/`
- **Samples**: `datasets/anli/samples.json`

### Download Instructions
Using Python/HuggingFace:
```python
from datasets import load_dataset

dset = load_dataset('anli')
dset.save_to_disk('datasets/anli')
```

### Notes
- Saved datasets are excluded from git; keep only samples and docs under version control.
- For quick checks, load via `from datasets import load_from_disk` and open the saved directory.
