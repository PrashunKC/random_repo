# GPT Model Trainer

## Overview
Training pipeline for fine-tuning GPT models on the Nvidia Nemotron dataset.

## Directory Structure
```
.
├── model_trainer_v_1.py    # Main training script
├── saved_model/           # Model checkpoints
├── .cache/               # Dataset cache
└── README.md            # Documentation
```

## Requirements
- Python 3.8+
- PyTorch with CUDA
- Transformers library
- Hugging Face datasets