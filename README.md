# GPT Model Trainer

A Python-based training pipeline for fine-tuning GPT models on the Nvidia Nemotron dataset.

## Features

- Multi-phase training with progressive learning rates
- Memory-optimized data loading and processing
- CUDA optimizations for better GPU utilization
- Dataset validation and analysis tools
- Progress tracking with detailed statistics
- Checkpoint management and safe model saving
- Interactive dataset management system

## Requirements

- Python 3.8+
- PyTorch with CUDA support
- Transformers library
- Datasets library
- tqdm for progress bars
- CUDA-capable GPU with 8GB+ VRAM

## Installation

```bash
pip install torch transformers datasets tqdm
```

## Usage

1. Set up your environment variables:
```python
os.environ["HUGGINGFACE_TOKEN"] = "your_token_here"
```

2. Run the training script:
```bash
python model_trainer_v_1.py
```

3. Replace [User] with the user you are currently logged into your computer as.

4. Follow the interactive prompts to manage datasets and start training.

5. Chat with the AI by running chat.py

## Configuration

The training pipeline is configured through the `DATASET_CONFIGS` dictionary in the script.
You can modify training parameters like batch size, learning rate, and model checkpointing
frequency in the `train()` function.

## License

MIT License - Feel free to use and modify as needed.