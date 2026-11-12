# GPT model trainer

This project contains an interactive PyTorch/Transformers pipeline for
fine-tuning causal language models on configured JSON data, plus a terminal
chat client for a saved model. The current implementation is experimental and
requires local configuration before it can run on another computer.

## Files

| File | Purpose |
| --- | --- |
| [`model_trainer_v_1_1.py`](model_trainer_v_1_1.py) | Interactive model selection, dataset management, phased training, validation, and checkpoint saving |
| [`chat.py`](chat.py) | Loads `saved_model/final_model` and generates terminal responses |
| [`documentation.md`](documentation.md) | Detailed pipeline and configuration notes |
| [`.config`](.config) | Ignore rules for caches, checkpoints, virtual environments, and generated model files |

Generated model files are intentionally ignored by Git. Do not commit model
weights, dataset caches, or access tokens.

## Requirements

- Python 3.8 or newer.
- PyTorch; CUDA support is recommended for practical training.
- `transformers`, `datasets`, and `tqdm`.
- `accelerate` and `bitsandbytes` may be required for selected device/precision
  paths.
- Enough disk space for the selected base model, dataset cache, and
  checkpoints. GPU memory requirements depend on the model and precision.

Create an isolated environment and install the baseline dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch transformers datasets tqdm accelerate
```

Install the PyTorch build appropriate for the machine (especially CUDA)
according to the current PyTorch installation instructions.

## Configure before running

Review `DATASET_CONFIGS` and `TEST_MODE_CONFIGS` in
`model_trainer_v_1_1.py`. They currently contain machine-specific Windows
cache paths and a hard-coded dataset configuration, so they will not work
unchanged on another system. Replace those paths with local JSON files or
adapt the configuration to the desired Hugging Face dataset.

The script reads `HUGGINGFACE_TOKEN` when calling `load_dataset`, but the
current source assigns the literal placeholder `HUGGINGFACE_TOKEN` during
import. For private datasets, configure authentication through the supported
Hugging Face login flow and update that assignment to use a real environment
variable without committing the token. Never paste an access token into this
README or source control.

The cache is written below `.cache/huggingface`. Final models and checkpoints
are written below `saved_model` (as configured by the script).

## Train

```bash
python model_trainer_v_1_1.py
```

Choose one of the interactive modes:

1. Full training mode.
2. Test mode with reduced data/chunk settings.
3. CPU mode.

The pipeline then selects a base model or saved checkpoint, precision, dataset
management options, and each configured training phase. It saves phase
checkpoints and a final model in a Hugging Face-compatible directory.

For a first validation, use test mode and confirm that the configured dataset
files exist before attempting a full run. Training can consume substantial
GPU memory and disk space.

## Chat with a saved model

After a successful run creates `saved_model/final_model`:

```bash
python chat.py
```

The client uses FP16 and automatic device mapping, truncates prompts to the
GPT-2 context limit, and generates sampled responses. Its inference code
currently moves tokenized inputs to CUDA, so CPU-only inference may require a
small code adjustment in `ModelChat.generate_response`.

## Known limitations

- Configuration is embedded in Python rather than supplied through a CLI or
  config file.
- Dataset paths are currently machine-specific.
- The trainer is interactive and is not suitable for unattended jobs without
  refactoring its prompts.
- The training script enables aggressive CUDA tuning at import time.
- Model and dataset quality, licensing, and privacy must be evaluated before
  using external data.

## License

See [`LICENSE`](LICENSE) for the applicable license terms.
