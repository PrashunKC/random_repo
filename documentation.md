# Training pipeline reference

## Execution flow

`model_trainer_v_1_1.py` performs these operations:

1. Defines the supported base models and scans `saved_model` for reusable
   checkpoints.
2. Configures Hugging Face and PyTorch cache locations.
3. Selects a device and precision interactively.
4. Optionally checks, downloads, validates, and analyzes dataset files.
5. Loads a fresh model or an existing final checkpoint.
6. Processes each entry in `DATASET_CONFIGS` as a training phase.
7. Saves phase checkpoints and the final tokenizer/model pair.

The `TEST_MODE_CONFIGS` mapping is intended for a small smoke test. It is not
a guarantee that the configured files exist or that the resulting model is
useful.

## Dataset configuration

Each phase configuration contains:

- `name`: dataset name passed to `datasets.load_dataset`.
- `split`: split name requested from the dataset.
- `text_column`: field containing training text.
- `data_files`: local files or dataset inputs used by the phase.
- `max_samples`: optional upper bound for processing.
- `trust_remote_code`: whether the dataset loader may execute its loader code.

The checked-in mappings contain absolute Windows cache paths from the original
development machine. Update every `data_files` entry before running elsewhere.
Prefer explicit local data files or a documented dataset revision, and review
remote dataset code before enabling `trust_remote_code`.

## Checkpoints and inference

The trainer writes phase checkpoints and `final_model` below its configured
checkpoint directory. A valid final directory must contain the model
configuration/weights and tokenizer files expected by
`GPT2LMHeadModel.from_pretrained` and `GPT2Tokenizer.from_pretrained`.

`chat.py` loads that directory, assigns an EOS token as padding when needed,
and generates one sampled continuation per prompt. The terminal commands are:

- `quit` exits.
- `clear` clears the visible terminal area.
- Any other non-empty line is sent to the model.

## Precision and hardware

The menu exposes INT8, FP16, BF16, FP32, FP64, and an experimental FP128
label. The FP128 option is only a logging easter egg; PyTorch does not train
the model in IEEE FP128. BF16 depends on GPU support, and INT8 depends on
`bitsandbytes`. CPU mode is available but may be slow and some inference
paths still assume CUDA.

## Security and reproducibility

- Keep Hugging Face credentials outside source control.
- Do not commit `.cache`, `saved_model`, `*.pt`, `*.bin`, or run logs.
- Record the base model revision, dataset revision, tokenizer, precision, and
  configuration used for a reproducible experiment.
- Treat downloaded datasets and remote loader code as untrusted inputs.
- Check the license and privacy requirements of training data before use.

## Troubleshooting

| Symptom | Likely cause |
| --- | --- |
| Dataset file missing | Absolute paths in the phase mappings do not exist on this machine |
| CUDA out-of-memory | Select a smaller model/precision, reduce batch/chunk sizes, or use CPU |
| Tokenizer padding error | The selected model has no pad token; the trainer normally maps it to EOS |
| Chat model not found | A successful training run has not created `saved_model/final_model` |
| Authentication failure | Hugging Face credentials are absent or the placeholder assignment is still active |
