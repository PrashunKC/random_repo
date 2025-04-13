import os
import sys
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset
from typing import Dict, List, Tuple, Optional, Generator, Any
import shutil
import time
import gc
from torch.utils.data import DataLoader
from itertools import islice
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
import json

AVAILABLE_MODELS = {
    "gpt2": {"name": "gpt2", "description": "Small GPT-2 (124M parameters)"},
    "gpt2-medium": {"name": "gpt2-medium", "description": "Medium GPT-2 (355M parameters)"},
    "gpt2-large": {"name": "gpt2-large", "description": "Large GPT-2 (774M parameters)"},
    "EleutherAI/gpt-neo-125M": {"name": "EleutherAI/gpt-neo-125M", "description": "GPT-Neo 125M"},
    "facebook/opt-125m": {"name": "facebook/opt-125m", "description": "Meta OPT 125M"},
    "bigscience/bloom-560m": {"name": "bigscience/bloom-560m", "description": "BLOOM 560M"}
}

def scan_saved_models():
    """Scan for saved models in the checkpoints directory"""
    saved_models = {}
    
    # Check model_checkpoint_dir for saved models
    if os.path.exists(model_checkpoint_dir):
        # Check for final model
        final_model_path = os.path.join(model_checkpoint_dir, "final_model")
        if os.path.exists(final_model_path):
            saved_models["final_model"] = {
                "name": final_model_path, 
                "description": "Final saved model"
            }
            
        # Check for phase checkpoints directories
        for item in os.listdir(model_checkpoint_dir):
            checkpoint_dir = os.path.join(model_checkpoint_dir, item)
            if os.path.isdir(checkpoint_dir) and item.endswith("_checkpoints"):
                phase_name = item.replace("_checkpoints", "")
                
                # Look for individual checkpoints
                for ckpt in os.listdir(checkpoint_dir):
                    if ckpt.startswith("checkpoint_") and os.path.isdir(os.path.join(checkpoint_dir, ckpt)):
                        chunk_num = ckpt.replace("checkpoint_", "")
                        model_name = f"{phase_name}_checkpoint_{chunk_num}"
                        saved_models[model_name] = {
                            "name": os.path.join(checkpoint_dir, ckpt),
                            "description": f"Checkpoint {chunk_num} from {phase_name}"
                        }
    
    return saved_models

def select_model(include_saved=True):
    """Allow user to select a model to train - both base models and saved checkpoints"""
    all_models = {}
    
    # Add pre-trained models
    for k, v in AVAILABLE_MODELS.items():
        all_models[k] = v
    
    # Add saved models if requested
    if include_saved:
        saved_models = scan_saved_models()
        if saved_models:
            print("\n=== Available Saved Models ===")
            for k, v in saved_models.items():
                all_models[k] = v
    
    # Display selection menu
    print("\n=== Select Model ===")
    for i, (model_id, info) in enumerate(all_models.items(), 1):
        print(f"{i}. {model_id}: {info['description']}")
    
    while True:
        try:
            choice = int(input(f"\nSelect model (1-{len(all_models)}): ").strip())
            if 1 <= choice <= len(all_models):
                selected_model_id = list(all_models.keys())[choice-1]
                selected_model_path = all_models[selected_model_id]["name"]
                print(f"Selected model: {selected_model_id}")
                return selected_model_id, selected_model_path
            else:
                print(f"Please enter a number between 1 and {len(all_models)}")
        except ValueError:
            print("Please enter a valid number")

# Add this function after the select_model function
def select_precision():
    """Allow user to select precision mode for training"""
    print("\n=== Select Precision Mode ===")
    precision_modes = [
        {"name": "8-bit Quantization (Q8)", "id": "int8", "description": "Fastest, lowest memory usage, may affect convergence"},
        {"name": "Half Precision (FP16)", "id": "fp16", "description": "Good balance of speed and accuracy"},
        {"name": "BFloat16 (BF16)", "id": "bf16", "description": "Better numerical stability than FP16"},
        {"name": "Full Precision (FP32)", "id": "fp32", "description": "Standard precision, higher memory usage"},
        {"name": "Double Precision (FP64)", "id": "fp64", "description": "Highest precision, extreme memory usage"},
    ]
    
    # Easter egg
    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
        precision_modes.append({"name": "Precision go brrrt (FP128)", "id": "fp128", "description": "🚀 Experimental ultra-high precision"})
    
    for i, mode in enumerate(precision_modes, 1):
        print(f"{i}. {mode['name']}: {mode['description']}")
    
    while True:
        try:
            choice = int(input(f"\nSelect precision (1-{len(precision_modes)}): ").strip())
            if 1 <= choice <= len(precision_modes):
                selected = precision_modes[choice-1]
                print(f"Selected precision: {selected['name']}")
                return selected['id']
            else:
                print(f"Please enter a number between 1 and {len(precision_modes)}")
        except ValueError:
            print("Please enter a valid number")

os.environ["HUGGINGFACE_TOKEN"] = ""  # Add your token here
os.environ["HF_HOME"] = "./.cache/huggingface"

# Configure extreme memory optimization
def configure_extreme_memory_savings():
    """Configure PyTorch for extreme memory efficiency"""
    # Most aggressive CUDA memory settings
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
        "expandable_segments:True,"
        "max_split_size_mb:64,"  # Increased from 32
        "garbage_collection_threshold:0.8,"  # Less aggressive GC
        "roundup_power2_divisions:32"  # More granular memory blocks
    )
    
    # CUDA optimizations
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Memory management
    torch.cuda.empty_cache()
    if hasattr(torch.cuda, 'memory'):
        torch.cuda.memory.set_per_process_memory_fraction(0.85)
    
    # Force garbage collection
    gc.collect()

# Apply extreme memory optimizations at startup
configure_extreme_memory_savings()

TEST_MODE_CONFIGS = {
    "phase1": {
        "name": "json",
        "split": "code",
        "text_column": "text",
        "data_files": {
            "code": r"C:\Users\nitro V16\.cache\huggingface\hub\datasets--nvidia--Llama-Nemotron-Post-Training-Dataset\blobs\58796a4ae89504bf0f1a5b073253b6dd06f5bdd8a1b5de3316939f81b9dfe960"
        },
        "max_samples": 1000,
        "trust_remote_code": True,
        "available_splits": ["code"]
    },
    "phase2": {
        "name": "json",
        "split": "combined",
        "text_column": "text",
        "data_files": {
            "code": r"C:\Users\nitro V16\.cache\huggingface\hub\datasets--nvidia--Llama-Nemotron-Post-Training-Dataset\blobs\dc8cbf0e4b729335ca70c0bf9a41b93bcaa7db6b011be21cd922bbde2f5fb65f",
            "math": r"C:\Users\nitro V16\.cache\huggingface\hub\datasets--nvidia--Llama-Nemotron-Post-Training-Dataset\blobs\84c561a510cac2652d787a8c6530635a51079679b0fac6c6430e85330da0ff74",
            "science": r"C:\Users\nitro V16\.cache\huggingface\hub\datasets--nvidia--Llama-Nemotron-Post-Training-Dataset\blobs\04e58d912b66a2b0190c4833885c64d95b1a05a16c91bc53403642e5cfae3e0c"
        },
        "max_samples": 500,
        "trust_remote_code": True,
        "available_splits": ["code", "math", "science"]
    },
    "phase3": {
        "name": "json",
        "split": "combined",
        "text_column": "text",
        "data_files": {
            "chat": r"C:\Users\nitro V16\.cache\huggingface\hub\datasets--nvidia--Llama-Nemotron-Post-Training-Dataset\blobs\71a04ec30b93abaf9121eca67baa5f2d9304adc3a5a28b38121b2ac68056d12b",
            "safety": r"C:\Users\nitro V16\.cache\huggingface\hub\datasets--nvidia--Llama-Nemotron-Post-Training-Dataset\blobs\e3ffe6f7a469d276d6f1af7568f116ce9cabe28274a949c6ffdd1f4c240f75c8"
        },
        "max_samples": 250,
        "trust_remote_code": True,
        "available_splits": ["chat", "safety"]
    }
}

def optimize_cuda_settings():
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'

def resolve_symlink(file_path):
    try:
        if os.path.islink(file_path):
            return os.path.realpath(file_path)
        return file_path
    except Exception as e:
        print(f"Warning: Could not resolve symlink {file_path}: {e}")
        return file_path

def get_device_settings(force_cpu=False):
    if force_cpu:
        device = torch.device("cpu")
        print("🖥️ Using CPU for training (may be slower)")
        return device, {}
    
    if torch.cuda.is_available() and not force_cpu:
        device = torch.device("cuda")
        print(f"🚀 Using GPU: {torch.cuda.get_device_name()}")
        cuda_settings = {
            'matmul.allow_tf32': True,
            'benchmark': True,
            'enabled': True,
            'allow_fp16_reduced_precision_reduction': True,
            'allow_tf32': True
        }
        return device, cuda_settings
    else:
        device = torch.device("cpu")
        print("⚠️ No GPU detected, falling back to CPU")
        return device, {}

DATASET_CONFIGS = {
    "phase1": {
        "name": "json",
        "split": "combined",  # Fixed duplicate key issue
        "text_column": "text",
        "data_files": {
            "chat": r"C:\Users\nitro V16\.cache\huggingface\hub\datasets--nvidia--Llama-Nemotron-Post-Training-Dataset\blobs\71a04ec30b93abaf9121eca67baa5f2d9304adc3a5a28b38121b2ac68056d12b",
            "safety": r"C:\Users\nitro V16\.cache\huggingface\hub\datasets--nvidia--Llama-Nemotron-Post-Training-Dataset\blobs\e3ffe6f7a469d276d6f1af7568f116ce9cabe28274a949c6ffdd1f4c240f75c8"
        },
        "max_samples": 25000,
        "trust_remote_code": True,
        "available_splits": ["chat", "safety"]
    }
}

def clear_dataset_cache(config: Dict) -> bool:
    try:
        cache_dir = os.path.join(".cache", "huggingface", "datasets")
        dataset_path = os.path.join(cache_dir, config['name'].replace("/", "___"))
        if os.path.exists(dataset_path):
            shutil.rmtree(dataset_path)
            print(f"Cleared cache at {dataset_path}")
            return True
        return False
    except Exception as e:
        print(f"Error clearing cache: {e}")
        return False

def download_dataset(config: Dict, show_progress=True) -> bool:
    try:
        dataset = load_dataset(
            config['name'],
            split=config.get('split', 'train'),
            streaming=False,
            trust_remote_code=True,
            token=os.getenv("HUGGINGFACE_TOKEN"),
            cache_dir="./.cache/huggingface"
        )
        
        if show_progress:
            print(f"Dataset info:")
            print(f"- Size: {len(dataset):,} samples")
            print(f"- Features: {dataset.features}")
            
        return True
    except Exception as e:
        if show_progress:
            print(f"Error downloading dataset: {e}")
        return False

def check_dataset_status(config: Dict) -> Tuple[bool, Optional[str]]:
    try:
        if config['name'] == 'json':
            for split_type, file_path in config['data_files'].items():
                if not os.path.exists(file_path):
                    return False, f"File not found: {file_path}"
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        first_line = f.readline().strip()
                        json.loads(first_line)
                except Exception as e:
                    return False, f"Error reading {split_type}: {str(e)}"
                    
            return True, None
        else:
            def check_with_timeout():
                dataset = load_dataset(
                    config['name'],
                    split=config.get('split', 'train'),
                    streaming=True,
                    trust_remote_code=True
                )
                next(iter(dataset))
                return True
                
            import threading
            result = {"success": False, "error": None}
            
            def run_check():
                try:
                    result["success"] = check_with_timeout()
                except Exception as e:
                    result["error"] = str(e)
            
            thread = threading.Thread(target=run_check)
            thread.start()
            thread.join(timeout=10)
            
            if thread.is_alive():
                return False, "Dataset check timed out"
                
            return result["success"], result["error"]
            
    except Exception as e:
        return False, str(e)

def validate_downloads(config: Dict) -> bool:
    print("\nValidating downloads...")
    
    phases = list(config.keys())
    all_valid = True
    
    for phase in phases:
        print(f"\nValidating {phase}...")
        phase_config = config[phase]
        
        try:
            if phase_config['name'] == 'json':
                for split_type, file_paths in phase_config['data_files'].items():
                    if isinstance(file_paths, str):
                        file_paths = [file_paths]
                        
                    for file_path in file_paths:
                        file_path = resolve_symlink(file_path)
                        if os.path.exists(file_path):
                            with open(file_path, 'r', encoding='utf-8') as f:
                                first_line = f.readline().strip()
                                try:
                                    import json
                                    json.loads(first_line)
                                    print(f"Validated {split_type} file: {file_path}")
                                except json.JSONDecodeError as e:
                                    raise ValueError(f"Invalid JSON in {file_path}: {e}")
                        else:
                            raise FileNotFoundError(f"File not found: {file_path}")
                print(f"{phase} validation passed")
            else:
                dataset = load_dataset(
                    phase_config['name'],
                    split=phase_config.get('split', 'train'),
                    streaming=True,
                    trust_remote_code=True
                )
                next(iter(dataset))
                print(f"{phase} validation passed")
            
        except Exception as e:
            print(f"Validation failed for {phase}: {str(e)}")
            all_valid = False
            
    if not all_valid:
        print("\nValidation failed for some datasets")
        return False
            
    return True

def extract_training_text(file_path):
    """Analyze a single JSON file and directly show what content is available"""
    print(f"\n🔍 Examining file: {file_path}")
    
    samples_extracted = 0
    examples = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # Examine the first 5 records to understand structure
            for _ in range(5):
                line = f.readline().strip()
                if not line:
                    continue
                    
                data = json.loads(line)
                examples.append(data)
                print(f"Keys in record: {list(data.keys())}")
                
                # Try to find the actual content
                for key in data.keys():
                    value = data[key]
                    preview = str(value)[:50] + "..." if len(str(value)) > 50 else str(value)
                    print(f"  {key}: {preview}")
    except Exception as e:
        print(f"Error reading file: {e}")
        
    print("\n")
    return examples

def verify_dataset_files(config):
    """Verify all dataset files exist before training starts"""
    print("\nVerifying dataset files...")
    all_exist = True
    
    for phase, phase_config in config.items():
        if phase_config['name'] == 'json':
            for split_type, file_path in phase_config['data_files'].items():
                if not os.path.exists(file_path):
                    print(f"❌ Missing file for {phase}/{split_type}: {file_path}")
                    all_exist = False
                else:
                    print(f"✅ Found {phase}/{split_type}: {file_path}")
                    # Extract and analyze content structure
                    examples = extract_training_text(file_path)
                    
                    # Update the text_column based on what we find
                    if examples:
                        example = examples[0]
                        # Look for the most promising text field
                        for key in example.keys():
                            if isinstance(example[key], str) and len(example[key]) > 100:
                                print(f"🔥 Found good text content in field: '{key}'")
                                phase_config['text_column'] = key
                                print(f"Setting text_column to '{key}' for {phase}")
                                break
    
    return all_exist

def manage_datasets():
    while True:
        print("\n=== Dataset Management System ===")
        print("1. View dataset status")
        print("2. Download specific datasets")
        print("3. Download all datasets")
        print("4. Clear cache")
        print("5. Validate downloads")
        print("6. Force redownload all")
        print("7. Exit")
        
        choice = input("\nEnter choice (1-7): ").strip()
        
        if choice == "1":
            print("\nChecking dataset status...")
            for phase, config in DATASET_CONFIGS.items():
                print(f"\n{phase}: {config['name']}")
                print("Splits:", ', '.join(config.get('available_splits', ['default'])))
                
                status, error = check_dataset_status(config)
                if status:
                    print("Status: ✅ Available")
                else:
                    print("Status: ❌ Not available")
                    if error:
                        print(f"Error: {error}")
                
                if config['name'] == 'json':
                    for split_type, file_path in config['data_files'].items():
                        if os.path.exists(file_path):
                            size_mb = os.path.getsize(file_path) / (1024 * 1024)
                            print(f"{split_type}: {size_mb:.1f} MB")
                    
        elif choice == "2":
            print("\nAvailable datasets:")
            for i, (phase, config) in enumerate(DATASET_CONFIGS.items(), 1):
                status, _ = check_dataset_status(config)
                print(f"{i}. {phase}: {config['name']} - {'[Downloaded]' if status else '[Not Downloaded]'}")
            
            try:
                selections = input("\nEnter dataset numbers (comma-separated): ").strip()
                indices = [int(x.strip()) for x in selections.split(",")]
                
                for idx in indices:
                    if 1 <= idx <= len(DATASET_CONFIGS):
                        phase = list(DATASET_CONFIGS.keys())[idx-1]
                        config = DATASET_CONFIGS[phase]
                        print(f"\nDownloading {phase}...")
                        
                        if clear_dataset_cache(config):
                            print("Cleared existing cache")
                        if download_dataset(config, show_progress=True):
                            print(f"Successfully downloaded {phase}")
                        else:
                            print(f"Failed to download {phase}")
            except ValueError:
                print("Invalid input. Please enter numbers separated by commas.")
                
        elif choice == "3":
            for phase, config in DATASET_CONFIGS.items():
                print(f"\nDownloading {phase}...")
                if clear_dataset_cache(config):
                    print("Cleared existing cache")
                if download_dataset(config, show_progress=True):
                    print(f"Successfully downloaded {phase}")
                else:
                    print(f"Failed to download {phase}")
                    
        elif choice == "4":
            for phase, config in DATASET_CONFIGS.items():
                if clear_dataset_cache(config):
                    print(f"Cleared cache for {phase}")
                else:
                    print(f"No cache found for {phase}")
                    
        elif choice == "5":
            if validate_downloads(DATASET_CONFIGS):
                print("All downloads validated successfully")
            else:
                print("Validation failed for some datasets")
                    
        elif choice == "6":
            print("\n=== Force Redownload All Datasets ===")
            print("This will delete all cached datasets and redownload them.")
            confirm = input("Are you sure? (y/n): ").strip().lower()
            
            if confirm == 'y':
                for phase, config in DATASET_CONFIGS.items():
                    print(f"\nProcessing {phase}...")
                    if clear_dataset_cache(config):
                        print(f"Cleared cache for {phase}")
                    print(f"Redownloading {phase}...")
                    if download_dataset(config, show_progress=True):
                        print(f"Successfully downloaded {phase}")
                    else:
                        print(f"Failed to download {phase}")
            
        elif choice == "7":
            break
            
        else:
            print("Invalid choice")

# === GPU Settings ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CUDA Optimizations
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Memory Management
    torch.cuda.empty_cache()
    if hasattr(torch.cuda, "memory"):
        torch.cuda.memory.set_per_process_memory_fraction(0.85)

# Memory Allocator Config
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = (
    'expandable_segments:True,'
    'max_split_size_mb:64,'  # Increased from 32
    'garbage_collection_threshold:0.8,'  # Less aggressive GC
    'roundup_power2_divisions:32'
)

# === Settings ===
model_checkpoint_dir = os.path.join(os.path.dirname(__file__), "saved_model")
os.makedirs(model_checkpoint_dir, exist_ok=True)
dataset_path = r"C:\Users\nitro V16\.cache\huggingface\hub\datasets--nvidia--Llama-Nemotron-Post-Training-Dataset\snapshots\8e1e47a67ced79723ad0735efc5a45f8bb5aabd6\SFT"

def print_memory_stats():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        
        print("\nGPU Memory Status:")
        print(f"Allocated: {allocated/1024**2:.1f}MB")
        print(f"Reserved: {reserved/1024**2:.1f}MB")
        
        free_mem, total_mem = torch.cuda.mem_get_info()
        print(f"Free: {free_mem/1024**2:.1f}MB")
        print(f"Total: {total_mem/1024**2:.1f}MB")
        
        # Clear cache if memory is tight
        if free_mem / total_mem < 0.2:
            torch.cuda.empty_cache()

def print_gpu_info():
    if torch.cuda.is_available():
        print("\nGPU Information:")
        print(f"Device: {torch.cuda.get_device_name()}")
        print(f"Memory Allocated: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
        print(f"Memory Cached: {torch.cuda.memory_reserved()/1024**2:.1f}MB")
        print(f"Memory Summary:")
        print(torch.cuda.memory_summary())

class GPTDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
        self._length = len(encodings['input_ids'])

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}

    def __len__(self):
        return self._length

def create_dataset(tokenizer, path):
    print("Loading and preprocessing data directly on GPU...")
    
    chunk_size = 1024  # Much smaller chunks
    all_outputs = []
    
    with open(path, 'r', encoding='utf-8') as f:
        while True:
            chunks = []
            for _ in range(chunk_size):
                line = f.readline()
                if not line:
                    break
                if line.strip():
                    chunks.append(line.strip())
            
            if not chunks:
                break
            
            # First tokenize on CPU
            outputs = tokenizer(
                chunks,
                truncation=True,
                max_length=64,  # Use shorter sequence length
                padding='max_length',
                return_tensors='pt'
            )
            
            # Then move tensors to GPU
            outputs = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v 
                for k, v in outputs.items()
            }
            
            outputs['labels'] = outputs['input_ids'].clone()
            all_outputs.append(outputs)
            print_memory_stats()
    
    print("Combining processed chunks...")
    # Combine on GPU efficiently
    combined_outputs = {
        k: torch.cat([out[k] for out in all_outputs], dim=0) 
        if isinstance(all_outputs[0][k], torch.Tensor)
        else all_outputs[0][k]
        for k in all_outputs[0].keys()
    }
    
    # Clear memory
    all_outputs = None
    torch.cuda.empty_cache()
    
    dataset = GPTDataset(combined_outputs)
    print(f"Dataset processed. Size: {len(dataset)} samples")
    return dataset

def process_batch(examples, text_column):
    # Convert to list if not already
    if isinstance(examples[text_column], str):
        examples[text_column] = [examples[text_column]]
        
    # Create tensors for each sequence    
    sequences = []
    masks = []
    
    # First pass - convert to tensors
    for text in examples[text_column]:
        seq = torch.tensor([ord(c) for c in text], dtype=torch.long)
        mask = torch.ones_like(seq)
        sequences.append(seq)
        masks.append(mask)
    
    # Pad sequences to max length in batch using pad_sequence
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
    padded_masks = pad_sequence(masks, batch_first=True, padding_value=0)
    
    return {
        "input_ids": padded_sequences,
        "attention_mask": padded_masks
    }

def try_decode_line(line: bytes) -> Optional[str]:
    encodings = [
        'utf-8-sig',
        'utf-8',
        'latin1',
        'cp1252',
        'ascii'
    ]
    
    for encoding in encodings:
        try:
            return line.decode(encoding).strip()
        except UnicodeDecodeError:
            continue
    return None

def freeze_layers(model, num_layers_to_freeze):
    """Freeze lower layers of the model to save memory"""
    # For GPT2, layers are in transformer.h
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        print(f"Freezing {num_layers_to_freeze} out of {len(model.transformer.h)} layers")
        for i, layer in enumerate(model.transformer.h):
            if i < num_layers_to_freeze:
                for param in layer.parameters():
                    param.requires_grad = False
                # Optional: Move frozen layers to CPU
                if i < num_layers_to_freeze // 2:  # Move half of frozen layers to CPU
                    layer.to('cpu')
                    print(f"Layer {i} moved to CPU")
    return model

def memory_efficient_forward(model, batch, device):
    """Handle forward pass with extreme memory efficiency"""
    # Split batch into micro-batches of just 1 sample if needed
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    
    batch_size = input_ids.shape[0]
    all_losses = []
    
    # Process each sample individually if batch size > 1
    if batch_size > 1:
        for i in range(batch_size):
            # Create a micro-batch
            micro_batch = {
                'input_ids': input_ids[i:i+1].to(device),
                'attention_mask': attention_mask[i:i+1].to(device),
                'labels': input_ids[i:i+1].to(device)
            }
            
            # Forward pass
            with torch.autocast(device_type='cuda'):
                outputs = model(**micro_batch)
                all_losses.append(outputs.loss)
            
            # Free memory immediately
            del micro_batch
            torch.cuda.empty_cache()
        
        # Average the losses
        loss = torch.mean(torch.stack(all_losses))
        return type('outputs', (), {'loss': loss})()  # Return a dummy object with loss attribute
    else:
        # If already batch size 1, just do normal forward pass
        batch['labels'] = batch['input_ids'].clone()
        with torch.autocast(device_type='cuda'):
            return model(**batch)

def load_or_download_dataset(phase_config, tokenizer, start_percent, chunk_percent=10):
    """Load dataset with better debugging and memory efficiency"""
    torch.cuda.empty_cache()  # Clear GPU cache before loading
    
    if phase_config['name'] == 'json':
        for split_type, file_path in phase_config['data_files'].items():
            dynamic_print(f"Loading {split_type} split from {chunk_percent}% chunk")
            
            # Calculate chunk boundaries
            file_size = os.path.getsize(file_path)
            chunk_size = int((chunk_percent / 100.0) * file_size)
            start_pos = int((start_percent / 100.0) * file_size)
            
            batch_counter = 0
            sample_counter = 0
            current_batch = {'input_ids': [], 'attention_mask': []}
            batch_size = 4
            max_length = 128
            
            with open(file_path, 'rb') as f:
                f.seek(start_pos)
                
                # Debug: inspect first line to understand file structure
                sample_line = f.readline()
                sample_text = try_decode_line(sample_line)
                if sample_text:
                    try:
                        sample_json = json.loads(sample_text)
                        
                        # Check if expected text column exists
                        if phase_config['text_column'] not in sample_json:
                            dynamic_print(f"Column '{phase_config['text_column']}' not found. Trying alternatives...", end="\n")
                            
                            # Check for alternative columns
                            for alt_col in ['text', 'content', 'input', 'message', 'conversation']:
                                if alt_col in sample_json:
                                    phase_config['text_column'] = alt_col
                                    dynamic_print(f"Using '{alt_col}' instead", end="\n")
                                    break
                    except Exception as e:
                        dynamic_print(f"Error examining JSON: {e}", end="\n")
                
                # Return to start position
                f.seek(start_pos)
                bytes_read = 0
                
                with tqdm(total=chunk_size, desc="Loading", unit='B', unit_scale=True, position=1, leave=False) as pbar:
                    while bytes_read < chunk_size:
                        try:
                            line = f.readline()
                            if not line:
                                break
                                
                            text = try_decode_line(line)
                            if text:
                                try:
                                    item = json.loads(text)
                                    
                                    # Try both the configured column and fallbacks
                                    content = None
                                    
                                    # First try the configured column
                                    if phase_config['text_column'] in item:
                                        content = item[phase_config['text_column']]
                                    
                                    # If that failed, try common alternatives
                                    if not content:
                                        for key in ['text', 'content', 'value', 'input', 'conversation', 'message']:
                                            if key in item and item[key]:
                                                content = item[key]
                                                if sample_counter == 0:  # Only print once
                                                    dynamic_print(f"Using alternative field: {key}", end="\n")
                                                break
                                    
                                    # Special handling for nested structures
                                    if not content and 'messages' in item and isinstance(item['messages'], list):
                                        # Try to extract from chat format
                                        messages = []
                                        for msg in item['messages']:
                                            if 'content' in msg:
                                                messages.append(msg['content'])
                                        if messages:
                                            content = " ".join(messages)
                                    
                                    if content and isinstance(content, str) and len(content.strip()) > 20:
                                        # Process on CPU first
                                        tokenized = tokenizer(
                                            content,
                                            truncation=True,
                                            max_length=128,
                                            padding='max_length',
                                            return_tensors='pt'
                                        )
                                        
                                        current_batch['input_ids'].append(tokenized['input_ids'][0])
                                        current_batch['attention_mask'].append(tokenized['attention_mask'][0])
                                        sample_counter += 1
                                        
                                        dynamic_print(f"Processing data: {sample_counter} samples, {batch_counter} batches")
                                        
                                        # Yield full batches
                                        if len(current_batch['input_ids']) >= batch_size:
                                            batch = {
                                                'input_ids': torch.stack(current_batch['input_ids']),
                                                'attention_mask': torch.stack(current_batch['attention_mask'])
                                            }
                                            batch_counter += 1
                                            yield batch
                                            current_batch = {'input_ids': [], 'attention_mask': []}
                                        
                                except Exception:
                                    # Skip errors silently to avoid console spam
                                    pass
                                        
                        except Exception:
                            # Skip errors silently
                            pass
                            
                        bytes_read += len(line)
                        pbar.update(len(line))
                
                # Yield remaining batch
                if current_batch['input_ids']:
                    batch = {
                        'input_ids': torch.stack(current_batch['input_ids']),
                        'attention_mask': torch.stack(current_batch['attention_mask'])
                    }
                    yield batch
                
            dynamic_print(f"Completed loading: {sample_counter} samples in {batch_counter + (1 if current_batch['input_ids'] else 0)} batches", end="\n")

def prefetch_batches(dataset_stream, n_prefetch=3):
    """Prefetch batches to increase GPU utilization"""
    prefetch_queue = []
    
    # Prime the queue
    for _ in range(n_prefetch):
        try:
            batch = next(dataset_stream)
            prefetch_queue.append(batch)
        except StopIteration:
            break
    
    # Yield and refill
    while prefetch_queue:
        # Yield the first item
        yield prefetch_queue.pop(0)
        
        # Try to get a new item
        try:
            batch = next(dataset_stream)
            prefetch_queue.append(batch)
        except StopIteration:
            continue

def prepare_batch(batch_items):
    max_len = max(b["input_ids"].size(1) for b in batch_items)
    
    batch_data = {
        "input_ids": torch.stack([
            torch.nn.functional.pad(
                b["input_ids"],
                (0, max_len - b["input_ids"].size(1))
            ) for b in batch_items
        ]),
        "attention_mask": torch.stack([
            torch.nn.functional.pad(
                b["attention_mask"],
                (0, max_len - b["attention_mask"].size(1))
            ) for b in batch_items
        ])
    }
    
    return {k: v.to('cuda') for k, v in batch_data.items()}

def validate_dataset(dataset: Dataset) -> bool:
    sample_size = 0
    samples = []
    
    try:
        for item in islice(dataset, 1000):
            samples.append(item)
            sample_size += 1
            
        if sample_size < 2:
            print(f"WARNING: Dataset sample too small ({sample_size} samples)")
            return False

        text_field = None
        for field in ['content', 'text', 'problem']:
            if field in samples[0]:
                text_field = field
                break
                
        if text_field is None:
            print(f"WARNING: No recognizable text field found in: {list(samples[0].keys())}")
            return False

        empty_count = 0
        for item in samples:
            if not item[text_field] or not item[text_field].strip():
                empty_count += 1
                
        if empty_count > sample_size * 0.1:
            print(f"WARNING: Too many empty texts ({empty_count}/{sample_size})")
            return False

        print(f"Dataset validation passed on {sample_size} sample items")
        return True

    except Exception as e:
        print(f"Error during dataset validation: {e}")
        return False

class CustomDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, features):
        batch = self.tokenizer.pad(
            features,
            padding='max_length',
            max_length=64,  # Smaller max length
            return_tensors="pt"
        )
        
        # Move directly to GPU and use half precision
        batch = {k: v.cuda().half() for k, v in batch.items()}
        return batch

def analyze_dataset(dataset_path, quick_mode=False):
    print("\n=== Dataset Analysis ===")
    
    if quick_mode:
        print("Running in QUICK ANALYSIS mode (sampling only)")
        
    try:
        total_stats = {
            'total_chars': 0,
            'total_words': 0,
            'total_lines': 0,
            'unique_words': set()
        }
        
        jsonl_files = []
        for root, _, files in os.walk(dataset_path):
            for file in files:
                if file.endswith('.jsonl'):
                    jsonl_files.append(os.path.join(root, file))
        
        for file_path in jsonl_files:
            file_name = os.path.basename(file_path)
            print(f"\nAnalyzing {file_name}...")
            
            if quick_mode:
                estimated_size = os.path.getsize(file_path)
                with open(file_path, 'r', encoding='utf-8') as f:
                    sample_lines = []
                    for _ in range(500):
                        line = f.readline()
                        if not line:
                            break
                        sample_lines.append(line)
                    
                    f.seek(estimated_size // 2)
                    f.readline()
                    for _ in range(250):
                        line = f.readline()
                        if not line:
                            break
                        sample_lines.append(line)
                    
                    f.seek(max(0, estimated_size - 50000))
                    f.readline()
                    for _ in range(250):
                        line = f.readline()
                        if not line:
                            break
                        sample_lines.append(line)
                
                print(f"Quick sampling: {len(sample_lines)} lines")
                for line in tqdm(sample_lines, desc="Processing samples"):
                    try:
                        if not line.strip():
                            continue
                        content = json.loads(line)
                        text = content.get('text', content.get('content', content.get('input', '')))
                        
                        total_stats['total_chars'] += len(text)
                        words = text.split()
                        total_stats['total_words'] += len(words)
                        total_stats['unique_words'].update(w.lower() for w in words)
                        total_stats['total_lines'] += 1
                        
                    except Exception:
                        continue
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    total_lines = sum(1 for _ in tqdm(f, desc="Counting lines", unit=" lines"))
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in tqdm(f, total=total_lines, desc="Processing", unit=" samples"):
                        try:
                            if not line.strip():
                                continue
                                
                            content = json.loads(line)
                            text = content.get('text', content.get('content', content.get('input', '')))
                            
                            total_stats['total_chars'] += len(text)
                            words = text.split()
                            total_stats['total_words'] += len(words)
                            total_stats['unique_words'].update(w.lower() for w in words)
                            total_stats['total_lines'] += 1
                            
                        except Exception:
                            continue
        
        print("\n=== Dataset Statistics ===")
        if quick_mode:
            print("NOTE: Statistics based on sampled data, not the entire dataset")
        print(f"Total characters: {total_stats['total_chars']:,}")
        print(f"Total words: {total_stats['total_words']:,}")
        print(f"Unique words: {len(total_stats['unique_words']):,}")
        print(f"Total samples analyzed: {total_stats['total_lines']:,}")
        
        return True
        
    except Exception as e:
        print(f"Error analyzing dataset: {e}")
        return False

def validate_phase(model, tokenizer, phase_config):
    """Fixed validation function that ensures all tensors are on the correct device"""
    try:
        print("\nStarting validation...")
        all_losses = []
        validation_batch_size = 1
        current_batch = {
            'input_ids': [],
            'attention_mask': []
        }
        
        # Make sure model is on CUDA
        device = next(model.parameters()).device
        print(f"Model is on device: {device}")
        model.eval()

        file_path = next(iter(phase_config['data_files'].values()))
        samples_processed = 0
        max_validation_samples = 100

        with open(file_path, 'rb') as f:
            with tqdm(total=max_validation_samples, desc="Processing validation data") as pbar:
                while samples_processed < max_validation_samples:
                    try:
                        line = f.readline()
                        if not line:
                            break

                        text = try_decode_line(line)
                        if not text:
                            continue

                        try:
                            item = json.loads(text)
                            content = None
                            
                            # Extract text content from JSON
                            if phase_config['text_column'] in item:
                                content = item[phase_config['text_column']]
                                
                            if not content:
                                for key in ['text', 'content', 'value', 'input', 'conversation', 'message']:
                                    if key in item and item[key]:
                                        content = item[key]
                                        break

                            if content and isinstance(content, str) and len(content.strip()) > 20:
                                # Process on CPU first
                                tokenized = tokenizer(
                                    content,
                                    truncation=True,
                                    max_length=64,
                                    padding='max_length',
                                    return_tensors='pt'
                                )
                                
                                # Move tensors to CPU before appending (critical fix)
                                tokenized = {k: v.to('cpu') for k, v in tokenized.items()}
                                
                                current_batch['input_ids'].append(tokenized['input_ids'][0])
                                current_batch['attention_mask'].append(tokenized['attention_mask'][0])
                                samples_processed += 1
                                pbar.update(1)

                                if len(current_batch['input_ids']) >= validation_batch_size:
                                    # Stack tensors while they're all on CPU
                                    batch = {
                                        'input_ids': torch.stack(current_batch['input_ids']),
                                        'attention_mask': torch.stack(current_batch['attention_mask'])
                                    }
                                    
                                    # Move entire batch to device at once
                                    batch = {k: v.to(device) for k, v in batch.items()}
                                    batch['labels'] = batch['input_ids'].clone()

                                    with torch.no_grad():
                                        outputs = model(**batch)
                                        all_losses.append(outputs.loss.item())

                                    current_batch = {
                                        'input_ids': [],
                                        'attention_mask': []
                                    }
                                    torch.cuda.empty_cache()

                        except json.JSONDecodeError:
                            continue

                    except Exception as e:
                        print(f"\nError processing validation line: {str(e)}")
                        continue

                # Process any remaining items
                if current_batch['input_ids']:
                    batch = {
                        'input_ids': torch.stack(current_batch['input_ids']),
                        'attention_mask': torch.stack(current_batch['attention_mask'])
                    }
                    batch = {k: v.to(device) for k, v in batch.items()}
                    batch['labels'] = batch['input_ids'].clone()

                    with torch.no_grad():
                        outputs = model(**batch)
                        all_losses.append(outputs.loss.item())

        # Calculate average loss and perplexity
        if all_losses:
            avg_loss = sum(all_losses) / len(all_losses)
            perplexity = torch.exp(torch.tensor(avg_loss)).item()
            
            print("\nValidation Results:")
            print(f"Average Loss: {avg_loss:.4f}")
            print(f"Perplexity: {perplexity:.4f}")
            
            return {
                'loss': avg_loss,
                'perplexity': perplexity
            }
        else:
            print("\nNo validation results - no valid batches processed")
            return None

    except Exception as e:
        print(f"Error during validation: {e}")
        return None

def safe_save_model(model, tokenizer, save_path):
    """More robust model saving with better error handling and alternative approaches"""
    temp_dir = None
    model_cpu = model.to('cpu')  # Move to CPU first
    torch.cuda.empty_cache()
    
    try:
        # Generate a unique temp directory name with timestamp
        timestamp = int(time.time())
        temp_dir = f"{save_path}_temp_{timestamp}"
        os.makedirs(temp_dir, exist_ok=True)
        
        print(f"Saving model to temporary location: {temp_dir}")
        model_cpu.save_pretrained(temp_dir, safe_serialization=True)
        tokenizer.save_pretrained(temp_dir)
        
        # First, try direct renaming
        try:
            if os.path.exists(save_path):
                # Try to remove existing directory
                print(f"Removing existing directory: {save_path}")
                shutil.rmtree(save_path)
                # Wait a moment for filesystem to catch up
                time.sleep(1)
                
            # Rename temp to final
            os.rename(temp_dir, save_path)
            print(f"Successfully saved model to {save_path}")
            return True
            
        except PermissionError:
            # If renaming fails, try copying files instead
            print(f"Permission error during rename. Trying copy method instead...")
            
            # Create the target directory if it doesn't exist
            os.makedirs(save_path, exist_ok=True)
            
            # Copy files from temp to destination
            for item in os.listdir(temp_dir):
                source = os.path.join(temp_dir, item)
                dest = os.path.join(save_path, item)
                
                if os.path.isfile(source):
                    shutil.copy2(source, dest)
                else:
                    if os.path.exists(dest):
                        shutil.rmtree(dest)
                    shutil.copytree(source, dest)
                    
            print(f"Successfully copied model to {save_path}")
            return True
            
    except Exception as e:
        print(f"Error saving model: {e}")
        return False
    finally:
        # Clean up temp directory
        try:
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"Warning: Could not remove temp directory {temp_dir}: {e}")
        
        # Move model back to original device if training will continue
        if hasattr(model, 'device_of_original_model'):
            model.to(model.device_of_original_model)
        else:
            model.to('cuda')

def save_checkpoint(model, tokenizer, phase_name, chunk_num):
    """Fixed checkpoint saving to avoid permission issues"""
    checkpoint_dir = os.path.join(model_checkpoint_dir, f"{phase_name}_checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{chunk_num}")
    
    # Save using our improved function
    if safe_save_model(model, tokenizer, checkpoint_path):
        print(f"\nSaved checkpoint for {phase_name} at chunk {chunk_num}")
        return True
    else:
        print(f"Failed to save checkpoint for {phase_name} at chunk {chunk_num}")
        return False

def train_model_phase(model, tokenizer, phase_config, phase_name, base_lr, chunk_size, device, precision_mode="fp16"):
    """Train model on a single phase with more stable optimization"""
    print(f"\nInitializing {phase_name} training on {device}")
    
    # Store original device for later restoration
    model.device_of_original_model = next(model.parameters()).device
    
    # Memory settings
    max_length = 128
    batch_size = 4
    gradient_accumulation_steps = 16
    
    # Move model to device explicitly
    model = model.to(device)
    model.gradient_checkpointing_enable()
    model.train()
    
    # FIX: Configure model to avoid attention mask overflow
    if hasattr(model, 'config'):
        # These settings help with numerical stability
        model.config.scale_attn_by_inverse_layer_idx = False
        model.config.reorder_and_upcast_attn = True  # Critical for FP16
    
    # Optimizer with improved stability
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=base_lr * 0.5,  # Lower learning rate
        weight_decay=0.001,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Better scheduler - smoother transitions
    from transformers import get_cosine_schedule_with_warmup
    num_batches_estimate = 5000
    warmup_steps = int(num_batches_estimate * 0.05)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps,
        num_training_steps=num_batches_estimate
    )
    
    # Use updated GradScaler format
    use_mixed_precision = precision_mode in ["fp16", "bf16"] and device.type == "cuda"
    scaler = torch.amp.GradScaler('cuda') if use_mixed_precision else None

    # Training loop
    step = 0
    metrics = {'loss': [], 'perplexity': [], 'samples': 0}
    
    # FIX: Add hooks for safe attention masking
    def attention_mask_hook(module, args):
        if len(args) > 1 and args[1] is not None:  # args[1] is attention_mask
            return (args[0], args[1].to(torch.float32), *args[2:])
        return args
    
    # Apply hooks to relevant attention modules
    if precision_mode == "fp16":
        for name, module in model.named_modules():
            if "attn" in name.lower() and hasattr(module, "register_forward_pre_hook"):
                module.register_forward_pre_hook(attention_mask_hook)
                
    # Process each batch
    for batch in prefetch_batches(load_or_download_dataset(phase_config, tokenizer, start_percent=0, chunk_percent=chunk_size)):
        step += 1

        # Forward pass
        batch = {k: v.to(device) for k, v in batch.items()}
        batch['labels'] = batch['input_ids'].clone()
        
        # FIX: Explicitly cast attention_mask to float32 for stability
        if 'attention_mask' in batch and precision_mode == "fp16":
            batch['attention_mask'] = batch['attention_mask'].to(torch.float32)
            
        try:
            if use_mixed_precision:
                dtype = torch.bfloat16 if precision_mode == "bf16" else torch.float16
                with torch.autocast(device_type='cuda', dtype=dtype):
                    outputs = model(**batch)
                    loss = outputs.loss
            else:
                outputs = model(**batch)
                loss = outputs.loss
                
            # Error checking
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"WARNING: Bad loss value detected - skipping batch")
                continue
                
            # Backward pass
            if use_mixed_precision:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
    
            # Record metrics and continue with training
            loss_val = loss.item()
            metrics['loss'].append(loss_val)
            metrics['perplexity'].append(torch.exp(torch.tensor(loss_val)).item())
            metrics['samples'] += batch['input_ids'].size(0)
            
            # Update schedule and clear gradients
            scheduler.step()
            optimizer.zero_grad()
            
            if step % 10 == 0:
                print(f"Step {step}: Loss = {loss_val:.4f}")
                
        except RuntimeError as e:
            print(f"Error during batch processing (skipping): {e}")
            continue
    
    # Return training metrics
    avg_loss = sum(metrics['loss']) / len(metrics['loss']) if metrics['loss'] else float('inf')
    avg_perplexity = sum(metrics['perplexity']) / len(metrics['perplexity']) if metrics['perplexity'] else float('inf')
    
    return {
        'loss': avg_loss,
        'perplexity': avg_perplexity,
        'samples': metrics['samples']
    }

def train(test_mode=False, force_cpu=False):
    # Apply extreme memory optimizations
    configure_extreme_memory_savings()
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    print("\nRunning in", "TEST MODE" if test_mode else "FULL TRAINING MODE")
    
    # Get device settings
    device, cuda_settings = get_device_settings(force_cpu)
    
    # Apply CUDA settings if using GPU
    if device.type == "cuda":
        for setting_name, value in cuda_settings.items():
            if hasattr(torch.backends.cudnn, setting_name):
                setattr(torch.backends.cudnn, setting_name, value)
            elif hasattr(torch.backends.cuda, setting_name):
                setattr(torch.backends.cuda, setting_name, value)
    
    # Select model
    base_model_id, base_model_path = select_model()
    
    # Select precision mode
    precision_mode = select_precision()
    
    # Set chunk size based on mode
    chunk_size = 10 if test_mode else 20
    
    if test_mode:
        print("⚠️ Test mode enabled: Using reduced dataset and settings")
        active_configs = TEST_MODE_CONFIGS
        base_lr = 5e-5  # Smaller base learning rate than 1e-4
        base_batch_size = 1  # Even smaller for test mode
    else:
        print("📚 Full training mode: Using complete dataset")
        active_configs = DATASET_CONFIGS
        base_lr = 5e-5  # Smaller base learning rate than 1e-4
        base_batch_size = 1  # Minimum batch size

    print("\nWould you like to manage datasets before training? (y/n)")
    if input().strip().lower() == 'y':
        manage_datasets()
    
    print("\nDataset analysis options:")
    print("1. Skip analysis (fastest, start training immediately)")
    print("2. Quick analysis (sample data only)")
    print("3. Full analysis (slow for large datasets)")
    analysis_choice = input("Select option (1/2/3): ").strip()
    
    if analysis_choice == "3":
        if not analyze_dataset(dataset_path):
            print("Dataset might be insufficient for good results!")
            user_input = input("Continue anyway? (y/n): ")
            if user_input.lower() != 'y':
                return
    elif analysis_choice == "2":
        analyze_dataset(dataset_path, quick_mode=True)
    else:
        print("Skipping dataset analysis")
    
    # Verify dataset files
    if not verify_dataset_files(active_configs):
        print("\n⚠️ Some dataset files are missing! Continue anyway? (y/n)")
        if input().strip().lower() != 'y':
            return

    # Initialize model and tokenizer
    final_model_path = os.path.join(model_checkpoint_dir, "final_model")
    model = None
    tokenizer = None

    if os.path.exists(final_model_path):
        try:
            model = GPT2LMHeadModel.from_pretrained(final_model_path)
            tokenizer = GPT2Tokenizer.from_pretrained(final_model_path)
            print("Loaded existing model for continued training")
        except Exception as e:
            print(f"Error loading saved model: {e}")
    
    if model is None:
        try:
            print(f"Loading model with {precision_mode} precision...")
            
            # Base loading kwargs
            load_kwargs = {
                'use_cache': False,  # Disable past key values caching during training
            }
            
            if precision_mode == "int8" and device.type == "cuda":
                try:
                    import bitsandbytes as bnb
                    print(f"Using 8-bit quantization for {base_model_id}")
                    load_kwargs['load_in_8bit'] = True
                    model = AutoModelForCausalLM.from_pretrained(base_model_path, **load_kwargs)
                except (ImportError, ValueError) as e:
                    print(f"Failed to load in 8-bit: {e}")
                    print("Falling back to FP16")
                    precision_mode = "fp16"  # Fall back to fp16
            
            # Load the model with the appropriate precision
            if precision_mode == "fp16" and device.type == "cuda":
                model = AutoModelForCausalLM.from_pretrained(base_model_path, **load_kwargs)
                model = model.half()  # Convert to FP16
                print("Model converted to half precision (FP16)")
                
            elif precision_mode == "bf16" and device.type == "cuda":
                if torch.cuda.is_bf16_supported():
                    model = AutoModelForCausalLM.from_pretrained(base_model_path, **load_kwargs)
                    model = model.to(torch.bfloat16)
                    print("Model converted to bfloat16 precision (BF16)")
                else:
                    print("BF16 not supported on this GPU, falling back to FP16")
                    model = AutoModelForCausalLM.from_pretrained(base_model_path, **load_kwargs)
                    model = model.half()
                    
            elif precision_mode == "fp32":
                model = AutoModelForCausalLM.from_pretrained(base_model_path, **load_kwargs)
                print("Model loaded in full precision (FP32)")
                
            elif precision_mode == "fp64":
                model = AutoModelForCausalLM.from_pretrained(base_model_path, **load_kwargs)
                model = model.double()  # Convert to FP64
                print("Model converted to double precision (FP64)")
                
            elif precision_mode == "fp128":
                # Easter egg - doesn't actually use FP128 since it's not supported
                model = AutoModelForCausalLM.from_pretrained(base_model_path, **load_kwargs)
                print("🚀🚀🚀 PRECISION GO BRRRT! 🚀🚀🚀")
                print("(Actually using FP32 with fancy logging)")
                
            model.gradient_checkpointing_enable()  # Trade speed for memory
            
            # Freeze layers if needed
            freeze_percent = 0  # Reduced from 0.6 (freeze 0% of layers)
            if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
                layers_to_freeze = int(len(model.transformer.h) * freeze_percent)
                model = freeze_layers(model, layers_to_freeze)
            
            tokenizer = AutoTokenizer.from_pretrained(base_model_path)
            print(f"Started with fresh {base_model_id} model")
        except Exception as e:
            print(f"Error initializing model: {e}")
            return

    # Ensure tokenizer has proper padding token
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

    # Train on each phase
    for phase_name, phase_config in active_configs.items():
        print(f"\n{'='*20} Starting {phase_name} {'='*20}")
        
        # Train phase
        phase_metrics = train_model_phase(
            model=model,
            tokenizer=tokenizer,
            phase_config=phase_config,
            phase_name=phase_name,
            base_lr=base_lr,
            chunk_size=chunk_size if not test_mode else 10,
            device=device,
            precision_mode=precision_mode  # Pass precision mode
        )
        
        # Phase summary
        print(f"\n{phase_name} Training Summary:")
        print(f"Average Loss: {phase_metrics['loss']:.4f}")
        print(f"Average Perplexity: {phase_metrics['perplexity']:.4f}")
        print(f"Total Samples: {phase_metrics['samples']}")
        
        # Save phase model
        save_path = os.path.join(model_checkpoint_dir, f"checkpoint_{phase_name}")
        safe_save_model(model, tokenizer, save_path)
        
        # Optional validation
        print("\nValidate this phase? (y/n)")
        if input().strip().lower() == 'y':
            validate_phase(model, tokenizer, phase_config)

    # Save final model
    safe_save_model(model, tokenizer, final_model_path)
    print(f"\nAll training phases completed! Model saved to: {final_model_path}")

def dynamic_print(message, end="\r"):
    """Print a message that updates in-place rather than adding new lines"""
    # Clear the current line first
    terminal_width = os.get_terminal_size().columns
    sys.stdout.write("\r" + " " * terminal_width)
    sys.stdout.write("\r" + message)
    sys.stdout.flush()
    if end != "\r":
        sys.stdout.write(end)
        sys.stdout.flush()

if __name__ == "__main__":
    print("\nGPT Model Training Pipeline")
    print("1. Full Training Mode (GPU)")
    print("2. Test Mode (Quick validation)")
    print("3. CPU Training Mode")
    choice = input("\nSelect mode (1/2/3): ").strip()
    
    test_mode = choice == "2"
    force_cpu = choice == "3"
    train(test_mode=test_mode, force_cpu=force_cpu)
    print("\nTraining process completed.")
    print("Model saved and ready for inference.")