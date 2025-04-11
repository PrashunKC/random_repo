import os
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
from typing import Dict, List, Tuple, Optional, Generator, Any
import shutil
import time
from torch.utils.data import DataLoader
from itertools import islice
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
import json

os.environ["HUGGINGFACE_TOKEN"] = "your_token_here"  # Remove actual token before pushing
os.environ["HF_HOME"] = "./.cache/huggingface"  # Set local cache directory

# Add to top of file after imports
def optimize_cuda_settings():
    torch.cuda.empty_cache()  # Clear existing cache
    
    # Basic performance settings 
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    
    # Simple memory strategy
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

# Add this helper function if needed
def resolve_symlink(file_path):
    """Resolve symlink to actual file path"""
    try:
        if os.path.islink(file_path):
            return os.path.realpath(file_path)
        return file_path
    except Exception as e:
        print(f"Warning: Could not resolve symlink {file_path}: {e}")
        return file_path

# Update dataset configs with a better code dataset
DATASET_CONFIGS = {
    "phase1": {
        "name": "json",
        "split": "code",
        "text_column": "text",
        "data_files": {
            "code": r"C:\Users\[User]\.cache\huggingface\hub\datasets--nvidia--Llama-Nemotron-Post-Training-Dataset\snapshots\8e1e47a67ced79723ad0735efc5a45f8bb5aabd6\SFT\code\code_v1.jsonl"
        },
        "max_samples": 50000,
        "trust_remote_code": True,
        "available_splits": ["code"]
    },
    "phase2": {
        "name": "json",
        "split": "combined",
        "text_column": "text",
        "data_files": {
            "code": r"C:\Users\[User]\.cache\huggingface\hub\datasets--nvidia--Llama-Nemotron-Post-Training-Dataset\snapshots\8e1e47a67ced79723ad0735efc5a45f8bb5aabd6\SFT\code\code_v1.1.jsonl",
            "math": r"C:\Users\[User]\.cache\huggingface\hub\datasets--nvidia--Llama-Nemotron-Post-Training-Dataset\snapshots\8e1e47a67ced79723ad0735efc5a45f8bb5aabd6\SFT\math\math_v1.1.jsonl",
            "science": r"C:\Users\[User]\.cache\huggingface\hub\datasets--nvidia--Llama-Nemotron-Post-Training-Dataset\snapshots\8e1e47a67ced79723ad0735efc5a45f8bb5aabd6\SFT\science\science.jsonl"
        },
        "field": None,
        "max_samples": 50000,
        "trust_remote_code": True,
        "available_splits": ["code", "math", "science"]
    },
    "phase3": {
        "name": "json",
        "split": "combined",
        "text_column": "text",
        "data_files": {
            "chat": r"C:\Users\[User]\.cache\huggingface\hub\datasets--nvidia--Llama-Nemotron-Post-Training-Dataset\snapshots\8e1e47a67ced79723ad0735efc5a45f8bb5aabd6\SFT\chat\chat.jsonl",
            "safety": r"C:\Users\[User]\.cache\huggingface\hub\datasets--nvidia--Llama-Nemotron-Post-Training-Dataset\snapshots\8e1e47a67ced79723ad0735efc5a45f8bb5aabd6\SFT\safety\safety.jsonl"
        },
        "max_samples": 25000,
        "trust_remote_code": True,
        "available_splits": ["chat", "safety"]
    }
}

def clear_dataset_cache(config: Dict) -> bool:
    """Clear cached dataset files"""
    try:
        # Use local cache directory
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
    """Download dataset with direct loading"""
    try:
        # Load dataset directly with authentication
        dataset = load_dataset(
            config['name'],
            split=config.get('split', 'train'),
            streaming=False,  # Set to False for direct download
            trust_remote_code=True,
            token=os.getenv("HUGGINGFACE_TOKEN"),
            cache_dir="./.cache/huggingface"  # Use local cache directory
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
    """Check dataset status with detailed info"""
    try:
        # Remove use_auth_token from load_dataset call
        dataset = load_dataset(
            config['name'],
            config.get('config'),
            split=config.get('split', 'train'),
            streaming=True,
            trust_remote_code=True
        )
        next(iter(dataset))
        return True, None
    except Exception as e:
        return False, str(e)

def validate_downloads(config: Dict) -> bool:
    """Validate downloaded datasets with improved error handling"""
    print("\nValidating downloads...")
    
    phases = ['phase1', 'phase3', 'phase2']
    all_valid = True
    
    for phase in phases:
        print(f"\nValidating {phase}...")
        phase_config = config[phase]
        
        try:
            if phase_config['name'] == 'json':
                # Handle local JSON files
                for split_type, file_paths in phase_config['data_files'].items():
                    # Ensure file_paths is a list
                    if isinstance(file_paths, str):
                        file_paths = [file_paths]
                        
                    for file_path in file_paths:
                        file_path = resolve_symlink(file_path)
                        if os.path.exists(file_path):
                            # Try to read first line of JSON file
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
                # Standard dataset validation
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

# Update manage_datasets menu
def manage_datasets():
    """Interactive dataset management system with enhanced error handling"""
    while True:
        print("\n=== Dataset Management System ===")
        print("1. View dataset status")
        print("2. Download specific datasets")
        print("3. Download all datasets")
        print("4. Clear cache")
        print("5. Validate downloads")
        print("6. Force redownload all")  # New option
        print("7. Exit")
        
        choice = input("\nEnter choice (1-7): ").strip()
        
        if choice == "1":
            for phase, config in DATASET_CONFIGS.items():
                status, error = check_dataset_status(config)
                print(f"\n{phase}: {config['name']}")
                print(f"Status: {'Available' if status else 'Not downloaded'}")
                if error:
                    print(f"Error: {error}")
                    
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
                    # First clear cache
                    if clear_dataset_cache(config):
                        print(f"Cleared cache for {phase}")
                    # Then force download
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
assert torch.cuda.is_available(), "This script requires a CUDA-enabled GPU"
device = torch.device("cuda")

# CUDA Optimizations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
torch.backends.cudnn.allow_tf32 = True

# Memory Management
torch.cuda.empty_cache()
torch.cuda.memory.set_per_process_memory_fraction(0.95)

# Memory Allocator Config
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = (
    'expandable_segments:True,'
    'max_split_size_mb:128,'
    'garbage_collection_threshold:0.8'
)

# === Settings ===
model_checkpoint_dir = os.path.join(os.path.dirname(__file__), "saved_model")
dataset_path =r"C:\Users\[User]\.cache\huggingface\hub\datasets--nvidia--Llama-Nemotron-Post-Training-Dataset\snapshots\8e1e47a67ced79723ad0735efc5a45f8bb5aabd6\SFT"

def print_memory_stats():
    """Enhanced memory statistics monitoring"""
    allocated = torch.cuda.memory_allocated(0)
    reserved = torch.cuda.memory_reserved(0)
    max_allocated = torch.cuda.max_memory_allocated(0)
    
    print("\nGPU Memory Statistics:")
    print(f"Currently Allocated: {allocated/1024**2:.2f} MB")
    print(f"Currently Reserved: {reserved/1024**2:.2f} MB")
    print(f"Max Allocated: {max_allocated/1024**2:.2f} MB")
    
    # Get more detailed memory info
    if torch.cuda.is_available():
        free_mem, total_mem = torch.cuda.mem_get_info(0)
        print(f"Total GPU Memory: {total_mem/1024**2:.2f} MB")
        print(f"Free GPU Memory: {free_mem/1024**2:.2f} MB")
        print(f"Used GPU Memory: {(total_mem-free_mem)/1024**2:.2f} MB")

# Move GPTDataset class to module level
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
    
    chunk_size = 8192  # Process in smaller chunks
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
                max_length=512,
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

# Move process_batch to the module level
def process_batch(examples, text_column):
    """Process a batch of examples."""
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

def load_or_download_dataset(
    phase_config: Dict, 
    start_percent: int, 
    chunk_percent: int = 10
) -> Generator[Dataset, None, None]:
    """Load dataset with direct file processing and progress tracking"""
    try:
        if phase_config['name'] == 'json':
            # Process local JSON files directly
            for split_type, file_path in phase_config['data_files'].items():
                print(f"\nProcessing {split_type} split from {file_path}")
                
                # Count total lines first for progress bar
                with open(file_path, 'r', encoding='utf-8') as f:
                    total_lines = sum(1 for _ in f)
                
                current_batch = []
                batch_size = 32
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    # Use tqdm for progress tracking
                    pbar = tqdm(f, total=total_lines, desc=f"Loading {split_type}")
                    for line in pbar:
                        try:
                            item = json.loads(line)
                            text = item.get(phase_config['text_column'], '')
                            if text:
                                processed = process_batch(
                                    {phase_config['text_column']: text}, 
                                    phase_config['text_column']
                                )
                                if processed["input_ids"].numel() > 0:
                                    current_batch.append(processed)
                                
                            if len(current_batch) >= batch_size:
                                try:
                                    yield prepare_batch(current_batch)
                                except Exception as e:
                                    print(f"\nError preparing batch: {e}")
                                current_batch = []
                                torch.cuda.empty_cache()
                                
                        except json.JSONDecodeError:
                            continue
                        except Exception as e:
                            print(f"\nError processing line: {e}")
                            continue
                            
                    # Handle remaining items
                    if current_batch:
                        try:
                            yield prepare_batch(current_batch)
                        except Exception as e:
                            print(f"\nError preparing final batch: {e}")
                            
        else:
            # Handle Hub datasets
            dataset = load_dataset(
                phase_config['name'],
                split=phase_config.get('split', 'train'),
                streaming=True,
                trust_remote_code=True
            )
            
            text_column = phase_config.get('text_column', 'text')
            current_batch = []
            batch_size = 32
            
            for item in tqdm(dataset, desc="Processing dataset"):
                if text_column not in item:
                    continue
                
                processed = process_batch({text_column: item[text_column]}, text_column)
                if processed["input_ids"].numel() > 0:
                    current_batch.append(processed)
                
                if len(current_batch) >= batch_size:
                    try:
                        yield prepare_batch(current_batch)
                    except Exception as e:
                        print(f"\nError preparing batch: {e}")
                    current_batch = []
                    torch.cuda.empty_cache()
                    
    except Exception as e:
        print(f"\nError in dataset loading: {e}")
        return None

def prepare_batch(batch_items):
    """Helper to prepare batches with proper padding"""
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
    """Validate dataset meets minimum requirements"""
    
    # For streaming datasets, accumulate a sample to check size
    sample_size = 0
    samples = []
    
    try:
        # Take first 1000 samples for validation
        for item in islice(dataset, 1000):
            samples.append(item)
            sample_size += 1
            
        if sample_size < 2:  # Changed from 1000 to allow initial testing
            print(f"WARNING: Dataset sample too small ({sample_size} samples)")
            return False

        # Get text field based on dataset structure
        text_field = None
        for field in ['content', 'text', 'problem']:
            if field in samples[0]:
                text_field = field
                break
                
        if text_field is None:
            print(f"WARNING: No recognizable text field found in: {list(samples[0].keys())}")
            return False

        # Check sample texts
        empty_count = 0
        for item in samples:
            if not item[text_field] or not item[text_field].strip():
                empty_count += 1
                
        if empty_count > sample_size * 0.1:  # Allow up to 10% empty
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
            max_length=512,
            return_tensors="pt"
        )
        
        # Move directly to GPU and use half precision
        batch = {k: v.cuda().half() for k, v in batch.items()}
        return batch

def analyze_dataset(dataset_path):
    """Analyze datasets from specified directory with progress tracking"""
    print("\n=== Dataset Analysis ===")
    
    try:
        total_stats = {
            'total_chars': 0,
            'total_words': 0,
            'total_lines': 0,
            'unique_words': set()
        }
        
        # First collect all JSONL files
        jsonl_files = []
        for root, _, files in os.walk(dataset_path):
            for file in files:
                if file.endswith('.jsonl'):
                    jsonl_files.append(os.path.join(root, file))
        
        # Process each file with progress bars
        for file_path in jsonl_files:
            file_name = os.path.basename(file_path)
            print(f"\nAnalyzing {file_name}...")
            
            # First count lines for the progress bar
            with open(file_path, 'r', encoding='utf-8') as f:
                total_lines = sum(1 for _ in tqdm(f, desc="Counting lines", unit=" lines"))
            
            # Then process the file with a progress bar
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in tqdm(f, total=total_lines, desc="Processing", unit=" samples"):
                    try:
                        if not line.strip():
                            continue
                            
                        content = json.loads(line)
                        text = content.get('text', content.get('content', content.get('input', '')))
                        
                        if isinstance(text, list):
                            text = ' '.join(str(t) for t in text)
                        
                        # Update statistics
                        total_stats['total_chars'] += len(text)
                        words = text.split()
                        total_stats['total_words'] += len(words)
                        total_stats['unique_words'].update(w.lower() for w in words)
                        total_stats['total_lines'] += 1
                        
                    except json.JSONDecodeError:
                        print(f"\nWarning: Invalid JSON in {file_name}")
                        continue
                    except Exception as e:
                        print(f"\nError processing line in {file_name}: {e}")
                        continue

        # Print consolidated statistics with formatting
        print("\n=== Dataset Statistics ===")
        print(f"Total characters: {total_stats['total_chars']:,}")
        print(f"Total words: {total_stats['total_words']:,}")
        print(f"Unique words: {len(total_stats['unique_words']):,}")
        print(f"Total samples: {total_stats['total_lines']:,}")
        
        if total_stats['total_lines'] > 0:
            avg_words = total_stats['total_words'] / total_stats['total_lines']
            vocab_richness = (len(total_stats['unique_words']) / total_stats['total_words']) * 100
            print(f"Average words per sample: {avg_words:.2f}")
            print(f"Vocabulary richness: {vocab_richness:.2f}%")

        # Validation checks
        is_valid = True
        if total_stats['total_chars'] < 100000:
            print("\nWARNING: Dataset might be too small for effective training!")
            is_valid = False
        if len(total_stats['unique_words']) < 5000:
            print("\nWARNING: Limited vocabulary might affect model quality!")
            is_valid = False
        if total_stats['total_lines'] < 1000:
            print("\nWARNING: Small number of training samples!")
            is_valid = False
            
        return is_valid
        
    except Exception as e:
        print(f"Error analyzing dataset: {e}")
        return False

def validate_phase(model, tokenizer, phase_config):
    """Validate model performance on a phase"""
    try:
        # Load validation dataset
        val_dataset = load_dataset(
            phase_config['name'],
            config=phase_config.get('config'),
            split=phase_config.get('split', 'train')
        )
        
        # Get the text field based on dataset structure
        if 'problem' in val_dataset.features:
            text_field = 'problem'
        elif 'text' in val_dataset.features:
            text_field = 'text'
        else:
            text_field = list(val_dataset.features.keys())[0]
        
        # Tokenize validation data
        val_encodings = tokenizer(
            val_dataset[text_field][:1000],  # Take first 1000 samples
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )
        
        val_dataset = GPTDataset(val_encodings)
        # Run validation
        training_args = TrainingArguments(
            output_dir="./validation",
            per_device_eval_batch_size=8,
            remove_unused_columns=False
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            eval_dataset=val_dataset
        )
        
        metrics = trainer.evaluate()
        print("\nValidation Results:")
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")
            
        return metrics
        
    except Exception as e:
        print(f"Error during validation: {e}")
        return None

def safe_save_model(model, tokenizer, save_path):
    """Safely save model with retries"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Ensure directory exists
            os.makedirs(save_path, exist_ok=True)
            
            # Clear any open handles
            torch.cuda.empty_cache()
            
            # Save model and tokenizer
            model.save_pretrained(save_path, safe_serialization=True)
            tokenizer.save_pretrained(save_path)
            return True
        except Exception as e:
            print(f"Save attempt {attempt + 1} failed: {e}")
            time.sleep(1)  # Wait before retry
    return False

# Update training loop to handle streamed data
def train():
    print("\nWould you like to manage datasets before training? (y/n)")
    if input().strip().lower() == 'y':
        manage_datasets()
    
    if not analyze_dataset(dataset_path):
        print("Dataset might be insufficient for good results!")
        user_input = input("Continue anyway? (y/n): ")
        if user_input.lower() != 'y':
            return

    # Enhanced CUDA settings for larger dataset
    torch.cuda.empty_cache()
    torch.cuda.memory.set_per_process_memory_fraction(0.95)  # Slightly reduced for stability
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    
    # Progressive learning rates and batch sizes
    base_lr = 5e-5
    base_batch_size = 8
    
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
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        print("Started with fresh model")

    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

    for phase_num, (phase_name, phase_config) in enumerate(DATASET_CONFIGS.items(), 1):
        print(f"\n{'='*20} Starting {phase_name} {'='*20}")
        
        phase_lr = base_lr * (0.85 ** (phase_num-1))
        phase_batch_size = max(2, base_batch_size - (phase_num // 3))
        
        chunk_size = min(10, max(5, int(100000 / phase_config.get('max_samples', float('inf')) * 10)))
        
        # Add overall progress tracking
        with tqdm(total=100, desc=f"{phase_name} Progress") as pbar:
            current_progress = 0
            
            for chunk_start in range(0, 100, chunk_size):
                dataset_stream = load_or_download_dataset(
                    phase_config, 
                    start_percent=chunk_start, 
                    chunk_percent=chunk_size
                )
                
                for batch_dataset in dataset_stream:
                    print("\nBefore batch processing:")
                    print_memory_stats()
                    
                    if batch_dataset is None or len(batch_dataset) == 0:
                        print(f"No more data in {phase_name}")
                        break
                        
                    if not validate_dataset(batch_dataset):
                        print("Skipping invalid batch")
                        continue
                        
                    print(f"\nTraining on {phase_name} chunk {chunk_start}%-{chunk_start+chunk_size}%")
                    print(f"Learning rate: {phase_lr:.2e}, Batch size: {phase_batch_size}")
                    
                    training_args = TrainingArguments(
                        output_dir=os.path.join(model_checkpoint_dir, f"phase_{phase_num}_chunk_{chunk_start}"),
                        num_train_epochs=1,
                        per_device_train_batch_size=8,  # Reduced batch size
                        gradient_accumulation_steps=8,   # Increased accumulation
                        save_strategy="steps",
                        save_steps=2000,                 # Save less frequently
                        fp16=True,
                        learning_rate=phase_lr,
                        logging_steps=200,
                        warmup_ratio=0.1,
                        weight_decay=0.01,
                        gradient_checkpointing=True,
                        # Memory optimizations
                        dataloader_num_workers=8,        # Workers go brrrrttttttt!!!!!
                        dataloader_pin_memory=True,
                        max_grad_norm=1.0,
                        report_to=["tensorboard"],
                        # Cache settings
                        load_best_model_at_end=False,    # Disable additional model loading
                        save_total_limit=2               # Keep only 2 checkpoints
                    )
                    
                    trainer = Trainer(
                        model=model,
                        args=training_args,
                        train_dataset=batch_dataset,
                        data_collator=CustomDataCollator(tokenizer)
                    )
                    
                    trainer.train()
                    
                    print("\nAfter batch processing:")
                    print_memory_stats()
                    
                    # Clear cache if memory usage is high
                    if torch.cuda.memory_allocated() > 0.8 * torch.cuda.get_device_properties(0).total_memory:
                        print("Clearing CUDA cache due to high memory usage")
                        torch.cuda.empty_cache()
                    
                    # Save checkpoint after each batch
                    checkpoint_path = os.path.join(
                        model_checkpoint_dir, 
                        f"checkpoint_phase{phase_num}_chunk{chunk_start}"
                    )
                    if not safe_save_model(model, tokenizer, checkpoint_path):
                        print("Warning: Failed to save checkpoint")
                    
                    del trainer
                    del batch_dataset
                    torch.cuda.empty_cache()
                    print_memory_stats()
                
                # Update progress
                progress_increment = chunk_size
                current_progress += progress_increment
                pbar.update(progress_increment)
                
                # Show GPU stats periodically
                if chunk_start % 20 == 0:
                    print_memory_stats()
            
        # Optional phase validation
        print(f"\nCompleted {phase_name}")
        print("Would you like to validate the model on this phase? (y/n)")
        if input().strip().lower() == 'y':
            validate_phase(model, tokenizer, phase_config)

    # Save final model
    safe_save_model(model, tokenizer, final_model_path)
    print(f"\nAll training phases completed! Model saved to: {final_model_path}")

if __name__ == "__main__":
    train()