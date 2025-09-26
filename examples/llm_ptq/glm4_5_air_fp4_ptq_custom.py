#!/usr/bin/env python3
"""
GLM-4.5 Air FP4 Post-Training Quantization with Custom Dataset Support
GLM-4.5 Air FP4 quantization script with custom dataset support

Supported dataset formats:
1. Standard datasets: cnn_dailymail, pile, wikipedia, etc.
2. Custom dialogue datasets: formats containing ipt_text, ans_text, concat_ids and other fields
"""

import argparse
import json
import pickle
import os
import time
import torch
import tarfile
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Union
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from huggingface_hub import login

import modelopt.torch.quantization as mtq
from modelopt.torch.utils.dataset_utils import create_forward_loop, get_dataset_dataloader
from modelopt.torch.export import export_hf_checkpoint


class CustomCalibrationDataset(Dataset):
    """Custom format calibration dataset"""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 512, 
                 num_samples: Optional[int] = None, calibration_mode: str = "full"):
        """
        Args:
            data_path: Data file path (supports .pkl, .json, .jsonl)
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
            num_samples: Number of samples to use, None means use all
            calibration_mode: Calibration mode
                - "full": Use complete conversation (concat_ids or ipt_ids+ans_ids)
                - "input_only": Use input part only (ipt_ids)  
                - "output_only": Use output part only (ans_ids)
                - "mixed": Randomly mix different parts
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.calibration_mode = calibration_mode
        
        print(f"🔄 Loading custom dataset: {data_path}")
        print(f"   Calibration mode: {calibration_mode}")
        
        # Handle tar files
        original_data_path = data_path
        temp_dir = None
        
        if data_path.endswith('.tar'):
            print(f"   Extracting tar file: {data_path}")
            temp_dir = tempfile.mkdtemp()
            
            with tarfile.open(data_path, 'r') as tar:
                tar.extractall(temp_dir)
                # Find jsonl file in extracted content
                for file in os.listdir(temp_dir):
                    if file.endswith('.jsonl'):
                        data_path = os.path.join(temp_dir, file)
                        print(f"   Using extracted file: {file}")
                        break
                else:
                    raise ValueError("No .jsonl file found in tar archive")
        
        # Load data
        if data_path.endswith('.pkl'):
            with open(data_path, 'rb') as f:
                self.data = pickle.load(f)
        elif data_path.endswith('.json'):
            with open(data_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
        elif data_path.endswith('.jsonl'):
            self.data = []
            print(f"   Reading JSONL file (this may take a while for large files)...")
            
            lines_read = 0
            json_objects_found = 0
            
            current_json = ""
            brace_count = 0
            in_string = False
            escape_next = False
            
            with open(data_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    lines_read = line_num
                    
                    
                    current_json += line
                    
                    # Track brace balance to find complete JSON objects
                    for char in line:
                        if escape_next:
                            escape_next = False
                            continue
                        
                        if char == '\\':
                            escape_next = True
                            continue
                        
                        if char == '"' and not escape_next:
                            in_string = not in_string
                        
                        if not in_string:
                            if char == '{':
                                brace_count += 1
                            elif char == '}':
                                brace_count -= 1
                                
                                if brace_count == 0 and '{' in current_json:
                                    try:
                                        data = json.loads(current_json.strip())
                                        self.data.append(data)
                                        json_objects_found += 1
                                        current_json = ""
                                        
                                        if num_samples and len(self.data) >= num_samples * 2:
                                            print(f"   Found enough samples ({len(self.data)}), stopping read")
                                            break
                                        
                                    except json.JSONDecodeError:
                                        current_json = ""
                                        brace_count = 0
                                    break
                    
                    if num_samples and len(self.data) >= num_samples * 2:
                        break
            
            print(f"   Read {lines_read} lines, found {json_objects_found} JSON objects")
            
        else:
            raise ValueError(f"Unsupported file format: {data_path}")
        
        # Clean up temp directory
        if temp_dir:
            import shutil
            shutil.rmtree(temp_dir)
            print(f"   Cleaned up temporary directory")
        
        print(f"   Original data count: {len(self.data)}")
        
        # Data quality filtering
        if self._has_finish_reason():
            original_length = len(self.data)
            self.data = [item for item in self.data if item.get('finish_reason') == 'stop']
            print(f"   Filtered data count: {len(self.data)} (filtered out {original_length - len(self.data)} abnormally terminated samples)")
        
        # Limit sample count
        if num_samples is not None and num_samples < len(self.data):
            self.data = self.data[:num_samples]
            print(f"   Using sample count: {len(self.data)}")
    
    def _has_finish_reason(self) -> bool:
        """Check if data contains finish_reason field"""
        return len(self.data) > 0 and 'finish_reason' in self.data[0]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Select different data based on calibration mode
        if self.calibration_mode == "full":
            # Use complete conversation
            if 'concat_ids' in item:
                input_ids = item['concat_ids']
            elif 'ipt_ids' in item and 'ans_ids' in item:
                input_ids = item['ipt_ids'] + item['ans_ids']
            elif 'ipt_text' in item and 'ans_text' in item:
                # If no preprocessed IDs, tokenize on the fly
                full_text = item['ipt_text'] + item['ans_text']
                input_ids = self.tokenizer.encode(full_text, add_special_tokens=False, return_tensors=None)
            else:
                raise KeyError("Missing required fields in data")
                
        elif self.calibration_mode == "input_only":
            # Use input part only
            if 'ipt_ids' in item:
                input_ids = item['ipt_ids']
            elif 'ipt_text' in item:
                input_ids = self.tokenizer.encode(item['ipt_text'], add_special_tokens=False, return_tensors=None)
            else:
                raise KeyError("Missing input field in data")
            
        elif self.calibration_mode == "output_only":
            # Use output part only
            if 'ans_ids' in item:
                input_ids = item['ans_ids']
            elif 'ans_text' in item:
                input_ids = self.tokenizer.encode(item['ans_text'], add_special_tokens=False, return_tensors=None)
            else:
                raise KeyError("Missing output field in data")
                
        elif self.calibration_mode == "mixed":
            # Random selection
            import random
            choice = random.choice(["full", "input_only", "output_only"])
            return self.__getitem_with_mode(idx, choice)
        
        # Truncate to maximum length
        input_ids = input_ids[:self.max_length]
        
        # Convert to tensor and padding
        input_ids = torch.tensor(input_ids, dtype=torch.long)

        return {"input_ids": input_ids}
    
    def __getitem_with_mode(self, idx, mode):
        """Get data using specified mode"""
        original_mode = self.calibration_mode
        self.calibration_mode = mode
        result = self.__getitem__(idx)
        self.calibration_mode = original_mode
        return result


def create_custom_dataloader(data_path: str, tokenizer, batch_size: int = 8, 
                           num_samples: int = 512, max_sample_length: int = 512, 
                           device: str = "cuda", calibration_mode: str = "full"):
    """
    Create DataLoader for custom calibration data
    
    Args:
        data_path: Custom data file path
        tokenizer: HuggingFace tokenizer
        batch_size: Batch size
        num_samples: Number of calibration samples
        max_sample_length: Maximum sequence length
        device: Device
        calibration_mode: Calibration mode
    """
    
    # Create dataset
    dataset = CustomCalibrationDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_length=max_sample_length,
        num_samples=num_samples,
        calibration_mode=calibration_mode
    )
    
    def collate_fn(batch):
        """Batch processing function"""
        # pad to same length
        input_ids = [item["input_ids"] for item in batch]
        max_len = max([len(ids) for ids in input_ids])
        if max_len > max_sample_length:
            max_len = max_sample_length
        for i in range(len(input_ids)):
            if len(input_ids[i]) < max_len:
                pad_length = max_len - len(input_ids[i])
                input_ids[i] = torch.cat([
                    input_ids[i],
                    torch.full((pad_length,), tokenizer.pad_token_id, dtype=torch.long)
                ])
            else:
                input_ids[i] = input_ids[i][:max_len]
        input_ids = torch.stack(input_ids)

        # Check device availability before moving data
        if device == "cuda" and not torch.cuda.is_available():
            print("⚠️  CUDA requested but not available, using CPU")
            actual_device = "cpu"
        else:
            actual_device = device
            
        input_ids = input_ids.to(actual_device)
        return {"input_ids": input_ids}
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    return dataloader


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="GLM-4.5 Air FP4 quantization script (with custom dataset support)")
    
    # Model parameters
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="THUDM/glm-4-9b-chat",
        help="GLM model name or path"
    )
    
    # Dataset parameters
    parser.add_argument(
        "--dataset_type", 
        type=str, 
        default="standard",
        choices=["standard", "custom"],
        help="Dataset type: standard (standard dataset) or custom (custom dataset)"
    )
    parser.add_argument(
        "--dataset_name", 
        type=str, 
        default="cnn_dailymail",
        help="Standard dataset name (used when dataset_type=standard)"
    )
    parser.add_argument(
        "--custom_data_path", 
        type=str, 
        default=None,
        help="Custom data file path (used when dataset_type=custom), supports .tar, .jsonl, .json, .pkl"
    )
    parser.add_argument(
        "--calibration_mode", 
        type=str, 
        default="full",
        choices=["full", "input_only", "output_only", "mixed"],
        help="Calibration mode (only effective for custom datasets)"
    )
    
    # Calibration parameters
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=1,
        help="Batch size"
    )
    parser.add_argument(
        "--calib_samples", 
        type=int, 
        default=1000,
        help="Number of calibration samples"
    )
    parser.add_argument(
        "--max_sample_length", 
        type=int, 
        default=512,
        help="Maximum sample length"
    )
    
    # Quantization parameters
    parser.add_argument(
        "--quant_format", 
        type=str, 
        default="fp4",
        choices=["fp4", "nvfp4", "fp8", "int8", "int4_awq", "mxfp4", "mxfp8"],
        help="Quantization format"
    )
    
    # Output parameters
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./quantized_glm4_5_air_fp4/",
        help="Quantized model output directory"
    )
    
    # Device parameters
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda",
        help="Computing device"
    )
    
    # Test parameters
    parser.add_argument(
        "--test_prompt", 
        type=str, 
        default="Hello, please introduce the development history of artificial intelligence.",
        help="Test prompt text"
    )
    parser.add_argument(
        "--max_new_tokens", 
        type=int, 
        default=50,
        help="Maximum number of new tokens to generate"
    )
    
    # HuggingFace related
    parser.add_argument(
        "--hf_token", 
        type=str, 
        default=None,
        help="HuggingFace access token"
    )
    
    return parser.parse_args()


def check_cuda_availability():
    """Check CUDA availability and return appropriate device"""
    if torch.cuda.is_available():
        print(f"✅ CUDA is available, GPU count: {torch.cuda.device_count()}")
        return "cuda"
    else:
        print("⚠️  CUDA is not available, falling back to CPU")
        print("   Note: CPU quantization will be much slower")
        return "cpu"


def setup_model_and_tokenizer(model_name: str, device: str = "cuda"):
    """Load model and tokenizer"""
    print(f"🔄 Loading model: {model_name}")
    
    # Check actual device availability
    actual_device = check_cuda_availability()
    if device == "cuda" and actual_device == "cpu":
        print(f"⚠️  Requested CUDA but not available, using CPU instead")
        device = actual_device
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        trust_remote_code=True,
        padding_side="left"
    )
    
    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("✅ Set pad_token to eos_token")
    
    # Configure tokenizer to not return token_type_ids for GLM models
    if hasattr(tokenizer, 'model_input_names'):
        if 'token_type_ids' in tokenizer.model_input_names:
            tokenizer.model_input_names.remove('token_type_ids')
            print("✅ Removed token_type_ids from tokenizer output")
    
    # Load model with appropriate settings based on device
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
    else:
        # CPU fallback with lower precision
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # Use float32 for CPU
            device_map="cpu",
            trust_remote_code=True
        )
    
    print(f"✅ Model loaded successfully, device: {next(model.parameters()).device}")
    return model, tokenizer


def get_quantization_config(quant_format: str):
    """Get quantization configuration"""
    config_map = {
        "fp4": mtq.NVFP4_DEFAULT_CFG,  # Use NVFP4 as the default FP4 config
        "nvfp4": mtq.NVFP4_DEFAULT_CFG, 
        "fp8": mtq.FP8_DEFAULT_CFG,
        "int8": mtq.INT8_DEFAULT_CFG,
        "int4_awq": mtq.INT4_AWQ_CFG,
        "mxfp4": mtq.MXFP4_DEFAULT_CFG,
        "mxfp8": mtq.MXFP8_DEFAULT_CFG,
    }
    
    if quant_format not in config_map:
        available_formats = list(config_map.keys())
        raise ValueError(f"Unsupported quantization format: {quant_format}. Available formats: {available_formats}")
    
    return config_map[quant_format]


def create_calibration_dataloader_enhanced(args, tokenizer):
    """Create enhanced calibration data loader"""
    if args.dataset_type == "standard":
        # Use standard dataset
        print(f"🔄 Using standard dataset: {args.dataset_name}")
        dataloader = get_dataset_dataloader(
            dataset_name=args.dataset_name,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            num_samples=args.calib_samples,
            max_sample_length=args.max_sample_length,
            device=args.device,
        )
    elif args.dataset_type == "custom":
        # Use custom dataset
        if not args.custom_data_path:
            raise ValueError("Must specify --custom_data_path when using custom dataset")
        
        print(f"🔄 Using custom dataset: {args.custom_data_path}")
        dataloader = create_custom_dataloader(
            data_path=args.custom_data_path,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            num_samples=args.calib_samples,
            max_sample_length=args.max_sample_length,
            device=args.device,
            calibration_mode=args.calibration_mode
        )
    else:
        raise ValueError(f"Unsupported dataset type: {args.dataset_type}")
    
    print(f"✅ Calibration data loader created, total batches: {len(dataloader)}")
    return dataloader


def quantize_model(model, quant_cfg, forward_loop):
    """Quantize model"""
    print(f"🔄 Starting model quantization...")
    start_time = time.time()
    
    quantized_model = mtq.quantize(model, quant_cfg, forward_loop=forward_loop)
    
    elapsed_time = time.time() - start_time
    print(f"✅ Model quantization completed, time elapsed: {elapsed_time:.2f}s")
    
    return quantized_model


def test_quantized_model(model, tokenizer, test_prompt: str, max_new_tokens: int):
    """Test quantized model"""
    print(f"🔄 Testing quantized model...")
    print(f"   Test prompt: {test_prompt}")
    
    # Only compile model if CUDA is available
    if torch.cuda.is_available():
        try:
            model = torch.compile(model)
            print("✅ Model compiled for faster inference")
        except Exception as e:
            print(f"⚠️  Model compilation failed: {e}, continuing without compilation")
    else:
        print("ℹ️  Skipping model compilation on CPU")
    
    # Prepare input
    inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
    
    # Remove unnecessary keys that GLM model doesn't use
    if 'token_type_ids' in inputs:
        del inputs['token_type_ids']
    
    # Generate output
    with torch.no_grad():
        start_time = time.time()
        outputs = model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.8,
            pad_token_id=tokenizer.eos_token_id
        )
        generation_time = time.time() - start_time
    
    # Decode output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print(f"✅ Generation completed, time elapsed: {generation_time:.2f}s")
    print(f"📝 Generated text:")
    print(f"   {generated_text}")
    print("-" * 50)


def export_quantized_model(model, tokenizer, output_dir: str):
    """Export quantized model"""
    print(f"🔄 Exporting quantized model to: {output_dir}")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Export model
    export_hf_checkpoint(model, export_dir=output_dir)
    
    # Save tokenizer
    tokenizer.save_pretrained(output_dir)
    
    print(f"✅ Model export completed")
    
    # Show exported files
    print(f"📁 Exported files:")
    for file_path in Path(output_dir).iterdir():
        if file_path.is_file():
            file_size = file_path.stat().st_size / (1024 * 1024)  # MB
            print(f"   {file_path.name}: {file_size:.2f} MB")


def main():
    """Main function"""
    args = parse_args()
    
    print("🚀 GLM-4.5 Air FP4 quantization started (with custom dataset support)")
    print("=" * 70)
    
    # Check and adjust device setting
    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA requested but not available, switching to CPU")
        args.device = "cpu"
    
    print(f"🔧 Using device: {args.device}")
    
    # Parameter validation
    if args.dataset_type == "custom" and not args.custom_data_path:
        print("❌ Error: Must specify --custom_data_path when using custom dataset")
        return
    
    # Login to HuggingFace (if needed)
    if args.hf_token:
        login(token=args.hf_token)
        print("✅ HuggingFace login successful")
    
    try:
        # 1. Load model and tokenizer
        model, tokenizer = setup_model_and_tokenizer(args.model_name, args.device)
        
        # 2. Create calibration data loader
        dataloader = create_calibration_dataloader_enhanced(args, tokenizer)
        
        # 3. Create forward loop
        print("🔄 Creating forward loop...")
        # show sample length
        print(f"data")
        # forward_loop = create_forward_loop(dataloader=dataloader)
        print("🔄 Creating forward loop with KV cache disabled...")

        def forward_loop(model):
            """Custom forward loop to disable KV cache during calibration."""
            # model.eval()
            for batch in tqdm(dataloader):
                with torch.no_grad():
                    model(**batch, use_cache=False)
        print("✅ Forward loop created")
        
        # 4. Get quantization configuration
        quant_cfg = get_quantization_config(args.quant_format)
        print(f"✅ Quantization config: {args.quant_format}")
        
        # show data in dataloader
        for batch in dataloader:
            print(batch['input_ids'].shape)
            print(batch)
            
            break  # Show only the first batch            
        
        # show element number in dataloader
        total_elements = 0
        i = 0
        for batch in dataloader:
            total_elements += batch['input_ids'].size(0)
            batch_length = batch['input_ids'].size(1)
            if i % 10 == 0:
                print(f"Batch {i}: Batch size: {batch['input_ids'].size(0)}, Sequence length: {batch_length}")
            i += 1
        print(f"Total elements in dataloader: {total_elements}")
        
        # 5. Quantize model
        quantized_model = quantize_model(model, quant_cfg, forward_loop)
        
        # # # 6. Test quantized model
        # # test_quantized_model(
        # #     model=quantized_model,
        # #     tokenizer=tokenizer,
        # #     test_prompt=args.test_prompt,
        # #     max_new_tokens=args.max_new_tokens
        # # )
        
        # 7. Export quantized model
        export_quantized_model(quantized_model, tokenizer, args.output_dir)
        
        print("=" * 70)
        print("🎉 GLM-4.5 Air FP4 quantization completed!")
        print(f"📁 Quantized model saved to: {args.output_dir}")
        
        # Show configuration used
        print(f"\n📋 Configuration used:")
        print(f"   Model: {args.model_name}")
        print(f"   Device: {args.device}")
        print(f"   Dataset type: {args.dataset_type}")
        if args.dataset_type == "custom":
            print(f"   Custom data path: {args.custom_data_path}")
            print(f"   Calibration mode: {args.calibration_mode}")
        else:
            print(f"   Standard dataset: {args.dataset_name}")
        print(f"   Quantization format: {args.quant_format}")
        print(f"   Calibration samples: {args.calib_samples}")
        
    except Exception as e:
        print(f"❌ Error occurred during quantization: {str(e)}")
        raise


if __name__ == "__main__":
    main()

