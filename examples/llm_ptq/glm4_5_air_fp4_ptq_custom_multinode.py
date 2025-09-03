"""
GLM-4.5 Air FP4 quantization script with custom dataset support, adapted for multi-node execution.

Supported dataset formats:
1. Standard datasets: cnn_dailymail, pile, wikipedia, etc.
2. Custom dialogue datasets: formats containing ipt_text, ans_text, concat_ids and other fields
"""

import argparse
import json
import os
import pickle
import tarfile
import tempfile
import time
from pathlib import Path

import torch
import torch.distributed as dist
from huggingface_hub import login
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import modelopt.torch.quantization as mtq
from modelopt.torch.export import export_hf_checkpoint
from modelopt.torch.utils.dataset_utils import get_dataset_dataloader


class CustomCalibrationDataset(Dataset):
    """Custom format calibration dataset"""

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 512,
        num_samples: int | None = None,
        calibration_mode: str = "full",
    ):
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
        rank = dist.get_rank() if dist.is_initialized() else 0

        if rank == 0:
            print(f"🔄 Loading custom dataset: {data_path}")
            print(f"   Calibration mode: {calibration_mode}")

        # Handle tar files
        temp_dir = None
        if data_path.endswith(".tar"):
            if rank == 0:
                print(f"   Extracting tar file: {data_path}")
                temp_dir = tempfile.mkdtemp()
                with tarfile.open(data_path, "r") as tar:
                    tar.extractall(temp_dir)
                    # Find jsonl file in extracted content
                    for file in os.listdir(temp_dir):
                        if file.endswith(".jsonl"):
                            data_path = os.path.join(temp_dir, file)
                            print(f"   Using extracted file: {file}")
                            break
                    else:
                        raise ValueError("No .jsonl file found in tar archive")
            if dist.is_initialized():
                dist.barrier()  # Wait for rank 0 to extract
                # Broadcast the new data_path to other ranks
                data_path_list = [data_path]
                dist.broadcast_object_list(data_path_list, src=0)
                data_path = data_path_list[0]

        # Load data
        if data_path.endswith(".pkl"):
            with open(data_path, "rb") as f:
                self.data = pickle.load(f)
        elif data_path.endswith(".json"):
            with open(data_path, encoding="utf-8") as f:
                self.data = json.load(f)
        elif data_path.endswith(".jsonl"):
            self.data = []
            if rank == 0:
                print("   Reading JSONL file (this may take a while for large files)...")

            lines_read = 0
            json_objects_found = 0

            current_json = ""
            brace_count = 0
            in_string = False
            escape_next = False

            with open(data_path, encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    lines_read = line_num
                    current_json += line
                    for char in line:
                        if escape_next:
                            escape_next = False
                            continue
                        if char == "\\":
                            escape_next = True
                            continue
                        if char == '"' and not escape_next:
                            in_string = not in_string
                        if not in_string:
                            if char == "{":
                                brace_count += 1
                            elif char == "}":
                                brace_count -= 1
                                if brace_count == 0 and "{" in current_json:
                                    try:
                                        data = json.loads(current_json.strip())
                                        self.data.append(data)
                                        json_objects_found += 1
                                        current_json = ""
                                        if num_samples and len(self.data) >= num_samples * 2:
                                            if rank == 0:
                                                print(
                                                    f"   Found enough samples ({len(self.data)}), stopping read"
                                                )
                                            break
                                    except json.JSONDecodeError:
                                        current_json = ""
                                        brace_count = 0
                                    break
                    if num_samples and len(self.data) >= num_samples * 2:
                        break
            if rank == 0:
                print(f"   Read {lines_read} lines, found {json_objects_found} JSON objects")
        else:
            raise ValueError(f"Unsupported file format: {data_path}")

        # Clean up temp directory on rank 0
        if rank == 0 and temp_dir:
            import shutil

            shutil.rmtree(temp_dir)
            print("   Cleaned up temporary directory")

        if rank == 0:
            print(f"   Original data count: {len(self.data)}")

        # Data quality filtering
        if self._has_finish_reason():
            original_length = len(self.data)
            self.data = [item for item in self.data if item.get("finish_reason") == "stop"]
            if rank == 0:
                print(
                    f"   Filtered data count: {len(self.data)} (filtered out "
                    f"{original_length - len(self.data)} abnormally terminated samples)"
                )

        # Limit sample count
        if num_samples is not None and num_samples < len(self.data):
            self.data = self.data[:num_samples]
            if rank == 0:
                print(f"   Using sample count: {len(self.data)}")

    def _has_finish_reason(self) -> bool:
        """Check if data contains finish_reason field"""
        return len(self.data) > 0 and "finish_reason" in self.data[0]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        if self.calibration_mode == "full":
            if "concat_ids" in item:
                input_ids = item["concat_ids"]
            elif "ipt_ids" in item and "ans_ids" in item:
                input_ids = item["ipt_ids"] + item["ans_ids"]
            elif "ipt_text" in item and "ans_text" in item:
                full_text = item["ipt_text"] + item["ans_text"]
                input_ids = self.tokenizer.encode(
                    full_text, add_special_tokens=False, return_tensors=None
                )
            else:
                raise KeyError("Missing required fields in data")
        elif self.calibration_mode == "input_only":
            if "ipt_ids" in item:
                input_ids = item["ipt_ids"]
            elif "ipt_text" in item:
                input_ids = self.tokenizer.encode(
                    item["ipt_text"], add_special_tokens=False, return_tensors=None
                )
            else:
                raise KeyError("Missing input field in data")
        elif self.calibration_mode == "output_only":
            if "ans_ids" in item:
                input_ids = item["ans_ids"]
            elif "ans_text" in item:
                input_ids = self.tokenizer.encode(
                    item["ans_text"], add_special_tokens=False, return_tensors=None
                )
            else:
                raise KeyError("Missing output field in data")
        elif self.calibration_mode == "mixed":
            import random

            choice = random.choice(["full", "input_only", "output_only"])
            return self.__getitem_with_mode(idx, choice)

        input_ids = input_ids[: self.max_length]
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        return {"input_ids": input_ids}

    def __getitem_with_mode(self, idx, mode):
        original_mode = self.calibration_mode
        self.calibration_mode = mode
        result = self.__getitem__(idx)
        self.calibration_mode = original_mode
        return result


def create_custom_dataloader(
    data_path: str,
    tokenizer,
    batch_size: int = 8,
    num_samples: int = 512,
    max_sample_length: int = 512,
    device: str = "cuda",
    calibration_mode: str = "full",
):
    dataset = CustomCalibrationDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_length=max_sample_length,
        num_samples=num_samples,
        calibration_mode=calibration_mode,
    )

    def collate_fn(batch):
        input_ids = [item["input_ids"] for item in batch]
        max_len = max(len(ids) for ids in input_ids)
        max_len = min(max_len, max_sample_length)
        for i in range(len(input_ids)):
            if len(input_ids[i]) < max_len:
                pad_length = max_len - len(input_ids[i])
                input_ids[i] = torch.cat(
                    [
                        input_ids[i],
                        torch.full((pad_length,), tokenizer.pad_token_id, dtype=torch.long),
                    ]
                )
            else:
                input_ids[i] = input_ids[i][:max_len]
        input_ids = torch.stack(input_ids)

        actual_device = device
        if device == "cuda" and not torch.cuda.is_available():
            if dist.get_rank() == 0:
                print("⚠️  CUDA requested but not available, using CPU")
            actual_device = "cpu"

        input_ids = input_ids.to(actual_device)
        return {"input_ids": input_ids}

    sampler = DistributedSampler(dataset, shuffle=True) if dist.is_initialized() else None
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=sampler is None,
        collate_fn=collate_fn,
        num_workers=0,
        sampler=sampler,
    )
    return dataloader


def parse_args():
    parser = argparse.ArgumentParser(
        description="GLM-4.5 Air FP4 quantization script (with multi-node support)"
    )
    parser.add_argument(
        "--model_name", type=str, default="THUDM/glm-4-9b-chat", help="GLM model name or path"
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="standard",
        choices=["standard", "custom"],
        help="Dataset type",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="cnn_dailymail",
        help="Standard dataset name",
    )
    parser.add_argument(
        "--custom_data_path",
        type=str,
        default=None,
        help="Custom data file path",
    )
    parser.add_argument(
        "--calibration_mode",
        type=str,
        default="full",
        choices=["full", "input_only", "output_only", "mixed"],
        help="Calibration mode for custom datasets",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument(
        "--calib_samples", type=int, default=1000, help="Number of calibration samples"
    )
    parser.add_argument("--max_sample_length", type=int, default=512, help="Maximum sample length")
    parser.add_argument(
        "--quant_format",
        type=str,
        default="fp4",
        choices=["fp4", "nvfp4", "fp8", "int8", "int4_awq", "mxfp4", "mxfp8"],
        help="Quantization format",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./quantized_glm4_5_air_fp4/",
        help="Output directory",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Computing device")
    parser.add_argument(
        "--test_prompt",
        type=str,
        default="Hello, please introduce the development history of artificial intelligence.",
        help="Test prompt text",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=50, help="Maximum new tokens for generation"
    )
    parser.add_argument("--hf_token", type=str, default=None, help="HuggingFace access token")
    return parser.parse_args()


def check_cuda_availability():
    rank = dist.get_rank() if dist.is_initialized() else 0
    if torch.cuda.is_available():
        if rank == 0:
            print(f"✅ CUDA is available, GPU count: {torch.cuda.device_count()}")
        return "cuda"
    else:
        if rank == 0:
            print("⚠️  CUDA is not available, falling back to CPU")
            print("   Note: CPU quantization will be much slower")
        return "cpu"


def setup_model_and_tokenizer(model_name: str, device: str = "cuda"):
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank == 0:
        print(f"🔄 Loading model: {model_name}")

    actual_device = check_cuda_availability()
    if device == "cuda" and actual_device == "cpu":
        if rank == 0:
            print("⚠️  Requested CUDA but not available, using CPU instead")
        device = actual_device

    if rank == 0:
        AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, trust_remote_code=True
        )
    if dist.is_initialized():
        dist.barrier()

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        if rank == 0:
            print("✅ Set pad_token to eos_token")
    if hasattr(tokenizer, "model_input_names") and "token_type_ids" in tokenizer.model_input_names:
        tokenizer.model_input_names.remove("token_type_ids")
        if rank == 0:
            print("✅ Removed token_type_ids from tokenizer output")

    if device == "cuda":
        with torch.device(f"cuda:{torch.cuda.current_device()}"):
            model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.bfloat16, trust_remote_code=True
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True,
        )

    if rank == 0:
        print(f"✅ Model loaded successfully, device: {next(model.parameters()).device}")
    return model, tokenizer


def get_quantization_config(quant_format: str):
    config_map = {
        "fp4": mtq.NVFP4_DEFAULT_CFG,
        "nvfp4": mtq.NVFP4_DEFAULT_CFG,
        "fp8": mtq.FP8_DEFAULT_CFG,
        "int8": mtq.INT8_DEFAULT_CFG,
        "int4_awq": mtq.INT4_AWQ_CFG,
        "mxfp4": mtq.MXFP4_DEFAULT_CFG,
        "mxfp8": mtq.MXFP8_DEFAULT_CFG,
    }
    if quant_format not in config_map:
        raise ValueError(
            f"Unsupported quantization format: {quant_format}. "
            f"Available formats: {list(config_map.keys())}"
        )
    return config_map[quant_format]


def create_calibration_dataloader_enhanced(args, tokenizer):
    rank = dist.get_rank() if dist.is_initialized() else 0
    if args.dataset_type == "standard":
        if rank == 0:
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
        if not args.custom_data_path:
            raise ValueError("Must specify --custom_data_path when using custom dataset")
        if rank == 0:
            print(f"🔄 Using custom dataset: {args.custom_data_path}")
        dataloader = create_custom_dataloader(
            data_path=args.custom_data_path,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            num_samples=args.calib_samples,
            max_sample_length=args.max_sample_length,
            device=args.device,
            calibration_mode=args.calibration_mode,
        )
    else:
        raise ValueError(f"Unsupported dataset type: {args.dataset_type}")

    if rank == 0:
        print(f"✅ Calibration data loader created, total batches: {len(dataloader)}")
    return dataloader


def quantize_model(model, quant_cfg, forward_loop):
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank == 0:
        print("🔄 Starting model quantization...")
    start_time = time.time()

    if dist.is_initialized():
        dist.barrier()

    quantized_model = mtq.quantize(model, quant_cfg, forward_loop=forward_loop)

    elapsed_time = time.time() - start_time
    if rank == 0:
        print(f"✅ Model quantization completed, time elapsed: {elapsed_time:.2f}s")
    return quantized_model


def test_quantized_model(model, tokenizer, test_prompt: str, max_new_tokens: int):
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank != 0:
        return

    print("🔄 Testing quantized model...")
    print(f"   Test prompt: {test_prompt}")

    if torch.cuda.is_available():
        try:
            model = torch.compile(model)
            print("✅ Model compiled for faster inference")
        except Exception as e:
            print(f"⚠️  Model compilation failed: {e}, continuing without compilation")
    else:
        print("INFO: Skipping model compilation on CPU")

    inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
    if "token_type_ids" in inputs:
        del inputs["token_type_ids"]

    with torch.no_grad():
        start_time = time.time()
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.8,
            pad_token_id=tokenizer.eos_token_id,
        )
        generation_time = time.time() - start_time

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"✅ Generation completed, time elapsed: {generation_time:.2f}s")
    print("📝 Generated text:")
    print(f"   {generated_text}")
    print("-" * 50)


def export_quantized_model(model, tokenizer, output_dir: str):
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    if rank == 0:
        print(f"🔄 Exporting quantized model to: {output_dir}")
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    if dist.is_initialized():
        dist.barrier()

    export_hf_checkpoint(model, export_dir=output_dir, rank=rank, world_size=world_size)

    if rank == 0:
        tokenizer.save_pretrained(output_dir)
        print("✅ Model export completed")
        print("📁 Exported files:")
        for file_path in Path(output_dir).iterdir():
            if file_path.is_file():
                file_size = file_path.stat().st_size / (1024 * 1024)
                print(f"   {file_path.name}: {file_size:.2f} MB")


def main():
    args = parse_args()

    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))

    if world_size > 1:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
        if rank == 0:
            print(f"✅ Distributed process group initialized. World size: {world_size}")

    if rank == 0:
        print("🚀 GLM-4.5 Air FP4 quantization started (with multi-node support)")
        print("=" * 70)

    if args.device == "cuda" and not torch.cuda.is_available():
        if rank == 0:
            print("⚠️  CUDA requested but not available, switching to CPU")
        args.device = "cpu"

    if rank == 0:
        print(f"🔧 Using device: {args.device}")

    if args.dataset_type == "custom" and not args.custom_data_path:
        if rank == 0:
            print("❌ Error: Must specify --custom_data_path when using custom dataset")
        return

    if args.hf_token and rank == 0:
        login(token=args.hf_token)
        print("✅ HuggingFace login successful")

    try:
        model, tokenizer = setup_model_and_tokenizer(args.model_name, args.device)
        dataloader = create_calibration_dataloader_enhanced(args, tokenizer)

        if rank == 0:
            print("🔄 Creating forward loop with KV cache disabled...")

        def forward_loop(model):
            model.eval()
            for batch in tqdm(dataloader, disable=(rank != 0)):
                with torch.no_grad():
                    model(**batch, use_cache=False)

        if rank == 0:
            print("✅ Forward loop created")

        quant_cfg = get_quantization_config(args.quant_format)
        if rank == 0:
            print(f"✅ Quantization config: {args.quant_format}")

        quantized_model = quantize_model(model, quant_cfg, forward_loop)

        test_quantized_model(
            model=quantized_model,
            tokenizer=tokenizer,
            test_prompt=args.test_prompt,
            max_new_tokens=args.max_new_tokens,
        )

        export_quantized_model(quantized_model, tokenizer, args.output_dir)

        if rank == 0:
            print("=" * 70)
            print("🎉 GLM-4.5 Air FP4 quantization completed!")
            print(f"📁 Quantized model saved to: {args.output_dir}")
            print("\n📋 Configuration used:")
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
        print(f"❌ Error occurred on rank {rank}: {e!s}")
        raise


if __name__ == "__main__":
    main()

