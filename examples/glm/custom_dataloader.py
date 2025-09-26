#!/usr/bin/env python3
"""
GLM-4.5 Air FP4 Post-Training Quantization with Custom Dataset Support
GLM-4.5 Air FP4 quantization script with custom dataset support

Supported dataset formats:
1. Standard datasets: cnn_dailymail, pile, wikipedia, etc.
2. Custom dialogue datasets: formats containing ipt_text, ans_text, concat_ids and other fields
"""

import json
import os
import pickle
import tarfile
import tempfile

import torch
from torch.utils.data import DataLoader, Dataset


class CustomCalibrationDataset(Dataset):
    """Custom format calibration dataset"""

    def __init__(
        self,
        data_path: str,
        tokenizer=None,
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

        print(f"🔄 Loading custom dataset: {data_path}")
        print(f"   Calibration mode: {calibration_mode}")

        # Handle tar files
        original_data_path = data_path
        temp_dir = None

        if data_path.endswith(".tar"):
            print(f"   Extracting tar file: {data_path}")
            temp_dir = tempfile.mkdtemp(dir="/mnt/data")

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

        # Load data
        if data_path.endswith(".pkl"):
            with open(data_path, "rb") as f:
                self.data = pickle.load(f)
        elif data_path.endswith(".json"):
            with open(data_path, encoding="utf-8") as f:
                self.data = json.load(f)
        elif data_path.endswith(".jsonl"):
            self.data = []
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

                    # Track brace balance to find complete JSON objects
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

            print(f"   Read {lines_read} lines, found {json_objects_found} JSON objects")

        else:
            raise ValueError(f"Unsupported file format: {data_path}")

        # Clean up temp directory
        if temp_dir:
            import shutil

            shutil.rmtree(temp_dir)
            print("   Cleaned up temporary directory")

        print(f"   Original data count: {len(self.data)}")

        # Data quality filtering
        if self._has_finish_reason():
            original_length = len(self.data)
            self.data = [item for item in self.data if item.get("finish_reason") == "stop"]
            print(
                f"   Filtered data count: {len(self.data)} (filtered out {original_length - len(self.data)} abnormally terminated samples)"
            )

        # Limit sample count
        if num_samples is not None and num_samples < len(self.data):
            self.data = self.data[:num_samples]
            print(f"   Using sample count: {len(self.data)}")

    def _has_finish_reason(self) -> bool:
        """Check if data contains finish_reason field"""
        return len(self.data) > 0 and "finish_reason" in self.data[0]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Select different data based on calibration mode
        if self.calibration_mode == "full":
            # Use complete conversation
            if "concat_ids" in item:
                input_ids = item["concat_ids"]
            elif "ipt_ids" in item and "ans_ids" in item:
                input_ids = item["ipt_ids"] + item["ans_ids"]
            elif "ipt_text" in item and "ans_text" in item:
                # If no preprocessed IDs, tokenize on the fly
                full_text = item["ipt_text"] + item["ans_text"]
                input_ids = self.tokenizer.encode(
                    full_text, add_special_tokens=False, return_tensors=None
                )
            else:
                raise KeyError("Missing required fields in data")

        elif self.calibration_mode == "input_only":
            # Use input part only
            if "ipt_ids" in item:
                input_ids = item["ipt_ids"]
            elif "ipt_text" in item:
                input_ids = self.tokenizer.encode(
                    item["ipt_text"], add_special_tokens=False, return_tensors=None
                )
            else:
                raise KeyError("Missing input field in data")

        elif self.calibration_mode == "output_only":
            # Use output part only
            if "ans_ids" in item:
                input_ids = item["ans_ids"]
            elif "ans_text" in item:
                input_ids = self.tokenizer.encode(
                    item["ans_text"], add_special_tokens=False, return_tensors=None
                )
            else:
                raise KeyError("Missing output field in data")

        elif self.calibration_mode == "mixed":
            # Random selection
            import random

            choice = random.choice(["full", "input_only", "output_only"])
            return self.__getitem_with_mode(idx, choice)

        # Truncate to maximum length
        input_ids = input_ids[: self.max_length]

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


def create_custom_dataloader(
    data_path: str,
    tokenizer,
    batch_size: int = 8,
    num_samples: int = 512,
    max_sample_length: int = 512,
    device: str = "cuda",
    calibration_mode: str = "full",
):
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
        calibration_mode=calibration_mode,
    )

    def collate_fn(batch):
        """Batch processing function"""
        # pad to same length
        input_ids = [item["input_ids"] for item in batch]
        max_len = max([len(ids) for ids in input_ids])
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
        dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0
    )

    return dataloader
