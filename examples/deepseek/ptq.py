# Adapted from https://github.com/deepseek-ai/DeepSeek-V3/blob/2f7b80eecebf3d1c84da5a0d465f6639ea175012/inference/model.py
# MIT License

# Copyright (c) 2023 DeepSeek

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import json
import os
import sys
from pathlib import Path
from typing import Literal

import torch
import torch.distributed as dist
import torch.nn.functional as F
from safetensors.torch import load_model
from tqdm import tqdm
from transformers import AutoTokenizer

import modelopt.torch.quantization as mtq
from modelopt.torch.export.model_config import KV_CACHE_FP8
from modelopt.torch.export.quant_utils import get_quant_config
from modelopt.torch.quantization.nn import TensorQuantizer
from modelopt.torch.quantization.utils import (
    is_quantized_column_parallel_linear,
    is_quantized_parallel_linear,
    is_quantized_row_parallel_linear,
)
from modelopt.torch.utils.dataset_utils import get_dataset_dataloader
from modelopt.torch.utils.distributed import ParallelState

sys.path.append(str(Path(__file__).resolve().parent))
import glm4_hf_pp as deekseep_model
from kernel import act_quant, fp8_gemm, weight_dequant


def monkey_patch_deepseek_model():
    # This function is specific to the original DeepSeek model and not compatible
    # with the HuggingFace model structure. The quantization logic would need
    # to be adapted specifically for the HF model's modules.
    pass


def load_deepseek_model(model_path: str, batch_size: int):
    """Loads the deepseek model to memory."""
    # get distributed info
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)

    # run with bf16
    torch.set_default_dtype(torch.bfloat16)

    # get config and build model
    model_args = deekseep_model.ModelArgs(model_path=model_path, max_batch_size=batch_size)
    # The device is handled internally by the Transformer class now
    model = deekseep_model.Transformer(model_args)

    # monkey path the model defition for qunatization
    monkey_patch_deepseek_model()

    # load_model is no longer needed as from_pretrained is used inside the new Transformer class
    print(f"Model loaded via from_pretrained on rank {rank}")
    return model


def ptq(
    model,
    tokenizer,
    quant_cfg: str,
    batch_size: int,
    calib_size: int,
    mla_quant: str | None = None,
):
    """Runs Deepseek model PTQ and returns the quantized model."""

    # quantize the model
    ## create dataset
    device = next(model.parameters()).device
    calib_dataset = get_dataset_dataloader(
        dataset_name="cnn_dailymail",
        tokenizer=tokenizer,
        batch_size=batch_size,
        num_samples=calib_size,
        device=device,
    )

    ## define calib loop
    def calibrate_loop(model):
        for data in tqdm(calib_dataset):
            model(data["input_ids"])

    ## handle DeepSeek model structures
    transformer = model.model if hasattr(model, "model") else model

    # make sure all processes are ready before starting the calibration
    dist.barrier()

    ## quant config
    mtq_cfg = getattr(mtq, quant_cfg)

    # disable head that corresponds to lm_head (for the huggingface checkpoint)
    mtq_cfg["quant_cfg"]["*head*"] = {"enable": False}

    allowed_mla_quant = [None, "per_tensor_fp8"]
    assert mla_quant in allowed_mla_quant, f"mla_quant must be {allowed_mla_quant}"

    if not mla_quant:
        mtq_cfg["quant_cfg"]["*attn*"] = {"enable": False}
    elif mla_quant == "per_tensor_fp8":
        mtq_cfg["quant_cfg"]["*attn*weight_quantizer"] = {"num_bits": (4, 3), "axis": None}
        mtq_cfg["quant_cfg"]["*attn*input_quantizer"] = {"num_bits": (4, 3), "axis": None}

    if args.enable_wo_quant and "FP4" in quant_cfg:
        mtq_cfg["quant_cfg"]["*wo*weight_quantizer"] = mtq_cfg["quant_cfg"]["*input_quantizer"]
        mtq_cfg["quant_cfg"]["*wo*input_quantizer"] = mtq_cfg["quant_cfg"]["*weight_quantizer"]
    ## ptq
    transformer = mtq.quantize(transformer, mtq_cfg, calibrate_loop)
    if int(os.environ["LOCAL_RANK"]) == 0:
        mtq.print_quant_summary(transformer)

    return model


def save_amax_and_quant_config(model, output_path: str, enable_fp8_kvcache: bool):
    """Saves the amax values of the model to the output path."""
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))

    if rank == 0 and not os.path.exists(output_path):
        os.mkdir(output_path)

    dist.barrier()

    # save amax
    def state_dict_filter(state_dict):
        return {key: value for key, value in state_dict.items() if "amax" in key or "quant" in key}

    # save quantization results
    torch.save(
        state_dict_filter(model.state_dict()),
        os.path.join(output_path, f"amax_dict_rank{rank}-mp{world_size}.pt"),
    )

    quant_config = get_quant_config(model.named_modules())

    if enable_fp8_kvcache:
        quant_config["quantization"]["kv_cache_quant_algo"] = KV_CACHE_FP8

    all_quant_configs = [None] * dist.get_world_size()
    dist.all_gather_object(all_quant_configs, quant_config)

    if rank == 0:
        exclude_modules = set()
        quantized_layers = {}

        for quant_config_rank in all_quant_configs:
            assert quant_config_rank is not None
            if "exclude_modules" in quant_config_rank["quantization"]:
                exclude_modules.update(quant_config_rank["quantization"]["exclude_modules"])
            if "quantized_layers" in quant_config_rank["quantization"]:
                quantized_layers.update(quant_config_rank["quantization"]["quantized_layers"])

        if exclude_modules:
            quant_config["quantization"]["exclude_modules"] = sorted(exclude_modules)
            # add the last layer to the exlcude module as the mtp is not loaded in the quantized model
            quant_config["quantization"]["exclude_modules"].append(f"layers.{len(model.layers)}*")
        if quantized_layers:
            quant_config["quantization"]["quantized_layers"] = quantized_layers

        with open(os.path.join(output_path, "hf_quant_config.json"), "w") as f:
            json.dump(quant_config, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--model_path", type=str, required=True, help="path to converted FP8 ckpt")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config_671B.json",
        help="config file for the model.",
    )
    parser.add_argument("--quant_cfg", type=str, required=True, help="target quantization config.")
    parser.add_argument(
        "--output_path", type=str, required=True, help="target quantization config."
    )
    parser.add_argument("--batch_size", type=int, default=1, help="batch size for quantization.")
    parser.add_argument("--calib_size", type=int, default=1, help="samples for calibration.")
    parser.add_argument("--enable_fp8_kvcache", action="store_true", help="enable fp8 kvcache.")
    parser.add_argument("--enable_wo_quant", action="store_true", help="enable MLA wo quant.")
    parser.add_argument("--trust_remote_code", action="store_true", help="trust remote code.")

    args = parser.parse_args()
    model = load_deepseek_model(args.model_path, args.batch_size)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=args.trust_remote_code
    )
    model = ptq(model, tokenizer, args.quant_cfg, args.batch_size, args.calib_size)
    save_amax_and_quant_config(model, args.output_path, args.enable_fp8_kvcache)
