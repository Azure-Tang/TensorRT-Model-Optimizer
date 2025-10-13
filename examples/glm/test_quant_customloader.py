import argparse
import json
import os

import glm4_hf_pp as deekseep_model
import torch
import torch.distributed as dist
from custom_dataloader import create_custom_dataloader
from safetensors.torch import save_file
from tqdm import tqdm
from transformers import AutoTokenizer

import modelopt.torch.quantization as mtq
from modelopt.torch.export.unified_export_hf import _export_hf_checkpoint
from modelopt.torch.utils.dataset_utils import get_dataset_dataloader

rank_env = os.getenv("RANK", None)
DEBUG_PP = os.getenv("DEBUG_PP", "0") == "1"


def _dlog(msg: str):
    if DEBUG_PP:
        r = os.getenv("RANK", "?")
        print(f"[PP][rank {r}] {msg}", flush=True)


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

    print(f"Model loaded via from_pretrained on rank {rank}")
    return model


def main():
    model_name = "zai-org/GLM-4.5"

    parser = argparse.ArgumentParser("HF GLM4-MoE wrapper demo: load and generate")
    parser.add_argument(
        "--model_path",
        type=str,
        default="zai-org/GLM-4.5",
        help="HuggingFace model path or repo id",
    )
    parser.add_argument("--prompt", type=str, default="你好, GLM-4.5! 请用一句话介绍你自己.")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument(
        "--do_sample", action="store_true", help="use multinomial sampling instead of greedy"
    )
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature for sampling")
    # save path
    parser.add_argument(
        "--export_dir",
        type=str,
        default="/data/numa0/downloaded_models/glm4-5-nvfp4-calib",
        help="Directory to save the exported checkpoint and configs",
    )
    # Dataset parameters
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="standard",
        choices=["standard", "custom"],
        help="Dataset type: standard (standard dataset) or custom (custom dataset)",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="cnn_dailymail",
        help="Standard dataset name (used when dataset_type=standard)",
    )
    parser.add_argument(
        "--custom_data_path",
        type=str,
        default="/data/numa0/zbz2/primary_synced/data_cal/glm4p5-calibration.jsonl.tar",
        help="Custom data file path (used when dataset_type=custom)",
    )
    parser.add_argument(
        "--calibration_mode",
        type=str,
        default="full",
        choices=["full", "input_only", "output_only", "mixed"],
        help="Calibration mode (only effective for custom datasets)",
    )

    # Calibration parameters
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument(
        "--calib_samples", type=int, default=10000000, help="Number of calibration samples"
    )
    parser.add_argument("--max_sample_length", type=int, default=512, help="Maximum sample length")

    args = parser.parse_args()

    rank = int(os.getenv("RANK", "0"))

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = load_deepseek_model(args.model_path, batch_size=1)

    config = mtq.NVFP4_DEFAULT_CFG
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    batch_size = args.batch_size
    num_samples = args.calib_samples

    if rank == 0:
        if args.dataset_type == "custom":
            _dlog(f"Using custom dataset from {args.custom_data_path}")
            calib_dataset = create_custom_dataloader(
                data_path=args.custom_data_path,
                tokenizer=tokenizer,
                batch_size=batch_size,
                num_samples=num_samples,
                max_sample_length=args.max_sample_length,
                calibration_mode=args.calibration_mode,
                device=model.device,
            )
        else:
            _dlog(f"Using standard dataset {args.dataset_name}")
            calib_dataset = get_dataset_dataloader(
                dataset_name="cnn_dailymail",
                tokenizer=tokenizer,
                batch_size=batch_size,
                num_samples=num_samples,
            )
        _dlog(f"Calib dataset created on rank {rank}, length: {len(calib_dataset)}")
    else:
        calib_dataset = 0  # 其他 rank 不需要数据集
        import time
        _dlog("sleep to wait rank 0 to finish dataset creation")

        # sleep 20min to wait for rank 0 to finish dataset creation
        time.sleep(3 * 60)

    _dlog("Ready to start calibration")
    dist.barrier()  # 确保所有进程在此同步

    def calibrate_loop(model):
        rank = int(os.getenv("RANK", "0"))
        world_size = int(os.getenv("WORLD_SIZE", "1"))
        dev = next(model.parameters()).device
        _dlog("=====enter calibrate_loop====")

        if world_size == 1:
            _dlog("Single GPU mode")
            # Single-GPU case: iterate normally
            for data in tqdm(calib_dataset):
                model(data["input_ids"])
        # Pipeline-parallel case
        elif rank == 0:
            _dlog("Multi-GPU mode, rank 0")
            # Rank 0 drives calibration, iterates real data, and broadcasts shape
            for data in tqdm(calib_dataset):
                tokens = data["input_ids"]
                shape = torch.tensor(tokens.shape, device=dev, dtype=torch.int64)
                _dlog(f"Broadcasting shape {shape.tolist()}")
                dist.broadcast(shape, src=0)
                _dlog(f"Running forward on real data {tokens.shape}")
                model(tokens.to(dev))
            # Send stop signal
            stop_signal = torch.tensor([-1, -1], device=dev, dtype=torch.int64)
            dist.broadcast(stop_signal, src=0)
        else:
            _dlog(f"Multi-GPU mode, rank {rank}")
            # Other ranks receive shape, create dummy tensor, and call forward
            while True:
                shape = torch.empty(2, device=dev, dtype=torch.int64)
                _dlog("Receiving shape")
                dist.broadcast(shape, src=0)
                _dlog(f"Received shape {shape.tolist()}")
                bsz, seqlen = shape[0].item(), shape[1].item()
                if bsz == -1:
                    # Received stop signal
                    break
                # Create dummy input just to trigger forward pass
                dummy_tokens = torch.zeros((bsz, seqlen), dtype=torch.long, device=dev)
                model(dummy_tokens)

    # def forward_loop(model):
    #     for data in tqdm(calib_dataset):
    #         print(
    #             f"Calibrating on rank {rank} with input shape {data['input_ids'].shape}, device {data['input_ids'].device}"
    #         )
    #         model(data["input_ids"].to(model.device))

    # PTQ with in-place replacement to quantized modules
    print(f"=====Start quantization on rank {rank}")
    model = mtq.quantize(model, config, calibrate_loop)
    print(f"=====Quantization done on rank {rank}")
    dist.barrier()
    print(f"=====Quantization barrier on rank {rank}")

    # mtq.print_quant_summary(model)

    export_dir = args.export_dir + f"{num_samples}-samples"

    # 1. 每个 rank 独立计算自己那部分模型的导出状态字典和量化配置
    #    _export_hf_checkpoint 内部的 dummy forward pass 需要所有 rank 参与

    # 1.1. 先修复缺失的 _amax 值 (参考 quantize_to_nvfp4.py 的处理方式)
    print(f"[Rank {rank}] Fixing missing _amax values...")

    def fix_missing_amax(model):
        """修复缺失的 _amax 值，使用合理的默认值"""
        from modelopt.torch.quantization.nn import TensorQuantizer

        # 收集所有有效的 amax 值
        valid_amax_values = []
        for name, module in model.named_modules():
            if (
                isinstance(module, TensorQuantizer)
                and hasattr(module, "_amax")
                and module._amax is not None
            ):
                valid_amax_values.append(module._amax.clone())

        # 如果没有任何有效的 amax 值, 使用默认值
        if not valid_amax_values:
            default_amax = torch.tensor(1.0, dtype=torch.float32)
        else:
            # 使用所有有效 amax 值的最大值作为默认值
            default_amax = torch.max(torch.stack(valid_amax_values))

        # 修复缺失的 _amax
        fixed_count = 0
        for name, module in model.named_modules():
            if isinstance(module, TensorQuantizer):
                if not hasattr(module, "_amax") or module._amax is None:
                    module._amax = default_amax.clone().to(next(model.parameters()).device)
                    fixed_count += 1

        if fixed_count > 0:
            print(
                f"[Rank {rank}] Fixed {fixed_count} missing _amax values "
                f"with default value {default_amax.item():.6f}"
            )
        return fixed_count

    fix_missing_amax(model)

    print(f"[Rank {rank}] Starting local model export processing...")
    post_state_dict, hf_quant_config = _export_hf_checkpoint(model, model.config.torch_dtype)
    print(f"[Rank {rank}] Finished local model export processing.")

    # 2. 同步所有进程, 确保大家都完成了上面的计算
    dist.barrier()
    print(f"[Rank {rank}] Barrier after local export processing.")

    # 3. 每个 rank 将自己的 state_dict 保存到独立的分片文件中
    world_size = dist.get_world_size()
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    # 格式化文件名, 例如 model-00001-of-00002.safetensors
    shard_file_name = f"model-{rank + 1:05d}-of-{world_size:05d}.safetensors"
    shard_file_path = os.path.join(export_dir, shard_file_name)

    # 每台机器的 local rank 0 负责创建目录
    if local_rank == 0:
        os.makedirs(export_dir, exist_ok=True)
    dist.barrier()  # 确保目录已创建

    print(f"[Rank {rank}] Saving its shard to {shard_file_path}...")
    # save_file 会处理目录创建, 但为了分布式安全, 最好由 rank 0 创建
    save_file(post_state_dict, shard_file_path, metadata={"format": "pt"})
    print(f"[Rank {rank}] Shard saved.")

    # 4. 收集所有 rank 的量化配置和权重 key 列表到 rank 0
    all_quant_configs = [None] * world_size
    all_keys = [None] * world_size

    my_keys = list(post_state_dict.keys())

    dist.gather_object(hf_quant_config, all_quant_configs if rank == 0 else None, dst=0)
    dist.gather_object(my_keys, all_keys if rank == 0 else None, dst=0)

    # 5. 只在 rank 0 上进行合并元数据和保存全局配置文件
    if rank == 0:
        print("[Rank 0] Starting to merge metadata and save global configs...")

        # a. 创建 model.safetensors.index.json
        weight_map = {}
        for r, key_list in enumerate(all_keys):
            shard_name = f"model-{r + 1:05d}-of-{world_size:05d}.safetensors"
            for key in key_list:
                weight_map[key] = shard_name

        index_json = {"metadata": {}, "weight_map": weight_map}
        index_json_path = os.path.join(export_dir, "model.safetensors.index.json")
        with open(index_json_path, "w") as f:
            json.dump(index_json, f, indent=4)
        print(f"[Rank 0] Saved model index to {index_json_path}")

        # b. 合并量化配置
        final_quant_config = all_quant_configs[0]
        exclude_modules = set(final_quant_config.get("quantization", {}).get("exclude_modules", []))
        for config in all_quant_configs:
            if config:
                mods = config.get("quantization", {}).get("exclude_modules", [])
                exclude_modules.update(mods)
        if "quantization" not in final_quant_config:
            final_quant_config["quantization"] = {}
        final_quant_config["quantization"]["exclude_modules"] = sorted(exclude_modules)

        # c. 保存原始模型配置和 tokenizer 配置
        model.config.save_pretrained(export_dir)
        tokenizer.save_pretrained(export_dir)

        # d. 更新 config.json 以包含量化信息
        config_path = os.path.join(export_dir, "config.json")
        with open(config_path) as f:
            config_data = json.load(f)

        config_data["quantization_config"] = final_quant_config

        with open(config_path, "w") as f:
            json.dump(config_data, f, indent=4)

        print(f"[Rank 0] Checkpoint successfully saved to {export_dir}")

    # with torch.inference_mode():
    #     export_hf_checkpoint(
    #         model,  # The quantized model.
    #         export_dir=export_dir,  # The directory where the exported files will be stored.
    #     )

    # # ====== 构造输入 ======
    # messages = [
    #     {"role": "user", "content": "你好，GLM-4.5！请用一句话介绍你自己。"}
    # ]
    # text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # inputs = tokenizer(text, return_tensors="pt").to(model.device)

    # # 移除模型不接受的 token_type_ids 参数
    # if "token_type_ids" in inputs:
    #     del inputs["token_type_ids"]

    # # ====== 生成回复 ======
    # with torch.no_grad():
    #     outputs = model.generate(
    #         **inputs,
    #         max_new_tokens=128,
    #         do_sample=False
    #     )

    # response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    # print(f"模型回复: {response}")


if __name__ == "__main__":
    main()
