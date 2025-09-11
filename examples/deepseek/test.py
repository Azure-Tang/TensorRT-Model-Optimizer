import argparse
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import AutoTokenizer

sys.path.append(str(Path(__file__).resolve().parent))
import glm4_hf_pp as deekseep_model

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
    parser = argparse.ArgumentParser("HF GLM4-MoE wrapper demo: load and forward")
    parser.add_argument("--model_path", default= "/mnt/data/models/GLM-4.5-Air-Test/",type=str, required=True, help="HuggingFace model path or repo id")
    parser.add_argument("--prompt", type=str, default="你好，GLM-4.5！请用一句话介绍你自己。")
    args = parser.parse_args()

    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = load_deepseek_model(args.model_path, batch_size=1)

    # 构造输入（简单 prompt -> input_ids）
    text = args.prompt
    input_ids = tokenizer(text, return_tensors="pt")["input_ids"].to(model.device)

    # 前向：仅最后一个 rank 有返回（多卡 PP 时）
    with torch.no_grad():
        logits = model(input_ids)

    if world_size == 1 or rank == world_size - 1:
        # 打印结果信息
        print(f"rank {rank}: logits shape = {None if logits is None else tuple(logits.shape)}")
        if logits is not None:
            # 取最后一个 token 的 top-5 预测
            last_token_logits = logits[0, -1]
            topk = torch.topk(last_token_logits, k=5)
            ids = topk.indices.tolist()
            toks = tokenizer.convert_ids_to_tokens(ids)
            print("top-5 ids:", ids)
            print("top-5 toks:", toks)

if __name__ == "__main__":
    main()
