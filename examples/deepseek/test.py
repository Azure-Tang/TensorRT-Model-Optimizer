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
    parser = argparse.ArgumentParser("HF GLM4-MoE wrapper demo: load and generate")
    parser.add_argument(
        "--model_path",
        type=str,
        default="/mnt/data/models/GLM-4.5-Air-Test/",
        help="HuggingFace model path or repo id",
    )
    parser.add_argument("--prompt", type=str, default="你好, GLM-4.5! 请用一句话介绍你自己.")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument(
        "--do_sample", action="store_true", help="use multinomial sampling instead of greedy"
    )
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature for sampling")
    args = parser.parse_args()

    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = load_deepseek_model(args.model_path, batch_size=1)

    # 构造输入(简单 prompt -> input_ids)
    text = args.prompt
    input_ids = tokenizer(text, return_tensors="pt")["input_ids"].to(model.device)
    print(f"intpu_ids.device: {input_ids.device}, model.device: {model.device}")
    # 准备增量生成的状态
    eos = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else None
    generated = input_ids.clone()

    # 生成循环: 最后一个 rank 采样并打印(多卡 PP 时), 然后广播给其他 rank
    with torch.no_grad():
        finished = torch.tensor([0], device=model.device, dtype=torch.int32)
        for _ in range(args.max_new_tokens):
            logits = model(generated)
            # 在流水线或单卡情况下, 只有最后一个 rank 拿到非 None 的 logits
            new_token = None
            if logits is not None:
                last_token_logits = logits[0, -1] / max(args.temperature, 1e-6)
                if args.do_sample:
                    probs = torch.softmax(last_token_logits, dim=-1)
                    new_token = torch.multinomial(probs, num_samples=1)
                else:
                    new_token = torch.argmax(last_token_logits, dim=-1, keepdim=True)
                # 形状统一为 [1, 1]
                if new_token.dim() == 1:
                    new_token = new_token.view(1, 1)
            # 分布式同步: 由最后一个 rank 广播 token 与 finished 标记
            if world_size > 1 and dist.is_initialized():
                if logits is None:
                    # 非最后一段 rank 占位, 准备接收
                    new_token = torch.empty((1, 1), device=model.device, dtype=torch.long)
                dist.broadcast(new_token, src=world_size - 1)
            # 拼接到序列
            generated = torch.cat([generated, new_token], dim=1)
            # 终止条件: 如果采样到 eos
            if eos is not None and int(new_token.item()) == int(eos):
                finished.fill_(1)
            # 广播结束标志
            if world_size > 1 and dist.is_initialized():
                dist.broadcast(finished, src=world_size - 1)
            if int(finished.item()) == 1:
                break

    dist.barrier()
    print(f"====Generation finished on rank {rank}====")
    # 仅最后一个 rank 打印完整回复
    if world_size == 1 or rank == world_size - 1:
        full_text = tokenizer.decode(generated[0], skip_special_tokens=True)
        print("==== 模型完整回复 ====")
        print(full_text)

    # destroy the process group
    if world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
