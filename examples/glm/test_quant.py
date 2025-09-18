import argparse
import os

import glm4_hf_pp as deekseep_model
import torch
import torch.distributed as dist
from tqdm import tqdm
from transformers import AutoTokenizer

import modelopt.torch.quantization as mtq
from modelopt.torch.utils.dataset_utils import get_dataset_dataloader

# set python env proxy
os.environ["http_proxy"] = "http://117.133.60.227:20181"
os.environ["https_proxy"] = "http://117.133.60.227:20181"
os.environ["ALL_PROXY"] = "http://117.133.60.227:20181"

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
    model_name = "/mnt/data/models/GLM-4.5-Air-Test/"
    # model_name = "zai-org/GLM-4.5-Air"

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

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = load_deepseek_model(args.model_path, batch_size=1)

    config = mtq.NVFP4_DEFAULT_CFG
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    batch_size = 1
    num_samples = 1

    calib_dataset = get_dataset_dataloader(
        dataset_name="cnn_dailymail",
        tokenizer=tokenizer,
        batch_size=batch_size,
        num_samples=num_samples,
    )

    def calibrate_loop(model):
        rank = int(os.getenv("RANK", "0"))
        world_size = int(os.getenv("WORLD_SIZE", "1"))
        dev = next(model.parameters()).device
        _dlog("=====enter calibrate_loop====")

        if world_size == 1:
            # Single-GPU case: iterate normally
            for data in tqdm(calib_dataset):
                model(data["input_ids"])
        # Pipeline-parallel case
        elif rank == 0:
            # Rank 0 drives calibration, iterates real data, and broadcasts shape
            for data in tqdm(calib_dataset):
                tokens = data["input_ids"]
                shape = torch.tensor(tokens.shape, device=dev, dtype=torch.int64)
                dist.broadcast(shape, src=0)
                model(tokens.to(dev))
            # Send stop signal
            stop_signal = torch.tensor([-1, -1], device=dev, dtype=torch.int64)
            dist.broadcast(stop_signal, src=0)
        else:
            # Other ranks receive shape, create dummy tensor, and call forward
            while True:
                shape = torch.empty(2, device=dev, dtype=torch.int64)
                dist.broadcast(shape, src=0)
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

    from modelopt.torch.export import export_hf_checkpoint

    export_dir = "/data/numa0/downloaded_models/GLM-4.5-Air-nvfp4-tp4-test"
    with torch.inference_mode():
        export_hf_checkpoint(
            model,  # The quantized model.
            export_dir=export_dir,  # The directory where the exported files will be stored.
        )

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
