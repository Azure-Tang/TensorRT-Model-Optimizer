# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM

# DEBUG 日志开关: 设置环境变量 DEBUG_PP=1 开启
DEBUG_PP = os.getenv("DEBUG_PP", "0") == "1"


def _dlog(msg: str):
    if DEBUG_PP:
        r = os.getenv("RANK", "?")
        print(f"[PP][rank {r}] {msg}", flush=True)


@dataclass
class ModelArgs:
    model_path: str
    max_batch_size: int = 1


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.rank = int(os.getenv("RANK", "0"))
        self.world_size = int(os.getenv("WORLD_SIZE", "1"))
        # Prefer LOCAL_RANK for correct device mapping on multi-node
        self.local_rank = int(os.getenv("LOCAL_RANK", str(self.rank)))
        self.device = f"cuda:{self.local_rank}"

        # Load config to determine model structure
        config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)

        # Load full model on CPU to avoid OOM on a single GPU
        # and to easily distribute layers.
        full_model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            config=config,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
        )

        num_layers = config.num_hidden_layers
        self.config = full_model.config
        # total layers across all ranks (metadata)
        self.total_layers = num_layers
        self.layers_per_rank = (num_layers + self.world_size - 1) // self.world_size
        self.start_layer = self.rank * self.layers_per_rank
        self.end_layer = min(self.start_layer + self.layers_per_rank, num_layers)
        _dlog(
            f"init: total_layers={self.total_layers}, local_layers=[{self.start_layer},{self.end_layer})"
        )

        # Assign modules to ranks with proper naming structure
        # Create a model wrapper to maintain correct module names
        self.model = nn.Module()

        if self.rank == 0:
            self.model.embed_tokens = full_model.model.embed_tokens.to(self.device)

        self.model.layers = nn.ModuleDict()
        self._layer_order: list[int] = []
        for layer_idx in range(self.start_layer, self.end_layer):
            self.model.layers[str(layer_idx)] = full_model.model.layers[layer_idx].to(self.device)
            self._layer_order.append(layer_idx)

        if self.rank == self.world_size - 1:
            self.model.norm = full_model.model.norm.to(self.device)
            self.lm_head = full_model.lm_head.to(self.device)

        self.hidden_size = config.hidden_size
        # Only rank 0 computes rotary position embeddings; others receive cos/sin via pipeline
        if self.rank == 0:
            self.model.rotary_emb = full_model.model.rotary_emb.to(self.device)
            # rope dimension = 2 * len(inv_freq)
            self.rope_dim = int(self.model.rotary_emb.inv_freq.shape[-1] * 2)
        else:
            self.model.rotary_emb = None
            # Deduce rope_dim from config (equals head_dim)
            self.rope_dim = getattr(
                config, "head_dim", config.hidden_size // config.num_attention_heads
            )

    def forward(self, tokens: torch.Tensor, start_pos: int = 0):
        # Get batch size and sequence length
        batch_size, seq_len = tokens.shape
        _dlog(f"forward enter: tokens shape={tuple(tokens.shape)}, start_pos={start_pos}")

        # Create position_ids
        position_ids = torch.arange(
            start_pos, start_pos + seq_len, dtype=torch.long, device=self.device
        ).unsqueeze(0)

        # Create attention mask.
        attention_mask = torch.full(
            (batch_size, 1, seq_len, seq_len), -10000.0, device=self.device, dtype=torch.bfloat16
        )
        attention_mask = torch.triu(attention_mask, diagonal=1)

        # Pipeline execution
        if self.rank == 0:
            hidden_states = self.model.embed_tokens(tokens)
            _dlog(f"after embed: hidden_states shape={tuple(hidden_states.shape)}")
        else:
            # Receive hidden_states from the previous rank
            hidden_states = torch.empty(
                (batch_size, seq_len, self.hidden_size), device=self.device, dtype=torch.bfloat16
            )
            _dlog("recv hidden_states from prev...")
            dist.recv(hidden_states, src=self.rank - 1, tag=0)
            _dlog(f"recv hidden_states done: shape={tuple(hidden_states.shape)}")

        # Create/receive position embeddings per HF implementation
        if self.rank == 0:
            cos, sin = self.model.rotary_emb(hidden_states, position_ids)
            cos = cos.contiguous()
            sin = sin.contiguous()
            _dlog(f"pos_emb computed on rank0: cos={tuple(cos.shape)}, sin={tuple(sin.shape)}")
        else:
            # First receive rope_dim header then allocate cos/sin accordingly
            rope_dim_hdr = torch.empty((1,), device=self.device, dtype=torch.int32)
            _dlog("recv rope_dim header...")
            dist.recv(rope_dim_hdr, src=self.rank - 1, tag=10)
            rope_dim = int(rope_dim_hdr.item())
            _dlog(f"rope_dim={rope_dim}")
            cos = torch.empty(
                (batch_size, seq_len, rope_dim),
                device=self.device,
                dtype=torch.bfloat16,
            )
            sin = torch.empty(
                (batch_size, seq_len, rope_dim),
                device=self.device,
                dtype=torch.bfloat16,
            )
            _dlog("recv cos...")
            dist.recv(cos, src=self.rank - 1, tag=11)
            _dlog("recv sin...")
            dist.recv(sin, src=self.rank - 1, tag=12)
            _dlog(f"recv cos/sin done: cos={tuple(cos.shape)}, sin={tuple(sin.shape)}")
        position_embeddings = (cos, sin)

        # Common processing for all ranks
        for idx, layer_idx in enumerate(self._layer_order):
            layer = self.model.layers[str(layer_idx)]
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=None,
                cache_position=None,
                output_attentions=False,
                use_cache=False,
                position_embeddings=position_embeddings,
            )
            if (idx == 0 or idx == len(self._layer_order) - 1) and DEBUG_PP:
                _dlog(f"after layer {layer_idx}: h={tuple(hidden_states.shape)}")

        if self.rank == self.world_size - 1:
            # Final rank computes logits and returns them
            hidden_states = self.model.norm(hidden_states)
            logits = self.lm_head(hidden_states)
            _dlog(f"final logits shape={tuple(logits.shape)}")
            return logits
        else:
            # Intermediate ranks send hidden_states and rope to the next rank (with tags)
            _dlog("send hidden_states to next...")
            dist.send(hidden_states, dst=self.rank + 1, tag=0)
            # Send rope_dim header so the receiver can allocate exact buffers
            rope_dim_hdr = torch.tensor(
                [position_embeddings[0].shape[-1]], device=self.device, dtype=torch.int32
            )
            _dlog(f"send rope_dim={int(rope_dim_hdr.item())}")
            dist.send(rope_dim_hdr, dst=self.rank + 1, tag=10)
            _dlog("send cos...")
            dist.send(position_embeddings[0].contiguous(), dst=self.rank + 1, tag=11)
            _dlog("send sin...")
            dist.send(position_embeddings[1].contiguous(), dst=self.rank + 1, tag=12)
            _dlog("send done")
            return None
