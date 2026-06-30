# coding=utf-8
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch Qwen3-VL model."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm

from transformers.activations import ACT2FN
from transformers.utils import logging

import os
# local_rank = int(os.environ.get("LOCAL_RANK", -1))

# logger = logging.get_logger(__name__)



class VisionMlp(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, hidden_act: str) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = ACT2FN[hidden_act]
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class VisionSdpaAttentionSimple(nn.Module):
    def __init__(self, dim: int, num_heads: int = 16) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        B, L, C = hidden_states.shape
        q, k, v = self.qkv(hidden_states).reshape(B, L, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4).unbind(0)

        attn_output = F.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0
        )
        attn_output = attn_output.transpose(1, 2).reshape(B, L, -1)
        attn_output = self.proj(attn_output)
        return attn_output


class Qwen3VLVisionBlockSimple(nn.Module):
    def __init__(self, embed_dim=1280, num_heads=16, mlp_ratio=4, hidden_act="quick_gelu", attn_implementation="sdpa"):
        super().__init__()
        self.norm1 = LayerNorm(embed_dim, eps=1e-6)
        self.norm2 = LayerNorm(embed_dim, eps=1e-6)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.attn = VisionSdpaAttentionSimple(embed_dim, num_heads)
        self.mlp = VisionMlp(dim=embed_dim, hidden_dim=mlp_hidden_dim, hidden_act=hidden_act)

    def forward(self, hidden_states, cu_seqlens, rotary_pos_emb=None, position_embeddings=None):
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
            position_embeddings=position_embeddings,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class Qwen3VLVisionConnectorSimple(nn.Module):
    def __init__(self, depth, seq_len, embed_dim=1280, num_heads=16, mlp_ratio=4, hidden_act="quick_gelu", attn_implementation="sdpa"):
        super().__init__()
        self.blocks = nn.ModuleList([Qwen3VLVisionBlockSimple(embed_dim, num_heads, mlp_ratio, hidden_act, attn_implementation) for _ in range(depth)])

        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, embed_dim))

    def forward(self, hidden_states, cu_seqlens=None, rotary_pos_emb=None):
        B, L, C = hidden_states.shape
        hidden_states = hidden_states + self.pos_embed

        if cu_seqlens is None:
            actual_lengths = torch.full((B,), L, dtype=torch.long)
            cu_seqlens = torch.cat([torch.tensor([0]), actual_lengths.cumsum(dim=0)])
        for block in self.blocks:
            hidden_states = block(hidden_states, cu_seqlens, rotary_pos_emb)
        return hidden_states
