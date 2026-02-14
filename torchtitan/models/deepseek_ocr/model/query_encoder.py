# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Qwen2-style query encoder with mixed bi-directional/causal attention masks.
# Adapted from DeepSeek-OCR-2's qwen2_d2e.py.

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .args import QueryEncoderArgs


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x).to(dtype)


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding."""

    def __init__(self, dim: int, max_position_embeddings: int = 131072, base: float = 1000000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [bs, num_heads, seq_len, head_dim]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(
            position_ids.shape[0], -1, 1
        )
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to q and k."""
    cos = cos.unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin.unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class QueryEncoderAttention(nn.Module):
    """Multi-head attention with grouped-query attention (GQA) support."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_kv_heads: int,
        attention_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_dim // num_heads
        self.num_key_value_groups = num_heads // num_kv_heads
        self.attention_dropout = attention_dropout

        self.q_proj = nn.Linear(hidden_dim, num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(hidden_dim, num_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(hidden_dim, num_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_dim, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Repeat k/v heads for grouped-query attention
        if self.num_key_value_groups > 1:
            key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
            value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)

        # Use a numerically stable attention implementation to avoid non-finite
        # gradients seen with masked SDPA in this mixed attention pattern.
        q = query_states.float()
        k = key_states.float()
        v = value_states.float()
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            if attention_mask.dtype == torch.bool:
                attn_scores = attn_scores.masked_fill(~attention_mask, float("-inf"))
            else:
                attn_scores = attn_scores + attention_mask.to(attn_scores.dtype)

        # Stabilize softmax and sanitize any all-masked row artifacts.
        attn_scores = attn_scores - torch.amax(attn_scores, dim=-1, keepdim=True)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = torch.nan_to_num(attn_probs, nan=0.0, posinf=0.0, neginf=0.0)
        if self.training and self.attention_dropout > 0.0:
            attn_probs = F.dropout(attn_probs, p=self.attention_dropout)

        attn_output = torch.matmul(attn_probs, v).to(query_states.dtype)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_dim)

        return self.o_proj(attn_output)


class QueryEncoderMLP(nn.Module):
    """MLP with SiLU activation (SwiGLU variant)."""

    def __init__(self, hidden_dim: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_dim, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class QueryEncoderLayer(nn.Module):
    """Single transformer layer for the query encoder."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_kv_heads: int,
        intermediate_size: int,
        rms_norm_eps: float,
        attention_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.self_attn = QueryEncoderAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            attention_dropout=attention_dropout,
        )
        self.mlp = QueryEncoderMLP(hidden_dim=hidden_dim, intermediate_size=intermediate_size)
        self.input_layernorm = RMSNorm(hidden_dim, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_dim, eps=rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        # Self-attention with residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states, attention_mask=attention_mask, position_embeddings=position_embeddings
        )
        hidden_states = residual + hidden_states

        # MLP with residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class QueryEncoder(nn.Module):
    """
    Qwen2-style decoder-as-encoder with mixed bi-directional/causal attention masks.

    The key insight from DeepSeek-OCR is using a mixed attention pattern:
    - Image tokens (type_id=0): bi-directional attention (can attend to all image tokens)
    - Query tokens (type_id=1): causal attention + cross-attend to ALL image tokens

    This creates a query-based bottleneck where only query tokens are output,
    which are then projected to the LLM dimension.
    """

    def __init__(self, args: QueryEncoderArgs) -> None:
        super().__init__()
        self.args = args
        self.hidden_dim = args.hidden_dim

        # Transformer layers
        self.layers = nn.ModuleList([
            QueryEncoderLayer(
                hidden_dim=args.hidden_dim,
                num_heads=args.num_heads,
                num_kv_heads=args.num_kv_heads,
                intermediate_size=args.intermediate_size,
                rms_norm_eps=args.rms_norm_eps,
                attention_dropout=args.attention_dropout,
            )
            for _ in range(args.num_layers)
        ])

        # Normalize image features before entering the stacked transformer.
        self.input_norm = RMSNorm(args.hidden_dim, eps=args.rms_norm_eps)
        self.norm = RMSNorm(args.hidden_dim, eps=args.rms_norm_eps)

        # Learnable query embeddings for different image sizes
        self.query_768 = nn.Embedding(args.num_query_tokens_768, args.hidden_dim)
        self.query_1024 = nn.Embedding(args.num_query_tokens_1024, args.hidden_dim)

        # Rotary embeddings
        self.rotary_emb = RotaryEmbedding(
            dim=args.hidden_dim // args.num_heads,
            max_position_embeddings=131072,
            base=args.rope_theta,
        )

    def _create_mixed_attention_mask(
        self,
        num_image_tokens: int,
        num_query_tokens: int,
        device: torch.device,
        batch_size: int,
    ) -> torch.Tensor:
        """
        Create the mixed bi-directional/causal attention mask.

        The mask has the following structure:
        - Image tokens (first num_image_tokens): bi-directional (attend to all image tokens)
        - Query tokens (last num_query_tokens): causal + cross-attend to ALL image tokens

        Returns:
            Boolean keep-mask of shape [batch_size, 1, total_len, total_len].
            True indicates an allowed attention position.
        """
        total_len = num_image_tokens + num_query_tokens

        masks = []
        for _ in range(batch_size):
            mask = torch.full(
                (total_len, total_len),
                fill_value=False,
                dtype=torch.bool,
                device=device,
            )

            # Image tokens: bi-directional (attend to all image tokens)
            # mask[0:num_image, 0:num_image] = True
            mask[:num_image_tokens, :num_image_tokens] = True

            # Query tokens: causal attention + cross-attend to all images
            for i in range(num_query_tokens):
                q_idx = num_image_tokens + i
                # Attend to all image tokens
                mask[q_idx, :num_image_tokens] = True
                # Causal for query tokens (attend to self and previous queries)
                mask[q_idx, num_image_tokens : q_idx + 1] = True

            masks.append(mask)

        # Stack and add head dimension: [B, 1, total_len, total_len]
        return torch.stack(masks, dim=0).unsqueeze(1)

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the query encoder.

        Args:
            image_features: Features from SAM encoder with shape [B, C, H, W]
                where C=896 (after SAM's downsampling convs).

        Returns:
            Query tokens with shape [B, num_query_tokens, hidden_dim]
        """
        # Flatten spatial dimensions: [B, C, H, W] -> [B, H*W, C]
        x = image_features.flatten(2).transpose(1, 2)
        x = self.input_norm(x)

        bs, num_image_tokens, _ = x.shape

        # Select query embeddings based on image size
        if num_image_tokens == 144:  # 12x12 grid (from 768x768 image)
            query_embeddings = self.query_768.weight
        elif num_image_tokens == 256:  # 16x16 grid (from 1024x1024 image)
            query_embeddings = self.query_1024.weight
        else:
            # Fallback: use larger query set
            query_embeddings = self.query_1024.weight

        num_query_tokens = query_embeddings.shape[0]

        # Expand queries for batch: [num_query, hidden] -> [B, num_query, hidden]
        batch_queries = query_embeddings.unsqueeze(0).expand(bs, -1, -1)

        # Concatenate: [image_tokens | query_tokens]
        x_combined = torch.cat([x, batch_queries], dim=1)

        # Create mixed attention mask
        attention_mask = self._create_mixed_attention_mask(
            num_image_tokens=num_image_tokens,
            num_query_tokens=num_query_tokens,
            device=x_combined.device,
            batch_size=bs,
        )

        # Create position ids
        total_len = num_image_tokens + num_query_tokens
        position_ids = torch.arange(total_len, device=x_combined.device).unsqueeze(0).expand(bs, -1)

        # Get rotary embeddings
        position_embeddings = self.rotary_emb(x_combined, position_ids)

        # Apply transformer layers
        hidden_states = x_combined
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
            )

        # Apply final norm
        hidden_states = self.norm(hidden_states)

        # Return only query tokens (the "causal flow")
        query_output = hidden_states[:, num_image_tokens:, :]

        return query_output

    def init_weights(self) -> None:
        """Initialize weights."""

        def _init_weights(module: nn.Module):
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.trunc_normal_(module.weight, std=0.02)

        self.apply(_init_weights)

        # Initialize RMSNorm
        for module in self.modules():
            if isinstance(module, RMSNorm):
                nn.init.ones_(module.weight)
