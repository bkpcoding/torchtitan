# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn

from torchtitan.models.attention import (
    FlexAttentionWrapper,
    ScaledDotProductAttentionWrapper,
    VarlenAttentionWrapper,
    VarlenMetadata,
)
from torchtitan.models.llama3.model.model import apply_rotary_emb, repeat_kv
from torchtitan.protocols.model import AttentionMasksType

from .args import MoTModelArgs


class ModalityUntiedAttention(nn.Module):
    """
    Modality-specific multi-head attention for MoT.

    Creates separate wq, wk, wv, wo projections and attention norms for each
    modality. After modality-specific QKV projection, performs global
    self-attention across all tokens, then routes outputs back through
    modality-specific output projections.

    Args:
        model_args: MoT model configuration

    Attributes:
        n_modalities: Number of modality experts
        n_heads: Number of attention heads
        n_kv_heads: Number of key/value heads (for multi-query attention)
        n_rep: Number of repetitions for key/value heads
        head_dim: Dimension of each attention head
        local_experts_wq/wk/wv/wo: Modality-specific linear projections
        local_experts_attention_norm: Modality-specific output norms
        inner_attention: Attention computation wrapper (flex/varlen/sdpa)
    """

    def __init__(self, model_args: MoTModelArgs):
        super().__init__()
        self.n_modalities = model_args.n_modalities
        self.n_heads = model_args.n_heads
        self.n_kv_heads = (
            model_args.n_heads
            if model_args.n_kv_heads is None
            else model_args.n_kv_heads
        )
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = model_args.dim // model_args.n_heads

        # Create modality-specific query projections
        self.local_experts_wq = nn.ModuleList(
            [
                nn.Linear(
                    model_args.dim, model_args.n_heads * self.head_dim, bias=False
                )
                for _ in range(self.n_modalities)
            ]
        )

        # Create modality-specific key projections
        self.local_experts_wk = nn.ModuleList(
            [
                nn.Linear(model_args.dim, self.n_kv_heads * self.head_dim, bias=False)
                for _ in range(self.n_modalities)
            ]
        )

        # Create modality-specific value projections
        self.local_experts_wv = nn.ModuleList(
            [
                nn.Linear(model_args.dim, self.n_kv_heads * self.head_dim, bias=False)
                for _ in range(self.n_modalities)
            ]
        )

        # Create modality-specific output projections
        self.local_experts_wo = nn.ModuleList(
            [
                nn.Linear(
                    model_args.n_heads * self.head_dim, model_args.dim, bias=False
                )
                for _ in range(self.n_modalities)
            ]
        )

        # Optional QK normalization per modality
        self.qk_normalization = model_args.qk_normalization
        if model_args.qk_normalization:
            self.local_experts_q_normalization = nn.ModuleList(
                [
                    nn.RMSNorm(self.head_dim, eps=model_args.norm_eps)
                    for _ in range(self.n_modalities)
                ]
            )
            self.local_experts_k_normalization = nn.ModuleList(
                [
                    nn.RMSNorm(self.head_dim, eps=model_args.norm_eps)
                    for _ in range(self.n_modalities)
                ]
            )

        # Modality-specific output norms (moved from TransformerBlock)
        self.local_experts_attention_norm = nn.ModuleList(
            [
                nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)
                for _ in range(self.n_modalities)
            ]
        )

        # Inner attention mechanism (reuse torchtitan's wrappers)
        self.attn_type = model_args.attn_type
        match self.attn_type:
            case "flex":
                self.inner_attention = FlexAttentionWrapper()
            case "varlen":
                self.inner_attention = VarlenAttentionWrapper()
            case _:
                self.inner_attention = ScaledDotProductAttentionWrapper()

    def forward(
        self,
        x: torch.Tensor,  # (bs, seqlen, dim)
        freqs_cis: torch.Tensor,  # RoPE frequencies
        attention_masks: AttentionMasksType | None,
        modality_masks: list[torch.Tensor],  # List of (bs*seqlen,) boolean masks
    ) -> torch.Tensor:
        """
        Forward pass with modality-specific routing.

        Flow:
        1. Route tokens to modality-specific wq/wk/wv projections
        2. Merge Q, K, V from all modalities back to sequence order
        3. Apply RoPE to queries and keys
        4. Compute global self-attention across all modalities
        5. Route attention output to modality-specific wo projections
        6. Apply modality-specific output norms

        Args:
            x: Input tensor of shape (bs, seqlen, dim)
            freqs_cis: Precomputed RoPE frequencies
            attention_masks: Optional attention masks (for flex/varlen)
            modality_masks: List of boolean masks indicating which tokens
                           belong to each modality

        Returns:
            Output tensor of shape (bs, seqlen, dim)
        """
        bs, seqlen, _ = x.shape

        # Flatten for token-level routing
        x_flat = x.view(-1, x.size(-1))  # (bs*seqlen, dim)

        # Step 1: Process QKV through modality-specific projections
        expert_outputs_xq, expert_outputs_xk, expert_outputs_xv = [], [], []
        for i in range(self.n_modalities):
            # Extract tokens for this modality
            expert_input = x_flat[modality_masks[i]]

            # Project through modality-specific wq, wk, wv
            xq = self.local_experts_wq[i](expert_input)
            xk = self.local_experts_wk[i](expert_input)
            xv = self.local_experts_wv[i](expert_input)

            # Optional: Apply QK normalization
            if self.qk_normalization:
                # Reshape for per-head normalization
                xq_reshaped = xq.view(-1, self.n_heads, self.head_dim)
                xk_reshaped = xk.view(-1, self.n_kv_heads, self.head_dim)

                # Apply normalization
                xq_reshaped = self.local_experts_q_normalization[i](xq_reshaped)
                xk_reshaped = self.local_experts_k_normalization[i](xk_reshaped)

                # Reshape back
                xq = xq_reshaped.view(-1, self.n_heads * self.head_dim)
                xk = xk_reshaped.view(-1, self.n_kv_heads * self.head_dim)

            expert_outputs_xq.append(xq)
            expert_outputs_xk.append(xk)
            expert_outputs_xv.append(xv)

        # Step 2: Merge modality outputs back to original sequence order
        xq = self._merge_modalities(
            expert_outputs_xq, modality_masks, bs * seqlen, self.n_heads * self.head_dim
        )
        xk = self._merge_modalities(
            expert_outputs_xk,
            modality_masks,
            bs * seqlen,
            self.n_kv_heads * self.head_dim,
        )
        xv = self._merge_modalities(
            expert_outputs_xv,
            modality_masks,
            bs * seqlen,
            self.n_kv_heads * self.head_dim,
        )

        # Reshape for multi-head attention
        # Use -1 instead of n_heads to infer actual local heads after TP sharding
        xq = xq.view(bs, seqlen, -1, self.head_dim)
        xk = xk.view(bs, seqlen, -1, self.head_dim)
        xv = xv.view(bs, seqlen, -1, self.head_dim)

        # Step 3: Apply RoPE (rotary position embeddings)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # Step 4: Repeat k/v for multi-query attention
        keys = repeat_kv(xk, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
        values = repeat_kv(xv, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)

        # Transpose for attention computation
        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xk = keys.transpose(1, 2)
        xv = values.transpose(1, 2)

        # Step 5: Global attention across all modalities
        match self.attn_type:
            case "flex":
                from torch.nn.attention.flex_attention import BlockMask

                assert isinstance(attention_masks, BlockMask), attention_masks
                output = self.inner_attention(xq, xk, xv, block_mask=attention_masks)
            case "varlen":
                assert isinstance(attention_masks, VarlenMetadata), attention_masks
                output = self.inner_attention(
                    xq,
                    xk,
                    xv,
                    self.head_dim,
                    attention_masks,
                )
            case "sdpa":
                assert attention_masks is None
                output = self.inner_attention(xq, xk, xv)
            case _:
                raise ValueError(f"Unknown attention type: {self.attn_type}")

        # Step 6: Process output through modality-specific wo projections
        output = output.transpose(1, 2).contiguous()  # (bs, seqlen, n_heads, head_dim)
        output = output.view(bs, seqlen, -1)
        output_flat = output.view(-1, output.size(-1))

        # Route through modality-specific output projections and norms
        expert_outputs = []
        for i in range(self.n_modalities):
            expert_input = output_flat[modality_masks[i]]
            expert_output = self.local_experts_wo[i](expert_input)
            expert_output = self.local_experts_attention_norm[i](expert_output)
            expert_outputs.append(expert_output)

        # Merge and reshape
        output = self._merge_modalities(
            expert_outputs, modality_masks, bs * seqlen, x.size(-1)
        )
        return output.view(bs, seqlen, -1)

    def _merge_modalities(
        self,
        expert_outputs: list[torch.Tensor],
        modality_masks: list[torch.Tensor],
        total_tokens: int,
        dim: int,
    ) -> torch.Tensor:
        """
        Merge modality-specific outputs back to original sequence order.

        Args:
            expert_outputs: List of modality-specific outputs
            modality_masks: List of boolean masks
            total_tokens: Total number of tokens (bs * seqlen)
            dim: Output dimension

        Returns:
            Merged tensor of shape (total_tokens, dim)
        """
        merged = torch.empty(
            (total_tokens, dim),
            device=expert_outputs[0].device,
            dtype=expert_outputs[0].dtype,
        )
        for i in range(len(expert_outputs)):
            merged[modality_masks[i]] = expert_outputs[i]
        return merged

    def init_weights(self, init_std: float):
        """Initialize all modality-specific weights."""
        for i in range(self.n_modalities):
            # Initialize Q, K, V projections with std=0.02
            for linear in (
                self.local_experts_wq[i],
                self.local_experts_wk[i],
                self.local_experts_wv[i],
            ):
                nn.init.trunc_normal_(linear.weight, mean=0.0, std=0.02)

            # Initialize output projection with layer-specific std
            nn.init.trunc_normal_(self.local_experts_wo[i].weight, mean=0.0, std=init_std)

            # Initialize norms
            self.local_experts_attention_norm[i].reset_parameters()

            # Initialize QK norms if enabled
            if self.qk_normalization:
                self.local_experts_q_normalization[i].reset_parameters()
                self.local_experts_k_normalization[i].reset_parameters()
