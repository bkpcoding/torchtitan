# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
from torch import nn

from .args import MoTModelArgs


class FeedForward(nn.Module):
    """
    Standard feed-forward network module (SwiGLU activation).

    This is the base FFN that will be instantiated for each modality.
    Uses SwiGLU: w2(silu(w1(x)) * w3(x))

    Args:
        dim: Input/output dimension
        hidden_dim: Hidden dimension
        multiple_of: Ensure hidden_dim is multiple of this value
        ffn_dim_multiplier: Optional multiplier for hidden dimension
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: float | None,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # Custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """SwiGLU feed-forward: w2(silu(w1(x)) * w3(x))"""
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

    def init_weights(self, init_std: float):
        """Initialize weights with truncated normal distribution."""
        nn.init.trunc_normal_(self.w1.weight, mean=0.0, std=0.02)
        for linear in (self.w2, self.w3):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)


class ModalityUntiedFeedForward(nn.Module):
    """
    Modality-specific feed-forward network for MoT.

    Creates separate FFN experts for each modality, enabling specialized
    processing. Tokens are deterministically routed to their modality-specific
    expert using modality_masks.

    Args:
        model_args: MoT model configuration

    Attributes:
        n_modalities: Number of modality experts
        local_experts: ModuleList of FeedForward networks (one per modality)
        local_experts_ffn_norm: ModuleList of RMSNorm layers (one per modality)
    """

    def __init__(self, model_args: MoTModelArgs):
        super().__init__()
        self.n_modalities = model_args.n_modalities

        # Calculate hidden dimension (same as llama3)
        hidden_dim = 4 * model_args.dim
        hidden_dim = int(2 * hidden_dim / 3)
        if model_args.ffn_dim_multiplier is not None:
            hidden_dim = int(model_args.ffn_dim_multiplier * hidden_dim)
        hidden_dim = model_args.multiple_of * (
            (hidden_dim + model_args.multiple_of - 1) // model_args.multiple_of
        )

        # Create modality-specific FFN experts
        self.local_experts = nn.ModuleList(
            [
                FeedForward(
                    dim=model_args.dim,
                    hidden_dim=hidden_dim,
                    multiple_of=model_args.multiple_of,
                    ffn_dim_multiplier=model_args.ffn_dim_multiplier,
                )
                for _ in range(self.n_modalities)
            ]
        )

        # Modality-specific FFN norms (moved from TransformerBlock)
        self.local_experts_ffn_norm = nn.ModuleList(
            [
                nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)
                for _ in range(self.n_modalities)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,  # (bs, seqlen, dim)
        modality_masks: list[torch.Tensor],  # List of (bs*seqlen,) boolean masks
    ) -> torch.Tensor:
        """
        Route tokens to modality-specific FFN experts.

        Args:
            x: Input tensor of shape (bs, seqlen, dim)
            modality_masks: List of boolean masks indicating which tokens
                           belong to each modality. Each mask has shape (bs*seqlen,)

        Returns:
            Output tensor of shape (bs, seqlen, dim) with modality-specific
            FFN applied to each token
        """
        bs, seqlen, dim = x.shape
        x_flat = x.view(-1, dim)  # (bs*seqlen, dim)

        # Process each modality separately
        expert_outputs = []
        for i in range(self.n_modalities):
            # Extract tokens for this modality
            expert_input = x_flat[modality_masks[i]]  # (num_tokens_i, dim)

            # Apply FFN + norm for this modality
            expert_output = self.local_experts[i](expert_input)
            expert_output = self.local_experts_ffn_norm[i](expert_output)

            expert_outputs.append(expert_output)

        # Merge outputs back to original sequence order
        merged = torch.empty_like(x_flat)
        for i in range(self.n_modalities):
            merged[modality_masks[i]] = expert_outputs[i]

        return merged.view(bs, seqlen, dim)

    def init_weights(self, init_std: float):
        """Initialize all modality-specific FFN weights."""
        for expert in self.local_experts:
            expert.init_weights(init_std)
        for norm in self.local_experts_ffn_norm:
            norm.reset_parameters()
