# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn

from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.models.attention import (
    create_attention_mask,
    create_varlen_metadata_for_document,
)
from torchtitan.models.llama3.model.model import precompute_freqs_cis
from torchtitan.protocols.model import AttentionMasksType
from torchtitan.protocols.train_spec import ModelProtocol

from .args import MoTModelArgs
from .attention import ModalityUntiedAttention
from .feedforward import ModalityUntiedFeedForward


class MoTTransformerBlock(nn.Module):
    """
    Transformer block with modality-specific attention and feed-forward.

    Key differences from standard TransformerBlock:
    - Norms are inside attention/FFN modules (MoT pattern)
    - Requires modality_masks in forward pass for routing

    Args:
        layer_id: Layer index for depth-based initialization
        model_args: MoT model configuration

    Attributes:
        layer_id: Layer index
        dim: Model dimension
        attention: Modality-untied attention module
        feed_forward: Modality-untied feed-forward module
        weight_init_std: Standard deviation for weight initialization
    """

    def __init__(self, layer_id: int, model_args: MoTModelArgs):
        super().__init__()
        self.layer_id = layer_id
        self.dim = model_args.dim

        # Modality-untied components
        self.attention = ModalityUntiedAttention(model_args)
        self.feed_forward = ModalityUntiedFeedForward(model_args)

        # Weight initialization scaling (depth-dependent or uniform)
        if model_args.depth_init:
            self.weight_init_std = 0.02 / (2 * (layer_id + 1)) ** 0.5
        else:
            self.weight_init_std = 0.02 / (2 * model_args.n_layers) ** 0.5

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        modality_masks: list[torch.Tensor],
    ) -> torch.Tensor:
        """
        Forward pass through transformer block with modality routing.

        Note: Norms are applied inside attention/FFN modules.

        Args:
            x: Input tensor of shape (bs, seqlen, dim)
            freqs_cis: Precomputed RoPE frequencies
            attention_masks: Optional attention masks (for flex/varlen)
            modality_masks: List of boolean masks for modality routing

        Returns:
            Output tensor of shape (bs, seqlen, dim)
        """
        # Attention with residual connection
        h = x + self.attention(x, freqs_cis, attention_masks, modality_masks)

        # Feed-forward with residual connection
        out = h + self.feed_forward(h, modality_masks)

        return out

    def init_weights(self):
        """Initialize all weights in this block."""
        self.attention.init_weights(self.weight_init_std)
        self.feed_forward.init_weights(self.weight_init_std)


class MoTTransformer(nn.Module, ModelProtocol):
    """
    Mixture-of-Transformers model for multi-modal generation.

    MoT uses modality-specific parameters in each transformer layer,
    enabling efficient multi-modal generation with sparse computation.
    Each modality gets its own attention and FFN parameters, but all
    modalities interact through global self-attention.

    Implements ModelProtocol for torchtitan compatibility.

    Args:
        model_args: MoT model configuration

    Attributes:
        model_args: Configuration object
        vocab_size: Vocabulary size
        n_layers: Number of transformer layers
        tok_embeddings: Token embedding layer (shared across modalities)
        layers: ModuleDict of transformer blocks
        norm: Output normalization layer (shared)
        output: Output projection layer (shared)
        freqs_cis: Precomputed RoPE frequencies
        attn_type: Attention type (sdpa, flex, varlen)
        attn_mask_type: Attention mask type (causal, block_causal)
    """

    def __init__(self, model_args: MoTModelArgs):
        super().__init__()
        self.model_args = model_args
        self.vocab_size = model_args.vocab_size
        self.n_layers = model_args.n_layers

        # Token embeddings (shared across modalities)
        self.tok_embeddings = nn.Embedding(model_args.vocab_size, model_args.dim)

        # Transformer layers with modality-specific parameters
        self.layers = nn.ModuleDict(
            {str(i): MoTTransformerBlock(i, model_args) for i in range(model_args.n_layers)}
        )

        # Output layers (shared across modalities)
        self.norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)
        self.output = nn.Linear(model_args.dim, model_args.vocab_size, bias=False)

        # Precompute RoPE frequencies
        self.freqs_cis = precompute_freqs_cis(
            model_args.dim // model_args.n_heads,
            model_args.max_seq_len * 2,
            model_args.rope_theta,
            model_args.rope_scaling_args,
        )

        # Attention configuration
        self.attn_type = model_args.attn_type
        self.attn_mask_type = model_args.attn_mask_type

    def forward(
        self,
        tokens: torch.Tensor,  # (bs, seqlen)
        modality_masks: list[torch.Tensor],  # List of (bs*seqlen,) boolean tensors
        attention_masks: AttentionMasksType | None = None,
    ) -> torch.Tensor:
        """
        Forward pass with modality routing.

        Args:
            tokens: Input token IDs of shape (bs, seqlen)
            modality_masks: List of boolean masks indicating which tokens
                           belong to each modality. Each mask has shape (bs*seqlen,)
            attention_masks: Optional attention masks (for flex/varlen attention)

        Returns:
            Logits tensor of shape (bs, seqlen, vocab_size)
        """
        bs, seqlen = tokens.shape

        # Embed tokens
        h = self.tok_embeddings(tokens)

        # Move freqs_cis to same device as input
        freqs_cis = self.freqs_cis[:seqlen].to(h.device)

        # Forward through all transformer layers
        for layer in self.layers.values():
            h = layer(h, freqs_cis, attention_masks, modality_masks)

        # Output projection
        h = self.norm(h)
        output = self.output(h)

        return output

    def init_weights(self, buffer_device: torch.device | None = None):
        """
        Initialize all model weights.

        Required by ModelProtocol.

        Args:
            buffer_device: Optional device for buffers (unused)
        """
        # Initialize embeddings
        nn.init.trunc_normal_(self.tok_embeddings.weight, mean=0.0, std=0.02)

        # Initialize all transformer layers
        for layer in self.layers.values():
            layer.init_weights()

        # Initialize output layers
        self.norm.reset_parameters()
        nn.init.trunc_normal_(self.output.weight, mean=0.0, std=0.02)

    def get_attention_masks(
        self,
        input_batch: torch.Tensor,
        tokenizer: BaseTokenizer,
        extra_inputs: dict[str, torch.Tensor] | None = None,
    ) -> AttentionMasksType:
        """
        Create attention masks for flex/varlen attention.

        Optional method required by ModelProtocol if using flex/varlen attention.

        Args:
            input_batch: Input token IDs
            tokenizer: Tokenizer for special token detection
            extra_inputs: Optional extra inputs (unused)

        Returns:
            Attention masks (BlockMask for flex, VarlenMetadata for varlen, None for sdpa)
        """
        if self.attn_type == "flex":
            return create_attention_mask(
                input_batch,
                tokenizer,
                self.attn_mask_type,
            )
        elif self.attn_type == "varlen":
            return create_varlen_metadata_for_document(
                input_batch,
                tokenizer,
            )
        return None
