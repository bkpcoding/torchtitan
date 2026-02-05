# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
DeepSeek-OCR Transformer model combining:
- SAM ViT-B vision encoder
- Qwen2-style query encoder with mixed attention masks
- DeepSeek-V3 MoE LLM backbone
"""

from typing import cast, Optional

import torch
from torch import nn

from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.models.deepseek_v3.model.model import (
    DeepSeekV3Model,
    TransformerBlock,
)
from torchtitan.protocols.model import AttentionMasksType

from .args import DeepSeekOCRModelArgs, SpecialTokens
from .projector import build_projector
from .query_encoder import QueryEncoder
from .sam_encoder import SAMViTEncoder


def scatter_vision_tokens(
    h: torch.Tensor,
    tokens: torch.Tensor,
    vision_tokens: torch.Tensor,
    image_token_id: int,
) -> torch.Tensor:
    """
    Scatter vision tokens into the text embedding sequence at image placeholder positions.

    Args:
        h: Text embeddings of shape [B, S, D].
        tokens: Input token ids of shape [B, S].
        vision_tokens: Vision tokens to scatter of shape [B, N, D] where N is num query tokens.
        image_token_id: Token ID used for image placeholders.

    Returns:
        Updated embeddings with vision tokens scattered at image positions.
    """
    B, S, D = h.shape
    device = h.device

    # Find image placeholder positions
    img_mask = tokens == image_token_id  # [B, S]

    # We need to scatter N vision tokens per batch item
    # The number of <image> tokens should match the number of vision tokens
    num_vision_tokens = vision_tokens.shape[1]

    for b in range(B):
        img_positions = torch.where(img_mask[b])[0]
        num_img_positions = img_positions.shape[0]

        if num_img_positions == 0:
            continue

        if num_img_positions != num_vision_tokens:
            # This can happen if the sequence was truncated or there's a mismatch
            # Use the minimum to avoid index errors
            num_to_scatter = min(num_img_positions, num_vision_tokens)
            h[b, img_positions[:num_to_scatter], :] = vision_tokens[b, :num_to_scatter, :]
        else:
            h[b, img_positions, :] = vision_tokens[b, :, :]

    return h


class DeepSeekOCRTransformer(DeepSeekV3Model):
    """
    DeepSeek-OCR Vision-Language Model.

    Architecture:
    1. SAM ViT-B encodes the image with bi-directional attention
    2. Query encoder uses mixed attention (bi-directional for images, causal + cross-attention for queries)
    3. Only query tokens are output and projected to LLM dimension
    4. Vision tokens are scattered into the text sequence at <image> positions
    5. DeepSeek-V3 MoE backbone processes the combined sequence
    """

    def __init__(self, model_args: DeepSeekOCRModelArgs):
        super().__init__(model_args)

        self.model_args = model_args

        # Vision components
        self.sam_encoder: Optional[SAMViTEncoder] = SAMViTEncoder(model_args.sam_encoder)
        self.query_encoder: Optional[QueryEncoder] = QueryEncoder(model_args.query_encoder)

        # Projector: query_encoder.hidden_dim -> LLM dim
        proj_in_dim, proj_out_dim = model_args.get_projector_dims()
        self.projector: Optional[nn.Module] = build_projector(
            model_args.projector, proj_in_dim, proj_out_dim
        )

    def init_weights(self, buffer_device: torch.device | None = None) -> None:
        """Initialize all model weights."""
        # Initialize LLM backbone
        super().init_weights(buffer_device=buffer_device)

        # Initialize vision components
        if self.sam_encoder is not None:
            self.sam_encoder.init_weights()
        if self.query_encoder is not None:
            self.query_encoder.init_weights()
        if self.projector is not None:
            self.projector.init_weights()

    def get_attention_masks(
        self,
        input_batch: torch.Tensor,
        tokenizer: BaseTokenizer,
        extra_inputs: dict[str, torch.Tensor] | None = None,
    ) -> AttentionMasksType:
        """
        Get attention masks for the model.

        For DeepSeek-OCR, we use the standard LLM attention masks (causal or block_causal).
        The vision encoder and query encoder handle their own attention patterns internally.
        """
        return super().get_attention_masks(input_batch, tokenizer, extra_inputs)

    def forward(
        self,
        tokens: torch.Tensor,
        images: Optional[torch.Tensor] = None,
        special_tokens: Optional[SpecialTokens] = None,
        attention_masks: AttentionMasksType | None = None,
        positions: torch.Tensor | None = None,
    ):
        """
        Forward pass for DeepSeek-OCR.

        Args:
            tokens: Input token ids of shape [B, S].
            images: Optional input images of shape [B, C, H, W].
            special_tokens: Special tokens configuration for finding image placeholders.
            attention_masks: Attention masks for the LLM backbone.
            positions: Optional position indices for RoPE.

        Returns:
            Logits of shape [B, S, vocab_size].
        """
        # Get text embeddings
        h = self.tok_embeddings(tokens) if self.tok_embeddings is not None else tokens

        # Process images if provided
        if images is not None and self.sam_encoder is not None:
            # 1. SAM encoder: [B, C, H, W] -> [B, 896, H', W']
            image_features = self.sam_encoder(images)

            # 2. Query encoder with mixed attention: [B, 896, H', W'] -> [B, num_query, hidden_dim]
            vision_tokens = self.query_encoder(image_features)

            # 3. Project to LLM dimension: [B, num_query, hidden_dim] -> [B, num_query, dim]
            vision_tokens = self.projector(vision_tokens)

            # 4. Scatter vision tokens into the sequence at <image> positions
            if special_tokens is not None:
                h = scatter_vision_tokens(h, tokens, vision_tokens, special_tokens.img_id)

        # Apply LLM transformer layers
        for layer in self.layers.values():
            h = layer(h, self.freqs_cis, attention_masks, positions)

        # Final norm and output projection
        h = self.norm(h) if self.norm is not None else h
        output = self.output(h) if self.output is not None else h

        return output
