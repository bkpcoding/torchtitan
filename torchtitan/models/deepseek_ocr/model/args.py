# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import Tuple

from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.models.deepseek_v3.model.args import DeepSeekV3ModelArgs


@dataclass
class SpecialTokens:
    """Special tokens for OCR VLM."""

    img_token: str
    img_id: int
    boi_token: str
    boi_id: int
    eoi_token: str
    eoi_id: int
    pad_token: str
    pad_id: int
    ignore_id: int = -100  # Pytorch F.cross_entropy default

    @classmethod
    def from_tokenizer(cls, tokenizer: HuggingFaceTokenizer):
        SPECIAL_TOKENS_MAP = {
            "img": "<|image|>",
            "boi": "<|begin_of_image|>",
            "eoi": "<|end_of_image|>",
            "pad": "<|pad|>",
        }
        added_tokens = tokenizer.tokenizer.get_added_tokens_decoder()
        token_to_id = {tok.content: tok_id for tok_id, tok in added_tokens.items()}
        special_tokens_dict = {}
        for prefix, tok in SPECIAL_TOKENS_MAP.items():
            special_tokens_dict[f"{prefix}_token"] = tok
            special_tokens_dict[f"{prefix}_id"] = token_to_id[tok]
        return cls(**special_tokens_dict)


@dataclass
class SAMEncoderArgs:
    """Arguments for SAM ViT-B vision encoder."""

    img_size: int = 1024
    patch_size: int = 16
    in_chans: int = 3
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    out_chans: int = 256
    qkv_bias: bool = True
    use_abs_pos: bool = True
    use_rel_pos: bool = True
    rel_pos_zero_init: bool = True
    window_size: int = 14
    global_attn_indexes: Tuple[int, ...] = (2, 5, 8, 11)


@dataclass
class QueryEncoderArgs:
    """Arguments for Qwen2-style query encoder with mixed attention masks."""

    hidden_dim: int = 896
    num_layers: int = 24
    num_heads: int = 14
    num_kv_heads: int = 2
    intermediate_size: int = 4864
    num_query_tokens_768: int = 144  # For 768x768 images (12x12 grid after SAM)
    num_query_tokens_1024: int = 256  # For 1024x1024 images (16x16 grid after SAM)
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    attention_dropout: float = 0.0


@dataclass
class ProjectorArgs:
    """Arguments for MLP projector."""

    projector_type: str = "mlp_gelu"  # "linear" or "mlp_gelu"


@dataclass
class DeepSeekOCRModelArgs(DeepSeekV3ModelArgs):
    """
    Model arguments for DeepSeek-OCR combining SAM ViT-B, Qwen2 query encoder,
    and DeepSeek-V3 MoE backbone.
    """

    # Vision encoder (SAM ViT-B)
    sam_encoder: SAMEncoderArgs = field(default_factory=SAMEncoderArgs)

    # Query encoder (Qwen2-style decoder-as-encoder)
    query_encoder: QueryEncoderArgs = field(default_factory=QueryEncoderArgs)

    # Projector
    projector: ProjectorArgs = field(default_factory=ProjectorArgs)

    # Vision-specific settings
    # The SAM encoder outputs 896-dim features after conv layers
    # These get projected to match LLM dim
    vision_output_dim: int = 896  # SAM output after neck + conv layers

    def get_projector_dims(self) -> Tuple[int, int]:
        """Get projector input and output dimensions."""
        return self.query_encoder.hidden_dim, self.dim
