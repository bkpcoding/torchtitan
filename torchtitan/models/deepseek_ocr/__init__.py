# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
DeepSeek-OCR experiment: Vision-Language OCR model combining
SAM ViT-B encoder, Qwen2-style query encoder, and DeepSeek-V3 MoE backbone.
"""

from dataclasses import fields
from typing import Any

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.optimizer import build_optimizers_with_moe_load_balancing
from torchtitan.components.tokenizer import build_hf_tokenizer
from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.models.moe import MoEArgs
from torchtitan.protocols.train_spec import TrainSpec

from .datasets.ocr_datasets import build_ocr_dataloader
from .infra.parallelize import parallelize_deepseek_ocr
from .model.args import (
    DeepSeekOCRModelArgs,
    ProjectorArgs,
    QueryEncoderArgs,
    SAMEncoderArgs,
)
from .model.model import DeepSeekOCRTransformer

__all__ = [
    "parallelize_deepseek_ocr",
    "DeepSeekOCRModelArgs",
    "DeepSeekOCRTransformer",
    "deepseek_ocr_args",
]


def _get_dict(obj) -> dict[str, Any]:
    """Convert dataclass to dict, preserving nested dataclasses (unlike asdict)."""
    return {field.name: getattr(obj, field.name) for field in fields(obj)}


# Model configurations
deepseek_ocr_args = {
    # Debug model for testing
    "debugmodel": DeepSeekOCRModelArgs(
        # LLM backbone (small for debugging)
        vocab_size=2048,
        dim=256,
        inter_dim=1024,
        moe_inter_dim=256,
        n_layers=4,
        n_dense_layers=1,
        n_heads=8,
        moe_args=MoEArgs(
            num_experts=4,
            num_shared_experts=1,
            top_k=2,
            score_func="softmax",
            route_norm=False,
            score_before_experts=False,
        ),
        q_lora_rank=0,
        kv_lora_rank=256,
        qk_nope_head_dim=64,
        qk_rope_head_dim=32,
        v_head_dim=64,
        mscale=0.70,
        # Vision encoder (small for debugging)
        sam_encoder=SAMEncoderArgs(
            img_size=256,  # Small images for debugging
            patch_size=16,
            embed_dim=128,
            depth=4,
            num_heads=4,
            mlp_ratio=4.0,
            out_chans=64,
            window_size=7,
            global_attn_indexes=(1, 3),
        ),
        # Query encoder (small for debugging)
        query_encoder=QueryEncoderArgs(
            hidden_dim=128,
            num_layers=4,
            num_heads=4,
            num_kv_heads=2,
            intermediate_size=512,
            num_query_tokens_768=36,  # 6x6 for debug
            num_query_tokens_1024=64,  # 8x8 for debug
        ),
        # Projector
        projector=ProjectorArgs(projector_type="mlp_gelu"),
        vision_output_dim=128,
    ),
    # Full debug model with original SAM sizes
    "debugmodel_full_vision": DeepSeekOCRModelArgs(
        # LLM backbone (small for debugging)
        vocab_size=2048,
        dim=256,
        inter_dim=1024,
        moe_inter_dim=256,
        n_layers=4,
        n_dense_layers=1,
        n_heads=8,
        moe_args=MoEArgs(
            num_experts=4,
            num_shared_experts=1,
            top_k=2,
            score_func="softmax",
            route_norm=False,
            score_before_experts=False,
        ),
        q_lora_rank=0,
        kv_lora_rank=256,
        qk_nope_head_dim=64,
        qk_rope_head_dim=32,
        v_head_dim=64,
        mscale=0.70,
        # Vision encoder (original SAM ViT-B)
        sam_encoder=SAMEncoderArgs(),  # Default = SAM ViT-B
        # Query encoder (original Qwen2 style)
        query_encoder=QueryEncoderArgs(),  # Default = 24 layers, 896 dim
        # Projector
        projector=ProjectorArgs(projector_type="mlp_gelu"),
        vision_output_dim=896,
    ),
    # 16B model variant
    "16B": DeepSeekOCRModelArgs(
        # LLM backbone (same as DeepSeek-V3 16B)
        vocab_size=102400,
        dim=2048,
        inter_dim=10944,
        moe_inter_dim=1408,
        n_layers=27,
        n_dense_layers=1,
        n_heads=16,
        moe_args=MoEArgs(
            num_experts=64,
            num_shared_experts=2,
            top_k=6,
            score_func="softmax",
            route_norm=False,
            score_before_experts=False,
        ),
        q_lora_rank=0,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        mscale=0.70,
        attn_type="sdpa",
        attn_mask_type="causal",
        # Vision encoder (SAM ViT-B)
        sam_encoder=SAMEncoderArgs(),
        # Query encoder (Qwen2-style, 24 layers)
        query_encoder=QueryEncoderArgs(),
        # Projector
        projector=ProjectorArgs(projector_type="mlp_gelu"),
        vision_output_dim=896,
    ),
}


def get_train_spec() -> TrainSpec:
    """Get the TrainSpec for DeepSeek-OCR experiment."""
    return TrainSpec(
        model_cls=DeepSeekOCRTransformer,
        model_args=deepseek_ocr_args,
        parallelize_fn=parallelize_deepseek_ocr,
        pipelining_fn=pipeline_llm,
        build_optimizers_fn=build_optimizers_with_moe_load_balancing,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_ocr_dataloader,
        build_tokenizer_fn=build_hf_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
    )
