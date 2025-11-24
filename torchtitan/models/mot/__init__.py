# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.optimizer import build_optimizers
from torchtitan.components.tokenizer import build_hf_tokenizer
from torchtitan.components.validate import build_validator
from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.hf_datasets.text_datasets import build_text_dataloader
from torchtitan.protocols.train_spec import TrainSpec

from .infra.parallelize import parallelize_mot
from .model.args import MoTModelArgs
from .model.model import MoTTransformer

__all__ = [
    "parallelize_mot",
    "MoTModelArgs",
    "MoTTransformer",
    "mot_args",
]


# Model presets with different configurations
mot_args = {
    # Debug model: Small 2-modality model for testing
    "debugmodel": MoTModelArgs(
        dim=256,
        n_layers=6,
        n_heads=16,
        vocab_size=2048,
        rope_theta=500000,
        n_modalities=2,
        modality_names=["text", "image"],
    ),
    # Debug model with flex attention
    "debugmodel_flex_attn": MoTModelArgs(
        dim=256,
        n_layers=6,
        n_heads=16,
        vocab_size=2048,
        rope_theta=500000,
        n_modalities=2,
        modality_names=["text", "image"],
        attn_type="flex",
        attn_mask_type="block_causal",
    ),
    # Debug model with varlen attention
    "debugmodel_varlen_attn": MoTModelArgs(
        dim=256,
        n_layers=6,
        n_heads=16,
        vocab_size=2048,
        rope_theta=500000,
        n_modalities=2,
        modality_names=["text", "image"],
        attn_type="varlen",
        attn_mask_type="block_causal",
    ),
    # 8B model with 2 modalities (text + image)
    # Similar to Chameleon setting
    "8B_2modality": MoTModelArgs(
        dim=4096,
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        multiple_of=1024,
        rope_theta=500000,
        n_modalities=2,
        modality_names=["text", "image"],
    ),
    # 8B model with 3 modalities (text + image + speech)
    "8B_3modality": MoTModelArgs(
        dim=4096,
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        multiple_of=1024,
        rope_theta=500000,
        n_modalities=3,
        modality_names=["text", "image", "speech"],
    ),
    # 70B model with 2 modalities
    "70B_2modality": MoTModelArgs(
        dim=8192,
        n_layers=80,
        n_heads=64,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        multiple_of=4096,
        rope_theta=500000,
        n_modalities=2,
        modality_names=["text", "image"],
    ),
    # 70B model with 3 modalities
    "70B_3modality": MoTModelArgs(
        dim=8192,
        n_layers=80,
        n_heads=64,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        multiple_of=4096,
        rope_theta=500000,
        n_modalities=3,
        modality_names=["text", "image", "speech"],
    ),
}


def get_train_spec() -> TrainSpec:
    """
    Get the training specification for MoT model.

    Returns a TrainSpec that defines all components needed for training:
    - Model class and preset configurations
    - Parallelization function
    - Optimizer, scheduler, dataloader, tokenizer builders
    - Loss function
    """
    return TrainSpec(
        model_cls=MoTTransformer,
        model_args=mot_args,
        parallelize_fn=parallelize_mot,
        pipelining_fn=pipeline_llm,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_text_dataloader,  # Note: Will need custom dataloader for modality_masks
        build_tokenizer_fn=build_hf_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
        build_validator_fn=build_validator,
        state_dict_adapter=None,  # Can add later for HF checkpoint conversion
    )
