# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

from torch import nn
from torchtitan.config import JobConfig
from torchtitan.models.llama3.model.args import RoPEScalingArgs
from torchtitan.models.utils import get_dense_model_nparams_and_flops
from torchtitan.protocols.model import BaseModelArgs
from torchtitan.tools.logging import logger


@dataclass
class MoTModelArgs(BaseModelArgs):
    """
    Configuration for Mixture-of-Transformers (MoT) model.

    MoT uses modality-specific parameters in each transformer layer,
    enabling efficient multi-modal generation with sparse computation.
    """

    # Base transformer architecture
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: int | None = None
    vocab_size: int = 128256
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: float | None = None
    norm_eps: float = 1e-5

    # MoT-specific: Modality configuration
    n_modalities: int = 2  # Number of modalities (e.g., 2 for text+image)
    modality_names: list[str] = field(
        default_factory=lambda: ["text", "image"]
    )

    # Position encoding (RoPE)
    rope_theta: float = 10000
    rope_scaling_args: RoPEScalingArgs = field(default_factory=RoPEScalingArgs)

    # Attention configuration
    attn_type: str = "sdpa"  # sdpa, flex, varlen
    attn_mask_type: str = "causal"
    qk_normalization: bool = False  # Per-head QK normalization

    # Context length
    max_seq_len: int = 131072

    # Initialization
    depth_init: bool = True  # Use layer-specific or uniform initialization

    # Special tokens
    eos_id: int = 0

    def update_from_config(self, job_config: JobConfig, **kwargs) -> None:
        """Update model args based on job configuration."""
        seq_len = job_config.training.seq_len
        if seq_len > self.max_seq_len:
            logger.warning(
                f"Sequence length {seq_len} exceeds original maximum {self.max_seq_len}."
            )
        self.max_seq_len = seq_len

        if (
            job_config.parallelism.context_parallel_degree > 1
            and self.attn_type != "sdpa"
        ):
            raise NotImplementedError(
                "CP support for FlexAttention is still in progress."
            )

    def get_nparams_and_flops(
        self, model: nn.Module, seq_len: int
    ) -> tuple[int, float]:
        """
        Calculate total parameters and FLOPs for MoT model.

        Note: MoT has n_modalities * parameters per layer, but only processes
        each token through one modality's parameters (sparse activation).

        For now, we report total parameters (all modalities) but note that
        active FLOPs depend on modality distribution in the data.
        """
        # Use dense model calculation as baseline
        # In practice, active FLOPs = (1/n_modalities) * total_flops
        return get_dense_model_nparams_and_flops(
            self,
            model,
            2 * (self.dim // self.n_heads),
            seq_len,
        )
