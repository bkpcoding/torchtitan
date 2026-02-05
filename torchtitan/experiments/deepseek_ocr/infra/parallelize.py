# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Parallelization for DeepSeek-OCR Vision-Language Model.

This module applies parallelism strategies to the DeepSeek-OCR model:
- SAM encoder: FSDP only (relatively small, no TP needed)
- Query encoder: FSDP (TP can be added later)
- LLM backbone: Full parallelization (TP + EP + FSDP) via DeepSeek-V3 parallelization
"""

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard, MixedPrecisionPolicy

from torchtitan.config import JobConfig, TORCH_DTYPE_MAP
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import apply_ac
from torchtitan.models.deepseek_v3.infra.parallelize import (
    _op_sac_save_list,
    apply_non_moe_tp,
)
from torchtitan.models.llama3.infra.parallelize import (
    apply_compile,
    apply_ddp,
    disable_fsdp_gradient_division,
)
from torchtitan.models.llama4.infra.parallelize import apply_fsdp, apply_moe_ep_tp
from torchtitan.tools.logging import logger


def parallelize_deepseek_ocr(
    model: nn.Module,
    parallel_dims: ParallelDims,
    job_config: JobConfig,
):
    """
    Apply parallelization to DeepSeek-OCR model.

    Parallelization strategy:
    1. Vision components (SAM encoder, query encoder, projector): FSDP
    2. LLM backbone: TP + EP + FSDP (reuses DeepSeek-V3 parallelization)

    Args:
        model: DeepSeekOCRTransformer model instance.
        parallel_dims: Parallel dimensions configuration.
        job_config: Job configuration.

    Returns:
        Parallelized model.
    """
    assert (
        job_config.training.seq_len % parallel_dims.seq_len_divisor == 0
    ), f"""
        Sequence length {job_config.training.seq_len} must be divisible by the product of TP degree
        ({parallel_dims.tp}) and 2 * CP degree ({parallel_dims.cp}).
        """

    attn_type = getattr(model.model_args, "attn_type", "sdpa")
    if job_config.parallelism.context_parallel_degree > 1 and attn_type != "sdpa":
        raise NotImplementedError(
            f"Context Parallel only supports SDPA attention. "
            f"Got attn_type='{attn_type}'. "
            f"FlexAttention and varlen attention are not supported with CP."
        )

    # Apply TP to LLM backbone (non-MoE layers)
    if parallel_dims.tp_enabled:
        tp_mesh = parallel_dims.get_mesh("tp")
        apply_non_moe_tp(
            model,
            tp_mesh,
            loss_parallel=not job_config.parallelism.disable_loss_parallel,
            enable_float8_tensorwise_tp=False,
            cp_enabled=parallel_dims.cp_enabled,
        )

    # Apply EP + TP to MoE layers
    if parallel_dims.tp_enabled or parallel_dims.ep_enabled:
        apply_moe_ep_tp(
            model,
            tp_mesh=parallel_dims.get_optional_mesh("tp"),
            ep_mesh=parallel_dims.get_optional_mesh("ep"),
            etp_mesh=parallel_dims.get_optional_mesh("etp"),
            ep_etp_mesh=parallel_dims.get_optional_mesh(["ep", "etp"]),
            dual_pipe_v=False,  # Not using dual pipe for VLM
            use_deepep=False,
        )

    # Apply activation checkpointing
    model_compile_enabled = (
        job_config.compile.enable and "model" in job_config.compile.components
    )
    if job_config.activation_checkpoint.mode != "none":
        apply_ac(
            model,
            job_config.activation_checkpoint,
            model_compile_enabled=model_compile_enabled,
            op_sac_save_list=_op_sac_save_list,
        )
        # Also apply AC to vision components if they exist
        if model.sam_encoder is not None:
            apply_ac(model.sam_encoder, job_config.activation_checkpoint)
        if model.query_encoder is not None:
            apply_ac(model.query_encoder, job_config.activation_checkpoint)

    # Apply compile
    if model_compile_enabled:
        apply_compile(model, job_config.compile)
        if model.sam_encoder is not None:
            apply_compile(model.sam_encoder, job_config.compile)
        if model.query_encoder is not None:
            apply_compile(model.query_encoder, job_config.compile)

    # Apply FSDP
    dp_mesh: DeviceMesh | None = None
    if parallel_dims.fsdp_enabled or parallel_dims.ep_enabled:
        dp_mesh_names = (
            ["dp_replicate", "fsdp"] if parallel_dims.dp_replicate_enabled else ["fsdp"]
        )
        dp_mesh = parallel_dims.get_mesh(dp_mesh_names)

        edp_mesh_names = (
            ["dp_replicate", "efsdp"]
            if parallel_dims.dp_replicate_enabled
            else ["efsdp"]
        )
        edp_mesh = parallel_dims.get_optional_mesh(edp_mesh_names)

        apply_fsdp_to_deepseek_ocr(
            model,
            dp_mesh,
            param_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_reduce],
            pp_enabled=parallel_dims.pp_enabled,
            cpu_offload=job_config.training.enable_cpu_offload,
            reshard_after_forward_policy=job_config.parallelism.fsdp_reshard_after_forward,
            ep_degree=parallel_dims.ep,
            edp_mesh=edp_mesh,
            gradient_divide_factor=parallel_dims.fsdp_gradient_divide_factor,
        )

        if parallel_dims.dp_replicate_enabled:
            logger.info("Applied HSDP to the model")
        else:
            logger.info("Applied FSDP to the model")

        if job_config.training.enable_cpu_offload:
            logger.info("Applied CPU Offloading to the model")

    elif parallel_dims.dp_replicate_enabled:
        dp_mesh = parallel_dims.get_mesh("dp_replicate")
        if dp_mesh.ndim > 1:
            raise RuntimeError("DDP has not supported > 1D parallelism")
        apply_ddp(
            model,
            dp_mesh,
            enable_compile=model_compile_enabled,
        )

    return model


def apply_fsdp_to_deepseek_ocr(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    pp_enabled: bool,
    cpu_offload: bool = False,
    reshard_after_forward_policy: str = "default",
    ep_degree: int = 1,
    edp_mesh: DeviceMesh | None = None,
    gradient_divide_factor: float = 1.0,
):
    """
    Apply FSDP to DeepSeek-OCR model.

    This function applies FSDP to:
    1. SAM encoder blocks
    2. Query encoder layers
    3. Projector
    4. LLM token embeddings
    5. LLM transformer blocks (with special handling for MoE)
    6. LLM output layers

    Args:
        model: DeepSeekOCRTransformer model.
        dp_mesh: Data parallel device mesh.
        param_dtype: Parameter data type for mixed precision.
        reduce_dtype: Reduction data type for mixed precision.
        pp_enabled: Whether pipeline parallelism is enabled.
        cpu_offload: Whether to enable CPU offloading.
        reshard_after_forward_policy: FSDP reshard policy.
        ep_degree: Expert parallelism degree.
        edp_mesh: Expert data parallel mesh.
        gradient_divide_factor: Factor to divide gradients by.
    """
    mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
    fsdp_config = {"mesh": dp_mesh, "mp_policy": mp_policy}
    if cpu_offload:
        fsdp_config["offload_policy"] = CPUOffloadPolicy()

    match reshard_after_forward_policy:
        case "always":
            reshard_after_forward = True
        case "never":
            reshard_after_forward = False
        case "default":
            reshard_after_forward = not pp_enabled
        case _:
            raise ValueError(
                f"Invalid reshard_after_forward_policy: {reshard_after_forward_policy}."
            )

    # 1. FSDP for SAM encoder blocks
    if model.sam_encoder is not None:
        for block in model.sam_encoder.blocks:
            fully_shard(
                block,
                **fsdp_config,
                reshard_after_forward=reshard_after_forward,
            )
        # Shard the whole SAM encoder
        fully_shard(
            model.sam_encoder,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )

    # 2. FSDP for query encoder layers
    if model.query_encoder is not None:
        for layer in model.query_encoder.layers:
            fully_shard(
                layer,
                **fsdp_config,
                reshard_after_forward=reshard_after_forward,
            )
        # Shard the whole query encoder
        fully_shard(
            model.query_encoder,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )

    # 3. FSDP for projector
    if model.projector is not None:
        fully_shard(
            model.projector,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )

    # 4. FSDP for LLM backbone (reuse logic from DeepSeek-V3)
    # Token embeddings
    if model.tok_embeddings is not None:
        fully_shard(
            model.tok_embeddings,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )

    # Transformer blocks
    for layer_id, transformer_block in model.layers.items():
        if hasattr(transformer_block, "moe_enabled") and transformer_block.moe_enabled:
            # MoE layer: use expert data parallel mesh if available
            if ep_degree > 1 and edp_mesh is not None:
                fully_shard(
                    transformer_block,
                    mesh=edp_mesh,
                    mp_policy=mp_policy,
                    reshard_after_forward=reshard_after_forward,
                )
            else:
                fully_shard(
                    transformer_block,
                    **fsdp_config,
                    reshard_after_forward=reshard_after_forward,
                )
        else:
            # Dense layer
            fully_shard(
                transformer_block,
                **fsdp_config,
                reshard_after_forward=reshard_after_forward,
            )

    # Output layers
    if model.norm is not None and model.output is not None:
        fully_shard(
            [model.norm, model.output],
            **fsdp_config,
            reshard_after_forward=reshard_after_forward_policy == "always",
        )

    # Shard the whole model
    fully_shard(model, **fsdp_config)

    # Disable FSDP's automatic gradient division
    disable_fsdp_gradient_division(model)
