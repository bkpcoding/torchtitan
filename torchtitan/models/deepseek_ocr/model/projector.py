# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

from .args import ProjectorArgs


class LinearProjector(nn.Module):
    """Simple linear projection from vision to LLM dimension."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)

    def init_weights(self) -> None:
        nn.init.xavier_uniform_(self.proj.weight)


class MLPProjector(nn.Module):
    """
    MLP projector with GELU activation.

    Projects vision encoder output to LLM embedding dimension.
    Architecture: Linear -> GELU -> Linear
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(in_dim, in_dim, bias=True)
        self.w2 = nn.Linear(in_dim, out_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.w1(x)
        x = nn.functional.gelu(x)
        x = self.w2(x)
        return x

    def init_weights(self) -> None:
        nn.init.xavier_uniform_(self.w1.weight)
        if self.w1.bias is not None:
            nn.init.zeros_(self.w1.bias)
        nn.init.xavier_uniform_(self.w2.weight)
        if self.w2.bias is not None:
            nn.init.zeros_(self.w2.bias)


class SiLUProjector(nn.Module):
    """
    MLP projector with SiLU activation.

    Same architecture as used in torchtitan's VLM.
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(in_dim, in_dim, bias=True)
        self.w2 = nn.Linear(in_dim, out_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.w1(x)
        x = nn.functional.silu(x)
        x = self.w2(x)
        return x

    def init_weights(self) -> None:
        nn.init.xavier_uniform_(self.w1.weight)
        if self.w1.bias is not None:
            nn.init.zeros_(self.w1.bias)
        nn.init.xavier_uniform_(self.w2.weight)
        if self.w2.bias is not None:
            nn.init.zeros_(self.w2.bias)


def build_projector(args: ProjectorArgs, in_dim: int, out_dim: int) -> nn.Module:
    """
    Build a projector module based on configuration.

    Args:
        args: Projector configuration.
        in_dim: Input dimension (from query encoder).
        out_dim: Output dimension (LLM embedding dimension).

    Returns:
        Projector module.
    """
    if args.projector_type == "linear":
        return LinearProjector(in_dim, out_dim)
    elif args.projector_type == "mlp_gelu":
        return MLPProjector(in_dim, out_dim)
    elif args.projector_type == "mlp_silu":
        return SiLUProjector(in_dim, out_dim)
    else:
        raise ValueError(f"Unknown projector type: {args.projector_type}")
