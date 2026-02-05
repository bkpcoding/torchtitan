# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from .parallelize import apply_fsdp_to_deepseek_ocr, parallelize_deepseek_ocr

__all__ = [
    "parallelize_deepseek_ocr",
    "apply_fsdp_to_deepseek_ocr",
]
