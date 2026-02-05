# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from .ocr_datasets import (
    build_ocr_dataloader,
    HuggingFaceOCRDataset,
    OCR_DATASETS,
    OCRCollator,
)

__all__ = [
    "build_ocr_dataloader",
    "HuggingFaceOCRDataset",
    "OCR_DATASETS",
    "OCRCollator",
]
