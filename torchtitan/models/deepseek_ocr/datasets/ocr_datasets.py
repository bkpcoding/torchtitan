# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
OCR Dataset implementation for DeepSeek-OCR training.

This module provides dataset classes for handling OCR data including images and text.
It supports streaming datasets from HuggingFace.
"""

from dataclasses import asdict
import os
import json
from io import BytesIO
from typing import Any, Callable, Optional

import torch
from datasets import Dataset, load_dataset
from datasets.distributed import split_dataset_by_node
from PIL import Image
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset
from torchvision import transforms

from torchtitan.components.dataloader import ParallelAwareDataloader
from torchtitan.components.tokenizer import BaseTokenizer, HuggingFaceTokenizer
from torchtitan.config import JobConfig
from torchtitan.hf_datasets import DatasetConfig
from torchtitan.tools.logging import logger

from ..model.args import SpecialTokens


def _get_image_transform(img_size: int = 1024) -> transforms.Compose:
    """Get image preprocessing transforms for SAM encoder."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def _process_ocr_sample(
    sample: dict[str, Any],
    tokenizer: BaseTokenizer,
    img_size: int,
    seq_len: int,
    special_tokens: SpecialTokens,
    image_transform: transforms.Compose,
) -> dict[str, Any] | None:
    """
    Process a single OCR sample.

    Expected sample format:
    {
        "image": PIL Image or bytes,
        "text": str (the OCR text / ground truth)
    }

    Returns:
    {
        "input_ids": torch.Tensor [S],
        "labels": torch.Tensor [S],
        "pixel_values": torch.Tensor [C, H, W],
    }
    """
    try:
        # Get image
        image = sample.get("image")
        text = sample.get("text", "")

        if image is None:
            return None

        # Convert image to PIL if needed
        if isinstance(image, bytes):
            image = Image.open(BytesIO(image)).convert("RGB")
        elif hasattr(image, "convert"):
            image = image.convert("RGB")
        else:
            return None

        # Apply image transform
        pixel_values = image_transform(image)

        # Create text sequence: <boi><image>...<eoi> text
        # We need to reserve space for image tokens
        # For now, we use a single <image> placeholder that will be replaced
        # with the query tokens (144 or 256) during forward pass
        num_image_tokens = 256 if img_size == 1024 else 144

        # Build the input text with image placeholders
        img_placeholder = special_tokens.img_token * num_image_tokens
        input_text = f"{special_tokens.boi_token}{img_placeholder}{special_tokens.eoi_token}{text}"

        # Tokenize
        tokens = tokenizer.encode(input_text)

        # Truncate if needed
        if len(tokens) > seq_len:
            tokens = tokens[:seq_len]

        input_ids = torch.tensor(tokens, dtype=torch.long)
        labels = torch.tensor(tokens, dtype=torch.long)

        # Mask special tokens in labels (we don't predict image tokens)
        special_token_ids = torch.tensor([
            special_tokens.boi_id,
            special_tokens.eoi_id,
            special_tokens.img_id,
        ])
        labels = torch.where(
            torch.isin(labels, special_token_ids),
            special_tokens.ignore_id,
            labels,
        )

        return {
            "input_ids": input_ids,
            "labels": labels,
            "pixel_values": pixel_values,
        }

    except Exception as e:
        logger.warning(f"Error processing OCR sample: {e}")
        return None


def _process_synthdog_sample(
    sample: dict[str, Any],
    tokenizer: BaseTokenizer,
    img_size: int,
    seq_len: int,
    special_tokens: SpecialTokens,
    image_transform: transforms.Compose,
) -> dict[str, Any] | None:
    """Process a sample from SynthDOG dataset."""
    # SynthDOG format: {"image": ..., "ground_truth": {"gt_parse": ...}}
    gt = sample.get("ground_truth", {})
    if isinstance(gt, str):
        try:
            gt = json.loads(gt)
        except json.JSONDecodeError:
            gt = {}
    text = gt.get("gt_parse", {}).get("text_sequence", "")

    return _process_ocr_sample(
        {"image": sample.get("image"), "text": text},
        tokenizer=tokenizer,
        img_size=img_size,
        seq_len=seq_len,
        special_tokens=special_tokens,
        image_transform=image_transform,
    )


# Dataset configurations
OCR_DATASETS = {
    "synthdog-en": DatasetConfig(
        path="naver-clova-ix/synthdog-en",
        loader=lambda path: load_dataset(path, split="train", streaming=True),
        sample_processor=_process_synthdog_sample,
    ),
}


def _validate_ocr_dataset(
    dataset_name: str, dataset_path: str | None = None
) -> tuple[str, Callable, Callable]:
    """Validate dataset name and path."""
    if dataset_name not in OCR_DATASETS:
        raise ValueError(
            f"Dataset {dataset_name} is not supported. "
            f"Supported datasets are: {list(OCR_DATASETS.keys())}"
        )

    config = OCR_DATASETS[dataset_name]
    path = dataset_path or config.path
    logger.info(f"Preparing {dataset_name} dataset from {path}")
    return path, config.loader, config.sample_processor


class OCRCollator:
    """Collator for OCR batches."""

    def __init__(
        self,
        batch_size: int,
        seq_len: int,
        special_tokens: SpecialTokens,
    ):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.special_tokens = special_tokens

    def __call__(self, batch: list[dict[str, Any]]) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Collate a batch of samples."""
        # Filter out None samples
        batch = [s for s in batch if s is not None]
        if not batch:
            raise ValueError("Empty batch after filtering")

        # Pad sequences
        input_ids = []
        labels = []
        pixel_values = []

        for sample in batch:
            ids = sample["input_ids"]
            lbl = sample["labels"]

            # Pad to seq_len
            pad_len = self.seq_len - ids.shape[0]
            if pad_len > 0:
                ids = torch.cat([ids, torch.full((pad_len,), self.special_tokens.pad_id)])
                lbl = torch.cat([lbl, torch.full((pad_len,), self.special_tokens.ignore_id)])

            input_ids.append(ids)
            labels.append(lbl)
            pixel_values.append(sample["pixel_values"])

        input_ids = torch.stack(input_ids)
        labels = torch.stack(labels)
        images = torch.stack(pixel_values)

        # Shift for next-token prediction (standard LM objective)
        input_ids = input_ids[:, :-1]
        labels = labels[:, 1:]

        if os.getenv("TORCHTITAN_DISABLE_OCR_IMAGES") == "1":
            images = None
        input_dict = {
            "input": input_ids,
            "images": images,
            "special_tokens": self.special_tokens,
        }
        return input_dict, labels


class HuggingFaceOCRDataset(IterableDataset, Stateful):
    """HuggingFace OCR Dataset."""

    def __init__(
        self,
        dataset_name: str,
        dataset_path: str | None,
        tokenizer: BaseTokenizer,
        batch_size: int,
        seq_len: int,
        img_size: int,
        special_tokens: SpecialTokens,
        dp_rank: int = 0,
        dp_world_size: int = 1,
        infinite: bool = False,
    ) -> None:
        dataset_name = dataset_name.lower()

        path, dataset_loader, self.sample_processor = _validate_ocr_dataset(
            dataset_name, dataset_path
        )
        ds = dataset_loader(path)
        self._data = split_dataset_by_node(ds, dp_rank, dp_world_size)

        self._tokenizer = tokenizer
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.img_size = img_size
        self.special_tokens = special_tokens
        self.image_transform = _get_image_transform(img_size)
        self.infinite = infinite
        self._sample_idx = 0

    def __iter__(self):
        while True:
            for sample in self._get_data_iter():
                try:
                    self._sample_idx += 1

                    processed = self.sample_processor(
                        sample=sample,
                        tokenizer=self._tokenizer,
                        img_size=self.img_size,
                        seq_len=self.seq_len,
                        special_tokens=self.special_tokens,
                        image_transform=self.image_transform,
                    )
                    if processed is None:
                        continue

                    yield processed

                except Exception as e:
                    logger.warning(f"Error in iteration: {e}")
                    continue

            if not self.infinite:
                break
            else:
                self._sample_idx = 0

    def _get_data_iter(self):
        try:
            if not hasattr(self._data, "iterable_dataset"):
                if isinstance(self._data, Dataset) and (
                    self._sample_idx == len(self._data)
                ):
                    return iter([])

            it = iter(self._data)

            if self._sample_idx > 0:
                for _ in range(self._sample_idx):
                    next(it)

            return it
        except Exception as e:
            logger.error(f"Error in _get_data_iter: {e}")
            return iter([])

    def load_state_dict(self, state_dict):
        self._sample_idx = state_dict["sample_idx"]

    def state_dict(self):
        return {"sample_idx": self._sample_idx}


def build_ocr_dataloader(
    dp_world_size: int,
    dp_rank: int,
    tokenizer: HuggingFaceTokenizer,
    job_config: JobConfig,
    infinite: bool = True,
) -> ParallelAwareDataloader:
    """
    Build a data loader for OCR datasets.

    Args:
        dp_world_size: Data parallel world size.
        dp_rank: Data parallel rank.
        tokenizer: Tokenizer for text processing.
        job_config: Job configuration.
        infinite: Whether to loop infinitely.

    Returns:
        DataLoader with appropriate parallelism handling.
    """
    dataset_path = job_config.training.dataset_path
    batch_size = job_config.training.local_batch_size
    seq_len = job_config.training.seq_len

    # Get image size from data config or use default
    img_size = getattr(job_config.data, "img_size", 1024)

    special_tokens = SpecialTokens.from_tokenizer(tokenizer)

    dataset = HuggingFaceOCRDataset(
        dataset_name=job_config.training.dataset,
        dataset_path=dataset_path,
        tokenizer=tokenizer,
        batch_size=batch_size,
        seq_len=seq_len,
        img_size=img_size,
        special_tokens=special_tokens,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        infinite=infinite,
    )

    collate_fn = OCRCollator(
        batch_size=batch_size,
        seq_len=seq_len,
        special_tokens=special_tokens,
    )

    dataloader_kwargs = {
        **asdict(job_config.training.dataloader),
        "batch_size": batch_size,
        "collate_fn": collate_fn,
    }

    base_dataloader = ParallelAwareDataloader(
        dataset=dataset,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        **dataloader_kwargs,
    )

    return base_dataloader
