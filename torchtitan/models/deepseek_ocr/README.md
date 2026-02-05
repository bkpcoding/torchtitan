# DeepSeek-OCR: Vision-Language OCR Model Training

**Status: Experimental**

This experiment integrates the DeepSeek-OCR-2 architecture into TorchTitan for training vision-language OCR models from scratch with the DeepSeek-V3 MoE backbone.

## Architecture Overview

DeepSeek-OCR uses a **query-based bottleneck** architecture that differs from standard VLMs:

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────┐     ┌──────────────────┐
│   SAM ViT-B     │ ──▶ │  Query Encoder   │ ──▶ │  Projector  │ ──▶ │ DeepSeek-V3 MoE  │
│  (bi-directional)│     │ (mixed attention) │     │   (MLP)     │     │   (LLM backbone) │
└─────────────────┘     └──────────────────┘     └─────────────┘     └──────────────────┘
       768-dim              896-dim                 LLM-dim               Output
```

### Key Components

1. **SAM ViT-B Encoder**: Processes images with bi-directional attention (all patches can attend to each other)
2. **Query Encoder**: Qwen2-style decoder-as-encoder with mixed attention masks:
   - Image tokens: bi-directional attention
   - Query tokens: causal attention + cross-attend to ALL image tokens
   - **Only query tokens are output** (bottleneck)
3. **MLP Projector**: Projects query encoder output (896-dim) to LLM dimension
4. **DeepSeek-V3 MoE Backbone**: Mixture-of-Experts LLM for text generation

### Mixed Attention Mask

The query encoder uses a special attention pattern:

```
              Image Tokens    Query Tokens
            ┌─────────────┬─────────────┐
Image       │             │             │
Tokens      │   ✓ (all)   │     ✗       │
            ├─────────────┼─────────────┤
Query       │             │   Causal    │
Tokens      │   ✓ (all)   │     ▽       │
            └─────────────┴─────────────┘
```

## Installation

### Prerequisites

- Python >= 3.10
- PyTorch nightly (recommended) or PyTorch >= 2.5
- CUDA 12.x (for GPU training)

### Setup Environment

```bash
# Clone the repository
git clone https://github.com/bkpcoding/torchtitan.git
cd torchtitan

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install PyTorch nightly (CUDA 12.6 example)
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu126

# Install torchtitan dependencies
pip install -r requirements.txt

# Install DeepSeek-OCR specific dependencies
pip install -r torchtitan/models/deepseek_ocr/requirements-deepseek-ocr.txt
```

### Verify Installation

```bash
python -c "from torchtitan.models.deepseek_ocr import get_train_spec; print('DeepSeek-OCR installed successfully!')"
```

## Dataset Setup

### SynthDOG-EN (Default)

SynthDOG is a synthetic document OCR dataset that's automatically downloaded via HuggingFace datasets.

```bash
# The dataset will be automatically streamed during training
# No manual download required for streaming mode
```

To pre-download the dataset:

```python
from datasets import load_dataset
ds = load_dataset("naver-clova-ix/synthdog-en", split="train")
```

### Custom OCR Dataset

To use your own dataset, add a new entry to `OCR_DATASETS` in `datasets/ocr_datasets.py`:

```python
OCR_DATASETS = {
    "your_dataset": DatasetConfig(
        path="path/to/your/dataset",
        loader=lambda path: load_dataset(path, split="train"),
        sample_processor=_process_your_sample,  # Define this function
    ),
}
```

Expected sample format:
```python
{
    "image": PIL.Image or bytes,
    "text": str  # OCR ground truth
}
```

### Tokenizer Setup

Download a tokenizer for training:

```bash
# Option 1: Use test tokenizer (for debugging only)
# Already included at tests/assets/tokenizer

# Option 2: Download Llama tokenizer (recommended for real training)
python scripts/download_hf_assets.py --repo_id meta-llama/Llama-3.1-8B --assets tokenizer --hf_token=YOUR_HF_TOKEN

# Option 3: Use Qwen tokenizer (matches original DeepSeek-OCR)
python scripts/download_hf_assets.py --repo_id Qwen/Qwen2-7B --assets tokenizer --hf_token=YOUR_HF_TOKEN
```

## Training

### Quick Start (Debug Model)

Single GPU:
```bash
python torchtitan/train.py --config torchtitan/models/deepseek_ocr/train_configs/debug.toml
```

Multi-GPU with FSDP:
```bash
torchrun --nproc_per_node=4 torchtitan/train.py \
    --config torchtitan/models/deepseek_ocr/train_configs/debug.toml
```

### Configuration Options

Key configuration parameters in `train_configs/debug.toml`:

```toml
[model]
name = "deepseek_ocr"
flavor = "debugmodel"  # Options: debugmodel, debugmodel_full_vision, 16B
hf_assets_path = "./tests/assets/tokenizer"

[training]
local_batch_size = 4
seq_len = 2048
steps = 10
dataset = "synthdog-en"

[data]
img_size = 256  # Use 1024 for full resolution

[parallelism]
data_parallel_shard_degree = -1  # FSDP
tensor_parallel_degree = 1
expert_parallel_degree = 1
```

### Model Flavors

| Flavor | Description | Use Case |
|--------|-------------|----------|
| `debugmodel` | Small model with reduced vision encoder | Testing & debugging |
| `debugmodel_full_vision` | Small LLM with full SAM ViT-B | Vision encoder testing |
| `16B` | Full 16B parameter model | Production training |

### Distributed Training

**FSDP (Data Parallel):**
```bash
torchrun --nproc_per_node=8 torchtitan/train.py \
    --config torchtitan/models/deepseek_ocr/train_configs/debug.toml \
    --parallelism.data_parallel_shard_degree=-1
```

**Expert Parallelism (for MoE layers):**
```bash
torchrun --nproc_per_node=8 torchtitan/train.py \
    --config torchtitan/models/deepseek_ocr/train_configs/debug.toml \
    --parallelism.expert_parallel_degree=2 \
    --parallelism.tensor_parallel_degree=2
```

**Multi-Node Training:**
```bash
# On each node
torchrun \
    --nnodes=2 \
    --nproc_per_node=8 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    torchtitan/train.py \
    --config torchtitan/models/deepseek_ocr/train_configs/debug.toml
```

## Verification

### 1. Check Model Initialization

```python
import torch
from torchtitan.models.deepseek_ocr import (
    DeepSeekOCRModelArgs,
    DeepSeekOCRTransformer,
    deepseek_ocr_args,
)

# Load debug model config
args = deepseek_ocr_args["debugmodel"]
print(f"LLM dim: {args.dim}")
print(f"SAM embed_dim: {args.sam_encoder.embed_dim}")
print(f"Query hidden_dim: {args.query_encoder.hidden_dim}")

# Initialize model on meta device
with torch.device("meta"):
    model = DeepSeekOCRTransformer(args)

print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
```

### 2. Verify Attention Masks

```python
from torchtitan.models.deepseek_ocr.model.query_encoder import QueryEncoder
from torchtitan.models.deepseek_ocr.model.args import QueryEncoderArgs

args = QueryEncoderArgs(hidden_dim=128, num_layers=2, num_heads=4, num_kv_heads=2)
encoder = QueryEncoder(args)

# Check mask structure
num_image = 16
num_query = 8
mask = encoder._create_mixed_attention_mask(
    num_image, num_query, torch.float32, "cpu", batch_size=1
)

print(f"Mask shape: {mask.shape}")
print("Image tokens (first 16 rows) should attend to all images (0.0):")
print(mask[0, 0, :num_image, :num_image].sum())  # Should be 0

print("Query tokens attend causally to queries + all images:")
print(mask[0, 0, num_image:, :])  # Lower triangular for queries + all images
```

### 3. Run Training Test

```bash
# Quick sanity check (10 steps)
python torchtitan/train.py \
    --config torchtitan/models/deepseek_ocr/train_configs/debug.toml \
    --training.steps=10 \
    --metrics.log_freq=1
```

Expected output:
```
Step 1: loss=X.XXX, ...
Step 2: loss=X.XXX, ...
...
Step 10: loss=X.XXX, ...
```

Loss should decrease over training steps.

## Project Structure

```
torchtitan/models/deepseek_ocr/
├── __init__.py              # TrainSpec registration & model configs
├── README.md                # This file
├── requirements-deepseek-ocr.txt  # Additional dependencies
├── model/
│   ├── __init__.py
│   ├── args.py              # DeepSeekOCRModelArgs, SAMEncoderArgs, etc.
│   ├── model.py             # DeepSeekOCRTransformer (main model)
│   ├── sam_encoder.py       # SAM ViT-B vision encoder
│   ├── query_encoder.py     # Qwen2-style query encoder with mixed masks
│   └── projector.py         # MLP projector variants
├── infra/
│   ├── __init__.py
│   └── parallelize.py       # TP/FSDP/EP parallelization
├── datasets/
│   ├── __init__.py
│   └── ocr_datasets.py      # OCR dataloader & preprocessing
└── train_configs/
    └── debug.toml           # Debug training configuration
```

## Known Limitations

- Pipeline Parallelism is not yet fully tested with vision components
- FlexAttention for the query encoder is not yet implemented (uses SDPA)
- Context Parallel is not supported for vision encoder

## References

- [DeepSeek-OCR-2 Paper](https://arxiv.org/abs/XXX) (if available)
- [DeepSeek-V3 Model](https://github.com/deepseek-ai/DeepSeek-V3)
- [SAM (Segment Anything)](https://github.com/facebookresearch/segment-anything)
- [TorchTitan Documentation](https://github.com/pytorch/torchtitan)

## License

This code is released under the BSD 3-Clause License, consistent with TorchTitan.
