# Mixture-of-Transformers (MoT)

Implementation of Mixture-of-Transformers for multi-modal foundation models in TorchTitan.

## Overview

Mixture-of-Transformers (MoT) enables efficient multi-modal generation by using **modality-specific sparsity** in every non-embedding transformer layer. Each modality (text, image, speech, etc.) gets its own dedicated parameters for:
- Attention projections (wq, wk, wv, wo)
- Feed-forward networks (w1, w2, w3)
- Layer normalization

**Key Results** (from the paper):
- **Chameleon setting (text + image):** MoT matches dense baseline quality using just **55.8% of the FLOPs**
- **3-modality setting (text + image + speech):** MoT achieves dense-level quality with only **37.2% of the FLOPs**

## Architecture

```
MoTTransformer
├── tok_embeddings (shared)
├── layers
│   ├── MoTTransformerBlock[0..N]
│   │   ├── ModalityUntiedAttention
│   │   │   ├── local_experts_wq[modality_id]
│   │   │   ├── local_experts_wk[modality_id]
│   │   │   ├── local_experts_wv[modality_id]
│   │   │   ├── local_experts_wo[modality_id]
│   │   │   └── local_experts_attention_norm[modality_id]
│   │   └── ModalityUntiedFeedForward
│   │       ├── local_experts[modality_id] (FFN)
│   │       └── local_experts_ffn_norm[modality_id]
├── norm (shared)
└── output (shared)
```

## Available Model Configurations

| Model | Modalities | Layers | Dim | Heads | Context Length |
|-------|-----------|--------|-----|-------|----------------|
| `debugmodel` | 2 (text, image) | 6 | 256 | 16 | 2048 |
| `debugmodel_flex_attn` | 2 | 6 | 256 | 16 | 2048 |
| `debugmodel_varlen_attn` | 2 | 6 | 256 | 16 | 2048 |
| `8B_2modality` | 2 (text, image) | 32 | 4096 | 32 | 8192 |
| `8B_3modality` | 3 (text, image, speech) | 32 | 4096 | 32 | 8192 |
| `70B_2modality` | 2 | 80 | 8192 | 64 | 8192 |
| `70B_3modality` | 3 | 80 | 8192 | 64 | 8192 |

## Usage

### Training with Debug Model

```bash
# Using the debug configuration
torchrun --nproc_per_node=1 train.py \
  --config torchtitan/models/mot/train_configs/debug_mot.toml
```

### Training with 8B Model

```bash
# 8B model with 2 modalities, 4-way tensor parallelism
torchrun --nproc_per_node=4 train.py \
  --config torchtitan/models/mot/train_configs/mot_8B_2modality.toml
```

### Custom Configuration

You can specify any model configuration in your TOML file:

```toml
[model]
name = "mot"
flavor = "8B_3modality"  # or any other preset
```

## Key Features

### 1. Modality-Specific Parameters
Each modality has dedicated parameters, enabling specialized processing:
```python
# Each token is routed to its modality-specific expert
for modality_id in range(n_modalities):
    expert_input = x[modality_masks[modality_id]]
    expert_output = local_experts[modality_id](expert_input)
```

### 2. Global Self-Attention
After modality-specific QKV projection, all modalities interact through global self-attention:
```python
# QKV from all modalities merged
xq = merge_modalities(expert_outputs_xq, modality_masks)
xk = merge_modalities(expert_outputs_xk, modality_masks)
xv = merge_modalities(expert_outputs_xv, modality_masks)

# Global attention across all modalities
output = attention(xq, xk, xv)
```

### 3. Deterministic Routing
Unlike MoE (which learns routing), MoT uses deterministic routing based on `modality_masks`:
```python
modality_masks = [
    torch.tensor([True, False, True, False]),  # Text tokens
    torch.tensor([False, True, False, True]),  # Image tokens
]
```

### 4. Distributed Training Support
Full support for TorchTitan's distributed training features:
- **Tensor Parallelism (TP)**: Each modality expert is independently sharded
- **FSDP/HSDP**: Data parallelism with mixed precision
- **Activation Checkpointing**: Memory-efficient training
- **torch.compile**: Optimized compilation
- **Context Parallelism**: For long sequences

## Implementation Requirements

### Modality Masks
The model requires `modality_masks` to be provided in the forward pass. Each mask is a boolean tensor indicating which tokens belong to which modality.

**Example:**
```python
# Batch size = 2, sequence length = 4, 2 modalities
tokens = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])  # (2, 4)

# Modality masks (flattened: bs*seqlen = 8)
modality_masks = [
    torch.tensor([True, True, False, False, True, False, False, True]),   # Text
    torch.tensor([False, False, True, True, False, True, True, False]),   # Image
]

logits = model(tokens, modality_masks)
```

### Custom Dataloader
**Important:** You will need to implement a custom dataloader that provides `modality_masks` along with input tokens. The current implementation uses the standard text dataloader, which needs to be extended for multi-modal data.

See `build_text_dataloader` in `torchtitan/hf_datasets/text_datasets.py` for reference on how to create a custom dataloader.

## Architecture Details

### Differences from Standard Transformer
1. **Norms inside modules**: Unlike standard transformers where norms are in `TransformerBlock`, MoT places norms inside attention/FFN modules (one per modality)
2. **No learned routing**: Uses deterministic routing via `modality_masks`, not learned routing like MoE
3. **Shared embeddings**: Token embeddings and output projections are shared across modalities

### Differences from MoE (Mixture-of-Experts)
| Aspect | MoE | MoT |
|--------|-----|-----|
| Granularity | Expert FFN layers | Entire attention + FFN |
| Routing | Learned (router network) | Deterministic (modality_masks) |
| Specialization | Task/token-based | Modality-based |
| Use Case | Single modality, sparse compute | Multi-modal generation |

## Paper Reference

**Mixture-of-Transformers: A Sparse and Scalable Architecture for Multi-Modal Foundation Models**

Weixin Liang, Lili Yu, Liang Luo, Srinivasan Iyer, Ning Dong, Chunting Zhou, Gargi Ghosh, Mike Lewis, Wen-tau Yih, Luke Zettlemoyer, Xi Victoria Lin

*Transactions on Machine Learning Research (TMLR), March 2025*

Paper: https://arxiv.org/abs/2411.04996

## Next Steps

1. **Implement custom dataloader** that provides `modality_masks` for your multi-modal dataset
2. **Define modality boundaries** (e.g., which tokens are text vs image vs speech)
3. **Train and evaluate** on your multi-modal task
4. **Monitor efficiency**: MoT should achieve dense-level quality with 40-60% of FLOPs

## File Structure

```
torchtitan/models/mot/
├── README.md                   # This file
├── __init__.py                 # TrainSpec and model registration
├── model/
│   ├── args.py                 # MoTModelArgs configuration
│   ├── model.py                # MoTTransformer and MoTTransformerBlock
│   ├── attention.py            # ModalityUntiedAttention
│   └── feedforward.py          # ModalityUntiedFeedForward
├── infra/
│   └── parallelize.py          # Distributed training (TP, FSDP, etc.)
└── train_configs/
    ├── debug_mot.toml          # Debug configuration
    └── mot_8B_2modality.toml   # 8B model with 2 modalities
```
