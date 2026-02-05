# Singularity Container for DeepSeek-OCR

This directory contains a Singularity definition file for building a container with all dependencies pre-installed for DeepSeek-OCR training.

## Prerequisites

- Singularity 3.0+ installed on your system
- NVIDIA GPU with CUDA support
- Root access (for building) or access to a pre-built `.sif` file

## Building the Container

```bash
# Build the container (requires root/sudo)
sudo singularity build deepseek_ocr.sif deepseek_ocr.def

# Or build with fakeroot (if configured)
singularity build --fakeroot deepseek_ocr.sif deepseek_ocr.def
```

Build time is approximately 15-30 minutes depending on network speed.

## Using the Container

### Interactive Shell

```bash
# Start an interactive shell with GPU support
singularity shell --nv deepseek_ocr.sif

# Inside the container, activate the conda environment
source /opt/conda/etc/profile.d/conda.sh
conda activate deepseek_ocr
```

### Single GPU Training

```bash
singularity exec --nv deepseek_ocr.sif \
    /opt/conda/envs/deepseek_ocr/bin/python /opt/torchtitan/torchtitan/train.py \
    --config /opt/torchtitan/torchtitan/models/deepseek_ocr/train_configs/debug.toml
```

### Multi-GPU Training (Single Node)

```bash
singularity exec --nv deepseek_ocr.sif \
    /opt/conda/envs/deepseek_ocr/bin/torchrun \
    --nproc_per_node=4 \
    /opt/torchtitan/torchtitan/train.py \
    --config /opt/torchtitan/torchtitan/models/deepseek_ocr/train_configs/debug.toml
```

### Multi-Node Training (SLURM)

Create a SLURM script (`train_deepseek_ocr.slurm`):

```bash
#!/bin/bash
#SBATCH --job-name=deepseek_ocr
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Load singularity module if needed
# module load singularity

# Set up distributed training environment
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500

# Run training
srun singularity exec --nv deepseek_ocr.sif \
    /opt/conda/envs/deepseek_ocr/bin/torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=8 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    /opt/torchtitan/torchtitan/train.py \
    --config /opt/torchtitan/torchtitan/models/deepseek_ocr/train_configs/debug.toml \
    --job.dump_folder=/outputs/deepseek_ocr_$SLURM_JOB_ID
```

Submit with:
```bash
mkdir -p logs
sbatch train_deepseek_ocr.slurm
```

## Bind Mounts

To access data and save outputs outside the container:

```bash
singularity exec --nv \
    --bind /path/to/your/data:/data \
    --bind /path/to/outputs:/outputs \
    --bind /path/to/checkpoints:/checkpoints \
    deepseek_ocr.sif \
    /opt/conda/envs/deepseek_ocr/bin/python /opt/torchtitan/torchtitan/train.py \
    --config /opt/torchtitan/torchtitan/models/deepseek_ocr/train_configs/debug.toml \
    --job.dump_folder=/outputs
```

## Using Custom Config

You can bind mount your own config file:

```bash
singularity exec --nv \
    --bind /path/to/my_config.toml:/config/my_config.toml \
    deepseek_ocr.sif \
    /opt/conda/envs/deepseek_ocr/bin/python /opt/torchtitan/torchtitan/train.py \
    --config /config/my_config.toml
```

## Environment Variables

The container sets the following environment variables:

| Variable | Value | Description |
|----------|-------|-------------|
| `PYTHONPATH` | `/opt/torchtitan` | TorchTitan source directory |
| `CUDA_HOME` | `/usr/local/cuda` | CUDA installation path |
| `PATH` | `/opt/conda/bin:...` | Includes conda binaries |

## Container Contents

- **Base Image**: NVIDIA CUDA 12.4.1 with cuDNN on Ubuntu 22.04
- **Python**: 3.11 (via Miniconda)
- **PyTorch**: Nightly build with CUDA 12.4 support
- **TorchTitan**: Cloned from bkpcoding/torchtitan
- **Flash Attention**: Pre-installed (if build succeeds)

## Troubleshooting

### GPU not detected

Ensure you're using the `--nv` flag:
```bash
singularity exec --nv deepseek_ocr.sif nvidia-smi
```

### Out of memory during build

Try building with less parallelism:
```bash
SINGULARITY_TMPDIR=/path/to/large/tmp sudo -E singularity build deepseek_ocr.sif deepseek_ocr.def
```

### Permission issues

For shared HPC systems, you may need to:
1. Build on a system where you have sudo
2. Transfer the `.sif` file to the HPC cluster
3. Use `--fakeroot` if configured by admins

### Updating the container

To update with latest code:
```bash
# Rebuild the container
sudo singularity build --force deepseek_ocr.sif deepseek_ocr.def
```

Or create an overlay for modifications:
```bash
singularity overlay create --size 1024 overlay.img
singularity shell --nv --overlay overlay.img deepseek_ocr.sif
# Make changes inside...
```
