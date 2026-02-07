# DeepSeek-OCR Singularity Commands

Below are the exact commands used during the working training run (single GPU, 2 steps). These assume the `deepseek_ocr.sif` file lives in this directory.

```bash
module load singularity/4.1.5 cuda/12.6.3

HF_HOME=/tmp/hf_cache singularity exec --nv \
  --bind /gpfs/accounts/bucherb_owned_root/bucherb_owned1/bpatil/torchtitan:/opt/torchtitan \
  ./deepseek_ocr.sif \
  /bin/bash -lc 'LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 \
  python /opt/torchtitan/torchtitan/train.py \
    --job.custom-config-module torchtitan.custom_data_config \
    --job.config-file /opt/torchtitan/torchtitan/models/deepseek_ocr/train_configs/debug.toml \
    --model.flavor=debugmodel_full_vision \
    --data.img_size=1024 \
    --training.steps=2 \
    --training.local_batch_size=1 \
    --activation_checkpoint.mode=none'
```

Notes:
- Uses SynthDOG-EN streaming dataset from Hugging Face.
- The run relies on a custom config extension: `torchtitan.custom_data_config`.
- Uses `debugmodel_full_vision` and disables activation checkpointing for compatibility.
