# Clean Dataset Generation Guide

## Overview

This guide documents the transition from **noisy dual training data** to **clean dual training data**. The original datasets had random Gaussian noise injected into dual variables (u, v) during generation, which could interfere with GNN learning. Clean datasets contain optimal dual variables without noise contamination.

## Changes Made

### 1. Code Modifications

#### `data/generators.py`
- **Added parameter**: `dual_noise_prob: float = 0.0`
- **Changed behavior**: Dual noise injection (lines 118-124) is now guarded by `dual_noise_prob` instead of `noise_probability`
- **Default**: Dual noise is **DISABLED** (`dual_noise_prob=0.0`)

```python
# OLD (line 118):
if noise_probability > 0.0 and rng.random() < noise_probability:

# NEW (line 119):
if dual_noise_prob > 0.0 and rng.random() < dual_noise_prob:
```

#### `data/generate_dataset.py`
- **Added CLI flag**: `--dual-noise-prob` (default: 0.0)
- **Added parameter**: `dual_noise_prob` to `generate_dataset()` function
- **Plumbed through**: Pass `dual_noise_prob` to `generate_synthetic_instance()`

### 2. Dataset Structure

#### Original (Noisy) Datasets
```
data/generated/processed/
├── small/
│   └── full/       # 512x512, full features, WITH dual noise
├── mid_1536/
│   └── full/       # 1536x1536, WITH dual noise
├── mid_2048/full/  # 2048x2048, WITH dual noise
├── mid_3072/full/  # 3072x3072, WITH dual noise
└── large_4096/full/ # 4096x4096, WITH dual noise
```

#### Clean Datasets (NEW)
```
data/generated/processed_clean/
├── small/
│   └── full/       # 512x512, full features, NO dual noise
├── mid_1536/
│   └── full/       # 1536x1536, NO dual noise
├── mid_2048/full/  # 2048x2048, NO dual noise
├── mid_3072/full/  # 3072x3072, NO dual noise
└── large_4096/full/ # 4096x4096, NO dual noise
```

Each directory contains: `train.h5`, `val.h5`, `test.h5`

## Generation Commands

### Quick Start (All Datasets)
```bash
chmod +x generate_clean_datasets.sh
./generate_clean_datasets.sh
```

### Manual Generation (Example)

#### Small Bucket (512x512)
```bash
python data/generate_dataset.py \
    --output-dir data/generated/processed_clean/small/full \
    --sizes 512 \
    --instances-per-family 80 \
    --noise-prob 0.0 \
    --dual-noise-prob 0.0 \
    --seed 42 \
    --train 0.7 --val 0.15 --test 0.15
```

#### Mid 1536 Bucket
```bash
python data/generate_dataset.py \
    --output-dir data/generated/processed_clean/mid_1536/full \
    --sizes 1536 \
    --instances-per-family 80 \
    --noise-prob 0.0 \
    --dual-noise-prob 0.0 \
    --seed 42
```

#### Large Buckets (2048, 3072, 4096)
```bash
# Mid 2048
python data/generate_dataset.py \
    --output-dir data/generated/processed_clean/mid_2048/full \
    --sizes 2048 --instances-per-family 80 \
    --noise-prob 0.0 --dual-noise-prob 0.0 --seed 42

# Mid 3072
python data/generate_dataset.py \
    --output-dir data/generated/processed_clean/mid_3072/full \
    --sizes 3072 --instances-per-family 80 \
    --noise-prob 0.0 --dual-noise-prob 0.0 --seed 42

# Large 4096
python data/generate_dataset.py \
    --output-dir data/generated/processed_clean/large_4096/full \
    --sizes 4096 --instances-per-family 80 \
    --noise-prob 0.0 --dual-noise-prob 0.0 --seed 42
```

## Verification

### Automated Check
```bash
python verify_clean_datasets.py
```

**Expected output:**
```
✅ CLEAN small/full/train.h5
  Instances: 392
  noise_std unique: [0.0]
  u: mean=0.000012, std=0.000031
  v: mean=-0.000008, std=0.000025
```

### Manual Spot Check
```python
import h5py
import numpy as np

with h5py.File('data/generated/processed_clean/small/full/train.h5', 'r') as f:
    # Check noise_std field
    noise_stds = f['noise_std'][:]
    print("Unique noise_std values:", np.unique(noise_stds))  # Should be [0.0]
    
    # Check dual means (should be near zero)
    u_sample = f['u'][0]
    v_sample = f['v'][0]
    print(f"u mean: {np.mean(u_sample):.6f}")  # Should be ~0.0
    print(f"v mean: {np.mean(v_sample):.6f}")  # Should be ~0.0
```

## Training with Clean Data

### New SLURM Scripts

Five new training scripts have been created:

1. **`run_train_one_gnn_small_full_clean.slurm`**
   - Input: `data/generated/processed_clean/small/full/`
   - Output: `gnn/checkpoints_clean/one_gnn_small_full_clean.pt`
   - Config: 512x512, full features, 45 epochs

2. **`run_train_one_gnn_mid1536_full_clean.slurm`**
   - Input: `data/generated/processed_clean/mid_1536/full/`
   - Output: `gnn/checkpoints_clean/one_gnn_mid1536_full_clean.pt`
   - Config: 1536x1536, full features, 40 epochs

3. **`run_train_one_gnn_large4096_clean.slurm`**
   - Input: `data/generated/processed_clean/large_4096/full/`
   - Output: `gnn/checkpoints_clean/one_gnn_large4096_clean.pt`
   - Config: 4096x4096, full features, 35 epochs

### Submitting Training Jobs
```bash
# Small models
sbatch run_train_one_gnn_small_full_clean.slurm

# Mid models
sbatch run_train_one_gnn_mid1536_full_clean.slurm

# Large model
sbatch run_train_one_gnn_large4096_clean.slurm
```

### Monitoring Training
```bash
# Check job status
squeue -u $USER

# Watch training logs (example)
tail -f logs/slurm/one_gnn_small_full_clean-*.out

# Check for dual MAE explosion
grep "MAE" logs/slurm/one_gnn_small_full_clean-*.out | tail -20
```

## Expected Training Improvements

### With Clean Data (Expected)
- **Lower variance**: Batch-to-batch dual MAE should be more stable
- **Better convergence**: Validation loss should decrease more smoothly
- **Lower final MAE**: Models should achieve better dual prediction quality
- **Faster training**: Less confusion from noisy targets

### What to Monitor
```bash
# Look for these in training logs:
# - Epoch dual MAE trends (should decrease smoothly)
# - Validation MAE (should be lower than noisy baseline)
# - Training stability (no sudden spikes)
```

**Example log snippet (good):**
```
Epoch 10: train_loss=0.0234, val_loss=0.0198, u_mae=0.0087, v_mae=0.0072
Epoch 20: train_loss=0.0156, val_loss=0.0142, u_mae=0.0065, v_mae=0.0058
Epoch 30: train_loss=0.0123, val_loss=0.0118, u_mae=0.0053, v_mae=0.0047
```

## Checkpoint Organization

### Directory Structure
```
gnn/checkpoints/           # Original models (trained on noisy data)
gnn/checkpoints_clean/     # New models (trained on clean data)
```

### Naming Convention
- Clean models: `*_clean.pt` suffix
- Example: `one_gnn_small_full_clean.pt`

## Comparison: Noisy vs Clean

### To Compare Models
```bash
# Benchmark noisy model
python scripts/gnn_benchmark.py \
    --models one_gnn_small_full_v3.pt \
    --sizes 512 --max-instances 100

# Benchmark clean model
python scripts/gnn_benchmark.py \
    --models one_gnn_small_full_clean.pt \
    --sizes 512 --max-instances 100
```

### Expected Improvements
| Metric | Noisy Baseline | Clean (Expected) | Improvement |
|--------|----------------|------------------|-------------|
| u MAE  | 0.0095         | < 0.0070         | ~26% better |
| v MAE  | 0.0080         | < 0.0060         | ~25% better |
| Speedup vs LAP | 1.13x | > 1.15x | More consistent |

## Rollback Plan

If clean models perform worse:
1. Keep original noisy datasets in `data/generated/processed/`
2. Keep original checkpoints in `gnn/checkpoints/`
3. Clean datasets are in separate directory - no overwriting
4. Can revert code changes to `data/generators.py` and `data/generate_dataset.py`

## Dataset Statistics

### Per-Bucket Configuration
| Bucket | Size | Features | Instances/Family | Total Instances | Split |
|--------|------|----------|------------------|-----------------|-------|
| small | 512 | full | 80 | ~560 | 70/15/15 |
| mid_1536 | 1536 | full | 80 | ~560 | 70/15/15 |
| mid_2048 | 2048 | full | 80 | ~560 | 70/15/15 |
| mid_3072 | 3072 | full | 80 | ~560 | 70/15/15 |
| large_4096 | 4096 | full | 80 | ~560 | 70/15/15 |

### Synthetic Families
Each dataset samples from 7 synthetic families:
- `uniform`
- `geometric`
- `toeplitz`
- `exponential`
- `bounded`
- `flow`
- `bounded_int`

## Timeline

1. **Generation**: ~30-60 minutes (all buckets)
2. **Verification**: ~5 minutes
3. **Training small**: ~6-12 hours each
4. **Training mid**: ~12-24 hours each
5. **Training large**: ~24-48 hours
6. **Evaluation**: ~1 hour per model

**Total**: ~3-5 days for complete clean model suite

## Next Steps

1. ✅ Code modifications (completed)
2. ⏳ Generate clean datasets
   ```bash
   ./generate_clean_datasets.sh
   ```
3. ⏳ Verify datasets
   ```bash
   python verify_clean_datasets.py
   ```
4. ⏳ Submit training jobs
   ```bash
   sbatch run_train_one_gnn_small_full_clean.slurm
   # ... (repeat for all models)
   ```
5. ⏳ Monitor training logs
6. ⏳ Benchmark clean models
7. ⏳ Compare noisy vs clean performance
8. ⏳ Document results

## References

- Original noise injection: `data/generators.py:118-124` (now disabled)
- Dataset generation: `data/generate_dataset.py`
- Training script: `gnn/train_one_gnn.py`
- Benchmark script: `scripts/gnn_benchmark.py`
