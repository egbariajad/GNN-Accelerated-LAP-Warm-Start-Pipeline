# Progressive Multi-Size Clean Training - Quick Guide

## Overview
Single unified OneGNN model trained across 5 problem sizes (512-4096) with curriculum learning on clean datasets.

## Quick Start

```bash
# After clean dataset generation completes:
sbatch run_train_progressive_clean.slurm

# Monitor progress:
tail -f logs/slurm/progressive_clean-*.out
```

## Configuration

**Model**: OneGNN (hidden=192, layers=4, dropout=0.1, topk=24)  
**Sizes**: 512, 1536, 2048, 3072, 4096  
**Curriculum**: Epochs 1-3 use 512-2048, then add 3072-4096  
**Time**: ~3-6 hours

## Key Files

- `progressive_clean_config.yaml` - Configuration
- `train_progressive_clean.py` - Training script  
- `run_train_progressive_clean.slurm` - SLURM submission

## Output

Checkpoints saved to:
- `gnn/checkpoints_clean/progressive_clean_best.pt`
- `gnn/checkpoints_clean/progressive_clean_last.pt`

## Why Use This?

✅ Single model for all sizes  
✅ Better generalization  
✅ Curriculum learning  
✅ Clean data = better convergence  
✅ Simpler deployment  

See `PROGRESSIVE_CLEAN_TRAINING.md` for full details.
