# Progressive Training Strategy for 8-Hour Time Limits

This document outlines the comprehensive strategy to train the full dataset within the 8-hour SLURM time limit using bucket-aware batching and progressive training.

## Overview

The strategy implements:
1. **Bucket-aware batching**: Different batch sizes based on matrix dimensions
2. **Progressive training**: Sequential training stages that build upon each other
3. **Checkpoint resumption**: Ability to resume training from interruptions
4. **Memory optimization**: Careful batch size selection to avoid GPU OOM

## Dataset Analysis

Current dataset distribution:
- **Small (512-1024)**: 2,100 samples (512: 1,200, 1024: 900)
- **Mid 1536**: 480 samples  
- **Mid 2048**: 144 samples
- **Mid 3072**: 88 samples
- **Large 4096**: 30 samples

## Training Stages

### Stage 1: Small 512-only (4 hours)
- **Dataset**: `data/generated/processed/small/full/train.h5` (filtered to 512×512)
- **Batch size**: 8
- **Epochs**: 30
- **Samples**: 1,200
- **Memory**: ~2.5 GB
- **Output**: `gnn/checkpoints/small_512.pt`

### Stage 2: Small 1024-only (3 hours) 
- **Dataset**: `data/generated/processed/small/full/train.h5` (filtered to 1024×1024)
- **Batch size**: 2 (reduced due to memory constraints)
- **Epochs**: 20
- **Samples**: 900
- **Memory**: ~6 GB
- **Init from**: Stage 1 checkpoint
- **Output**: `gnn/checkpoints/small_1024.pt`

### Stage 3: Mid 1536 (6 hours)
- **Dataset**: `data/generated/processed/mid_1536/full/train.h5`
- **Batch size**: 4
- **Epochs**: 25  
- **Samples**: 480
- **Memory**: ~8 GB
- **Init from**: Stage 2 checkpoint
- **Output**: `gnn/checkpoints/mid_1536.pt`

### Stage 4: Mid 2048 (6 hours)
- **Dataset**: `data/generated/processed/mid_2048/full/train.h5`
- **Batch size**: 2
- **Epochs**: 25
- **Samples**: 144
- **Memory**: ~12 GB
- **Init from**: Stage 3 checkpoint
- **Output**: `gnn/checkpoints/mid_2048.pt`

### Stage 5: Mid 3072 (8 hours)
- **Dataset**: `data/generated/processed/mid_3072/full/train.h5`
- **Batch size**: 1
- **Epochs**: 25
- **Samples**: 88
- **Memory**: ~18 GB
- **Init from**: Stage 4 checkpoint
- **Output**: `gnn/checkpoints/mid_3072.pt`

### Stage 6: Large 4096 (6 hours)
- **Dataset**: `data/generated/processed/large_4096/full/train.h5`
- **Batch size**: 1
- **Epochs**: 20
- **Samples**: 30
- **Memory**: ~24 GB
- **Init from**: Stage 5 checkpoint
- **Output**: `gnn/checkpoints/large_4096.pt` (final model)

## Implementation

### Enhanced Training Script: `gnn/train_progressive.py`

Key features:
- **Size filtering**: `--filter-size` parameter to train on specific matrix sizes
- **Checkpoint resumption**: `--checkpoint-init` to load weights, `--checkpoint-save` for periodic saves
- **Resume capability**: `--resume` flag to continue from saved checkpoints
- **Memory efficient**: Uses filtered datasets to reduce memory overhead

### Progressive Orchestrator: `progressive_trainer.py`

Manages the entire training pipeline:
- Automatically discovers available datasets
- Creates appropriate training stages
- Handles checkpoint initialization between stages
- Provides dry-run capability for testing
- Comprehensive logging and progress tracking

### SLURM Scripts

#### Main Progressive Training: `run_progressive_train.slurm`
Runs the complete pipeline in a single 8-hour job:
```bash
sbatch run_progressive_train.slurm
```

#### Individual Stage Scripts:
- `run_train_stage_512.slurm`: Stage 1 only (4h)
- `run_train_stage_1024.slurm`: Stage 2 only (4h)  
- `run_train_stage_mid.slurm`: Mid stages with configurable bucket size (6-8h)

Example usage for individual stages:
```bash
# Stage 1: 512×512 matrices
sbatch run_train_stage_512.slurm

# Stage 2: 1024×1024 matrices (after Stage 1 completes)
sbatch run_train_stage_1024.slurm

# Stage 3: 1536×1536 matrices
BUCKET_SIZE=1536 sbatch run_train_stage_mid.slurm

# Stage 4: 2048×2048 matrices  
BUCKET_SIZE=2048 sbatch run_train_stage_mid.slurm
```

## Usage Instructions

### 1. Full Progressive Training (Recommended)
```bash
# Run complete pipeline
sbatch run_progressive_train.slurm

# Monitor progress
tail -f logs/slurm/progressive_train-*.out
```

### 2. Staged Training (Manual Control)
```bash
# Analyze datasets first
python analyze_training_strategy.py --memory-analysis

# List available stages
python progressive_trainer.py --list

# Run specific stages
python progressive_trainer.py --stages small_512_only small_1024_only

# Dry run to test
python progressive_trainer.py --dry-run
```

### 3. Individual Stage Control
```bash
# Run just the first stage
sbatch run_train_stage_512.slurm

# Then run second stage (uses checkpoint from first)
sbatch run_train_stage_1024.slurm
```

## Memory Management

### Batch Size Selection
The batch sizes are carefully chosen based on GPU memory constraints:

| Matrix Size | Batch Size | Est. Memory | Justification |
|-------------|------------|-------------|---------------|
| 512×512     | 8          | ~2.5 GB     | Efficient training |
| 1024×1024   | 2          | ~6 GB       | Memory constraint |
| 1536×1536   | 4          | ~8 GB       | Balanced approach |
| 2048×2048   | 2          | ~12 GB      | Memory constraint |
| 3072×3072   | 1          | ~18 GB      | Single sample fits |
| 4096×4096   | 1          | ~24 GB      | Maximum GPU usage |

### Environment Configuration
All scripts use:
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

This enables efficient GPU memory allocation and helps prevent fragmentation.

## Checkpoint Management

### Checkpoint Types
1. **Training checkpoints**: Full state for resumption (`*_checkpoint.pt`)
2. **Model checkpoints**: Final weights for next stage initialization (`*.pt`)

### Resume Capability
If training is interrupted:
```bash
# Resume from last checkpoint
python gnn/train_progressive.py --resume --checkpoint-save gnn/checkpoints/stage_checkpoint.pt
```

## Monitoring and Evaluation

### Progress Monitoring
```bash
# View training logs
tail -f logs/slurm/progressive_train-*.out

# Check GPU usage
nvidia-smi

# Monitor checkpoint creation
ls -la gnn/checkpoints/
```

### Evaluation After Each Stage
```bash
# Evaluate current model
python gnn/eval.py --checkpoint gnn/checkpoints/small_512.pt --test data/generated/processed/small/full/test.h5

# Run benchmarks
python scripts/main_benchmark.py --model gnn/checkpoints/small_512.pt
```

## Expected Timeline

| Stage | Duration | Cumulative | Status |
|-------|----------|------------|---------|
| 512×512 | 4h | 4h | ✓ Can run standalone |
| 1024×1024 | 3h | 7h | ✓ Fits in 8h limit |
| Mid 1536 | 6h | - | ⚠ Separate job |
| Mid 2048 | 6h | - | ⚠ Separate job |
| Mid 3072 | 8h | - | ⚠ Separate job |
| Large 4096 | 6h | - | ⚠ Separate job |

**Strategy**: Either run stages 1-2 together in 8h, or run each mid/large stage separately.

## Best Practices

1. **Always start with dry-run**: Test command generation before execution
2. **Monitor memory usage**: Watch `nvidia-smi` during training
3. **Use resumable training**: Always enable checkpoint saving
4. **Validate each stage**: Run evaluation after each completed stage
5. **Keep datasets clean**: Use `scripts/merge_h5.py` for clean data
6. **Fixed seeds**: Use `prepare_buckets.py --seed 42` for reproducible splits

## Troubleshooting

### Common Issues

1. **GPU OOM**: Reduce batch size or use gradient accumulation
2. **Dataset not found**: Run `scripts/merge_h5.py` to prepare datasets
3. **Checkpoint errors**: Ensure compatible model architecture across stages
4. **Time limit exceeded**: Split into more stages or reduce epochs

### Debug Commands
```bash
# Check dataset integrity
python -c "import h5py; f=h5py.File('data/generated/processed/small/full/train.h5'); print(f.keys(), len(f['n']))"

# Test memory usage
python gnn/train_progressive.py --train data/generated/processed/small/full/train.h5 --filter-size 512 --batch-size 1 --epochs 1

# Verify GPU
nvidia-smi
```

## Files Created/Modified

### New Files
- `gnn/train_progressive.py`: Enhanced training with bucket-aware batching
- `progressive_trainer.py`: Training orchestrator
- `analyze_training_strategy.py`: Dataset analysis tool
- `run_progressive_train.slurm`: Main progressive training job
- `run_train_stage_512.slurm`: Stage 1 job
- `run_train_stage_1024.slurm`: Stage 2 job  
- `run_train_stage_mid.slurm`: Mid bucket stages job

### Key Features Added
- Size-based filtering in dataset loading
- Checkpoint initialization and resumption
- Memory-efficient batch size selection
- Comprehensive progress tracking
- Dry-run testing capability

This strategy ensures efficient use of the 8-hour time limit while progressively training on increasingly complex matrix sizes, leading to a robust final model.