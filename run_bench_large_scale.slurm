#!/bin/bash
#SBATCH --job-name=bench_large_scale
#SBATCH --output=logs/slurm/bench_large_scale-%j.out
#SBATCH --error=logs/slurm/bench_large_scale-%j.err
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=64G

echo "=== Large-Scale Synthetic Benchmark (Sparse & Uniform) ==="
echo "Start time: $(date)"
echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo ""

# Activate environment
source /home/projects/nssl-prj10106/anaconda3/bin/activate base

# Run large-scale benchmark on synthetic data
# Test sizes: 4096, 8192, 16384
# Skip SciPy for 16384 (too slow)
python scripts/comprehensive_large_scale_benchmark.py \
  --model gnn/checkpoints_clean/progressive_clean_tie_best.pt \
  --sizes 4096 8192 16384 \
  --instances 3 \
  --sparse-density 0.3 \
  --output-dir results/large_scale_benchmark

echo ""
echo "End time: $(date)"
echo "=== Job Complete ==="
