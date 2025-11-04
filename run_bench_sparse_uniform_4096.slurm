#!/bin/bash
#SBATCH --job-name=bench_sparse_uniform_4096
#SBATCH --output=logs/slurm/bench_sparse_uniform_4096-%j.out
#SBATCH --error=logs/slurm/bench_sparse_uniform_4096-%j.err
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --mem=48G

echo "=== Comprehensive GNN Benchmark - LARGE_4096 Dataset (Sparse & Uniform) ==="
echo "Start time: $(date)"
echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo ""

# Activate environment
source /home/projects/nssl-prj10106/anaconda3/bin/activate base

# Run benchmark
python scripts/comprehensive_gnn_benchmark.py \
  --model progressive_clean_tie_best.pt \
  --datasets large_4096 \
  --filter-types sparse uniform \
  --repeats 3 \
  --output-dir results/comprehensive_benchmark

echo ""
echo "End time: $(date)"
echo "=== Job Complete ==="
