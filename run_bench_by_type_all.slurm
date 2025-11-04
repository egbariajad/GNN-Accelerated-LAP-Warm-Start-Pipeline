#!/bin/bash
#SBATCH --job-name=bench_by_type_all
#SBATCH --output=logs/slurm/bench_by_type_all-%j.out
#SBATCH --error=logs/slurm/bench_by_type_all-%j.err
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --time=362:00:00
#SBATCH --mem=48G

echo "=== Comprehensive GNN Benchmark BY TYPE - ALL Datasets (Sparse & Uniform) ==="
echo "Start time: $(date)"
echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo ""

# Activate environment
source /home/projects/nssl-prj10106/anaconda3/bin/activate base

# Run benchmark with type-specific analysis
python scripts/comprehensive_by_type_benchmark.py \
  --model gnn/checkpoints_clean/progressive_clean_tie_best.pt \
  --datasets small mid_1536 mid_2048 mid_3072 large_4096 \
  --filter-types sparse uniform \
  --repeats 3 \
  --output-dir results/comprehensive_benchmark_by_type

echo ""
echo "End time: $(date)"
echo "=== Job Complete ==="
