#!/bin/bash
#SBATCH --job-name=bench_pipeline_profile
#SBATCH --output=logs/slurm/bench_pipeline_profile-%j.out
#SBATCH --error=logs/slurm/bench_pipeline_profile-%j.err
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --time=16:00:00
#SBATCH --mem=48G

echo "=== Pipeline Profiling Benchmark (Sparse & Uniform) ==="
echo "Start time: $(date)"
echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo ""

# Activate environment
source /home/projects/nssl-prj10106/anaconda3/bin/activate base

# Run detailed pipeline profiling on all real datasets
python scripts/comprehensive_pipeline_profiling.py \
  --model gnn/checkpoints_clean/progressive_clean_tie_best.pt \
  --datasets small mid_1536 mid_2048 mid_3072 large_4096 \
  --filter-types sparse uniform \
  --repeats 5 \
  --output-dir results/pipeline_profiling

echo ""
echo "End time: $(date)"
echo "=== Job Complete ==="
