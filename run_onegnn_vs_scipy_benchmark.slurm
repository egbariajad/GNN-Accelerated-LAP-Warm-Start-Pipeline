#!/bin/bash
#SBATCH --job-name=bench_onegnn_vs_scipy
#SBATCH --output=logs/slurm/bench_onegnn_vs_scipy-%j.out
#SBATCH --error=logs/slurm/bench_onegnn_vs_scipy-%j.err
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=12:00:00

set -euo pipefail

echo "=== Benchmark: OneGNN vs SciPy ==="
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo "============================="

nvidia-smi

cd /home/projects/nssl-prj10106
NOW=$(date +%Y%m%d_%H%M)
mkdir -p logs/performance artifacts/reports scripts

# Run the benchmark
python3 scripts/bench_onegnn_vs_scipy.py \
  --data data/buckets \
  --ckpt checkpoints/one_gnn_debug_stabilized.pt \
  --sizes 1024 2048 3072 4096 \
  --per-size 5 \
  --topk 16 \
  --device cuda \
  --outdir artifacts/reports \
  > logs/performance/bench_vs_scipy_${NOW}.log 2>&1

status=$?
echo "Benchmark finished with status $status"
echo "Output log: logs/performance/bench_vs_scipy_${NOW}.log"
nvidia-smi
exit $status
