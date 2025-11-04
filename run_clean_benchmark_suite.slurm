#!/bin/bash
#SBATCH --job-name=clean-bench-suite
#SBATCH --partition=long
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --output=logs/slurm/clean-bench-suite-%j.out
#SBATCH --error=logs/slurm/clean-bench-suite-%j.err

set -euo pipefail

# Env
module load cuda/12.1
source ~/anaconda3/bin/activate
conda activate base

mkdir -p logs/slurm

cd /home/projects/nssl-prj10106

echo "Starting CLEAN benchmark suite at $(date) on ${SLURMD_NODENAME:-unknown}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits || true)"

# Optionally limit instances via env var
export CLEAN_BENCH_MAX_INSTANCES=${CLEAN_BENCH_MAX_INSTANCES:-1000000}

python scripts/clean_benchmark_suite.py 2>&1 | tee logs/clean_benchmarks_run_$(date +%Y%m%d_%H%M%S).log

echo "Finished at $(date)"

