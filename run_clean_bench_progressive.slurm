#!/bin/bash
#SBATCH --job-name=clean-prog
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --output=logs/slurm/clean-prog-%j.out
#SBATCH --error=logs/slurm/clean-prog-%j.err

set -euo pipefail
if command -v module >/dev/null 2>&1; then
  module load cuda/12.1
fi

PROJECT_DIR=/home/projects/nssl-prj10106
PYTHON_BIN="$PROJECT_DIR/anaconda3/bin/python"

if [ ! -x "$PYTHON_BIN" ]; then
  echo "Expected Python binary not found at $PYTHON_BIN" >&2
  exit 1
fi

mkdir -p logs/slurm
cd "$PROJECT_DIR"
export CLEAN_BENCH_MAX_INSTANCES=${CLEAN_BENCH_MAX_INSTANCES:-1000000}
"$PYTHON_BIN" scripts/clean_benchmark_suite.py --pairs progressive 2>&1 | tee logs/clean_bench_progressive_$(date +%Y%m%d_%H%M%S).log
