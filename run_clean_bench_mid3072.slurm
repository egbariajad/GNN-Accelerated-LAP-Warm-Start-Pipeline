#!/bin/bash
#SBATCH --job-name=clean-3072
#SBATCH --partition=long
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --output=logs/slurm/clean-3072-%j.out
#SBATCH --error=logs/slurm/clean-3072-%j.err

set -euo pipefail
module load cuda/12.1
source ~/anaconda3/bin/activate
conda activate base
mkdir -p logs/slurm
cd /home/projects/nssl-prj10106
export CLEAN_BENCH_MAX_INSTANCES=${CLEAN_BENCH_MAX_INSTANCES:-1000000}
python scripts/clean_benchmark_suite.py --pairs mid3072 2>&1 | tee logs/clean_bench_mid3072_$(date +%Y%m%d_%H%M%S).log

