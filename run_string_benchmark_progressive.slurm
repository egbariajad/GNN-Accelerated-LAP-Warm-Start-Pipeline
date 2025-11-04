#!/bin/bash
#SBATCH --job-name=string_prog
#SBATCH --output=logs/slurm/string_progressive-%j.out
#SBATCH --error=logs/slurm/string_progressive-%j.err
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --time=36:00:00

set -euo pipefail

cd /home/projects/nssl-prj10106

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

echo "==========================================="
echo "STRING Dataset - Progressive Models Test"
echo "==========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# Show GPU info
nvidia-smi

echo ""
echo "==========================================="
echo "Testing progressive training models"
echo "==========================================="
echo ""

# Run the benchmark with progressive models
python scripts/benchmark_string.py \
    --max-instances 20 \
    --models \
        progressive_clean_best.pt \
        progressive_clean_tie_best.pt

exit_code=$?

echo ""
echo "==========================================="
echo "Benchmark completed with exit code: $exit_code"
echo "End time: $(date)"
echo "==========================================="

if [ $exit_code -eq 0 ]; then
    echo ""
    echo "Results saved to logs/string_benchmark/"
    ls -lh logs/string_benchmark/string_benchmark_*.csv | tail -1
    ls -lh logs/string_benchmark/string_benchmark_*_summary.txt | tail -1
fi

exit $exit_code
