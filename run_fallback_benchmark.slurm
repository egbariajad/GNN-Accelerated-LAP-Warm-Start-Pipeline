#!/bin/bash
#SBATCH --job-name=fallback_benchmark
#SBATCH --output=logs/fallback_threshold_benchmark_%j.out
#SBATCH --error=logs/fallback_threshold_benchmark_%j.err
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

echo "========================================================================"
echo "SLURM Job: Fallback Threshold Benchmark"
echo "========================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "========================================================================"
echo ""

# Set environment
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Change to project directory
cd /home/projects/nssl-prj10106

# Create logs directory if it doesn't exist
mkdir -p logs

# Run the benchmark script
echo "Starting benchmark script..."
echo ""

./benchmark_fallback_thresholds.sh

EXIT_CODE=$?

echo ""
echo "========================================================================"
echo "Job Complete"
echo "========================================================================"
echo "Exit Code: $EXIT_CODE"
echo "End Time: $(date)"
echo "========================================================================"

exit $EXIT_CODE
