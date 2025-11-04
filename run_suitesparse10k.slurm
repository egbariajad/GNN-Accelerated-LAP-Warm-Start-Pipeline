#!/bin/bash
#SBATCH --job-name=suitesparse10k
#SBATCH --partition=long
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --output=logs/slurm/suitesparse10k-%j.out
#SBATCH --error=logs/slurm/suitesparse10k-%j.err

# Load environment
module load cuda/12.1
source ~/anaconda3/bin/activate
conda activate base

# Create logs directory if needed
mkdir -p logs/slurm

cd /home/projects/nssl-prj10106

echo "Starting SuiteSparse 10k processing at $(date)"
echo "Node: $SLURMD_NODENAME"
echo "RAM: $(free -h | grep Mem: | awk '{print $2}')"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits)"

# Run the generator
python scripts/test_10k_suitesparse.py 2>&1 | tee logs/suitesparse10k_$(date +%Y%m%d_%H%M%S).log

if [ $? -eq 0 ]; then
    echo "Generator completed successfully at $(date)"
    
    # Optional: Run benchmark if dataset created
    if [ -f "data/suitesparse_10k/test.h5" ]; then
        echo "Running benchmark..."
        python scripts/gnn_benchmark.py \
          --models one_gnn_mid1536_full_v2.pt \
          --sizes 10000 \
          --max-instances 5 \
          --checkpoint-dirs checkpoints gnn/checkpoints \
          --data-root data/suitesparse_10k 2>&1 | tee logs/benchmark_suitesparse10k_$(date +%Y%m%d_%H%M%S).log
    else
        echo "No test.h5 found, skipping benchmark"
    fi
else
    echo "Generator failed at $(date)"
fi

echo "Job completed at $(date)"
