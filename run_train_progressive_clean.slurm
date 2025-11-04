#!/bin/bash
#SBATCH --job-name=progressive_clean
#SBATCH --output=logs/slurm/progressive_clean-%j.out
#SBATCH --error=logs/slurm/progressive_clean-%j.err
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=72:00:00

set -euo pipefail

echo "========================================="
echo "Progressive Multi-Size Training - CLEAN"
echo "========================================="
echo "Training on: 512, 1536, 2048, 3072, 4096"
echo "Features: full (21 features)"
echo "Curriculum: Epochs 1-3 small, 4+ all sizes"
echo "Config: progressive_clean_config.yaml"
echo "Output: gnn/checkpoints_clean/progressive_clean_*.pt"
echo "========================================="
echo ""

nvidia-smi

cd /home/projects/nssl-prj10106

# Ensure checkpoints directory exists
mkdir -p gnn/checkpoints_clean

# Run progressive training
python train_progressive_clean.py \
    --config progressive_clean_config.yaml

status=$?
echo ""
echo "========================================="
echo "Training finished with status $status"
echo "========================================="
nvidia-smi
exit $status
