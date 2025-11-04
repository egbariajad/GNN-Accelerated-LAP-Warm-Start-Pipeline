#!/bin/bash
#SBATCH --job-name=progressive_tie_boost
#SBATCH --output=logs/slurm/progressive_tie_boost-%j.out
#SBATCH --error=logs/slurm/progressive_tie_boost-%j.err
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=36:00:00

set -euo pipefail

echo "========================================="
echo "Progressive Training with Tie-Focused Augmentation"
echo "========================================="
echo "Config: progressive_clean_tie_boost.yaml"
echo "Init checkpoint: gnn/checkpoints_clean/progressive_clean_best.pt"
echo "Datasets include tie-only boosts for 512/1536/2048"
echo "========================================="
echo ""

nvidia-smi

cd /home/projects/nssl-prj10106

mkdir -p gnn/checkpoints_clean

python train_progressive_clean.py \
    --config progressive_clean_tie_boost.yaml \
    --init-checkpoint gnn/checkpoints_clean/progressive_clean_best.pt

status=$?
echo ""
echo "========================================="
echo "Training finished with status $status"
echo "========================================="
nvidia-smi
exit $status
