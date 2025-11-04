#!/bin/bash
#SBATCH --job-name=one_gnn_small_full_clean
#SBATCH --output=logs/slurm/one_gnn_small_full_clean-%j.out
#SBATCH --error=logs/slurm/one_gnn_small_full_clean-%j.err
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24GB
#SBATCH --time=24:00:00

set -euo pipefail

echo "=== OneGNN Training - small/full CLEAN (no dual noise) ==="
echo "Train file: data/generated/processed_clean/small/full/train.h5"
echo "Val file:   data/generated/processed_clean/small/full/val.h5"
echo "Output:     gnn/checkpoints_clean/one_gnn_small_full_clean.pt"
echo "========================"

nvidia-smi

cd /home/projects/nssl-prj10106

# Create checkpoints directory
mkdir -p gnn/checkpoints_clean

python gnn/train_one_gnn.py \
    --train data/generated/processed_clean/small/full/train.h5 \
    --val data/generated/processed_clean/small/full/val.h5 \
    --epochs 45 \
    --batch-size 32 \
    --hidden 96 \
    --layers 3 \
    --dropout 0.1 \
    --lr 4e-4 \
    --weight-decay 1e-4 \
    --device cuda \
    --output gnn/checkpoints_clean/one_gnn_small_full_clean.pt

status=$?
echo "Training finished with status $status"
nvidia-smi
exit $status
