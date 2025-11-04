#!/bin/bash
#SBATCH --job-name=one_gnn_mid1536_full_clean
#SBATCH --output=logs/slurm/one_gnn_mid1536_full_clean-%j.out
#SBATCH --error=logs/slurm/one_gnn_mid1536_full_clean-%j.err
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --time=48:00:00

set -euo pipefail

echo "=== OneGNN Training - mid_1536/full CLEAN (no dual noise) ==="
echo "Train file: data/generated/processed_clean/mid_1536/full/train.h5"
echo "Val file:   data/generated/processed_clean/mid_1536/full/val.h5"
echo "Output:     gnn/checkpoints_clean/one_gnn_mid1536_full_clean.pt"
echo "========================"

nvidia-smi

cd /home/projects/nssl-prj10106

mkdir -p gnn/checkpoints_clean

python gnn/train_one_gnn.py \
    --train data/generated/processed_clean/mid_1536/full/train.h5 \
    --val data/generated/processed_clean/mid_1536/full/val.h5 \
    --epochs 40 \
    --batch-size 16 \
    --hidden 128 \
    --layers 3 \
    --dropout 0.1 \
    --lr 3e-4 \
    --weight-decay 1e-4 \
    --device cuda \
    --output gnn/checkpoints_clean/one_gnn_mid1536_full_clean.pt

status=$?
echo "Training finished with status $status"
nvidia-smi
exit $status
