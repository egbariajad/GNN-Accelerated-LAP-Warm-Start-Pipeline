#!/bin/bash
#SBATCH --job-name=one_gnn_mid2048_clean_h192L4
#SBATCH --output=logs/slurm/one_gnn_mid2048_clean_h192L4-%j.out
#SBATCH --error=logs/slurm/one_gnn_mid2048_clean_h192L4-%j.err
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=56GB
#SBATCH --time=48:00:00

set -euo pipefail

echo "=== OneGNN Training - mid_2048/full CLEAN (hidden=192, layers=4) ==="
echo "Train file: data/generated/processed_clean/mid_2048/full/train.h5"
echo "Val file:   data/generated/processed_clean/mid_2048/full/val.h5"
echo "Output:     gnn/checkpoints_clean/one_gnn_mid2048_clean_h192L4.pt"
echo "========================"

nvidia-smi

cd /home/projects/nssl-prj10106

mkdir -p gnn/checkpoints_clean

python gnn/train_one_gnn.py \
    --train data/generated/processed_clean/mid_2048/full/train.h5 \
    --val data/generated/processed_clean/mid_2048/full/val.h5 \
    --epochs 40 \
    --batch-size 12 \
    --hidden 192 \
    --layers 4 \
    --dropout 0.1 \
    --lr 2e-4 \
    --weight-decay 1e-4 \
    --device cuda \
    --output gnn/checkpoints_clean/one_gnn_mid2048_clean_h192L4.pt

status=$?
echo "Training finished with status $status"
nvidia-smi
exit $status
