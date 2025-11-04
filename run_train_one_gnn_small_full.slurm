#!/bin/bash
#SBATCH --job-name=one_gnn_small_full
#SBATCH --output=logs/slurm/one_gnn_small_full-%j.out
#SBATCH --error=logs/slurm/one_gnn_small_full-%j.err
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24GB
#SBATCH --time=24:00:00

set -euo pipefail

echo "=== OneGNN Training - small/full (full features) ==="
echo "Train file: data/generated/processed/small/full/train.h5"
echo "Val file:   data/generated/processed/small/full/val.h5"
echo "Output:     gnn/checkpoints/one_gnn_small_full_v3.pt"
echo "========================"

nvidia-smi

cd /home/projects/nssl-prj10106

python gnn/train_one_gnn.py \
    --train data/generated/processed/small/full/train.h5 \
    --val data/generated/processed/small/full/val.h5 \
    --epochs 45 \
    --batch-size 32 \
    --hidden 96 \
    --layers 3 \
    --dropout 0.1 \
    --lr 4e-4 \
    --weight-decay 1e-4 \
    --device cuda \
    --output gnn/checkpoints/one_gnn_small_full_v3.pt

status=$?
echo "Training finished with status $status"
nvidia-smi
exit $status
