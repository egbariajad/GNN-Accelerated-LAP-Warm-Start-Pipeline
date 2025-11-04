#!/bin/bash
#SBATCH --job-name=one_gnn_large4096_clean
#SBATCH --output=logs/slurm/one_gnn_large4096_clean-%j.out
#SBATCH --error=logs/slurm/one_gnn_large4096_clean-%j.err
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=72:00:00

set -euo pipefail

echo "=== OneGNN Training - large_4096 CLEAN (no dual noise) ==="
echo "Train file: data/generated/processed_clean/large_4096/full/train.h5"
echo "Val file:   data/generated/processed_clean/large_4096/full/val.h5"
echo "Output:     gnn/checkpoints_clean/one_gnn_large4096_clean.pt"
echo "========================"

nvidia-smi

cd /home/projects/nssl-prj10106

mkdir -p gnn/checkpoints_clean

python gnn/train_one_gnn.py \
    --train data/generated/processed_clean/large_4096/full/train.h5 \
    --val data/generated/processed_clean/large_4096/full/val.h5 \
    --epochs 35 \
    --batch-size 8 \
    --hidden 160 \
    --layers 3 \
    --dropout 0.1 \
    --lr 2e-4 \
    --weight-decay 1e-4 \
    --device cuda \
    --output gnn/checkpoints_clean/one_gnn_large4096_clean.pt

status=$?
echo "Training finished with status $status"
nvidia-smi
exit $status
