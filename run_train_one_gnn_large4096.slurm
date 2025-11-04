#!/bin/bash
#SBATCH --job-name=one_gnn_train_large4096
#SBATCH --output=logs/slurm/one_gnn_train_large4096-%j.out
#SBATCH --error=logs/slurm/one_gnn_train_large4096-%j.err
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=72:00:00

echo "=== OneGNN Training - Large 4096 ==="
echo "Command: $0 $@"
echo "Train file: data/generated/processed/large_4096/full/train.h5"
echo "Val file: data/generated/processed/large_4096/full/val.h5"  
echo "Output: gnn/checkpoints/one_gnn_large4096_v2.pt"
echo "========================"

nvidia-smi

# Navigate to project directory
cd /home/projects/nssl-prj10106

# Train OneGNN on large_4096 dataset
python gnn/train_one_gnn.py \
    --train data/generated/processed/large_4096/full/train.h5 \
    --epochs 60 \
    --batch-size 2 \
    --hidden 64 \
    --layers 3 \
    --dropout 0.2 \
    --lr 1e-4 \
    --weight-decay 1e-4 \
    --output gnn/checkpoints/one_gnn_large4096_v2.pt \
    --device cuda \
    --val data/generated/processed/large_4096/full/val.h5

echo "Training finished with status $?"
nvidia-smi