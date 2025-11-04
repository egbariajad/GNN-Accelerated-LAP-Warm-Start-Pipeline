#!/bin/bash
#SBATCH --job-name=one_gnn_train_mid3072
#SBATCH --output=logs/slurm/one_gnn_train_mid3072-%j.out
#SBATCH --error=logs/slurm/one_gnn_train_mid3072-%j.err
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48GB
#SBATCH --time=48:00:00

echo "=== OneGNN Training - Mid 3072 ==="
echo "Command: $0 $@"
echo "Train file: data/generated/processed/mid_3072/full/train.h5"
echo "Val file: data/generated/processed/mid_3072/full/val.h5"  
echo "Output: gnn/checkpoints/one_gnn_mid3072_v2.pt"
echo "========================"

nvidia-smi

# Navigate to project directory
cd /home/projects/nssl-prj10106

# Train OneGNN on mid_3072 dataset
python gnn/train_one_gnn.py \
    --train data/generated/processed/mid_3072/full/train.h5 \
    --epochs 60 \
    --batch-size 4 \
    --hidden 64 \
    --layers 3 \
    --dropout 0.15 \
    --lr 1.5e-4 \
    --weight-decay 1e-4 \
    --output gnn/checkpoints/one_gnn_mid3072_v2.pt \
    --device cuda \
    --val data/generated/processed/mid_3072/full/val.h5

echo "Training finished with status $?"
nvidia-smi