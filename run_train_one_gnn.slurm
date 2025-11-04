#!/bin/bash
#SBATCH --job-name=one_gnn_train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/one_gnn_train-%j.out
#SBATCH --error=logs/slurm/one_gnn_train-%j.err

if [ -f "$HOME/.bashrc" ]; then
    source "$HOME/.bashrc"
fi

cd /home/projects/nssl-prj10106 || exit 1

PYTHON_EXE="/home/projects/nssl-prj10106/anaconda3/bin/python"

mkdir -p logs/slurm gnn/checkpoints

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TRAIN_FILE=${TRAIN_FILE:-"data/generated/processed/small/full/train.h5"}
VAL_FILE=${VAL_FILE:-"data/generated/processed/small/full/val.h5"}
EPOCHS=${EPOCHS:-30}
BATCH_SIZE=${BATCH_SIZE:-32}
HIDDEN=${HIDDEN:-96}
LAYERS=${LAYERS:-3}
DROPOUT=${DROPOUT:-0.1}
LR=${LR:-5e-4}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-4}
OUTPUT=${OUTPUT:-"gnn/checkpoints/one_gnn_small.pt"}
DEVICE=${DEVICE:-cuda}

CMD="$PYTHON_EXE gnn/train_one_gnn.py --train $TRAIN_FILE --epochs $EPOCHS --batch-size $BATCH_SIZE \
    --hidden $HIDDEN --layers $LAYERS --dropout $DROPOUT --lr $LR --weight-decay $WEIGHT_DECAY \
    --output $OUTPUT --device $DEVICE"

if [ -n "$VAL_FILE" ]; then
    CMD="$CMD --val $VAL_FILE"
fi

if [ -n "$MIN_EPOCHS" ]; then
    CMD="$CMD --min-epochs $MIN_EPOCHS"
fi

if [ -n "$PATIENCE" ]; then
    CMD="$CMD --patience $PATIENCE"
fi

echo "=== OneGNN Training ==="
echo "Command: $CMD"
echo "Train file: $TRAIN_FILE"
echo "Val file: ${VAL_FILE:-None}"
echo "Output: $OUTPUT"
echo "========================"

nvidia-smi

$CMD

STATUS=$?

echo "Training finished with status $STATUS"
nvidia-smi

exit $STATUS
