#!/bin/bash
#SBATCH --job-name=progressive_train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/progressive_train-%j.out
#SBATCH --error=logs/slurm/progressive_train-%j.err

if [ -f "$HOME/.bashrc" ]; then
    source "$HOME/.bashrc"
fi

cd /home/projects/nssl-prj10106 || exit 1

PYTHON_EXE="/home/projects/nssl-prj10106/anaconda3/bin/python"

# Environment setup
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Create log directories
mkdir -p logs/slurm gnn/checkpoints

# Log GPU information
echo "=== GPU Information ==="
nvidia-smi
echo "======================="

# Progressive training configuration
STAGES=${STAGES:-""}  # Optional: specify specific stages
CHECKPOINT_DIR=${CHECKPOINT_DIR:-"gnn/checkpoints"}
DRY_RUN=${DRY_RUN:-""}

# Build command
CMD="$PYTHON_EXE progressive_trainer.py --base-dir . --checkpoint-dir $CHECKPOINT_DIR"

if [ -n "$STAGES" ]; then
    CMD="$CMD --stages $STAGES"
fi

if [ -n "$DRY_RUN" ]; then
    CMD="$CMD --dry-run"
fi

echo "Running progressive training..."
echo "Command: $CMD"
echo

# Execute progressive training
$CMD

# Log completion
echo
echo "=== Training Completion ==="
echo "Job completed at: $(date)"
echo "Final GPU state:"
nvidia-smi
echo "=========================="
