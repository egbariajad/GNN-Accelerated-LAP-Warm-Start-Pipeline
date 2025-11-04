#!/bin/bash
#SBATCH --job-name=gen_mid
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/gen_mid-%j.out
#SBATCH --error=logs/slurm/gen_mid-%j.err

if [ -f "$HOME/.bashrc" ]; then
    source "$HOME/.bashrc"
fi

cd /home/projects/nssl-prj10106 || exit 1

PYTHON_EXE="/home/projects/nssl-prj10106/anaconda3/bin/python"

$PYTHON_EXE data/generate_dataset.py \
    --output-dir data/buckets_raw/mid \
    --sizes 1536 2048 3072 \
    --instances-per-family 0 \
    --instances-map 1536:180 2048:120 3072:80 \
    --train 0.3 \
    --val 0.3 \
    --test 0.4 \
    --seed 1447 \
    --summary "$@"
