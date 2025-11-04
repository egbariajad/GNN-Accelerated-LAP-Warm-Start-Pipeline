#!/bin/bash
#SBATCH --job-name=gen_small
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm/gen_small-%j.out
#SBATCH --error=logs/slurm/gen_small-%j.err

if [ -f "$HOME/.bashrc" ]; then
    source "$HOME/.bashrc"
fi

cd /home/projects/nssl-prj10106 || exit 1

PYTHON_EXE="/home/projects/nssl-prj10106/anaconda3/bin/python"

$PYTHON_EXE data/generate_dataset.py \
    --output-dir data/buckets_raw/small \
    --sizes 512 1024 \
    --instances-per-family 0 \
    --instances-map 512:400 1024:300 \
    --train 0.5 \
    --val 0.4 \
    --test 0.1 \
    --seed 1337 \
    --summary "$@"
