#!/bin/bash
#
# Regenerate ALL clean datasets (no dual noise)
# This script creates data/generated/processed_clean/ with the same structure
# as the original noisy datasets but with --dual-noise-prob 0.0
#

set -e

BASE_DIR="data/generated/processed_clean"
SEED=42

echo "========================================="
echo "CLEAN DATASET GENERATION"
echo "========================================="
echo "Output base: $BASE_DIR"
echo "Dual noise: DISABLED (--dual-noise-prob 0.0)"
echo "Cost noise: DISABLED (--noise-prob 0.0)"
echo ""

# Create base directory
mkdir -p "$BASE_DIR"

# =========================================
# SMALL BUCKET (512x512)
# =========================================
echo ">>> Generating SMALL bucket (512x512) - FULL features"
python data/generate_dataset.py \
    --output-dir "$BASE_DIR/small/full" \
    --sizes 512 \
    --instances-per-family 80 \
    --noise-prob 0.0 \
    --dual-noise-prob 0.0 \
    --noise-std 0.15 \
    --seed $SEED \
    --train 0.7 --val 0.15 --test 0.15

# =========================================
# MID 1536 BUCKET
# =========================================
echo ""
echo ">>> Generating MID_1536 bucket (1536x1536) - FULL features"
python data/generate_dataset.py \
    --output-dir "$BASE_DIR/mid_1536/full" \
    --sizes 1536 \
    --instances-per-family 80 \
    --noise-prob 0.0 \
    --dual-noise-prob 0.0 \
    --noise-std 0.15 \
    --seed $SEED \
    --train 0.7 --val 0.15 --test 0.15

# =========================================
# MID 2048 BUCKET
# =========================================
echo ""
echo ">>> Generating MID_2048 bucket (2048x2048) - FULL features"
python data/generate_dataset.py \
    --output-dir "$BASE_DIR/mid_2048/full" \
    --sizes 2048 \
    --instances-per-family 80 \
    --noise-prob 0.0 \
    --dual-noise-prob 0.0 \
    --noise-std 0.15 \
    --seed $SEED \
    --train 0.7 --val 0.15 --test 0.15

# =========================================
# MID 3072 BUCKET
# =========================================
echo ""
echo ">>> Generating MID_3072 bucket (3072x3072) - FULL features"
python data/generate_dataset.py \
    --output-dir "$BASE_DIR/mid_3072/full" \
    --sizes 3072 \
    --instances-per-family 80 \
    --noise-prob 0.0 \
    --dual-noise-prob 0.0 \
    --noise-std 0.15 \
    --seed $SEED \
    --train 0.7 --val 0.15 --test 0.15

# =========================================
# LARGE 4096 BUCKET
# =========================================
echo ""
echo ">>> Generating LARGE_4096 bucket (4096x4096) - FULL features"
python data/generate_dataset.py \
    --output-dir "$BASE_DIR/large_4096/full" \
    --sizes 4096 \
    --instances-per-family 80 \
    --noise-prob 0.0 \
    --dual-noise-prob 0.0 \
    --noise-std 0.15 \
    --seed $SEED \
    --train 0.7 --val 0.15 --test 0.15

# =========================================
# SUMMARY
# =========================================
echo ""
echo "========================================="
echo "CLEAN DATASET GENERATION COMPLETE"
echo "========================================="
echo "Generated datasets:"
find "$BASE_DIR" -name "*.h5" -type f | while read f; do
    size=$(stat -f%z "$f" 2>/dev/null || stat -c%s "$f" 2>/dev/null || echo "?")
    size_mb=$((size / 1024 / 1024))
    echo "  $f (${size_mb} MB)"
done

echo ""
echo "Next steps:"
echo "1. Run verify_clean_datasets.py to check for noise contamination"
echo "2. Update SLURM scripts to point to $BASE_DIR"
echo "3. Retrain models using the clean datasets"
