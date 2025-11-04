#!/bin/bash
#
# Master Benchmark Submission Script
# Submits all comprehensive benchmarks for progressive_clean_tie_best.pt
# on sparse and uniform problem types across all dataset sizes
#

echo "========================================================================"
echo "COMPREHENSIVE BENCHMARK SUITE - Sparse & Uniform Problems"
echo "========================================================================"
echo "Model: progressive_clean_tie_best.pt"
echo "Problem Types: sparse, uniform"
echo "Datasets: small, mid_1536, mid_2048, mid_3072, large_4096"
echo "========================================================================"
echo ""

# Create logs directory if it doesn't exist
mkdir -p logs/slurm
mkdir -p results/comprehensive_benchmark
mkdir -p results/comprehensive_benchmark_by_type

echo "Submitting benchmark jobs..."
echo ""

# 1. Main comprehensive benchmark (all sizes combined)
echo "1️⃣  Submitting: Comprehensive GNN Benchmark (All Sizes)"
JOB1=$(sbatch run_bench_sparse_uniform_all.slurm | awk '{print $4}')
echo "   Job ID: $JOB1"
echo ""

# 2. Type-specific analysis benchmark  
echo "2️⃣  Submitting: By-Type Benchmark (Sparse vs Uniform)"
JOB2=$(sbatch run_bench_by_type_all.slurm | awk '{print $4}')
echo "   Job ID: $JOB2"
echo ""

echo "========================================================================"
echo "✅ All jobs submitted successfully!"
echo "========================================================================"
echo ""
echo "Monitor jobs with:"
echo "  squeue -u \$USER"
echo ""
echo "View logs in:"
echo "  logs/slurm/"
echo ""
echo "Results will be saved to:"
echo "  results/comprehensive_benchmark/"
echo "  results/comprehensive_benchmark_by_type/"
echo ""
echo "Job IDs:"
echo "  Comprehensive: $JOB1"
echo "  By-Type:       $JOB2"
echo ""
echo "========================================================================"
