#!/bin/bash
#
# Complete Benchmark Suite Submission
# Submits ALL comprehensive benchmarks for progressive_clean_tie_best.pt
#

echo "========================================================================"
echo "COMPLETE BENCHMARK SUITE - Sparse & Uniform Problems"
echo "========================================================================"
echo "Model: gnn/checkpoints_clean/progressive_clean_tie_best.pt"
echo "Problem Types: sparse, uniform"
echo "========================================================================"
echo ""

# Create output directories
mkdir -p logs/slurm
mkdir -p results/comprehensive_benchmark
mkdir -p results/comprehensive_benchmark_by_type
mkdir -p results/large_scale_benchmark
mkdir -p results/pipeline_profiling

echo "Submitting all benchmark jobs..."
echo ""

# 1. Main comprehensive benchmark (real data, all sizes)
echo "1️⃣  Comprehensive Benchmark (Real Data: 512-4096)"
JOB1=$(sbatch run_bench_sparse_uniform_all.slurm | awk '{print $4}')
echo "   Job ID: $JOB1"
echo ""

# 2. By-type analysis benchmark
echo "2️⃣  By-Type Analysis (Sparse vs Uniform comparison)"
JOB2=$(sbatch run_bench_by_type_all.slurm | awk '{print $4}')
echo "   Job ID: $JOB2"
echo ""

# 3. Large-scale synthetic benchmark
echo "3️⃣  Large-Scale Synthetic (4096, 8192, 16384)"
JOB3=$(sbatch run_bench_large_scale.slurm | awk '{print $4}')
echo "   Job ID: $JOB3"
echo ""

# 4. Pipeline profiling
echo "4️⃣  Pipeline Profiling (Detailed timing breakdown)"
JOB4=$(sbatch run_bench_pipeline_profile.slurm | awk '{print $4}')
echo "   Job ID: $JOB4"
echo ""

echo "========================================================================"
echo "✅ All 4 benchmark jobs submitted successfully!"
echo "========================================================================"
echo ""
echo "Job Summary:"
echo "  1. Comprehensive (real):     $JOB1  (~8-12 hours)"
echo "  2. By-Type (real):           $JOB2  (~8-12 hours)"
echo "  3. Large-Scale (synthetic):  $JOB3  (~12-24 hours)"
echo "  4. Pipeline Profiling:       $JOB4  (~12-16 hours)"
echo ""
echo "Monitor jobs:"
echo "  squeue -u \$USER"
echo "  ./monitor_all_benchmarks.sh"
echo ""
echo "Results will be saved to:"
echo "  results/comprehensive_benchmark/          # Real data benchmark"
echo "  results/comprehensive_benchmark_by_type/  # Type comparison"
echo "  results/large_scale_benchmark/            # Scaling to 16384"
echo "  results/pipeline_profiling/               # Detailed profiling"
echo ""
echo "========================================================================"
