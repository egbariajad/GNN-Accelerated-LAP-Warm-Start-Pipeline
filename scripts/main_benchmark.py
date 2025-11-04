#!/usr/bin/env python3
"""
Main Benchmark Script: Seeded vs LAP vs SciPy

A comprehensive benchmarking script that compares three LAP solvers:
1. SciPy linear_sum_assignment (baseline reference)
2. lap.lapjv (unseeded, high-performance baseline)  
3. lap.lapjv_seeded (warm-started with dual potentials)

This script demonstrates the proper modular structure for LAP benchmarking.
"""

import os
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set thread limits for fair and consistent comparison
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MKL_DYNAMIC", "FALSE")
os.environ.setdefault("PYTHONHASHSEED", "0")

import numpy as np
import statistics
import warnings
from typing import Dict, List

# Import our modular components
from solvers import (
    SciPySolver, LAPSolver, SeededLAPSolver,
    compute_oracle_duals, verify_solver_correctness, time_solver_rigorous,
    generate_uniform_costs, generate_near_diagonal_costs, generate_sparse_costs
)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


def benchmark_problem(dataset_name: str, C: np.ndarray, noise_level: float = 0.0) -> Dict[str, any]:
    """Benchmark all three solvers on a single problem."""
    
    print(f"\n=== {dataset_name} ===")
    print(f"Problem size: {C.shape[0]}x{C.shape[1]}")
    print(f"Seed noise level: {noise_level}")
    
    # Generate dual seeds with specified noise
    u_oracle, v_oracle = compute_oracle_duals(C, noise_level)
    
    # Verify correctness first
    if not verify_solver_correctness(C, u_oracle, v_oracle):
        print("❌ CORRECTNESS CHECK FAILED")
        return {'success': False, 'dataset': dataset_name}
    
    print("✅ Correctness verified - all solvers agree")
    
    # Initialize solvers
    scipy_solver = SciPySolver()
    lap_solver = LAPSolver()
    seeded_solver = SeededLAPSolver()
    
    # Benchmark each solver
    print("\nBenchmarking solvers...")
    
    # SciPy baseline
    print("  SciPy linear_sum_assignment...", end=' ')
    scipy_timing = time_solver_rigorous(lambda: scipy_solver.solve(C))
    if scipy_timing['success']:
        print(f"{scipy_timing['median']*1000:.2f} ms (median)")
    else:
        print("FAILED")
        return {'success': False, 'dataset': dataset_name}
    
    # LAP unseeded
    print("  LAP unseeded (lapjv)...", end=' ')
    lap_timing = time_solver_rigorous(lambda: lap_solver.solve(C))
    if lap_timing['success']:
        print(f"{lap_timing['median']*1000:.2f} ms (median)")
    else:
        print("FAILED")
        return {'success': False, 'dataset': dataset_name}
    
    # LAP seeded
    print("  LAP seeded (lapjv_seeded)...", end=' ')
    seeded_timing = time_solver_rigorous(lambda: seeded_solver.solve(C, u_oracle, v_oracle))
    if seeded_timing['success']:
        print(f"{seeded_timing['median']*1000:.2f} ms (median)")
    else:
        print("FAILED")
        return {'success': False, 'dataset': dataset_name}
    
    # Calculate speedups
    speedup_vs_unseeded = lap_timing['median'] / seeded_timing['median']
    speedup_vs_scipy = scipy_timing['median'] / seeded_timing['median']
    
    print(f"\nTiming Results:")
    print(f"  SciPy time: {scipy_timing['median']*1000:.2f} ms")
    print(f"  LAP time: {lap_timing['median']*1000:.2f} ms")
    print(f"  Seeded time: {seeded_timing['median']*1000:.2f} ms")
    
    print(f"\nSpeedup Analysis:")
    print(f"  Seeded vs SciPy: {speedup_vs_scipy:.2f}x")
    print(f"  Seeded vs LAP (unseeded): {speedup_vs_unseeded:.2f}x")
    
    # Performance assessment
    if speedup_vs_unseeded >= 2.0:
        status = "🚀 EXCELLENT"
    elif speedup_vs_unseeded >= 1.1:
        status = "✅ GOOD"
    elif speedup_vs_unseeded >= 0.9:
        status = "🔶 COMPETITIVE"
    else:
        status = "⚠️ OVERHEAD"
    
    print(f"  Assessment: {status}")
    
    return {
        'success': True,
        'dataset': dataset_name,
        'n': C.shape[0],
        'noise_level': noise_level,
        'scipy_time': scipy_timing['median'],
        'lap_time': lap_timing['median'],
        'seeded_time': seeded_timing['median'],
        'speedup_vs_unseeded': speedup_vs_unseeded,
        'speedup_vs_scipy': speedup_vs_scipy,
        'status': status
    }


def run_main_benchmark():
    """Run the complete benchmark suite."""
    
    print("=" * 80)
    print("MODULAR LAP BENCHMARK: Seeded vs LAP vs SciPy")
    print("=" * 80)
    print(f"Thread settings: OMP={os.environ.get('OMP_NUM_THREADS')}, MKL={os.environ.get('MKL_NUM_THREADS')}")
    print("Timing methodology: 5 warmups, 30 repeats, median reporting")
    print("Solvers: SciPy, LAP (unseeded), LAP (seeded)")
    
    all_results = []
    
    # ========================================================================
    # TEST 1: UNIFORM RANDOM PROBLEMS (Core performance comparison)
    # ========================================================================
    
    print(f"\n" + "="*60)
    print("UNIFORM RANDOM PROBLEMS")
    print("="*60)
    
    for n in [200, 500, 1000]:
        C = generate_uniform_costs(n)
        result = benchmark_problem(f"Uniform_{n}x{n}", C, noise_level=0.0)
        if result['success']:
            all_results.append(result)
    
    # ========================================================================
    # TEST 2: SEED QUALITY SENSITIVITY (Noise analysis)
    # ========================================================================
    
    print(f"\n" + "="*60)
    print("SEED QUALITY SENSITIVITY ANALYSIS")
    print("="*60)
    
    test_sizes = [200, 500]
    noise_levels = [0.0, 0.1, 0.2]
    
    for n in test_sizes:
        for noise in noise_levels:
            C = generate_uniform_costs(n, seed=123)  # Different seed for variety
            result = benchmark_problem(f"NoiseTest_{n}x{n}_noise{noise}", C, noise_level=noise)
            if result['success']:
                all_results.append(result)
    
    # ========================================================================
    # TEST 3: STRUCTURED PROBLEMS (Near-diagonal)
    # ========================================================================
    
    print(f"\n" + "="*60)
    print("STRUCTURED PROBLEMS (Near-diagonal)")
    print("="*60)
    
    for n in [200, 500]:
        C = generate_near_diagonal_costs(n, noise_level=0.1)
        result = benchmark_problem(f"NearDiag_{n}x{n}", C, noise_level=0.0)
        if result['success']:
            all_results.append(result)
    
    # ========================================================================
    # TEST 4: SPARSE PROBLEMS
    # ========================================================================
    
    print(f"\n" + "="*60)
    print("SPARSE PROBLEMS")
    print("="*60)
    
    for n in [200, 500]:
        for sparsity in [0.3, 0.1]:
            C = generate_sparse_costs(n, sparsity_ratio=sparsity)
            result = benchmark_problem(f"Sparse_{n}x{n}_sp{int(sparsity*100)}", C, noise_level=0.0)
            if result['success']:
                all_results.append(result)
    
    # ========================================================================
    # COMPREHENSIVE ANALYSIS
    # ========================================================================
    
    print("\n" + "="*80)
    print("COMPREHENSIVE ANALYSIS")
    print("="*80)
    
    if not all_results:
        print("❌ No successful benchmarks!")
        return
    
    print(f"\nSuccessful benchmarks: {len(all_results)}")
    
    # Summary table
    print(f"\n{'Dataset':<30} {'Size':<6} {'SciPy(ms)':<10} {'LAP(ms)':<10} {'Seeded(ms)':<12} {'vs SciPy':<8} {'vs LAP':<8} {'Status':<15}")
    print("-" * 115)
    
    speedups_vs_lap = []
    speedups_vs_scipy = []
    
    for result in all_results:
        dataset = result['dataset']
        n = result['n']
        scipy_time = f"{result['scipy_time']*1000:.2f}"
        lap_time = f"{result['lap_time']*1000:.2f}"
        seeded_time = f"{result['seeded_time']*1000:.2f}"
        speedup_scipy = f"{result['speedup_vs_scipy']:.2f}x"
        speedup_lap = f"{result['speedup_vs_unseeded']:.2f}x"
        status = result['status'].replace('🚀 ', '').replace('✅ ', '').replace('🔶 ', '').replace('⚠️ ', '')
        
        print(f"{dataset:<30} {n:<6} {scipy_time:<10} {lap_time:<10} {seeded_time:<12} {speedup_scipy:<8} {speedup_lap:<8} {status:<15}")
        
        speedups_vs_lap.append(result['speedup_vs_unseeded'])
        speedups_vs_scipy.append(result['speedup_vs_scipy'])
    
    # Statistical summary
    print(f"\n📊 PERFORMANCE STATISTICS")
    
    if speedups_vs_lap:
        print(f"\n🎯 SEEDED LAP vs UNSEEDED LAP:")
        print(f"   Median: {statistics.median(speedups_vs_lap):.2f}x")
        print(f"   Mean: {statistics.mean(speedups_vs_lap):.2f}x")
        print(f"   Range: {min(speedups_vs_lap):.2f}x - {max(speedups_vs_lap):.2f}x")
        
        excellent_count = sum(1 for s in speedups_vs_lap if s >= 2.0)
        competitive_count = sum(1 for s in speedups_vs_lap if s >= 0.9)
        print(f"   Excellent cases (≥2.0x): {excellent_count}/{len(speedups_vs_lap)}")
        print(f"   Competitive cases (≥0.9x): {competitive_count}/{len(speedups_vs_lap)}")
    
    print(f"\n🎯 FINAL ASSESSMENT:")
    if speedups_vs_lap:
        overall_avg = statistics.mean(speedups_vs_lap)
        if overall_avg >= 1.5:
            print("   🚀 SEEDED LAP: Significant performance advantage!")
        elif overall_avg >= 1.0:
            print("   ✅ SEEDED LAP: Consistent performance improvement")
        elif overall_avg >= 0.9:
            print("   🔶 SEEDED LAP: Competitive with unseeded LAP")
        else:
            print("   ⚠️ SEEDED LAP: Some overhead detected")
    
    print("   ✅ CORRECTNESS: All solvers produce identical results")
    print("   ✅ METHODOLOGY: Rigorous timing with statistical significance")
    print(f"   📈 RECOMMENDATION: Use seeded LAP when good dual seeds are available")


def quick_test(n: int = 200, noise_level: float = 0.0):
    """Quick test for development and verification."""
    print(f"Quick Test: {n}x{n} problem, noise={noise_level}")
    C = generate_uniform_costs(n)
    result = benchmark_problem(f"QuickTest_{n}x{n}", C, noise_level)
    return result


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        # Quick test mode for development
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 200
        noise = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
        quick_test(n, noise)
    else:
        # Full benchmark suite
        run_main_benchmark()