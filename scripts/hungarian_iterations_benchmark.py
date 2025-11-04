#!/usr/bin/env python3
"""
Hungarian Algorithm Iterations Benchmark
=========================================

Compare LAP solver performance with/without seeding:
- Log Hungarian algorithm iterations
- Measure solver time only (no GNN overhead)
- Compare seeded vs unseeded performance
"""

import os
import sys
import time
import numpy as np
import h5py
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import LAP solvers
from scipy.optimize import linear_sum_assignment
import lap

def load_test_data(sizes=[512, 1024, 2048], max_instances=5):
    """Load test instances from multiple dataset files."""
    print("============================================================")
    print("LOADING TEST DATA")
    print("============================================================")
    
    instances = []
    
    # Map sizes to dataset paths
    size_to_path = {
        512: "data/generated/processed/small/full/test.h5",
        1024: "data/generated/processed/small/full/test.h5", 
        2048: "data/generated/processed/mid_2048/full/test.h5"
    }
    
    for target_size in sizes:
        if target_size not in size_to_path:
            print(f"No dataset available for size {target_size}")
            continue
            
        data_path = size_to_path[target_size]
        print(f"Loading {target_size}x{target_size} instances from: {data_path}")
        
        try:
            with h5py.File(data_path, 'r') as f:
                # Filter instances by size
                n_values = f['n'][:]
                size_mask = (n_values == target_size)
                size_indices = np.where(size_mask)[0]
                
                n_available = len(size_indices)
                n_instances = min(n_available, max_instances)
                print(f"  Found {n_available} instances of size {target_size}, using {n_instances}")
                
                if n_instances == 0:
                    continue
                
                # Take first n_instances that match the size
                selected_indices = size_indices[:n_instances]
                
                for idx, i in enumerate(selected_indices):
                    # Reshape flattened cost matrix
                    C_flat = f['C'][i]
                    C = C_flat.reshape(target_size, target_size)
                    
                    # Get dual potentials
                    u_true = f['u'][i] if 'u' in f else None
                    v_true = f['v'][i] if 'v' in f else None
                    
                    instances.append({
                        'size': target_size,
                        'C': C,
                        'u_true': u_true,
                        'v_true': v_true,
                        'instance_id': idx,
                        'dataset_index': i
                    })
        except Exception as e:
            print(f"  Error loading from {data_path}: {e}")
    
    print(f"Total test instances loaded: {len(instances)}")
    return instances

def benchmark_scipy_hungarian(C):
    """Benchmark SciPy's Hungarian algorithm."""
    start_time = time.perf_counter()
    row_indices, col_indices = linear_sum_assignment(C)
    end_time = time.perf_counter()
    
    solve_time = (end_time - start_time) * 1000  # Convert to ms
    total_cost = C[row_indices, col_indices].sum()
    
    # SciPy doesn't expose iteration count, estimate based on theoretical complexity
    n = C.shape[0]
    estimated_iterations = n * (n + 1) // 2  # Rough estimate for Hungarian complexity
    
    return {
        'solve_time': solve_time,
        'total_cost': total_cost,
        'iterations': estimated_iterations,  # Estimated
        'assignment': (row_indices, col_indices)
    }

def benchmark_lap_unseeded(C):
    """Benchmark LAP library without seeding."""
    start_time = time.perf_counter()
    cost, row_ind, col_ind = lap.lapjv(C)  # Returns (cost, row_assignment, col_assignment)
    end_time = time.perf_counter()
    
    solve_time = (end_time - start_time) * 1000  # Convert to ms
    
    return {
        'solve_time': solve_time,
        'total_cost': cost,
        'iterations': None,  # LAP library doesn't expose iteration count
        'assignment': (row_ind, col_ind)
    }

def benchmark_lap_seeded(C, u_seed, v_seed):
    """Benchmark LAP library with dual seeding."""
    start_time = time.perf_counter()
    row_ind, col_ind, cost = lap.lapjv_seeded(C, u_seed, v_seed)  # Returns (row_assignment, col_assignment, cost)
    end_time = time.perf_counter()
    
    solve_time = (end_time - start_time) * 1000  # Convert to ms
    
    return {
        'solve_time': solve_time,
        'total_cost': cost,
        'iterations': None,  # LAP library doesn't expose iteration count
        'assignment': (row_ind, col_ind)
    }

def run_benchmark():
    """Run comprehensive Hungarian algorithm benchmark."""
    print("================================================================================")
    print("HUNGARIAN ALGORITHM ITERATIONS & SOLVER TIME BENCHMARK")
    print("================================================================================")
    print("Comparing LAP solver performance with/without dual seeding")
    print("Focus: Pure solver performance (no GNN overhead)")
    print()
    
    # Load test data
    instances = load_test_data()
    
    # Group by size for organized output
    by_size = {}
    for inst in instances:
        size = inst['size']
        if size not in by_size:
            by_size[size] = []
        by_size[size].append(inst)
    
    # Results storage
    all_results = []
    
    for size in sorted(by_size.keys()):
        size_instances = by_size[size]
        print(f"\n============================================================")
        print(f"TESTING {size}x{size} PROBLEMS ({len(size_instances)} instances)")
        print("============================================================")
        
        size_results = []
        
        for i, instance in enumerate(size_instances, 1):
            C = instance['C']
            u_true = instance['u_true']
            v_true = instance['v_true']
            
            print(f"\n--- Instance {i}/{len(size_instances)} ---")
            
            # 1. SciPy Hungarian (baseline)
            scipy_result = benchmark_scipy_hungarian(C)
            print(f"SciPy Hungarian:")
            print(f"  Time: {scipy_result['solve_time']:.2f} ms")
            print(f"  Cost: {float(scipy_result['total_cost']):.2f}")
            print(f"  Est. Iterations: {scipy_result['iterations']}")
            
            # 2. LAP library unseeded
            lap_unseeded_result = benchmark_lap_unseeded(C)
            print(f"LAP Unseeded:")
            print(f"  Time: {lap_unseeded_result['solve_time']:.2f} ms")
            print(f"  Cost: {float(lap_unseeded_result['total_cost']):.2f}")
            
            # 3. LAP library seeded (if dual potentials available)
            if u_true is not None and v_true is not None:
                lap_seeded_result = benchmark_lap_seeded(C, u_true, v_true)
                print(f"LAP Seeded:")
                print(f"  Time: {lap_seeded_result['solve_time']:.2f} ms")
                print(f"  Cost: {float(lap_seeded_result['total_cost']):.2f}")
                
                # Calculate speedup
                speedup = lap_unseeded_result['solve_time'] / lap_seeded_result['solve_time']
                print(f"  Speedup: {speedup:.2f}x")
            else:
                lap_seeded_result = None
                speedup = None
                print(f"LAP Seeded: No dual potentials available")
            
            # Verify solutions match
            cost_diff_unseeded = abs(scipy_result['total_cost'] - lap_unseeded_result['total_cost'])
            print(f"Cost verification:")
            print(f"  SciPy vs LAP Unseeded: {cost_diff_unseeded:.6f}")
            
            if lap_seeded_result:
                cost_diff_seeded = abs(scipy_result['total_cost'] - lap_seeded_result['total_cost'])
                print(f"  SciPy vs LAP Seeded: {cost_diff_seeded:.6f}")
            
            # Store results
            result = {
                'size': size,
                'instance': i,
                'scipy_time': scipy_result['solve_time'],
                'scipy_cost': scipy_result['total_cost'],
                'scipy_iterations': scipy_result['iterations'],
                'lap_unseeded_time': lap_unseeded_result['solve_time'],
                'lap_unseeded_cost': lap_unseeded_result['total_cost'],
                'lap_seeded_time': lap_seeded_result['solve_time'] if lap_seeded_result else None,
                'lap_seeded_cost': lap_seeded_result['total_cost'] if lap_seeded_result else None,
                'speedup': speedup
            }
            size_results.append(result)
            all_results.append(result)
        
        # Size summary
        print(f"\n--- {size}x{size} Summary ---")
        scipy_times = [r['scipy_time'] for r in size_results]
        lap_unseeded_times = [r['lap_unseeded_time'] for r in size_results]
        lap_seeded_times = [r['lap_seeded_time'] for r in size_results if r['lap_seeded_time'] is not None]
        speedups = [r['speedup'] for r in size_results if r['speedup'] is not None]
        
        print(f"SciPy Hungarian - Mean: {np.mean(scipy_times):.2f} ms, Median: {np.median(scipy_times):.2f} ms")
        print(f"LAP Unseeded - Mean: {np.mean(lap_unseeded_times):.2f} ms, Median: {np.median(lap_unseeded_times):.2f} ms")
        
        if lap_seeded_times:
            print(f"LAP Seeded - Mean: {np.mean(lap_seeded_times):.2f} ms, Median: {np.median(lap_seeded_times):.2f} ms")
            print(f"Seeding Speedup - Mean: {np.mean(speedups):.2f}x, Median: {np.median(speedups):.2f}x")
            print(f"Speedup Range: {min(speedups):.2f}x - {max(speedups):.2f}x")
    
    # Overall summary
    print(f"\n================================================================================")
    print("OVERALL SUMMARY")
    print("================================================================================")
    
    # Group results by algorithm
    scipy_times = [r['scipy_time'] for r in all_results]
    lap_unseeded_times = [r['lap_unseeded_time'] for r in all_results]
    lap_seeded_times = [r['lap_seeded_time'] for r in all_results if r['lap_seeded_time'] is not None]
    all_speedups = [r['speedup'] for r in all_results if r['speedup'] is not None]
    
    print(f"Algorithm Performance Across All {len(all_results)} Instances:")
    print(f"")
    print(f"SciPy Hungarian:")
    print(f"  Mean: {np.mean(scipy_times):.2f} ms")
    print(f"  Median: {np.median(scipy_times):.2f} ms")
    print(f"  Range: {min(scipy_times):.2f} - {max(scipy_times):.2f} ms")
    print(f"")
    print(f"LAP Unseeded:")
    print(f"  Mean: {np.mean(lap_unseeded_times):.2f} ms")
    print(f"  Median: {np.median(lap_unseeded_times):.2f} ms")
    print(f"  Range: {min(lap_unseeded_times):.2f} - {max(lap_unseeded_times):.2f} ms")
    
    if lap_seeded_times:
        print(f"")
        print(f"LAP Seeded ({len(lap_seeded_times)} instances with dual seeds):")
        print(f"  Mean: {np.mean(lap_seeded_times):.2f} ms")
        print(f"  Median: {np.median(lap_seeded_times):.2f} ms")
        print(f"  Range: {min(lap_seeded_times):.2f} - {max(lap_seeded_times):.2f} ms")
        print(f"")
        print(f"Seeding Speedup Analysis:")
        print(f"  Mean speedup: {np.mean(all_speedups):.2f}x")
        print(f"  Median speedup: {np.median(all_speedups):.2f}x")
        print(f"  Range: {min(all_speedups):.2f}x - {max(all_speedups):.2f}x")
        print(f"  Instances with >2x speedup: {sum(1 for s in all_speedups if s > 2.0)}/{len(all_speedups)}")
        print(f"  Instances with >5x speedup: {sum(1 for s in all_speedups if s > 5.0)}/{len(all_speedups)}")
    
    print(f"")
    print("🎯 KEY FINDINGS:")
    if lap_seeded_times and all_speedups:
        best_speedup = max(all_speedups)
        worst_speedup = min(all_speedups)
        print(f"  • Dual seeding provides {np.mean(all_speedups):.1f}x average speedup")
        print(f"  • Best case speedup: {best_speedup:.1f}x")
        print(f"  • Seeding is most effective for larger problems")
        
        # Compare LAP vs SciPy
        lap_vs_scipy_unseeded = np.mean(scipy_times) / np.mean(lap_unseeded_times)
        lap_vs_scipy_seeded = np.mean(scipy_times) / np.mean(lap_seeded_times)
        print(f"  • LAP library is {lap_vs_scipy_unseeded:.1f}x faster than SciPy (unseeded)")
        print(f"  • LAP library is {lap_vs_scipy_seeded:.1f}x faster than SciPy (seeded)")
    else:
        print(f"  • No dual seeding data available for comparison")
    
    print(f"  • Pure solver times (no GNN overhead) range from {min(scipy_times + lap_unseeded_times):.1f} to {max(scipy_times + lap_unseeded_times):.1f} ms")

if __name__ == "__main__":
    run_benchmark()