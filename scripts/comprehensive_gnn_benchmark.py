#!/usr/bin/env python3
"""
Comprehensive GNN Benchmark - Sparse & Uniform Only

Tests progressive_clean_tie_best.pt across all available dataset sizes,
filtering for SPARSE and UNIFORM problem types only.

Generates visualizations similar to the STRING benchmark:
1. Speedup distribution histograms (vs SciPy, vs LAP)
2. Execution time breakdown bar chart
3. Speedup vs problem size scatter plot
"""

import os
import sys
import argparse
from pathlib import Path
import time
import statistics
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import warnings

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Thread limits
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import torch
    from torch.amp import autocast
    torch_available = True
except ImportError:
    torch_available = False
    print("❌ PyTorch not available")
    sys.exit(1)

from scripts.gnn_benchmark import GNNPredictor
from solvers import SciPySolver, LAPSolver, SeededLAPSolver, time_solver_rigorous

warnings.filterwarnings('ignore')


def load_filtered_instances(
    data_dir: Path,
    dataset_name: str,
    filter_types: List[str] = None,
    max_instances: int = None
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, str, int]]:
    """Load instances from dataset, filtering by problem type.
    
    Args:
        data_dir: Root data directory
        dataset_name: Dataset name (e.g., 'small', 'mid_1536', etc.)
        filter_types: List of problem types to include (default: ['sparse', 'uniform'])
        max_instances: Maximum number of instances to load
        
    Returns:
        List of (C, u_true, v_true, family, size) tuples
    """
    if filter_types is None:
        filter_types = ['sparse', 'uniform']
    
    filter_types = [t.lower() for t in filter_types]
    
    test_path = data_dir / dataset_name / "full" / "test.h5"
    
    if not test_path.exists():
        print(f"⚠️  Test file not found: {test_path}")
        return []
    
    instances = []
    
    with h5py.File(test_path, 'r') as f:
        n_problems = len(f['C'])
        families = [t.decode() if isinstance(t, bytes) else t for t in f['family'][:]]
        
        indices = []
        for i, family in enumerate(families):
            if family.lower() in filter_types:
                indices.append(i)
        
        if max_instances:
            indices = indices[:max_instances]
        
        print(f"  Loading {len(indices)} instances (filtered from {n_problems} total)")
        
        for i in indices:
            C_flat = f['C'][i]
            u = f['u'][i]
            v = f['v'][i]
            family = families[i]
            n = int(f['n'][i])
            # Reshape flattened cost matrix to dense n×n
            C = C_flat.reshape(n, n)
            instances.append((C, u, v, family, n))
    
    return instances


def benchmark_instance(
    C: np.ndarray,
    u_true: np.ndarray,
    v_true: np.ndarray,
    family: str,
    gnn_predictor: GNNPredictor,
    scipy_solver: SciPySolver,
    lap_solver: LAPSolver,
    seeded_solver: SeededLAPSolver,
    num_repeats: int = 3
) -> Dict:
    """Benchmark a single instance with timing breakdown.
    
    Returns:
        Dictionary with detailed timing and quality metrics
    """
    n = C.shape[0]
    results = {
        'size': n,
        'family': family,
        'density': (C > 0).sum() / (n * n),
    }
    
    # 1. SciPy baseline
    scipy_timing = time_solver_rigorous(lambda: scipy_solver.solve(C), num_repeats=num_repeats)
    if not scipy_timing['success']:
        results['success'] = False
        return results
    scipy_time = scipy_timing['median']
    _, _, scipy_cost = scipy_solver.solve(C)
    results['scipy_time'] = scipy_time
    results['scipy_cost'] = scipy_cost
    
    # 2. Pure LAP
    lap_timing = time_solver_rigorous(lambda: lap_solver.solve(C), num_repeats=num_repeats)
    if not lap_timing['success']:
        results['success'] = False
        return results
    lap_time = lap_timing['median']
    _, _, lap_cost = lap_solver.solve(C)
    results['lap_time'] = lap_time
    results['lap_cost'] = lap_cost
    
    # 3. GNN inference
    try:
        t0 = time.perf_counter()
        u_pred, v_pred = gnn_predictor.predict(C)
        gnn_time = time.perf_counter() - t0
        results['gnn_time'] = gnn_time
        
        # 4. Seeded LAP
        seeded_timing = time_solver_rigorous(
            lambda: seeded_solver.solve(C, u_pred, v_pred),
            num_repeats=num_repeats
        )
        if not seeded_timing['success']:
            results['success'] = False
            return results
        seeded_time = seeded_timing['median']
        _, _, seeded_cost = seeded_solver.solve(C, u_pred, v_pred)
        results['seeded_time'] = seeded_time
        results['seeded_cost'] = seeded_cost
        
        # Compute metrics
        total_time = gnn_time + seeded_time
        results['total_time'] = total_time
        results['speedup_vs_scipy'] = scipy_time / total_time
        results['speedup_vs_lap'] = lap_time / total_time
        results['primal_gap'] = 100 * (seeded_cost - scipy_cost) / max(abs(scipy_cost), 1e-10)
        results['success'] = True
        
    except Exception as e:
        print(f"  ❌ GNN failed: {e}")
        results['success'] = False
        results['error'] = str(e)
    
    return results


def create_visualizations(
    results: List[Dict],
    output_path: Path,
    dataset_name: str
):
    """Generate visualizations matching the STRING benchmark style, showing all datasets."""
    
    successful = [r for r in results if r.get('success', False)]
    
    if not successful:
        print("⚠️  No successful results to visualize")
        return
    
    # Extract data
    speedups_scipy = [r['speedup_vs_scipy'] for r in successful]
    speedups_lap = [r['speedup_vs_lap'] for r in successful]
    sizes = [r['size'] for r in successful]
    primal_gaps = [r['primal_gap'] for r in successful]
    datasets = [r.get('dataset', 'unknown') for r in successful]
    
    # Create figure with 3 subplots
    fig = plt.figure(figsize=(20, 5))
    
    # ===== Subplot 1: Speedup Distribution vs SciPy =====
    ax1 = plt.subplot(1, 3, 1)
    
    mean_scipy = statistics.mean(speedups_scipy)
    median_scipy = statistics.median(speedups_scipy)
    
    ax1.hist(speedups_scipy, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
    ax1.axvline(median_scipy, color='green', linestyle='--', linewidth=2, label=f'Median: {median_scipy:.2f}×')
    ax1.axvline(mean_scipy, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_scipy:.2f}×')
    ax1.axvline(1.0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
    
    ax1.set_xlabel('Speedup vs SciPy', fontsize=13)
    ax1.set_ylabel('Frequency', fontsize=13)
    ax1.set_title('Speedup Distribution (vs SciPy)', fontsize=15, fontweight='bold')
    ax1.legend(fontsize=12)
    ax1.grid(axis='y', alpha=0.3)
    ax1.tick_params(labelsize=11)
    
    # ===== Subplot 2: Speedup Distribution vs LAP =====
    ax2 = plt.subplot(1, 3, 2)
    
    mean_lap = statistics.mean(speedups_lap)
    median_lap = statistics.median(speedups_lap)
    
    ax2.hist(speedups_lap, bins=20, color='orange', edgecolor='black', alpha=0.7)
    ax2.axvline(median_lap, color='green', linestyle='--', linewidth=2, label=f'Median: {median_lap:.2f}×')
    ax2.axvline(mean_lap, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_lap:.2f}×')
    ax2.axvline(1.0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
    
    ax2.set_xlabel('Speedup vs LAP', fontsize=13)
    ax2.set_ylabel('Frequency', fontsize=13)
    ax2.set_title('Speedup Distribution (vs LAP)', fontsize=15, fontweight='bold')
    ax2.legend(fontsize=12)
    ax2.grid(axis='y', alpha=0.3)
    ax2.tick_params(labelsize=11)
    
    # ===== Subplot 3: Speedup vs Problem Size (colored by primal gap) =====
    ax3 = plt.subplot(1, 3, 3)
    
    # Color by primal gap
    scatter = ax3.scatter(sizes, speedups_scipy, c=primal_gaps, cmap='RdYlGn_r', 
                         s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    # Add trend line
    z = np.polyfit(sizes, speedups_scipy, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(min(sizes), max(sizes), 100)
    ax3.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2, label='Trend')
    
    # Add 1.0x reference line
    ax3.axhline(1.0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
    
    ax3.set_xlabel('Problem Size (n)', fontsize=13)
    ax3.set_ylabel('Speedup vs SciPy', fontsize=13)
    ax3.set_title('Speedup vs Problem Size (colored by solution quality)', fontsize=15, fontweight='bold')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(labelsize=11)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label('Primal Gap (%)', fontsize=11)
    cbar.ax.tick_params(labelsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Visualization saved: {output_path}")


def run_benchmark(
    model_path: Path,
    data_dir: Path,
    dataset_names: List[str],
    filter_types: List[str] = None,
    max_instances: int = None,
    output_dir: Path = None,
    num_repeats: int = 3
):
    """Run comprehensive benchmark across multiple datasets."""
    
    if filter_types is None:
        filter_types = ['sparse', 'uniform']
    
    print("="*80)
    print("COMPREHENSIVE GNN BENCHMARK - Sparse & Uniform Only")
    print("="*80)
    print(f"Model: {model_path.name}")
    print(f"Datasets: {', '.join(dataset_names)}")
    print(f"Filter types: {', '.join(filter_types)}")
    print("="*80)
    
    # Setup output
    if output_dir is None:
        output_dir = project_root / "results" / "comprehensive_benchmark"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Load model
    print(f"\nLoading model...")
    try:
        gnn_predictor = GNNPredictor(str(model_path))
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Initialize solvers
    scipy_solver = SciPySolver()
    lap_solver = LAPSolver()
    seeded_solver = SeededLAPSolver()
    
    # Run benchmarks for each dataset
    all_results = []
    
    for dataset_name in dataset_names:
        print(f"\n{'='*80}")
        print(f"DATASET: {dataset_name}")
        print("="*80)
        
        # Load instances
        instances = load_filtered_instances(
            data_dir,
            dataset_name,
            filter_types,
            max_instances
        )
        
        if not instances:
            print(f"⚠️  No instances loaded for {dataset_name}")
            continue
        
        print(f"\nTesting {len(instances)} instances...")
        
        dataset_results = []
        for i, (C, u_true, v_true, family, size) in enumerate(instances):
            print(f"  [{i+1}/{len(instances)}] {family} {size}×{size}...", end=" ")
            
            result = benchmark_instance(
                C, u_true, v_true, family,
                gnn_predictor,
                scipy_solver, lap_solver, seeded_solver,
                num_repeats
            )
            result['dataset'] = dataset_name
            
            if result.get('success', False):
                print(f"✓ Speedup: {result['speedup_vs_scipy']:.2f}× (vs SciPy), {result['speedup_vs_lap']:.2f}× (vs LAP)")
                dataset_results.append(result)
                all_results.append(result)
            else:
                print("✗ Failed")
        
        # Dataset summary
        if dataset_results:
            print(f"\n{dataset_name} Summary:")
            print(f"  Successful: {len(dataset_results)}/{len(instances)}")
            speedups = [r['speedup_vs_scipy'] for r in dataset_results]
            print(f"  Speedup vs SciPy: mean={statistics.mean(speedups):.2f}×, median={statistics.median(speedups):.2f}×")
    
    # Overall summary
    print(f"\n{'='*80}")
    print("OVERALL SUMMARY")
    print("="*80)
    
    if not all_results:
        print("❌ No successful results!")
        return
    
    print(f"Total successful tests: {len(all_results)}")
    
    speedups_scipy = [r['speedup_vs_scipy'] for r in all_results]
    speedups_lap = [r['speedup_vs_lap'] for r in all_results]
    gaps = [r['primal_gap'] for r in all_results]
    
    print(f"\nSpeedup vs SciPy:")
    print(f"  Mean:   {statistics.mean(speedups_scipy):.2f}×")
    print(f"  Median: {statistics.median(speedups_scipy):.2f}×")
    print(f"  Range:  {min(speedups_scipy):.2f}× - {max(speedups_scipy):.2f}×")
    
    print(f"\nSpeedup vs LAP:")
    print(f"  Mean:   {statistics.mean(speedups_lap):.2f}×")
    print(f"  Median: {statistics.median(speedups_lap):.2f}×")
    print(f"  Range:  {min(speedups_lap):.2f}× - {max(speedups_lap):.2f}×")
    
    print(f"\nPrimal Gap:")
    print(f"  Mean:   {statistics.mean(gaps):.4f}%")
    print(f"  Median: {statistics.median(gaps):.4f}%")
    print(f"  Max:    {max(gaps):.4f}%")
    
    # Save detailed results
    import csv
    csv_path = output_dir / f"comprehensive_benchmark_{timestamp}.csv"
    with open(csv_path, 'w', newline='') as f:
        fieldnames = ['dataset', 'size', 'family', 'density', 'scipy_time', 'lap_time', 
                     'gnn_time', 'seeded_time', 'total_time', 'speedup_vs_scipy', 
                     'speedup_vs_lap', 'primal_gap']
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(all_results)
    
    print(f"\n✅ Results saved: {csv_path}")
    
    # Create combined visualization with ALL results
    print(f"\n{'='*80}")
    print("GENERATING COMBINED VISUALIZATION")
    print("="*80)
    combined_viz_path = output_dir / f"comprehensive_benchmark_{timestamp}.png"
    create_visualizations(all_results, combined_viz_path, "All Datasets Combined")
    
    # Generate per-dataset breakdown table
    print(f"\n{'='*80}")
    print("PER-DATASET BREAKDOWN")
    print("="*80)
    print(f"{'Dataset':<15} {'Count':<8} {'SciPy Mean':<12} {'SciPy Median':<12} {'LAP Mean':<12} {'Gap Mean':<12}")
    print("-" * 85)
    
    dataset_names = sorted(set(r['dataset'] for r in all_results))
    for ds_name in dataset_names:
        ds_results = [r for r in all_results if r['dataset'] == ds_name]
        if ds_results:
            scipy_speeds = [r['speedup_vs_scipy'] for r in ds_results]
            lap_speeds = [r['speedup_vs_lap'] for r in ds_results]
            gaps = [r['primal_gap'] for r in ds_results]
            
            print(f"{ds_name:<15} {len(ds_results):<8} {statistics.mean(scipy_speeds):<12.2f} "
                  f"{statistics.median(scipy_speeds):<12.2f} {statistics.mean(lap_speeds):<12.2f} "
                  f"{statistics.mean(gaps):<12.4f}")
    
    print(f"\n{'='*80}")
    print("BENCHMARK COMPLETE!")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive benchmark for progressive_clean_tie_best.pt on sparse/uniform problems"
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=project_root / "gnn" / "checkpoints_clean" / "progressive_clean_tie_best.pt",
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=project_root / "data" / "generated" / "processed_clean",
        help="Root data directory"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["small", "mid_1536", "mid_2048", "mid_3072", "large_4096"],
        help="Dataset names to test"
    )
    parser.add_argument(
        "--filter-types",
        nargs="+",
        default=["sparse", "uniform"],
        help="Problem types to include"
    )
    parser.add_argument(
        "--max-instances",
        type=int,
        help="Maximum instances per dataset"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory"
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of timing repeats"
    )
    
    args = parser.parse_args()
    
    if not args.model.exists():
        print(f"❌ Model not found: {args.model}")
        sys.exit(1)
    
    run_benchmark(
        model_path=args.model,
        data_dir=args.data_dir,
        dataset_names=args.datasets,
        filter_types=args.filter_types,
        max_instances=args.max_instances,
        output_dir=args.output_dir,
        num_repeats=args.repeats
    )


if __name__ == "__main__":
    main()
