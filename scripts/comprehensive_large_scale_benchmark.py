#!/usr/bin/env python3
"""
Large-Scale Synthetic Benchmark - Sparse & Uniform

Tests progressive_clean_tie_best.pt on synthetic sparse and uniform problems
at very large sizes (4096, 8192, 16384) to evaluate scaling behavior beyond training data.
"""

import os
import sys
import argparse
from pathlib import Path
import time
import statistics
from typing import Dict, List
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import torch
    torch_available = True
except ImportError:
    torch_available = False
    print("❌ PyTorch not available")
    sys.exit(1)

from scripts.gnn_large_scale_benchmark import GNNPredictor
from solvers import SciPySolver, LAPSolver, SeededLAPSolver, time_solver_rigorous

warnings.filterwarnings('ignore')


def generate_sparse_matrix(n: int, density: float, seed: int) -> np.ndarray:
    """Generate sparse cost matrix."""
    rng = np.random.RandomState(seed)
    C = np.zeros((n, n))
    mask = rng.random((n, n)) < density
    C[mask] = rng.uniform(0.1, 1.0, size=mask.sum())
    # Ensure feasibility - add small values to zero entries
    C[C == 0] = rng.uniform(0.01, 0.05, size=(C == 0).sum())
    return C


def generate_uniform_matrix(n: int, seed: int) -> np.ndarray:
    """Generate uniform random cost matrix."""
    rng = np.random.RandomState(seed)
    return rng.uniform(0, 1, size=(n, n))


def benchmark_instance(
    C: np.ndarray,
    problem_type: str,
    size: int,
    instance_id: int,
    gnn_predictor: GNNPredictor,
    scipy_solver: SciPySolver,
    lap_solver: LAPSolver,
    seeded_solver: SeededLAPSolver,
    compute_scipy: bool = True
) -> Dict:
    """Benchmark a single synthetic instance."""
    
    n = C.shape[0]
    results = {
        'size': n,
        'type': problem_type,
        'instance_id': instance_id,
        'density': (C > 0).sum() / (n * n),
    }
    
    try:
        # SciPy baseline (may be slow for large problems)
        if compute_scipy:
            print(f"    SciPy...", end=' ', flush=True)
            scipy_timing = time_solver_rigorous(lambda: scipy_solver.solve(C), num_repeats=3)
            if scipy_timing['success']:
                scipy_time = scipy_timing['median']
                _, _, scipy_cost = scipy_solver.solve(C)
                results['scipy_time'] = scipy_time
                results['scipy_cost'] = scipy_cost
                print(f"{scipy_time:.3f}s", end=' ')
            else:
                print("FAILED", end=' ')
                results['success'] = False
                return results
        else:
            print(f"    Skipping SciPy (too large)...", end=' ')
        
        # LAP baseline
        print(f"LAP...", end=' ', flush=True)
        lap_timing = time_solver_rigorous(lambda: lap_solver.solve(C), num_repeats=3)
        if lap_timing['success']:
            lap_time = lap_timing['median']
            _, _, lap_cost = lap_solver.solve(C)
            results['lap_time'] = lap_time
            results['lap_cost'] = lap_cost
            print(f"{lap_time:.3f}s", end=' ')
        else:
            print("FAILED", end=' ')
            results['success'] = False
            return results
        
        # GNN inference
        print(f"GNN...", end=' ', flush=True)
        t0 = time.perf_counter()
        u_pred, v_pred = gnn_predictor.predict(C)
        gnn_time = time.perf_counter() - t0
        results['gnn_time'] = gnn_time
        print(f"{gnn_time:.3f}s", end=' ')
        
        # Seeded LAP
        print(f"Seeded...", end=' ', flush=True)
        seeded_timing = time_solver_rigorous(
            lambda: seeded_solver.solve(C, u_pred, v_pred),
            num_repeats=3
        )
        if seeded_timing['success']:
            seeded_time = seeded_timing['median']
            _, _, seeded_cost = seeded_solver.solve(C, u_pred, v_pred)
            results['seeded_time'] = seeded_time
            results['seeded_cost'] = seeded_cost
            
            total_time = gnn_time + seeded_time
            results['total_time'] = total_time
            
            if compute_scipy:
                results['speedup_vs_scipy'] = scipy_time / total_time
                results['primal_gap'] = 100 * (seeded_cost - scipy_cost) / max(abs(scipy_cost), 1e-10)
                print(f"✓ {results['speedup_vs_scipy']:.2f}× speedup, {results['primal_gap']:.4f}% gap")
            else:
                results['speedup_vs_lap'] = lap_time / total_time
                print(f"✓ {results['speedup_vs_lap']:.2f}× vs LAP")
            
            results['success'] = True
        else:
            print("FAILED")
            results['success'] = False
            
    except Exception as e:
        print(f"ERROR: {e}")
        results['success'] = False
        results['error'] = str(e)
    
    return results


def create_scaling_visualization(results: List[Dict], output_path: Path):
    """Generate visualization showing scaling behavior."""
    
    successful = [r for r in results if r.get('success', False)]
    
    if not successful:
        print("⚠️  No successful results to visualize")
        return
    
    # Separate by type
    sparse_results = [r for r in successful if r['type'] == 'sparse']
    uniform_results = [r for r in successful if r['type'] == 'uniform']
    
    fig = plt.figure(figsize=(20, 10))
    
    # ===== Subplot 1: Execution time scaling =====
    ax1 = plt.subplot(2, 3, 1)
    
    for results_subset, label, color in [
        (sparse_results, 'Sparse', 'forestgreen'),
        (uniform_results, 'Uniform', 'coral')
    ]:
        if results_subset:
            # Group by size
            sizes = sorted(set(r['size'] for r in results_subset))
            scipy_times = []
            lap_times = []
            gnn_times = []
            seeded_times = []
            
            for size in sizes:
                size_results = [r for r in results_subset if r['size'] == size]
                if 'scipy_time' in size_results[0]:
                    scipy_times.append(statistics.mean([r['scipy_time'] for r in size_results]))
                lap_times.append(statistics.mean([r['lap_time'] for r in size_results]))
                gnn_times.append(statistics.mean([r['gnn_time'] for r in size_results]))
                seeded_times.append(statistics.mean([r['seeded_time'] for r in size_results]))
            
            if scipy_times:
                ax1.plot(sizes[:len(scipy_times)], scipy_times, 'o--', label=f'{label} SciPy', 
                        color=color, alpha=0.7, linewidth=2, markersize=8)
            ax1.plot(sizes, lap_times, 's--', label=f'{label} LAP', 
                    color=color, alpha=0.5, linewidth=1.5, markersize=6)
            
            # Stacked GNN+Seeded
            total_times = [g + s for g, s in zip(gnn_times, seeded_times)]
            ax1.plot(sizes, total_times, '^-', label=f'{label} GNN+LAP', 
                    color=color, linewidth=2.5, markersize=8)
    
    ax1.set_xlabel('Problem Size (n)', fontsize=13)
    ax1.set_ylabel('Time (seconds)', fontsize=13)
    ax1.set_title('Execution Time Scaling', fontsize=15, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    ax1.set_xscale('log')
    
    # ===== Subplot 2: Speedup scaling =====
    ax2 = plt.subplot(2, 3, 2)
    
    for results_subset, label, color in [
        (sparse_results, 'Sparse', 'forestgreen'),
        (uniform_results, 'Uniform', 'coral')
    ]:
        if results_subset:
            sizes = sorted(set(r['size'] for r in results_subset))
            speedups_scipy = []
            speedups_lap = []
            
            for size in sizes:
                size_results = [r for r in results_subset if r['size'] == size]
                if 'speedup_vs_scipy' in size_results[0]:
                    speedups_scipy.append(statistics.mean([r['speedup_vs_scipy'] for r in size_results]))
                if 'speedup_vs_lap' in size_results[0]:
                    speedups_lap.append(statistics.mean([r['speedup_vs_lap'] for r in size_results]))
            
            if speedups_scipy:
                ax2.plot(sizes[:len(speedups_scipy)], speedups_scipy, 'o-', 
                        label=f'{label} vs SciPy', color=color, linewidth=2.5, markersize=8)
            if speedups_lap:
                ax2.plot(sizes[:len(speedups_lap)], speedups_lap, 's--', 
                        label=f'{label} vs LAP', color=color, alpha=0.6, linewidth=2, markersize=6)
    
    ax2.axhline(1.0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
    ax2.set_xlabel('Problem Size (n)', fontsize=13)
    ax2.set_ylabel('Speedup', fontsize=13)
    ax2.set_title('Speedup Scaling', fontsize=15, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    
    # ===== Subplot 3: GNN overhead percentage =====
    ax3 = plt.subplot(2, 3, 3)
    
    for results_subset, label, color in [
        (sparse_results, 'Sparse', 'forestgreen'),
        (uniform_results, 'Uniform', 'coral')
    ]:
        if results_subset:
            sizes = sorted(set(r['size'] for r in results_subset))
            gnn_percentages = []
            
            for size in sizes:
                size_results = [r for r in results_subset if r['size'] == size]
                percentages = [100 * r['gnn_time'] / r['total_time'] for r in size_results]
                gnn_percentages.append(statistics.mean(percentages))
            
            ax3.plot(sizes, gnn_percentages, 'o-', label=label, 
                    color=color, linewidth=2.5, markersize=8)
    
    ax3.set_xlabel('Problem Size (n)', fontsize=13)
    ax3.set_ylabel('GNN Time (%)', fontsize=13)
    ax3.set_title('GNN Overhead Percentage', fontsize=15, fontweight='bold')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.set_xscale('log')
    
    # ===== Subplot 4: Time breakdown for largest size =====
    ax4 = plt.subplot(2, 3, 4)
    
    largest_size = max(r['size'] for r in successful)
    largest_results = [r for r in successful if r['size'] == largest_size]
    
    types = sorted(set(r['type'] for r in largest_results))
    x = np.arange(len(types))
    width = 0.35
    
    gnn_times = [statistics.mean([r['gnn_time'] for r in largest_results if r['type'] == t]) for t in types]
    seeded_times = [statistics.mean([r['seeded_time'] for r in largest_results if r['type'] == t]) for t in types]
    
    ax4.bar(x, gnn_times, width, label='GNN Inference', color='orange', alpha=0.8)
    ax4.bar(x, seeded_times, width, bottom=gnn_times, label='Seeded LAP', color='steelblue', alpha=0.8)
    
    ax4.set_xlabel('Problem Type', fontsize=13)
    ax4.set_ylabel('Time (seconds)', fontsize=13)
    ax4.set_title(f'Time Breakdown at n={largest_size}', fontsize=15, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(types)
    ax4.legend(fontsize=11)
    ax4.grid(axis='y', alpha=0.3)
    
    # ===== Subplot 5: Solution quality =====
    ax5 = plt.subplot(2, 3, 5)
    
    quality_results = [r for r in successful if 'primal_gap' in r]
    if quality_results:
        sparse_gaps = [r['primal_gap'] for r in quality_results if r['type'] == 'sparse']
        uniform_gaps = [r['primal_gap'] for r in quality_results if r['type'] == 'uniform']
        
        data = []
        labels = []
        if sparse_gaps:
            data.append(sparse_gaps)
            labels.append('Sparse')
        if uniform_gaps:
            data.append(uniform_gaps)
            labels.append('Uniform')
        
        bp = ax5.boxplot(data, labels=labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], ['forestgreen', 'coral']):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        ax5.set_ylabel('Primal Gap (%)', fontsize=13)
        ax5.set_title('Solution Quality Distribution', fontsize=15, fontweight='bold')
        ax5.grid(axis='y', alpha=0.3)
    
    # ===== Subplot 6: Memory scaling estimate =====
    ax6 = plt.subplot(2, 3, 6)
    
    sizes = sorted(set(r['size'] for r in successful))
    memory_gb = [n * n * 8 / (1024**3) for n in sizes]  # Float64 matrix size
    
    ax6.plot(sizes, memory_gb, 'o-', color='purple', linewidth=2.5, markersize=8)
    ax6.set_xlabel('Problem Size (n)', fontsize=13)
    ax6.set_ylabel('Memory (GB)', fontsize=13)
    ax6.set_title('Dense Matrix Memory Requirement', fontsize=15, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    ax6.set_xscale('log')
    ax6.set_yscale('log')
    
    # Add size annotations
    for size, mem in zip(sizes, memory_gb):
        ax6.annotate(f'{size}', (size, mem), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Large-scale visualization saved: {output_path}")


def run_large_scale_benchmark(
    model_path: Path,
    sizes: List[int],
    instances_per_size: int,
    sparse_density: float,
    output_dir: Path = None,
    compute_scipy: bool = True
):
    """Run large-scale synthetic benchmark."""
    
    print("="*80)
    print("LARGE-SCALE SYNTHETIC BENCHMARK - Sparse & Uniform")
    print("="*80)
    print(f"Model: {model_path.name}")
    print(f"Sizes: {', '.join(map(str, sizes))}")
    print(f"Instances per size: {instances_per_size}")
    print(f"Sparse density: {sparse_density}")
    print("="*80)
    
    # Setup output
    if output_dir is None:
        output_dir = project_root / "results" / "large_scale_benchmark"
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
    
    # Run benchmarks
    all_results = []
    
    for size in sizes:
        print(f"\n{'='*80}")
        print(f"SIZE: {size}×{size}")
        print("="*80)
        
        # Skip SciPy for very large problems
        skip_scipy = size > 10000 or not compute_scipy
        
        for instance_id in range(instances_per_size):
            # Test sparse
            print(f"\n  [Sparse {instance_id+1}/{instances_per_size}]")
            C_sparse = generate_sparse_matrix(size, sparse_density, seed=instance_id)
            result_sparse = benchmark_instance(
                C_sparse, 'sparse', size, instance_id,
                gnn_predictor, scipy_solver, lap_solver, seeded_solver,
                compute_scipy=not skip_scipy
            )
            if result_sparse.get('success', False):
                all_results.append(result_sparse)
            
            # Test uniform
            print(f"\n  [Uniform {instance_id+1}/{instances_per_size}]")
            C_uniform = generate_uniform_matrix(size, seed=instance_id)
            result_uniform = benchmark_instance(
                C_uniform, 'uniform', size, instance_id,
                gnn_predictor, scipy_solver, lap_solver, seeded_solver,
                compute_scipy=not skip_scipy
            )
            if result_uniform.get('success', False):
                all_results.append(result_uniform)
    
    # Analysis
    print(f"\n{'='*80}")
    print("LARGE-SCALE BENCHMARK SUMMARY")
    print("="*80)
    
    if not all_results:
        print("❌ No successful results!")
        return
    
    print(f"\nTotal successful tests: {len(all_results)}")
    
    # Per-size summary
    for size in sizes:
        size_results = [r for r in all_results if r['size'] == size]
        if size_results:
            print(f"\n{size}×{size}:")
            sparse = [r for r in size_results if r['type'] == 'sparse']
            uniform = [r for r in size_results if r['type'] == 'uniform']
            
            for subset, label in [(sparse, 'Sparse'), (uniform, 'Uniform')]:
                if subset:
                    if 'speedup_vs_scipy' in subset[0]:
                        speeds = [r['speedup_vs_scipy'] for r in subset]
                        print(f"  {label}: {statistics.mean(speeds):.2f}× speedup (vs SciPy)")
                    elif 'speedup_vs_lap' in subset[0]:
                        speeds = [r['speedup_vs_lap'] for r in subset]
                        print(f"  {label}: {statistics.mean(speeds):.2f}× speedup (vs LAP)")
    
    # Save results
    import csv
    csv_path = output_dir / f"large_scale_benchmark_{timestamp}.csv"
    with open(csv_path, 'w', newline='') as f:
        fieldnames = ['size', 'type', 'instance_id', 'density', 'scipy_time', 'lap_time',
                     'gnn_time', 'seeded_time', 'total_time', 'speedup_vs_scipy',
                     'speedup_vs_lap', 'primal_gap']
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(all_results)
    
    print(f"\n✅ Results saved: {csv_path}")
    
    # Generate visualization
    viz_path = output_dir / f"large_scale_benchmark_{timestamp}.png"
    create_scaling_visualization(all_results, viz_path)
    
    print(f"\n{'='*80}")
    print("LARGE-SCALE BENCHMARK COMPLETE!")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Large-scale synthetic benchmark (sparse & uniform)"
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=project_root / "gnn" / "checkpoints_clean" / "progressive_clean_tie_best.pt",
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=[4096, 8192, 16384],
        help="Problem sizes to test"
    )
    parser.add_argument(
        "--instances",
        type=int,
        default=3,
        help="Instances per size"
    )
    parser.add_argument(
        "--sparse-density",
        type=float,
        default=0.3,
        help="Density for sparse matrices"
    )
    parser.add_argument(
        "--no-scipy",
        action="store_true",
        help="Skip SciPy baseline (for very large problems)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    if not args.model.exists():
        print(f"❌ Model not found: {args.model}")
        sys.exit(1)
    
    run_large_scale_benchmark(
        model_path=args.model,
        sizes=args.sizes,
        instances_per_size=args.instances,
        sparse_density=args.sparse_density,
        output_dir=args.output_dir,
        compute_scipy=not args.no_scipy
    )


if __name__ == "__main__":
    main()
