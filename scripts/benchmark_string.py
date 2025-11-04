#!/usr/bin/env python3
"""
STRING Dataset Benchmark - Real-World PPI Testing

Benchmark GNN models on protein-protein interaction data from STRING database.
Tests the full pipeline (GNN inference + seeded LAP) against baselines.
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

# Thread limits for fair comparison
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import h5py

try:
    import torch
    torch_available = True
except ImportError:
    torch_available = False
    print("❌ PyTorch not available")
    sys.exit(1)

# Import benchmark infrastructure
from scripts.gnn_benchmark import GNNPredictor
from solvers import SciPySolver, LAPSolver, SeededLAPSolver, time_solver_rigorous

warnings.filterwarnings('ignore')


def load_string_instances(h5_path: Path, max_instances: int = None) -> List[Tuple[np.ndarray, str]]:
    """Load cost matrices from STRING dataset H5 file.
    
    Args:
        h5_path: Path to test.h5 file
        max_instances: Maximum number of instances to load
        
    Returns:
        List of (cost_matrix, instance_name) tuples
    """
    instances = []
    
    with h5py.File(h5_path, 'r') as f:
        keys = sorted(f.keys(), key=lambda x: int(x.split('_')[1]))
        
        if max_instances:
            keys = keys[:max_instances]
            
        print(f"Loading {len(keys)} instances from STRING dataset...")
        
        for key in keys:
            cost_matrix = f[key]['cost_matrix'][:]
            instances.append((cost_matrix, key))
            
    return instances


def benchmark_instance(
    cost_matrix: np.ndarray,
    instance_name: str,
    gnn_predictor: Optional[GNNPredictor],
    verbose: bool = False
) -> Dict:
    """Benchmark a single instance with all methods.
    
    Args:
        cost_matrix: Cost matrix (n×n)
        instance_name: Name/ID of instance
        gnn_predictor: GNN model for prediction (optional)
        verbose: Print detailed output
        
    Returns:
        Dictionary with benchmark results
    """
    n = cost_matrix.shape[0]
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Instance: {instance_name}")
        print(f"Size: {n} × {n}")
        print(f"{'='*60}")
    
    results = {
        'instance': instance_name,
        'size': n,
        'density': (cost_matrix > 0).sum() / (n * n),
    }
    
    # 1. SciPy baseline
    if verbose:
        print("\n1. SciPy (linear_sum_assignment)...")
    scipy_solver = SciPySolver()
    scipy_timing = time_solver_rigorous(lambda: scipy_solver.solve(cost_matrix), num_repeats=10)
    if not scipy_timing['success']:
        print(f"   ❌ SciPy failed: {scipy_timing.get('error')}")
        return results
    scipy_time = scipy_timing['median']
    _, _, scipy_cost = scipy_solver.solve(cost_matrix)  # Returns (rows, cols, cost)
    results['scipy_time'] = scipy_time
    results['scipy_cost'] = scipy_cost
    if verbose:
        print(f"   Time: {scipy_time:.4f}s | Cost: {scipy_cost:.2f}")
    
    # 2. Pure LAP solver
    if verbose:
        print("\n2. LAP (Hungarian algorithm)...")
    lap_solver = LAPSolver()
    lap_timing = time_solver_rigorous(lambda: lap_solver.solve(cost_matrix), num_repeats=10)
    if not lap_timing['success']:
        print(f"   ❌ LAP failed: {lap_timing.get('error')}")
        return results
    lap_time = lap_timing['median']
    _, _, lap_cost = lap_solver.solve(cost_matrix)  # Returns (rows, cols, cost)
    results['lap_time'] = lap_time
    results['lap_cost'] = lap_cost
    if verbose:
        print(f"   Time: {lap_time:.4f}s | Cost: {lap_cost:.2f}")
    
    # 3. GNN-seeded LAP (if model provided)
    if gnn_predictor:
        if verbose:
            print("\n3. GNN-Seeded LAP...")
        
        try:
            # GNN inference
            t0 = time.perf_counter()
            u_pred, v_pred = gnn_predictor.predict(cost_matrix)
            gnn_time = time.perf_counter() - t0
            
            # Seeded LAP
            seeded_solver = SeededLAPSolver()
            seeded_timing = time_solver_rigorous(
                lambda: seeded_solver.solve(cost_matrix, u_pred, v_pred),
                num_repeats=10
            )
            
            if not seeded_timing['success']:
                print(f"   ❌ Seeded LAP failed: {seeded_timing.get('error')}")
                return results
                
            seeded_time = seeded_timing['median']
            _, _, seeded_cost = seeded_solver.solve(cost_matrix, u_pred, v_pred)  # Returns (rows, cols, cost)
            
            total_time = gnn_time + seeded_time
            
            results['gnn_time'] = gnn_time
            results['seeded_time'] = seeded_time
            results['total_time'] = total_time
            results['seeded_cost'] = seeded_cost
            
            # Calculate metrics
            results['speedup_vs_scipy'] = scipy_time / total_time
            results['speedup_vs_lap'] = lap_time / total_time
            results['primal_gap'] = 100 * (seeded_cost - scipy_cost) / max(abs(scipy_cost), 1e-10)
            
            # Initialization quality
            init_cost = (cost_matrix * np.exp(u_pred[:, None] + v_pred[None, :])).sum()
            results['init_gap'] = 100 * (init_cost - scipy_cost) / max(abs(scipy_cost), 1e-10)
            
            if verbose:
                print(f"   GNN inference: {gnn_time:.4f}s")
                print(f"   Seeded LAP: {seeded_time:.4f}s")
                print(f"   Total time: {total_time:.4f}s")
                print(f"   Cost: {seeded_cost:.2f}")
                print(f"   Primal gap: {results['primal_gap']:.2f}%")
                print(f"   Speedup vs SciPy: {results['speedup_vs_scipy']:.2f}×")
                print(f"   Speedup vs LAP: {results['speedup_vs_lap']:.2f}×")
                
        except Exception as e:
            print(f"   ❌ GNN inference failed: {e}")
            results['error'] = str(e)
    
    return results


def run_string_benchmark(
    h5_path: Path,
    model_files: List[str] = None,
    checkpoint_dirs: List[Path] = None,
    max_instances: int = None,
    output_dir: Path = None,
    verbose: bool = True
):
    """Run full benchmark on STRING dataset.
    
    Args:
        h5_path: Path to STRING test.h5 file
        model_files: List of model filenames/paths to test
        checkpoint_dirs: Directories to search for models
        max_instances: Maximum instances to test
        output_dir: Directory for output files
        verbose: Print detailed progress
    """
    print("="*80)
    print("STRING DATASET BENCHMARK - Real-World PPI Testing")
    print("="*80)
    
    # Setup output directory
    if output_dir is None:
        output_dir = project_root / "logs" / "string_benchmark"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"string_benchmark_{timestamp}.csv"
    summary_path = output_dir / f"string_benchmark_{timestamp}_summary.txt"
    
    # Load instances
    instances = load_string_instances(h5_path, max_instances)
    print(f"\nLoaded {len(instances)} instances")
    print(f"Size range: {min(c.shape[0] for c, _ in instances)} - {max(c.shape[0] for c, _ in instances)}")
    
    # Default models if none specified
    if model_files is None:
        model_files = [
            "one_gnn_small_full_clean_h192L4.pt",
            "one_gnn_mid1536_full_clean_h192L4.pt",
            "one_gnn_mid2048_clean_h192L4.pt",
            "one_gnn_mid3072_clean_h192L4.pt",
        ]
    
    # Default checkpoint directories
    if checkpoint_dirs is None:
        checkpoint_dirs = [
            project_root / "gnn" / "checkpoints_clean",
            project_root / "gnn" / "checkpoints",
        ]
    
    # Find and load models
    models_to_test = []
    for model_name in model_files:
        found = False
        for ckpt_dir in checkpoint_dirs:
            model_path = ckpt_dir / model_name
            if model_path.exists():
                models_to_test.append(model_path)
                found = True
                break
        if not found:
            print(f"⚠️  Model not found: {model_name}")
    
    if not models_to_test:
        print("❌ No models found! Running baseline-only benchmark...")
    
    print(f"\nModels to test: {len(models_to_test)}")
    for mp in models_to_test:
        print(f"  - {mp.name}")
    
    # Run benchmarks
    all_results = []
    
    # First, run baseline-only (no GNN)
    print(f"\n{'='*80}")
    print("BASELINE BENCHMARKS (SciPy + LAP)")
    print(f"{'='*80}")
    
    for cost_matrix, instance_name in instances:
        result = benchmark_instance(cost_matrix, instance_name, None, verbose=verbose)
        result['model'] = 'baseline'
        all_results.append(result)
    
    # Then test each model
    for model_path in models_to_test:
        print(f"\n{'='*80}")
        print(f"MODEL: {model_path.name}")
        print(f"{'='*80}")
        
        try:
            predictor = GNNPredictor(str(model_path))
            
            for cost_matrix, instance_name in instances:
                result = benchmark_instance(cost_matrix, instance_name, predictor, verbose=verbose)
                result['model'] = model_path.name
                all_results.append(result)
                
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            continue
    
    # Save detailed results to CSV
    print(f"\n{'='*80}")
    print("SAVING RESULTS")
    print(f"{'='*80}")
    
    import csv
    with open(csv_path, 'w', newline='') as f:
        if all_results:
            # Collect all possible fieldnames from all results
            all_fieldnames = set()
            for result in all_results:
                all_fieldnames.update(result.keys())
            fieldnames = sorted(all_fieldnames)
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
    
    print(f"✅ Detailed results: {csv_path}")
    
    # Generate summary statistics
    with open(summary_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("STRING DATASET BENCHMARK SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Dataset: {h5_path}\n")
        f.write(f"Instances tested: {len(instances)}\n")
        f.write(f"Models tested: {len(models_to_test)}\n\n")
        
        # Per-model summary
        models = set(r['model'] for r in all_results)
        for model in sorted(models):
            model_results = [r for r in all_results if r['model'] == model]
            
            f.write(f"\nModel: {model}\n")
            f.write("-" * 60 + "\n")
            
            if model != 'baseline' and 'primal_gap' in model_results[0]:
                gaps = [r['primal_gap'] for r in model_results if 'primal_gap' in r]
                speedups_scipy = [r['speedup_vs_scipy'] for r in model_results if 'speedup_vs_scipy' in r]
                speedups_lap = [r['speedup_vs_lap'] for r in model_results if 'speedup_vs_lap' in r]
                
                if gaps:
                    f.write(f"  Primal Gap:\n")
                    f.write(f"    Mean:   {statistics.mean(gaps):.2f}%\n")
                    f.write(f"    Median: {statistics.median(gaps):.2f}%\n")
                    f.write(f"    Min:    {min(gaps):.2f}%\n")
                    f.write(f"    Max:    {max(gaps):.2f}%\n\n")
                
                if speedups_scipy:
                    f.write(f"  Speedup vs SciPy:\n")
                    f.write(f"    Mean:   {statistics.mean(speedups_scipy):.2f}×\n")
                    f.write(f"    Median: {statistics.median(speedups_scipy):.2f}×\n\n")
                
                if speedups_lap:
                    f.write(f"  Speedup vs LAP:\n")
                    f.write(f"    Mean:   {statistics.mean(speedups_lap):.2f}×\n")
                    f.write(f"    Median: {statistics.median(speedups_lap):.2f}×\n\n")
            
            # Baseline timing
            scipy_times = [r['scipy_time'] for r in model_results if 'scipy_time' in r]
            if scipy_times:
                f.write(f"  SciPy timing: {statistics.mean(scipy_times):.4f}s avg\n")
    
    print(f"✅ Summary: {summary_path}")
    print(f"\n{'='*80}")
    print("BENCHMARK COMPLETE!")
    print(f"{'='*80}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark GNN models on STRING protein-protein interaction dataset"
    )
    parser.add_argument(
        "--h5-path",
        type=Path,
        default=project_root / "data" / "string_dataset" / "processed" / "test.h5",
        help="Path to STRING test.h5 file",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="Model filenames to test (searches in checkpoint dirs)",
    )
    parser.add_argument(
        "--checkpoint-dirs",
        nargs="+",
        type=Path,
        help="Directories to search for models",
    )
    parser.add_argument(
        "--max-instances",
        type=int,
        help="Maximum number of instances to test",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for results",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity",
    )
    
    args = parser.parse_args()
    
    # Check if test file exists
    if not args.h5_path.exists():
        print(f"❌ Test file not found: {args.h5_path}")
        print("\nRun this to generate it:")
        print("  python scripts/prepare_string_dataset.py --num-instances 20")
        sys.exit(1)
    
    run_string_benchmark(
        h5_path=args.h5_path,
        model_files=args.models,
        checkpoint_dirs=args.checkpoint_dirs,
        max_instances=args.max_instances,
        output_dir=args.output_dir,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
