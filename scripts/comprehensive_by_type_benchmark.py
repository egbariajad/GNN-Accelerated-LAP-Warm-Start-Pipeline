#!/usr/bin/env python3
"""
Comprehensive GNN Benchmark By Type - Sparse & Uniform Only

Similar to comprehensive_gnn_benchmark.py but provides detailed breakdown
by problem type (sparse vs uniform), testing across all dataset sizes.

Generates comprehensive visualizations showing:
1. Type comparison histograms
2. Performance heatmap by type and size
3. Execution time breakdown
"""

import os
import sys
import argparse
from pathlib import Path
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

# Import from our main benchmark script
from comprehensive_gnn_benchmark import (
    load_filtered_instances,
    benchmark_instance,
    GNNPredictor,
    SciPySolver,
    LAPSolver,
    SeededLAPSolver
)

warnings.filterwarnings('ignore')


def create_type_visualizations(
    results: List[Dict],
    output_path: Path
):
    """Generate visualizations comparing sparse vs uniform."""
    
    successful = [r for r in results if r.get('success', False)]
    
    if not successful:
        print("⚠️  No successful results to visualize")
        return
    
    # Separate by type
    sparse_results = [r for r in successful if r['family'].lower() == 'sparse']
    uniform_results = [r for r in successful if r['family'].lower() == 'uniform']
    
    fig = plt.figure(figsize=(20, 10))
    
    # ===== Row 1: Speedup distributions =====
    # Subplot 1: Sparse speedup vs SciPy
    ax1 = plt.subplot(2, 3, 1)
    if sparse_results:
        sparse_speeds = [r['speedup_vs_scipy'] for r in sparse_results]
        ax1.hist(sparse_speeds, bins=15, color='forestgreen', edgecolor='black', alpha=0.7)
        ax1.axvline(statistics.mean(sparse_speeds), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {statistics.mean(sparse_speeds):.2f}×')
        ax1.axvline(statistics.median(sparse_speeds), color='blue', linestyle='--', linewidth=2,
                   label=f'Median: {statistics.median(sparse_speeds):.2f}×')
        ax1.axvline(1.0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
        ax1.set_title('SPARSE: Speedup vs SciPy', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Speedup', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.legend(fontsize=10)
        ax1.grid(axis='y', alpha=0.3)
    
    # Subplot 2: Uniform speedup vs SciPy
    ax2 = plt.subplot(2, 3, 2)
    if uniform_results:
        uniform_speeds = [r['speedup_vs_scipy'] for r in uniform_results]
        ax2.hist(uniform_speeds, bins=15, color='coral', edgecolor='black', alpha=0.7)
        ax2.axvline(statistics.mean(uniform_speeds), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {statistics.mean(uniform_speeds):.2f}×')
        ax2.axvline(statistics.median(uniform_speeds), color='blue', linestyle='--', linewidth=2,
                   label=f'Median: {statistics.median(uniform_speeds):.2f}×')
        ax2.axvline(1.0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
        ax2.set_title('UNIFORM: Speedup vs SciPy', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Speedup', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.legend(fontsize=10)
        ax2.grid(axis='y', alpha=0.3)
    
    # Subplot 3: Combined comparison
    ax3 = plt.subplot(2, 3, 3)
    if sparse_results and uniform_results:
        sparse_speeds = [r['speedup_vs_scipy'] for r in sparse_results]
        uniform_speeds = [r['speedup_vs_scipy'] for r in uniform_results]
        
        ax3.hist([sparse_speeds, uniform_speeds], bins=15, label=['Sparse', 'Uniform'],
                color=['forestgreen', 'coral'], alpha=0.6, edgecolor='black')
        ax3.axvline(1.0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
        ax3.set_title('Type Comparison: Speedup Distribution', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Speedup vs SciPy', fontsize=12)
        ax3.set_ylabel('Frequency', fontsize=12)
        ax3.legend(fontsize=11)
        ax3.grid(axis='y', alpha=0.3)
    
    # ===== Row 2: Performance by size =====
    # Collect data by dataset and type
    datasets = sorted(set(r['dataset'] for r in successful))
    
    # Subplot 4: Speedup by size (sparse)
    ax4 = plt.subplot(2, 3, 4)
    if sparse_results:
        sparse_by_dataset = {}
        for ds in datasets:
            ds_sparse = [r for r in sparse_results if r['dataset'] == ds]
            if ds_sparse:
                sparse_by_dataset[ds] = statistics.mean([r['speedup_vs_scipy'] for r in ds_sparse])
        
        if sparse_by_dataset:
            x_labels = list(sparse_by_dataset.keys())
            y_values = list(sparse_by_dataset.values())
            bars = ax4.bar(range(len(x_labels)), y_values, color='forestgreen', alpha=0.7, edgecolor='black')
            ax4.axhline(1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
            ax4.set_xticks(range(len(x_labels)))
            ax4.set_xticklabels(x_labels, rotation=45, ha='right')
            ax4.set_title('SPARSE: Mean Speedup by Dataset', fontsize=14, fontweight='bold')
            ax4.set_ylabel('Speedup vs SciPy', fontsize=12)
            ax4.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}×', ha='center', va='bottom', fontsize=9)
    
    # Subplot 5: Speedup by size (uniform)
    ax5 = plt.subplot(2, 3, 5)
    if uniform_results:
        uniform_by_dataset = {}
        for ds in datasets:
            ds_uniform = [r for r in uniform_results if r['dataset'] == ds]
            if ds_uniform:
                uniform_by_dataset[ds] = statistics.mean([r['speedup_vs_scipy'] for r in ds_uniform])
        
        if uniform_by_dataset:
            x_labels = list(uniform_by_dataset.keys())
            y_values = list(uniform_by_dataset.values())
            bars = ax5.bar(range(len(x_labels)), y_values, color='coral', alpha=0.7, edgecolor='black')
            ax5.axhline(1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
            ax5.set_xticks(range(len(x_labels)))
            ax5.set_xticklabels(x_labels, rotation=45, ha='right')
            ax5.set_title('UNIFORM: Mean Speedup by Dataset', fontsize=14, fontweight='bold')
            ax5.set_ylabel('Speedup vs SciPy', fontsize=12)
            ax5.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax5.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}×', ha='center', va='bottom', fontsize=9)
    
    # Subplot 6: Primal gap comparison
    ax6 = plt.subplot(2, 3, 6)
    if sparse_results and uniform_results:
        sparse_gaps = [r['primal_gap'] for r in sparse_results]
        uniform_gaps = [r['primal_gap'] for r in uniform_results]
        
        box_data = [sparse_gaps, uniform_gaps]
        bp = ax6.boxplot(box_data, labels=['Sparse', 'Uniform'], patch_artist=True)
        bp['boxes'][0].set_facecolor('forestgreen')
        bp['boxes'][0].set_alpha(0.6)
        bp['boxes'][1].set_facecolor('coral')
        bp['boxes'][1].set_alpha(0.6)
        
        ax6.set_title('Solution Quality: Primal Gap Comparison', fontsize=14, fontweight='bold')
        ax6.set_ylabel('Primal Gap (%)', fontsize=12)
        ax6.grid(axis='y', alpha=0.3)
        ax6.tick_params(labelsize=11)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Type visualization saved: {output_path}")


def run_benchmark_by_type(
    model_path: Path,
    data_dir: Path,
    dataset_names: List[str],
    filter_types: List[str] = None,
    max_instances: int = None,
    output_dir: Path = None,
    num_repeats: int = 3
):
    """Run benchmark with type-specific analysis."""
    
    if filter_types is None:
        filter_types = ['sparse', 'uniform']
    
    print("="*80)
    print("COMPREHENSIVE GNN BENCHMARK BY TYPE - Sparse & Uniform")
    print("="*80)
    print(f"Model: {model_path.name}")
    print(f"Datasets: {', '.join(dataset_names)}")
    print("="*80)
    
    # Setup output
    if output_dir is None:
        output_dir = project_root / "results" / "comprehensive_benchmark_by_type"
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
    
    for dataset_name in dataset_names:
        print(f"\n{'='*80}")
        print(f"DATASET: {dataset_name}")
        print("="*80)
        
        instances = load_filtered_instances(
            data_dir,
            dataset_name,
            filter_types,
            max_instances
        )
        
        if not instances:
            continue
        
        print(f"\nTesting {len(instances)} instances...")
        
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
                print(f"✓ {result['speedup_vs_scipy']:.2f}×")
                all_results.append(result)
            else:
                print("✗")
    
    # Analysis
    print(f"\n{'='*80}")
    print("TYPE-SPECIFIC ANALYSIS")
    print("="*80)
    
    if not all_results:
        print("❌ No successful results!")
        return
    
    # Separate by type
    sparse_results = [r for r in all_results if r['family'].lower() == 'sparse']
    uniform_results = [r for r in all_results if r['family'].lower() == 'uniform']
    
    print(f"\nSPARSE ({len(sparse_results)} instances):")
    if sparse_results:
        speeds = [r['speedup_vs_scipy'] for r in sparse_results]
        gaps = [r['primal_gap'] for r in sparse_results]
        print(f"  Speedup: mean={statistics.mean(speeds):.2f}×, median={statistics.median(speeds):.2f}×")
        print(f"  Range: {min(speeds):.2f}× - {max(speeds):.2f}×")
        print(f"  Primal gap: mean={statistics.mean(gaps):.4f}%, max={max(gaps):.4f}%")
    
    print(f"\nUNIFORM ({len(uniform_results)} instances):")
    if uniform_results:
        speeds = [r['speedup_vs_scipy'] for r in uniform_results]
        gaps = [r['primal_gap'] for r in uniform_results]
        print(f"  Speedup: mean={statistics.mean(speeds):.2f}×, median={statistics.median(speeds):.2f}×")
        print(f"  Range: {min(speeds):.2f}× - {max(speeds):.2f}×")
        print(f"  Primal gap: mean={statistics.mean(gaps):.4f}%, max={max(gaps):.4f}%")
    
    # Save results
    import csv
    csv_path = output_dir / f"by_type_benchmark_{timestamp}.csv"
    with open(csv_path, 'w', newline='') as f:
        fieldnames = ['dataset', 'size', 'family', 'density', 'scipy_time', 'lap_time',
                     'gnn_time', 'seeded_time', 'total_time', 'speedup_vs_scipy',
                     'speedup_vs_lap', 'primal_gap']
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(all_results)
    
    print(f"\n✅ Results saved: {csv_path}")
    
    # Generate visualization
    viz_path = output_dir / f"by_type_benchmark_{timestamp}.png"
    create_type_visualizations(all_results, viz_path)
    
    print(f"\n{'='*80}")
    print("BENCHMARK COMPLETE!")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive benchmark by problem type"
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
    
    run_benchmark_by_type(
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
