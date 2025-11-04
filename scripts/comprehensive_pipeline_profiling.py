#!/usr/bin/env python3
"""
Comprehensive Pipeline Profiling - Sparse & Uniform

Detailed profiling of the GNN+LAP pipeline on sparse and uniform problems
across all dataset sizes. Provides timing breakdown for every stage.
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

import numpy as np
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

from comprehensive_gnn_benchmark import load_filtered_instances, GNNPredictor
from solvers import SciPySolver, LAPSolver, SeededLAPSolver

warnings.filterwarnings('ignore')


def profile_instance(
    C: np.ndarray,
    u_true: np.ndarray,
    v_true: np.ndarray,
    family: str,
    gnn_predictor: GNNPredictor,
    num_repeats: int = 5
) -> Dict:
    """Detailed profiling of all pipeline stages."""
    
    n = C.shape[0]
    results = {
        'size': n,
        'family': family,
        'density': (C > 0).sum() / (n * n),
    }
    
    try:
        # Stage 1: SciPy baseline
        scipy_solver = SciPySolver()
        scipy_times = []
        for _ in range(num_repeats):
            t0 = time.perf_counter()
            rows, cols, cost = scipy_solver.solve(C)
            scipy_times.append(time.perf_counter() - t0)
        results['scipy_time'] = statistics.median(scipy_times)
        results['scipy_cost'] = cost
        
        # Stage 2: LAP baseline
        lap_solver = LAPSolver()
        lap_times = []
        for _ in range(num_repeats):
            t0 = time.perf_counter()
            rows, cols, cost = lap_solver.solve(C)
            lap_times.append(time.perf_counter() - t0)
        results['lap_time'] = statistics.median(lap_times)
        results['lap_cost'] = cost
        
        # Stage 3: GNN inference (detailed timing)
        # - Feature computation
        # - GPU transfer (if applicable)
        # - Forward pass
        # - V computation
        
        feature_times = []
        transfer_times = []
        forward_times = []
        v_comp_times = []
        
        for _ in range(num_repeats):
            # Feature computation
            t0 = time.perf_counter()
            if gnn_predictor.row_only:
                from gnn import compute_row_features, compute_row_features_torch
                if gnn_predictor.use_cuda:
                    cost_tensor = torch.from_numpy(C).float()
                    feature_times.append(time.perf_counter() - t0)
                    
                    # GPU transfer
                    t0 = time.perf_counter()
                    cost_tensor = cost_tensor.to(gnn_predictor.device)
                    row_feat_tensor = compute_row_features_torch(cost_tensor)
                    transfer_times.append(time.perf_counter() - t0)
                    
                    # Forward pass
                    t0 = time.perf_counter()
                    row_tensor = row_feat_tensor.unsqueeze(0)
                    cost_batch = cost_tensor.unsqueeze(0)
                    mask = torch.ones((1, n), dtype=torch.bool, device=gnn_predictor.device)
                    with torch.inference_mode():
                        with autocast('cuda', enabled=gnn_predictor.amp_enabled):
                            outputs = gnn_predictor.model(row_tensor, cost=cost_batch, mask=mask)
                    u_pred_tensor = outputs['u'].squeeze(0)
                    forward_times.append(time.perf_counter() - t0)
                    
                    # V computation
                    t0 = time.perf_counter()
                    v_pred_tensor = torch.min(cost_tensor - u_pred_tensor[:, None], dim=0)[0]
                    u_pred = u_pred_tensor.cpu().numpy()[:n]
                    v_pred = v_pred_tensor.cpu().numpy()
                    v_comp_times.append(time.perf_counter() - t0)
                else:
                    # CPU path
                    row_feat = compute_row_features(C)
                    feature_times.append(time.perf_counter() - t0)
                    
                    t0 = time.perf_counter()
                    row_tensor = torch.from_numpy(row_feat).float().unsqueeze(0)
                    cost_batch = torch.from_numpy(C).float().unsqueeze(0)
                    transfer_times.append(time.perf_counter() - t0)
                    
                    t0 = time.perf_counter()
                    mask = torch.ones((1, n), dtype=torch.bool)
                    with torch.inference_mode():
                        outputs = gnn_predictor.model(row_tensor, cost=cost_batch, mask=mask)
                    u_pred = outputs['u'].squeeze(0).numpy()[:n]
                    forward_times.append(time.perf_counter() - t0)
                    
                    t0 = time.perf_counter()
                    v_pred = np.min(C - u_pred[:, None], axis=0)
                    v_comp_times.append(time.perf_counter() - t0)
        
        results['feature_time'] = statistics.median(feature_times)
        results['transfer_time'] = statistics.median(transfer_times)
        results['forward_time'] = statistics.median(forward_times)
        results['v_comp_time'] = statistics.median(v_comp_times)
        results['gnn_total_time'] = sum([
            results['feature_time'],
            results['transfer_time'],
            results['forward_time'],
            results['v_comp_time']
        ])
        
        # Stage 4: Seeded LAP
        seeded_solver = SeededLAPSolver()
        seeded_times = []
        for _ in range(num_repeats):
            t0 = time.perf_counter()
            rows, cols, cost = seeded_solver.solve(C, u_pred, v_pred)
            seeded_times.append(time.perf_counter() - t0)
        results['seeded_time'] = statistics.median(seeded_times)
        results['seeded_cost'] = cost
        
        # Compute metrics
        results['pipeline_total_time'] = results['gnn_total_time'] + results['seeded_time']
        results['speedup_vs_scipy'] = results['scipy_time'] / results['pipeline_total_time']
        results['speedup_vs_lap'] = results['lap_time'] / results['pipeline_total_time']
        results['primal_gap'] = 100 * (results['seeded_cost'] - results['scipy_cost']) / max(abs(results['scipy_cost']), 1e-10)
        
        # Timing breakdown percentages
        total = results['pipeline_total_time']
        results['feature_pct'] = 100 * results['feature_time'] / total
        results['transfer_pct'] = 100 * results['transfer_time'] / total
        results['forward_pct'] = 100 * results['forward_time'] / total
        results['v_comp_pct'] = 100 * results['v_comp_time'] / total
        results['seeded_pct'] = 100 * results['seeded_time'] / total
        
        results['success'] = True
        
    except Exception as e:
        print(f"  ❌ Profiling failed: {e}")
        results['success'] = False
        results['error'] = str(e)
    
    return results


def create_profiling_visualization(results: List[Dict], output_path: Path):
    """Generate detailed profiling visualizations."""
    
    successful = [r for r in results if r.get('success', False)]
    
    if not successful:
        print("⚠️  No successful results to visualize")
        return
    
    fig = plt.figure(figsize=(20, 12))
    
    # ===== Subplot 1: Time breakdown by stage (stacked bar) =====
    ax1 = plt.subplot(2, 3, 1)
    
    # Group by dataset and type
    datasets = sorted(set(r.get('dataset', 'unknown') for r in successful))
    types = ['sparse', 'uniform']
    
    x = np.arange(len(datasets))
    width = 0.35
    
    for i, ptype in enumerate(types):
        type_results = [r for r in successful if r['family'].lower() == ptype]
        
        feature_times = []
        transfer_times = []
        forward_times = []
        v_comp_times = []
        seeded_times = []
        
        for ds in datasets:
            ds_results = [r for r in type_results if r.get('dataset', '') == ds]
            if ds_results:
                feature_times.append(statistics.mean([r['feature_time']*1000 for r in ds_results]))
                transfer_times.append(statistics.mean([r['transfer_time']*1000 for r in ds_results]))
                forward_times.append(statistics.mean([r['forward_time']*1000 for r in ds_results]))
                v_comp_times.append(statistics.mean([r['v_comp_time']*1000 for r in ds_results]))
                seeded_times.append(statistics.mean([r['seeded_time']*1000 for r in ds_results]))
            else:
                feature_times.append(0)
                transfer_times.append(0)
                forward_times.append(0)
                v_comp_times.append(0)
                seeded_times.append(0)
        
        offset = width * (i - 0.5)
        bottom = np.zeros(len(datasets))
        
        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc']
        labels = ['Features', 'Transfer', 'Forward', 'V Comp', 'Seeded LAP']
        
        for times, color, label in zip(
            [feature_times, transfer_times, forward_times, v_comp_times, seeded_times],
            colors, labels
        ):
            ax1.bar(x + offset, times, width, bottom=bottom, 
                   label=f'{ptype.capitalize()} {label}' if i == 0 else None,
                   color=color, alpha=0.7)
            bottom += times
    
    ax1.set_xlabel('Dataset', fontsize=12)
    ax1.set_ylabel('Time (ms)', fontsize=12)
    ax1.set_title('Pipeline Stage Breakdown', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets, rotation=45, ha='right')
    ax1.legend(fontsize=9)
    ax1.grid(axis='y', alpha=0.3)
    
    # ===== Subplot 2: Percentage breakdown =====
    ax2 = plt.subplot(2, 3, 2)
    
    stages = ['Features', 'Transfer', 'Forward', 'V Comp', 'Seeded LAP']
    stage_keys = ['feature_pct', 'transfer_pct', 'forward_pct', 'v_comp_pct', 'seeded_pct']
    
    sparse_pcts = [statistics.mean([r[k] for r in successful if r['family'].lower() == 'sparse']) 
                   for k in stage_keys]
    uniform_pcts = [statistics.mean([r[k] for r in successful if r['family'].lower() == 'uniform']) 
                    for k in stage_keys]
    
    x = np.arange(len(stages))
    width = 0.35
    
    ax2.bar(x - width/2, sparse_pcts, width, label='Sparse', color='forestgreen', alpha=0.7)
    ax2.bar(x + width/2, uniform_pcts, width, label='Uniform', color='coral', alpha=0.7)
    
    ax2.set_ylabel('Percentage of Total Time (%)', fontsize=12)
    ax2.set_title('Stage Time Distribution', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(stages, rotation=45, ha='right')
    ax2.legend(fontsize=11)
    ax2.grid(axis='y', alpha=0.3)
    
    # ===== Subplot 3: Speedup by size =====
    ax3 = plt.subplot(2, 3, 3)
    
    sizes = sorted(set(r['size'] for r in successful))
    
    for ptype, color in [('sparse', 'forestgreen'), ('uniform', 'coral')]:
        type_results = [r for r in successful if r['family'].lower() == ptype]
        speedups = [statistics.mean([r['speedup_vs_scipy'] for r in type_results if r['size'] == s])
                   for s in sizes]
        ax3.plot(sizes, speedups, 'o-', label=ptype.capitalize(), 
                color=color, linewidth=2.5, markersize=8)
    
    ax3.axhline(1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax3.set_xlabel('Problem Size', fontsize=12)
    ax3.set_ylabel('Speedup vs SciPy', fontsize=12)
    ax3.set_title('Overall Speedup', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    
    # ===== Subplot 4: GNN vs LAP time comparison =====
    ax4 = plt.subplot(2, 3, 4)
    
    for ptype, color in [('sparse', 'forestgreen'), ('uniform', 'coral')]:
        type_results = [r for r in successful if r['family'].lower() == ptype]
        
        gnn_times = [statistics.mean([r['gnn_total_time']*1000 for r in type_results if r['size'] == s])
                    for s in sizes]
        seeded_times = [statistics.mean([r['seeded_time']*1000 for r in type_results if r['size'] == s])
                       for s in sizes]
        
        ax4.plot(sizes, gnn_times, 'o-', label=f'{ptype.capitalize()} GNN', 
                color=color, linewidth=2, markersize=6)
        ax4.plot(sizes, seeded_times, 's--', label=f'{ptype.capitalize()} LAP', 
                color=color, alpha=0.5, linewidth=1.5, markersize=5)
    
    ax4.set_xlabel('Problem Size', fontsize=12)
    ax4.set_ylabel('Time (ms)', fontsize=12)
    ax4.set_title('GNN vs Seeded LAP Time', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')
    
    # ===== Subplot 5: Overhead analysis =====
    ax5 = plt.subplot(2, 3, 5)
    
    # GNN overhead vs total time
    for ptype, color in [('sparse', 'forestgreen'), ('uniform', 'coral')]:
        type_results = [r for r in successful if r['family'].lower() == ptype]
        
        overhead_pcts = [100 * statistics.mean([r['gnn_total_time'] / r['pipeline_total_time'] 
                                                for r in type_results if r['size'] == s])
                        for s in sizes]
        
        ax5.plot(sizes, overhead_pcts, 'o-', label=ptype.capitalize(), 
                color=color, linewidth=2.5, markersize=8)
    
    ax5.set_xlabel('Problem Size', fontsize=12)
    ax5.set_ylabel('GNN Time (% of total)', fontsize=12)
    ax5.set_title('GNN Overhead Trend', fontsize=14, fontweight='bold')
    ax5.legend(fontsize=11)
    ax5.grid(True, alpha=0.3)
    
    # ===== Subplot 6: Solution quality =====
    ax6 = plt.subplot(2, 3, 6)
    
    sparse_gaps = [r['primal_gap'] for r in successful if r['family'].lower() == 'sparse']
    uniform_gaps = [r['primal_gap'] for r in successful if r['family'].lower() == 'uniform']
    
    bp = ax6.boxplot([sparse_gaps, uniform_gaps], labels=['Sparse', 'Uniform'], 
                     patch_artist=True)
    bp['boxes'][0].set_facecolor('forestgreen')
    bp['boxes'][0].set_alpha(0.6)
    bp['boxes'][1].set_facecolor('coral')
    bp['boxes'][1].set_alpha(0.6)
    
    ax6.set_ylabel('Primal Gap (%)', fontsize=12)
    ax6.set_title('Solution Quality', fontsize=14, fontweight='bold')
    ax6.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Profiling visualization saved: {output_path}")


def run_pipeline_profiling(
    model_path: Path,
    data_dir: Path,
    dataset_names: List[str],
    filter_types: List[str],
    max_instances: int = None,
    output_dir: Path = None,
    num_repeats: int = 5
):
    """Run comprehensive pipeline profiling."""
    
    print("="*80)
    print("COMPREHENSIVE PIPELINE PROFILING - Sparse & Uniform")
    print("="*80)
    print(f"Model: {model_path.name}")
    print(f"Datasets: {', '.join(dataset_names)}")
    print(f"Profiling repeats: {num_repeats}")
    print("="*80)
    
    # Setup output
    if output_dir is None:
        output_dir = project_root / "results" / "pipeline_profiling"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Load model
    print(f"\nLoading model...")
    try:
        gnn_predictor = GNNPredictor(str(model_path))
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Run profiling
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
        
        print(f"\nProfiling {len(instances)} instances...")
        
        for i, (C, u_true, v_true, family, size) in enumerate(instances):
            print(f"  [{i+1}/{len(instances)}] {family} {size}×{size}...", end=" ")
            
            result = profile_instance(
                C, u_true, v_true, family,
                gnn_predictor,
                num_repeats
            )
            result['dataset'] = dataset_name
            
            if result.get('success', False):
                print(f"✓ {result['speedup_vs_scipy']:.2f}× speedup")
                all_results.append(result)
            else:
                print("✗")
    
    # Analysis
    print(f"\n{'='*80}")
    print("PROFILING SUMMARY")
    print("="*80)
    
    if not all_results:
        print("❌ No successful results!")
        return
    
    print(f"\nTotal profiled: {len(all_results)}")
    
    # Average timing breakdown
    print(f"\nAverage Pipeline Breakdown (all instances):")
    print(f"  Feature computation: {statistics.mean([r['feature_pct'] for r in all_results]):.1f}%")
    print(f"  GPU transfer:        {statistics.mean([r['transfer_pct'] for r in all_results]):.1f}%")
    print(f"  Forward pass:        {statistics.mean([r['forward_pct'] for r in all_results]):.1f}%")
    print(f"  V computation:       {statistics.mean([r['v_comp_pct'] for r in all_results]):.1f}%")
    print(f"  Seeded LAP:          {statistics.mean([r['seeded_pct'] for r in all_results]):.1f}%")
    
    # Save results
    import csv
    csv_path = output_dir / f"pipeline_profiling_{timestamp}.csv"
    with open(csv_path, 'w', newline='') as f:
        fieldnames = ['dataset', 'size', 'family', 'scipy_time', 'lap_time',
                     'feature_time', 'transfer_time', 'forward_time', 'v_comp_time',
                     'gnn_total_time', 'seeded_time', 'pipeline_total_time',
                     'speedup_vs_scipy', 'speedup_vs_lap', 'primal_gap',
                     'feature_pct', 'transfer_pct', 'forward_pct', 'v_comp_pct', 'seeded_pct']
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(all_results)
    
    print(f"\n✅ Results saved: {csv_path}")
    
    # Generate visualization
    viz_path = output_dir / f"pipeline_profiling_{timestamp}.png"
    create_profiling_visualization(all_results, viz_path)
    
    print(f"\n{'='*80}")
    print("PIPELINE PROFILING COMPLETE!")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive pipeline profiling"
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
        help="Dataset names to profile"
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
        "--repeats",
        type=int,
        default=5,
        help="Number of timing repeats per stage"
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
    
    run_pipeline_profiling(
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
