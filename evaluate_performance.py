#!/usr/bin/env python3
"""
Performance evaluation script for the trained GNN model.
Tests warm-start performance vs baseline solvers.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, List
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from gnn import DualGNN, compute_features
from gnn.train import LapDataset, collate
from torch.utils.data import DataLoader
from solvers.lap_solver import SeededLAPSolver
from solvers.dual_computation import compute_oracle_duals


def load_model(checkpoint_path: Path, device: torch.device) -> DualGNN:
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model = DualGNN(
        hidden_dim=checkpoint["hidden_dim"],
        layers=checkpoint["layers"], 
        heads=checkpoint["heads"],
        dropout=checkpoint["dropout"]
    )
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    return model


def evaluate_instance(
    model: DualGNN,
    cost_matrix: np.ndarray,
    device: torch.device
) -> Dict[str, float]:
    """Evaluate a single instance."""
    
    n = cost_matrix.shape[0]
    
    # Baseline: solve directly
    solver = SeededLAPSolver()
    
    start_time = time.perf_counter()
    baseline_assignment, baseline_cost = solver.solve(cost_matrix)
    baseline_time = time.perf_counter() - start_time
    
    # GNN prediction
    with torch.no_grad():
        # Compute features
        features = compute_features(cost_matrix)
        
        # Convert to tensors
        cost_tensor = torch.from_numpy(cost_matrix.astype(np.float32)).unsqueeze(0).to(device)
        row_feat = torch.from_numpy(features.row_feat.astype(np.float32)).unsqueeze(0).to(device)
        col_feat = torch.from_numpy(features.col_feat.astype(np.float32)).unsqueeze(0).to(device)
        edge_feat = torch.from_numpy(features.edge_feat.astype(np.float32)).unsqueeze(0).to(device)
        mask = torch.ones(1, n, dtype=torch.bool, device=device)
        
        # GNN prediction
        start_time = time.perf_counter()
        preds = model(edge_feat, row_feat, col_feat, mask=mask)
        u_pred = preds["u"][0].cpu().numpy()
        gnn_time = time.perf_counter() - start_time
        
        # Project to feasible duals
        u_pred = u_pred - u_pred.mean()  # Center
        cost_minus = cost_matrix - u_pred[:, None]
        v_proj = cost_minus.min(axis=0)
        
        # Warm-start solve
        reduced_costs = cost_matrix - u_pred[:, None] - v_proj[None, :]
        
        start_time = time.perf_counter()
        warmstart_assignment, warmstart_cost = solver.solve(reduced_costs)
        warmstart_solve_time = time.perf_counter() - start_time
        
        total_warmstart_time = gnn_time + warmstart_solve_time
    
    # Compute oracle duals for comparison
    oracle_u, oracle_v = compute_oracle_duals(cost_matrix, baseline_assignment)
    
    # Metrics
    results = {
        "n": n,
        "baseline_cost": baseline_cost,
        "warmstart_cost": warmstart_cost,
        "baseline_time": baseline_time,
        "warmstart_time": total_warmstart_time,
        "gnn_time": gnn_time,
        "solve_time": warmstart_solve_time,
        "speedup": baseline_time / total_warmstart_time,
        "cost_error": abs(warmstart_cost - baseline_cost),
        "optimal": abs(warmstart_cost - baseline_cost) < 1e-6,
        "u_mse": np.mean((u_pred - oracle_u) ** 2),
    }
    
    return results


def main():
    """Run performance evaluation."""
    
    print("🔬 GNN Warm-Start Performance Evaluation")
    print("=" * 50)
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load model
    checkpoint_path = Path("checkpoints/small_bucket_model.pt")
    if not checkpoint_path.exists():
        print(f"❌ Model checkpoint not found: {checkpoint_path}")
        return False
        
    model = load_model(checkpoint_path, device)
    print(f"✅ Model loaded: {checkpoint_path}")
    
    # Load test data
    test_paths = [
        Path("data/buckets/small/test.h5"),
        Path("data/tmp/test.h5"),
        Path("data/tmp/train.h5")  # Fallback
    ]
    
    test_path = None
    for path in test_paths:
        if path.exists():
            test_path = path
            break
    
    if not test_path:
        print("❌ No test data found")
        return False
        
    print(f"📊 Test data: {test_path}")
    
    # Load dataset
    dataset = LapDataset(test_path)
    loader = DataLoader(dataset, batch_size=1, collate_fn=collate)
    
    # Evaluate instances
    results = []
    n_evaluated = 0
    max_instances = 10  # Limit for quick evaluation
    
    print(f"\n🏃 Evaluating up to {max_instances} instances...")
    print("Instance | Size | Baseline | Warm-start | Speedup | Optimal | U-MSE")
    print("-" * 70)
    
    for batch in loader:
        if n_evaluated >= max_instances:
            break
            
        cost_matrix = batch.cost[0].numpy()
        n = int(batch.mask[0].sum().item())
        cost_matrix = cost_matrix[:n, :n]
        
        try:
            result = evaluate_instance(model, cost_matrix, device)
            results.append(result)
            
            print(f"{n_evaluated+1:8d} | {result['n']:4d} | "
                  f"{result['baseline_time']*1000:8.2f}ms | "
                  f"{result['warmstart_time']*1000:9.2f}ms | "
                  f"{result['speedup']:7.2f}x | "
                  f"{'✅' if result['optimal'] else '❌':7s} | "
                  f"{result['u_mse']:8.4f}")
            
            n_evaluated += 1
            
        except Exception as e:
            print(f"❌ Failed on instance {n_evaluated+1}: {e}")
            continue
    
    if not results:
        print("❌ No successful evaluations")
        return False
    
    # Summary statistics
    print("\n📈 Summary Statistics:")
    print("=" * 50)
    
    speedups = [r["speedup"] for r in results]
    optimal_rate = sum(r["optimal"] for r in results) / len(results)
    u_mses = [r["u_mse"] for r in results]
    
    print(f"Instances evaluated: {len(results)}")
    print(f"Optimal rate: {optimal_rate:.1%}")
    print(f"Speedup - Mean: {np.mean(speedups):.2f}x")
    print(f"Speedup - Median: {np.median(speedups):.2f}x") 
    print(f"Speedup - Range: [{np.min(speedups):.2f}x, {np.max(speedups):.2f}x]")
    print(f"Dual prediction MSE: {np.mean(u_mses):.4f}")
    
    # Performance assessment
    print(f"\n🎯 Performance Assessment:")
    if optimal_rate >= 0.9:
        print("✅ Excellent: >90% optimal solutions")
    elif optimal_rate >= 0.7:
        print("🟡 Good: >70% optimal solutions")
    else:
        print("🔴 Needs improvement: <70% optimal solutions")
        
    if np.median(speedups) >= 1.5:
        print("✅ Good speedup: >1.5x median acceleration")
    elif np.median(speedups) >= 1.0:
        print("🟡 Modest speedup: >1.0x (breaking even)")
    else:
        print("🔴 Slowdown: <1.0x (need optimization)")
    
    print(f"\n🚀 Next Steps:")
    print(f"1. Train on larger dataset once SLURM job completes")
    print(f"2. Test on larger problem sizes (1024, 2048, 4096)")
    print(f"3. Implement dual-first training curriculum for better convergence")
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)