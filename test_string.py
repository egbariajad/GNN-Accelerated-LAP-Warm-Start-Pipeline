#!/usr/bin/env python3
"""
Test the trained GNN model on STRING protein interaction dataset.
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
from data.processors import iter_real_instances


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
    
    # Compute features
    features = compute_features(cost_matrix)
    
    # Create dataset and dataloader
    dataset = LapDataset([cost_matrix], [features])
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate)
    
    # Get batch
    batch = next(iter(dataloader))
    batch = {k: v.to(device) for k, v in batch.items()}
    
    # Predict duals
    with torch.no_grad():
        pred_u, pred_v = model(batch)
        pred_u = pred_u.cpu().numpy().flatten()
        pred_v = pred_v.cpu().numpy().flatten()
    
    # Solve with predicted duals
    solver = SeededLAPSolver(cost_matrix, pred_u, pred_v)
    pred_obj, pred_time = solver.solve()
    
    # Solve without duals (baseline)
    baseline_solver = SeededLAPSolver(cost_matrix, np.zeros(n), np.zeros(n))
    baseline_obj, baseline_time = baseline_solver.solve()
    
    # Compute oracle duals
    oracle_u, oracle_v = compute_oracle_duals(cost_matrix)
    oracle_solver = SeededLAPSolver(cost_matrix, oracle_u, oracle_v)
    oracle_obj, oracle_time = oracle_solver.solve()
    
    return {
        "n": n,
        "pred_obj": pred_obj,
        "pred_time": pred_time,
        "baseline_obj": baseline_obj,
        "baseline_time": baseline_time,
        "oracle_obj": oracle_obj,
        "oracle_time": oracle_time,
        "speedup": baseline_time / pred_time if pred_time > 0 else float('inf'),
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device")
    parser.add_argument("--string-tax-id", type=str, default="9606", help="STRING taxonomy ID")
    parser.add_argument("--subset-size", type=int, default=512, help="Subset size for cost matrices")
    parser.add_argument("--num-matrices", type=int, default=10, help="Number of matrices to test")
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    model = load_model(args.checkpoint, device)
    
    print(f"Testing on STRING {args.string_tax_id} with {args.num_matrices} matrices of size {args.subset_size//2}x{args.subset_size//2}")
    
    results = []
    for instance in iter_real_instances(string_tax_id=args.string_tax_id, 
                                       string_subset_size=args.subset_size, 
                                       string_num_matrices=args.num_matrices):
        print(f"Evaluating {instance.name}")
        result = evaluate_instance(model, instance.cost, device)
        result["name"] = instance.name
        results.append(result)
        print(f"  Speedup: {result['speedup']:.2f}x, Pred time: {result['pred_time']:.3f}s, Baseline: {result['baseline_time']:.3f}s")
    
    # Summary
    speedups = [r["speedup"] for r in results]
    print("\nSummary:")
    print(f"Average speedup: {np.mean(speedups):.2f}x")
    print(f"Median speedup: {np.median(speedups):.2f}x")
    print(f"Min speedup: {min(speedups):.2f}x")
    print(f"Max speedup: {max(speedups):.2f}x")


if __name__ == "__main__":
    main()