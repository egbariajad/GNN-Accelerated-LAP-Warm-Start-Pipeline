#!/usr/bin/env python3
"""
Step-by-step visualization of the OneGNN → seeded LAP pipeline on a single cost matrix.

The script mirrors the flow used in `scripts/gnn_benchmark.py`, but prints every stage:
  1. Load cost matrix (random or from HDF5 dataset)
  2. Compute row features for OneGNN
  3. Run OneGNN forward pass to predict row duals `û`
  4. Reconstruct column duals `v̂ = min_j (C_ij - û_i)`
  5. Inspect reduced costs and (optionally) feasibility projection
  6. Solve with seeded LAP and compare against the cold-start LAP baseline

Example:
    python scripts/show_pipeline_walkthrough.py \\
        --model gnn/checkpoints/one_gnn_small_full.pt \\
        --dataset data/generated/processed_clean/small/full/test.h5 \\
        --index 0
"""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path
from typing import Optional, Tuple
import time

import numpy as np
import h5py

import torch

# Add project root for relative imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gnn import OneGNN, compute_row_features, compute_row_features_torch  # type: ignore
from solvers import LAPSolver, SeededLAPSolver  # type: ignore
from solvers.advanced_dual import check_dual_feasible, project_feasible  # type: ignore


def format_header(title: str) -> str:
    bar = "=" * 80
    return f"\n{bar}\n{title}\n{bar}"


def preview_matrix(matrix: np.ndarray, rows: int = 5, cols: int = 5) -> str:
    r = min(rows, matrix.shape[0])
    c = min(cols, matrix.shape[1])
    snippet = matrix[:r, :c]
    lines = ["top-left "
             f"{r}x{c} block (values rounded to 3 decimals):"]
    for row in snippet:
        lines.append("  " + " ".join(f"{val:8.3f}" for val in row))
    if matrix.shape[0] > r or matrix.shape[1] > c:
        lines.append("  ...")
    return "\n".join(lines)


def load_cost_matrix(dataset: Optional[Path], index: int, size: int, seed: int) -> Tuple[np.ndarray, dict]:
    """Load a cost matrix either from an HDF5 dataset or by sampling uniformly."""
    if dataset is not None:
        with h5py.File(dataset, "r") as f:
            n = int(f["n"][index])
            C_flat = f["C"][index]
            metadata = {
                "n": n,
                "family": f["family"][index].decode() if hasattr(f["family"][index], "decode") else f["family"][index],
                "tag": f["tag"][index].decode() if "tag" in f and hasattr(f["tag"][index], "decode") else "",
                "noise_std": float(f["noise_std"][index]) if "noise_std" in f else None,
            }
            matrix = np.asarray(C_flat, dtype=np.float64).reshape(n, n)
    else:
        rng = np.random.default_rng(seed)
        matrix = rng.uniform(0.0, 1.0, size=(size, size)).astype(np.float64)
        metadata = {"n": size, "family": "uniform_random", "tag": "", "noise_std": None}
    return matrix, metadata


def rows_cols_to_assignment(rows: np.ndarray, cols: np.ndarray, n: int) -> np.ndarray:
    """Convert LAP solver row/col outputs into a per-row column lookup."""
    assignment = np.full(n, -1, dtype=int)
    for r, c in zip(rows.astype(int), cols.astype(int)):
        if 0 <= r < n:
            assignment[r] = c
    return assignment


def load_one_gnn_checkpoint(path: Path, device: torch.device) -> Tuple[OneGNN, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    hidden = checkpoint.get("hidden_dim")
    layers = checkpoint.get("layers")
    dropout = checkpoint.get("dropout", 0.1)
    row_feat_dim = checkpoint.get("row_feat_dim") or compute_row_features(np.zeros((1, 1))).shape[1]

    if hidden is None or layers is None:
        raise ValueError(f"Checkpoint {path} missing OneGNN hyperparameters (hidden/layers).")

    model = OneGNN(
        in_dim=row_feat_dim,
        hidden=hidden,
        layers=layers,
        dropout=dropout,
    ).to(device)

    model.load_state_dict(state_dict)
    model.eval()

    meta = {
        "hidden": hidden,
        "layers": layers,
        "dropout": dropout,
        "row_feat_dim": row_feat_dim,
        "architecture": checkpoint.get("architecture", "one_gnn"),
    }
    return model, meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Path to OneGNN checkpoint.")
    parser.add_argument("--dataset", type=Path, help="Optional HDF5 dataset to read from.")
    parser.add_argument("--index", type=int, default=0, help="Dataset index to visualise.")
    parser.add_argument("--size", type=int, default=8, help="Matrix size when sampling random data.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for synthetic matrix.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-project", action="store_true", help="Skip feasibility projection stage.")
    parser.add_argument("--preview", type=int, default=5, help="Preview dimension for matrices/features.")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    print(format_header("Stage 0: Load Cost Matrix"))
    cost, meta = load_cost_matrix(args.dataset, args.index, args.size, args.seed)
    n = cost.shape[0]
    print(f"matrix size: {n} x {n}")
    print(f"family: {meta.get('family')} | tag: {meta.get('tag')} | noise_std: {meta.get('noise_std')}")
    print(preview_matrix(cost, rows=args.preview, cols=args.preview))
    print(f"stats: min={cost.min():.4f}, max={cost.max():.4f}, mean={cost.mean():.4f}")

    print(format_header("Stage 1: Feature Engineering (row features)"))
    if device.type == "cuda":
        cost_tensor = torch.from_numpy(cost).float().to(device)
        row_features_tensor = compute_row_features_torch(cost_tensor)
        row_features = row_features_tensor.cpu().numpy()
    else:
        row_features = compute_row_features(cost)
        cost_tensor = torch.from_numpy(cost).float().to(device)
        row_features_tensor = torch.from_numpy(row_features).float().to(device)

    print(f"row feature tensor shape: {row_features.shape}")
    print(f"first {args.preview} rows (rounded):")
    for i in range(min(args.preview, row_features.shape[0])):
        row_str = " ".join(f"{val:7.3f}" for val in row_features[i][:args.preview])
        print(f"  row {i:02d}: {row_str} ...")

    print(format_header("Stage 2: Load OneGNN Checkpoint"))
    model, model_info = load_one_gnn_checkpoint(args.model, device)
    print(textwrap.dedent(f"""
        model: OneGNN
          hidden: {model_info['hidden']}
          layers: {model_info['layers']}
          dropout: {model_info['dropout']}
          expected feature dim: {model_info['row_feat_dim']}
          device: {device}
    """).strip())

    print(format_header("Stage 3: GNN Forward Pass (predict row duals û)"))
    mask = torch.ones((1, n), dtype=torch.bool, device=device)
    row_batch = row_features_tensor.unsqueeze(0)
    cost_batch = cost_tensor.unsqueeze(0)

    start = time.perf_counter()
    with torch.inference_mode():
        outputs = model(row_batch, cost=cost_batch, mask=mask)
    torch.cuda.synchronize() if device.type == "cuda" else None
    elapsed = time.perf_counter() - start

    u_pred = outputs["u"].squeeze(0).detach().cpu().numpy()
    print(f"GNN runtime: {elapsed*1e3:.2f} ms")
    print(f"û shape: {u_pred.shape} | mean={u_pred.mean():.4f}, std={u_pred.std():.4f}")
    print(f"first {args.preview} entries of û: {u_pred[:args.preview]}")

    print(format_header("Stage 4: Compute Column Duals via Min-Trick"))
    v_pred = np.min(cost - u_pred[:, None], axis=0)
    print(f"v̂ shape: {v_pred.shape} | mean={v_pred.mean():.4f}, std={v_pred.std():.4f}")
    print(f"first {args.preview} entries of v̂: {v_pred[:args.preview]}")

    print(format_header("Stage 5: Reduced Cost Diagnostics"))
    reduced = cost - u_pred[:, None] - v_pred[None, :]
    red_min = float(reduced.min())
    red_max = float(reduced.max())
    red_mean = float(reduced.mean())
    print(f"reduced cost stats: min={red_min:.4e}, max={red_max:.4e}, mean={red_mean:.4e}")
    if reduced.min() < -1e-6:
        print("⚠️  Reduced costs dip below zero; duals are infeasible without projection.")
    else:
        print("Duals already satisfy feasibility checks (min reduced cost ≥ 0).")

    row_min = reduced.min(axis=1)
    col_min = reduced.min(axis=0)
    tight_mask = reduced <= 1e-8
    tight_per_row = tight_mask.sum(axis=1)
    tight_per_col = tight_mask.sum(axis=0)

    print(f"row minima range: {row_min.min():.4e} – {row_min.max():.4e}")
    print(f"column minima range: {col_min.min():.4e} – {col_min.max():.4e}")
    print(f"tight-edge density: {tight_mask.mean():.4%} (≤1e-8 residual)")
    print("first few rows:")
    for i in range(min(args.preview, n)):
        print(f"  row {i:02d}: min={row_min[i]:.3e} | tight_edges={int(tight_per_row[i])}")

    # Highlight rows with highest and lowest number of tight edges
    top_rows = np.argsort(-tight_per_row)[:min(5, n)]
    bottom_rows = np.argsort(tight_per_row)[:min(5, n)]
    print("rows with most tight edges:", ", ".join(f"{idx}(#{tight_per_row[idx]})" for idx in top_rows))
    print("rows with fewest tight edges:", ", ".join(f"{idx}(#{tight_per_row[idx]})" for idx in bottom_rows))

    # Store reduced costs for summary (updated after projection if needed)
    reduced_summary = {"min": red_min, "max": red_max, "mean": red_mean}

    if not args.no_project:
        print("\nApplying feasibility projection (iterative tightening)...")
        u_proj, v_proj = project_feasible(cost, u_pred, v_pred, max_rounds=75)
        reduced_proj = cost - u_proj[:, None] - v_proj[None, :]
        print(f"after projection: min={reduced_proj.min():.4e}, max={reduced_proj.max():.4e}")
        u_pred, v_pred = u_proj, v_proj
        reduced = reduced_proj
        tight_mask = reduced <= 1e-8
        tight_per_row = tight_mask.sum(axis=1)
        tight_per_col = tight_mask.sum(axis=0)
        reduced_summary = {
            "min": float(reduced.min()),
            "max": float(reduced.max()),
            "mean": float(reduced.mean()),
        }
    else:
        print("Skipping projection per --no-project flag.")

    try:
        check_dual_feasible(cost, u_pred, v_pred)
        print("✔ Dual feasibility check passed (min reduced cost ≥ 0).")
    except AssertionError as exc:
        print(f"✖ Dual feasibility check failed: {exc}")

    print(format_header("Stage 6: Seeded LAP Solve"))
    seeded_solver = None
    try:
        seeded_solver = SeededLAPSolver()
    except ImportError as exc:
        print(f"Seeded solver unavailable ({exc}); skipping seeded solve.")

    baseline_solver = LAPSolver()

    start = time.perf_counter()
    rows_base, cols_base, cost_baseline = baseline_solver.solve(cost)
    baseline_time = time.perf_counter() - start
    print(f"LAP baseline cost: {cost_baseline:.6f} | time: {baseline_time*1e3:.2f} ms")

    dual_lower = float(u_pred.sum() + v_pred.sum())
    print(f"dual lower bound (Σu + Σv): {dual_lower:.6f}")
    baseline_gap = cost_baseline - dual_lower
    print(f"baseline primal-dual gap: {baseline_gap:.6e}")
    baseline_assign = rows_cols_to_assignment(rows_base, cols_base, n)

    if seeded_solver is not None:
        start = time.perf_counter()
        rows_seed, cols_seed, cost_seeded = seeded_solver.solve(cost, u_pred, v_pred)
        seeded_time = time.perf_counter() - start
        print(f"Seeded LAP cost:  {cost_seeded:.6f} | time: {seeded_time*1e3:.2f} ms")
        print(f"Speedup vs baseline: {baseline_time / seeded_time if seeded_time > 0 else float('inf'):.2f}×")

        seeded_gap = cost_seeded - dual_lower
        print(f"seeded primal-dual gap: {seeded_gap:.6e}")

        seeded_assign = rows_cols_to_assignment(rows_seed, cols_seed, n)
        valid_mask = (baseline_assign >= 0) & (seeded_assign >= 0)
        mismatches = np.sum(baseline_assign[valid_mask] != seeded_assign[valid_mask])
        overlap = int(valid_mask.sum() - mismatches)
        print(f"assignment match with baseline: {overlap}/{int(valid_mask.sum())} rows identical")

        # Compare assignments if dimensions permit
        total_rows_compared = int(valid_mask.sum())
    else:
        print("Seeded LAP result unavailable.")
        seeded_time = None
        cost_seeded = None
        seeded_gap = None
        total_rows_compared = None
        overlap = None

    print(format_header("Stage 7: Summary"))
    summary = textwrap.dedent(f"""
        • Input size: n = {n}
        • OneGNN inference time: {elapsed*1e3:.2f} ms
        • Seeded solver available: {'yes' if seeded_solver is not None else 'no'}
        • Baseline LAP cost: {cost_baseline:.6f}
        {"• Seeded LAP cost:   {:.6f}".format(cost_seeded) if cost_seeded is not None else ""}
        {"• Total speedup (LAP / seeded): {:.2f}×".format(baseline_time / seeded_time) if seeded_time else ""}
        • Dual lower bound: {dual_lower:.6f}
        • Primal-dual gap (baseline): {baseline_gap:.6e}
        {"• Primal-dual gap (seeded):  {:.6e}".format(seeded_gap) if seeded_gap is not None else ""}
        • Min reduced cost after projection: {reduced_summary['min']:.4e}
        • Tight-edge density: {tight_mask.mean():.4%}
        {"• Matching overlap (seeded vs baseline): {}/{} rows".format(overlap, total_rows_compared) if overlap is not None else ""}
    """).strip()
    print(summary)


if __name__ == "__main__":
    main()
