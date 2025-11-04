"""Evaluation utilities for DualGNN models."""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

if __package__ is None or __package__ == "":  # Allow running as script
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:  # pragma: no cover - optional dependency
    import torch
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install torch to run evaluation") from exc

from gnn import DualGNN, compute_features
from gnn.train import LapDataset
from solvers import LAPSolver, SeededLAPSolver
from solvers.advanced_dual import check_dual_feasible, project_feasible


def greedy_primal_upper_np(cost: np.ndarray, reduced: np.ndarray) -> float:
    n = cost.shape[0]
    if n == 0:
        return 0.0

    assignment = -np.ones(n, dtype=int)
    used_cols: set[int] = set()
    row_order = np.argsort(reduced.min(axis=1))

    for row in row_order:
        for col in np.argsort(reduced[row]):
            if col not in used_cols:
                assignment[row] = int(col)
                used_cols.add(int(col))
                break
        if assignment[row] == -1:
            remaining = [c for c in range(n) if c not in used_cols]
            if remaining:
                best = min(remaining, key=lambda c: reduced[row, c])
                assignment[row] = int(best)
                used_cols.add(int(best))
            else:
                assignment[row] = int(np.argmin(reduced[row]))

    seen: dict[int, list[int]] = {}
    for r, c in enumerate(assignment):
        seen.setdefault(int(c), []).append(r)

    available = [c for c in range(n) if c not in used_cols]
    for col, rows in seen.items():
        if len(rows) <= 1:
            continue
        for row in rows[1:]:
            if available:
                best = min(available, key=lambda c: reduced[row, c])
                assignment[row] = int(best)
                available.remove(best)

    return float(cost[np.arange(n), assignment].sum())


def load_model(checkpoint_path: Path, device: torch.device) -> DualGNN:
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = DualGNN(
        hidden_dim=ckpt.get("hidden_dim", 128),
        layers=ckpt.get("layers", 4),
        heads=ckpt.get("heads", 4),
        dropout=ckpt.get("dropout", 0.0),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def evaluate(
    dataset_path: Path,
    checkpoint_path: Path,
    device: torch.device,
    limit: Optional[int] = None,
    project: bool = True,
) -> Dict[str, float]:
    dataset = LapDataset(dataset_path)
    model = load_model(checkpoint_path, device)

    lap_solver = LAPSolver()
    seeded_solver = SeededLAPSolver()

    total = 0
    feasible = 0
    speedups: list[float] = []
    u_mse: list[float] = []
    v_mse: list[float] = []
    cost_gaps: list[float] = []
    assist = 0
    pre_gaps: list[float] = []

    for idx in range(len(dataset)):
        item = dataset[idx]
        n = item["n"]
        feats = compute_features(item["cost"])
        edge_feat = torch.from_numpy(feats.edge_feat).unsqueeze(0).to(device)
        row_feat = torch.from_numpy(feats.row_feat).unsqueeze(0).to(device)
        col_feat = torch.from_numpy(feats.col_feat).unsqueeze(0).to(device)
        mask = torch.ones(1, n, dtype=torch.bool, device=device)

        with torch.no_grad():
            preds = model(edge_feat, row_feat, col_feat, mask=mask)

        u_hat = preds["u"].squeeze(0).cpu().numpy()[:n]
        v_hint = preds["v_hint"].squeeze(0).cpu().numpy()[:n]
        v_hat = np.min(item["cost"] - u_hat[:, None], axis=0)

        if project:
            u_hat, v_hat = project_feasible(item["cost"], u_hat, v_hat)

        try:
            check_dual_feasible(item["cost"], u_hat, v_hat)
            feasible += 1
        except AssertionError:
            pass

        mse_u = float(np.mean((u_hat - item["u"]) ** 2))
        mse_v = float(np.mean((v_hat - item["v"]) ** 2))
        u_mse.append(mse_u)
        v_mse.append(mse_v)

        start = time.perf_counter()
        _, _, base_cost = lap_solver.solve(item["cost"])
        base_time = time.perf_counter() - start

        start = time.perf_counter()
        _, _, seeded_cost = seeded_solver.solve(item["cost"], u_hat, v_hat)
        seeded_time = time.perf_counter() - start

        if seeded_time > 0:
            speedups.append(base_time / seeded_time)
        cost_gap = abs(seeded_cost - base_cost)
        cost_gaps.append(cost_gap)
        if cost_gap <= 1e-6:
            assist += 1

        reduced = item["cost"] - u_hat[:, None] - v_hat[None, :]
        primal_upper = greedy_primal_upper_np(item["cost"], reduced)
        dual_lower = float(u_hat.sum() + v_hat.sum())
        pre_gaps.append(primal_upper - dual_lower)

        total += 1
        if limit is not None and total >= limit:
            break

    return {
        "count": float(total),
        "feasible_rate": feasible / max(1, total),
        "avg_speedup": statistics.mean(speedups) if speedups else 0.0,
        "median_speedup": statistics.median(speedups) if speedups else 0.0,
        "avg_u_mse": statistics.mean(u_mse) if u_mse else 0.0,
        "avg_v_mse": statistics.mean(v_mse) if v_mse else 0.0,
        "avg_cost_gap": statistics.mean(cost_gaps) if cost_gaps else 0.0,
        "assist_rate": assist / max(1, total),
        "pre_solve_gap_median": statistics.median(pre_gaps) if pre_gaps else 0.0,
        "pre_solve_gap_mean": statistics.mean(pre_gaps) if pre_gaps else 0.0,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, required=True, help="Path to split (e.g., val.h5)")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-project", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    metrics = evaluate(
        dataset_path=args.split,
        checkpoint_path=args.checkpoint,
        device=device,
        limit=args.limit,
        project=not args.no_project,
    )
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")


if __name__ == "__main__":
    main()
