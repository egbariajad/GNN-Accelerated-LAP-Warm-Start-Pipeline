"""Baseline dual seed constructors for warm-start experiments.

All routines return dual potentials `(u, v)` that are feasible for the
original cost matrix. They purposefully avoid any edge pruning so the
resulting warm-start always operates on the full problem.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .advanced_dual import make_feasible_duals, project_feasible
from .dual_computation import dual_from_matching_diff_constraints


def seed_row_col_minima(
    C: np.ndarray,
    *,
    project_rounds: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Classic row/column minima warm-start.

    This solution is always feasible after projection and requires no solver.
    """

    C = np.asarray(C, dtype=np.float64)
    row_min = C.min(axis=1)
    u = row_min.copy()

    # Estimate column duals relative to row minima, then project to feasibility.
    col_slack = C - u[:, None]
    v = col_slack.min(axis=0)

    u, v = project_feasible(C, u, v, max_rounds=project_rounds)
    return u, v


def seed_greedy_matching(
    C: np.ndarray,
    *,
    project_rounds: int = 50,
    rng: Optional[np.random.Generator] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct duals from a greedy matching followed by feasibility projection."""

    C = np.asarray(C, dtype=np.float64)
    n = C.shape[0]
    rng = rng or np.random.default_rng()

    remaining_cols = set(range(n))
    rows = np.arange(n, dtype=np.int32)
    cols = np.full(n, -1, dtype=np.int32)

    for i in range(n):
        row_costs = C[i]
        candidate_cols = np.array(sorted(remaining_cols))
        best_idx = candidate_cols[np.argmin(row_costs[candidate_cols])]
        cols[i] = best_idx
        remaining_cols.remove(int(best_idx))

    # If greedy left unmatched columns (due to degeneracy), fill randomly.
    if remaining_cols:
        unused = list(remaining_cols)
        for idx in np.where(cols == -1)[0]:
            if not unused:
                break
            choice = rng.choice(unused)
            cols[idx] = int(choice)
            unused.remove(int(choice))

    # Repair duplicates if greedy selected the same column twice.
    unique_cols, counts = np.unique(cols, return_counts=True)
    duplicates = unique_cols[counts > 1]
    available = [c for c in range(n) if c not in cols]
    for dup in duplicates:
        dup_indices = np.where(cols == dup)[0]
        for dup_idx in dup_indices[1:]:
            if available:
                replacement = available.pop(0)
            else:
                replacement = dup
            cols[dup_idx] = replacement

    u, v, _ = dual_from_matching_diff_constraints(C, rows, cols)
    u, v = project_feasible(C, u, v, max_rounds=project_rounds)
    return u, v


def seed_noisy_optimal(
    C: np.ndarray,
    *,
    noise_std: float = 0.05,
    rng: Optional[np.random.Generator] = None,
    project_rounds: int = 75,
) -> tuple[np.ndarray, np.ndarray]:
    """Perturb oracle duals with small noise and re-project to feasibility."""

    rng = rng or np.random.default_rng()
    u_opt, v_opt = make_feasible_duals(C, noise_std=0.0)
    u_noisy = u_opt + rng.normal(0.0, noise_std, size=u_opt.shape)
    v_noisy = v_opt + rng.normal(0.0, noise_std, size=v_opt.shape)
    u_proj, v_proj = project_feasible(
        np.asarray(C, dtype=np.float64),
        u_noisy,
        v_noisy,
        max_rounds=project_rounds,
    )
    return u_proj, v_proj


__all__ = [
    "seed_row_col_minima",
    "seed_greedy_matching",
    "seed_noisy_optimal",
]
