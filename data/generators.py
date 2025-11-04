"""Synthetic dataset generators for DualGNN training."""

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np
import scipy.optimize

from solvers import (
    generate_uniform_costs,
    generate_metric_costs,
    generate_sparse_costs,
    generate_clustered_costs,
    generate_noisy_linear_costs,
)
from solvers.advanced_dual import make_feasible_duals, project_feasible
from solvers.dual_computation import dual_from_matching_diff_constraints


ArrayGenerator = Callable[[int, np.random.Generator], np.ndarray]


def _wrap_solver_generator(base_func: Callable[..., np.ndarray]) -> ArrayGenerator:
    """Wrap solvers.* generator to use numpy Generator for randomness."""

    def _runner(n: int, rng: np.random.Generator) -> np.ndarray:
        seed = int(rng.integers(0, np.iinfo(np.uint32).max))
        return base_func(n, seed=seed)

    return _runner


def _gen_low_rank(n: int, rng: np.random.Generator, rank: int = 12, sigma: float = 0.1) -> np.ndarray:
    a = rng.normal(0.0, 1.0, size=(n, rank))
    b = rng.normal(0.0, 1.0, size=(n, rank))
    c = a @ b.T + sigma * rng.normal(0.0, 1.0, size=(n, n))
    return np.maximum(c, 0.0).astype(np.float64)


def _gen_block(n: int, rng: np.random.Generator, blocks: int = 4, noise: float = 0.1) -> np.ndarray:
    seed = int(rng.integers(0, np.iinfo(np.uint32).max))
    return generate_clustered_costs(n, blocks=blocks, noise=noise, seed=seed)


def _gen_noisy_linear(n: int, rng: np.random.Generator, rank: int = 1, noise: float = 0.1) -> np.ndarray:
    seed = int(rng.integers(0, np.iinfo(np.uint32).max))
    return generate_noisy_linear_costs(n, rank=rank, noise=noise, seed=seed)


def _gen_tie_heavy(n: int, rng: np.random.Generator, bins: int = 5, jitter: float = 1e-6) -> np.ndarray:
    base = rng.integers(0, max(1, bins), size=(n, n)) / max(1, float(bins))
    noise = jitter * rng.uniform(0.0, 1.0, size=(n, n))
    return (base + noise).astype(np.float64)


def _gen_sparse_for_dataset(n: int, rng: np.random.Generator, sparsity: float = 0.3) -> np.ndarray:
    dense = _wrap_solver_generator(generate_uniform_costs)(n, rng)
    mask = rng.random(size=(n, n)) < sparsity

    # Ensure feasibility by keeping at least one entry per row/column.
    for i in range(n):
        if not mask[i].any():
            mask[i, rng.integers(0, n)] = True
    for j in range(n):
        if not mask[:, j].any():
            mask[rng.integers(0, n), j] = True

    dense[~mask] = 1e6
    return dense.astype(np.float64)


SYNTHETIC_FAMILIES: Dict[str, ArrayGenerator] = {
    "uniform": _wrap_solver_generator(generate_uniform_costs),
    "metric": _wrap_solver_generator(generate_metric_costs),
    "low_rank": _gen_low_rank,
    "block": _gen_block,
    "clustered": _gen_block,
    "noisy_linear": _gen_noisy_linear,
    "tie": _gen_tie_heavy,
    "sparse": _gen_sparse_for_dataset,
}


@dataclass
class SyntheticInstance:
    cost: np.ndarray
    rows: np.ndarray
    cols: np.ndarray
    u: np.ndarray
    v: np.ndarray
    family: str
    size: int
    noise_std: float
    tag: Optional[str] = None


def generate_synthetic_instance(
    family: str,
    n: int,
    rng: Optional[np.random.Generator] = None,
    noise_probability: float = 0.2,
    noise_std: float = 0.15,
    dual_noise_prob: float = 0.0,
) -> SyntheticInstance:
    """Generate cost matrix, duals, and optimal matching for a given family.
    
    Args:
        family: Name of synthetic family
        n: Problem size
        rng: Random number generator
        noise_probability: Legacy parameter (for cost matrix noise, if implemented)
        noise_std: Standard deviation for dual noise
        dual_noise_prob: Probability of applying dual noise (0.0 = clean duals)
    """

    if family not in SYNTHETIC_FAMILIES:
        raise KeyError(f"Unknown family '{family}'. Known families: {sorted(SYNTHETIC_FAMILIES)}")

    rng = rng or np.random.default_rng(0)
    generator = SYNTHETIC_FAMILIES[family]
    cost = np.asarray(generator(n, rng), dtype=np.float64)

    rows, cols = scipy.optimize.linear_sum_assignment(cost)
    try:
        u, v, _ = dual_from_matching_diff_constraints(cost, rows, cols)
    except (AssertionError, RuntimeError):
        u, v = make_feasible_duals(cost)

    applied_noise = 0.0
    # Dual noise injection (disabled by default for clean training)
    if dual_noise_prob > 0.0 and rng.random() < dual_noise_prob:
        applied_noise = noise_std
        u_noisy = u + rng.normal(0.0, noise_std, size=u.shape)
        v_noisy = v + rng.normal(0.0, noise_std, size=v.shape)
        u, v = project_feasible(cost, u_noisy, v_noisy, max_rounds=75)

    return SyntheticInstance(
        cost=cost,
        rows=rows.astype(np.int32),
        cols=cols.astype(np.int32),
        u=u.astype(np.float64),
        v=v.astype(np.float64),
        family=family,
        size=int(n),
        noise_std=float(applied_noise),
        tag=None,
    )


__all__ = [
    "SYNTHETIC_FAMILIES",
    "SyntheticInstance",
    "generate_synthetic_instance",
]
