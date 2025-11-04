"""
Dual Computation Module

Robust methods for computing optimal dual potentials for LAP problems.
Uses difference constraints and Bellman-Ford algorithm for maximum reliability.
"""

import numpy as np
import scipy.optimize
from typing import Tuple


def dual_from_matching_diff_constraints(C, row_ind, col_ind, tol=1e-12):
    """
    Reconstruct optimal dual potentials (u, v) from known optimal matching
    using difference constraints and Bellman-Ford algorithm.
    
    This is the most robust method for generating high-quality dual seeds.
    
    Args:
        C: Cost matrix
        row_ind: Row indices of optimal matching
        col_ind: Column indices of optimal matching  
        tol: Tolerance for numerical checks
        
    Returns:
        u, v, reduced_costs: Dual potentials and reduced cost matrix
    """
    C = np.asarray(C, dtype=float)
    m, n = C.shape
    assert len(row_ind) == len(col_ind)

    # Build constraints for Bellman–Ford over columns
    edges = []
    for ri, cj in zip(row_ind, col_ind):
        p = cj
        w = C[ri, :] - C[ri, p]
        edges.extend((p, j, float(w[j])) for j in range(n))

    v = np.zeros(n, dtype=float)  # gauge free
    for _ in range(n - 1):
        updated = False
        for a, b, w in edges:
            if v[b] > v[a] + w:
                v[b] = v[a] + w
                updated = True
        if not updated:
            break
    else:
        for a, b, w in edges:
            if v[b] > v[a] + w - tol:
                raise RuntimeError("Negative cycle while solving difference constraints for v.")

    u = np.full(m, np.nan, dtype=float)
    for ri, cj in zip(row_ind, col_ind):
        u[ri] = C[ri, cj] - v[cj]
    for i in range(m):
        if np.isnan(u[i]):
            u[i] = np.min(C[i, :] - v)

    # Optional gauge-fix for numerical stability
    shift = (np.mean(u) + np.mean(v)) / 2.0
    u -= shift
    v += shift

    # Verify dual feasibility
    red = C - u[:, None] - v[None, :]
    if np.any(red < -1e-8):
        raise AssertionError("Dual infeasible after reconstruction (negative reduced costs).")
    for ri, cj in zip(row_ind, col_ind):
        if abs(red[ri, cj]) > 1e-6:
            raise AssertionError("Complementary slackness violated on a matched edge.")
    
    return u, v, red


def compute_oracle_duals(C: np.ndarray, noise_level: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute oracle dual potentials with optional noise.
    Uses the robust difference constraints method.
    
    Args:
        C: Cost matrix
        noise_level: Amount of Gaussian noise to add to duals (for testing)
        
    Returns:
        u_star, v_star: Optimal dual potentials
    """
    n = C.shape[0]
    
    # Get optimal primal solution
    rows, cols = scipy.optimize.linear_sum_assignment(C)
    
    try:
        # Use the robust difference constraints method
        u_star, v_star, _ = dual_from_matching_diff_constraints(C, rows, cols)
    except (RuntimeError, AssertionError) as e:
        print(f"Warning: Difference constraints failed ({e}), using fallback method")
        # Fallback to simpler method
        u_star = np.zeros(n, dtype=np.float64)
        v_star = np.min(C, axis=0)
        
        # Make assignment constraints tight
        for r, c in zip(rows, cols):
            u_star[r] = C[r, c] - v_star[c]
    
    # Add noise if specified (for testing seed quality sensitivity)
    if noise_level > 0:
        np.random.seed(42)  # Consistent noise for reproducibility
        u_noise = np.random.normal(0, noise_level, n)
        v_noise = np.random.normal(0, noise_level, n)
        u_star += u_noise
        v_star += v_noise
    
    return u_star.astype(np.float64), v_star.astype(np.float64)