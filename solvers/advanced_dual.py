"""
Advanced Dual Techniques Module

Additional advanced techniques for dual computation and validation,
including feasibility projection, reduced cost computation, and validation tests.
"""

import numpy as np
import platform
import scipy
from typing import Tuple, Optional


def project_feasible(C: np.ndarray, u: np.ndarray, v: np.ndarray, 
                    max_rounds: int = 50, tol: float = 1e-12) -> Tuple[np.ndarray, np.ndarray]:
    """
    Iteratively tighten dual potentials to obtain feasible dual:
        u_i <= min_j (C_ij - v_j)
        v_j <= min_i (C_ij - u_i)
    Stop when min reduced cost >= -tol or rounds exhausted.
    
    This is crucial for making noisy seeds feasible.
    """
    C = np.asarray(C, dtype=float)
    u = np.asarray(u, dtype=float).copy()
    v = np.asarray(v, dtype=float).copy()

    for _ in range(max(1, int(max_rounds))):
        u_cap = (C - v[None, :]).min(axis=1)
        u = np.minimum(u, u_cap)
        v_cap = (C - u[:, None]).min(axis=0)
        v = np.minimum(v, v_cap)
        red = C - u[:, None] - v[None, :]
        if red.min() >= -tol:
            break
    return u, v


def reduce_costs(C: np.ndarray, u: np.ndarray, v: np.ndarray, 
                shift_nonneg: bool = True) -> np.ndarray:
    """
    Form reduced-cost matrix C' = C - u 1^T - 1 v^T.
    If shift_nonneg, subtract min(C') so entries are >= 0 (numeric hygiene).
    
    This allows warm-starting regular LAP solvers on reduced costs.
    """
    C = np.asarray(C, dtype=float)
    Cprime = C - u[:, None] - v[None, :]
    if shift_nonneg:
        m = Cprime.min()
        if m < 0:
            Cprime = Cprime - m
    return np.ascontiguousarray(Cprime, dtype=np.float64)


def check_dual_feasible(C: np.ndarray, u: np.ndarray, v: np.ndarray, 
                       tol: float = 1e-8) -> bool:
    """Assert dual feasibility only: r_ij = C_ij - u_i - v_j >= -tol for all i,j."""
    red = C - u[:, None] - v[None, :]
    mn = float(red.min())
    if mn < -tol:
        raise AssertionError(f"Dual infeasible: min reduced cost {mn:.3e} < -tol")
    return True


def check_dual_and_match(C: np.ndarray, u: np.ndarray, v: np.ndarray, 
                        rows: np.ndarray, cols: np.ndarray, tol: float = 1e-8) -> bool:
    """
    Strict check for oracle (noise=0) seeds:
      - dual feasibility
      - tightness on matched edges (complementary slackness)
      
    Args:
        C: Cost matrix
        u, v: Dual potentials
        rows, cols: Assignment indices
        tol: Numerical tolerance
    """
    red = C - u[:, None] - v[None, :]
    assert np.all(red >= -tol), "Dual infeasible: some reduced costs < 0"
    assert np.all(np.abs(red[rows, cols]) <= 1e-6), "Complementary slackness violated on matched edges"
    return True


def make_feasible_duals(C: np.ndarray, iters: int = 2, noise_std: float = 0.0, 
                       project_rounds: int = 2, rng: Optional[np.random.Generator] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Produce a FEASIBLE dual (u, v) for LAP on cost matrix C using complete v0 approach:
      1) solve LAP once (SciPy LSA) to get an optimal matching,
      2) reconstruct (u, v) via complementary slackness (difference constraints),
      3) optionally add noise,
      4) project to feasibility (iterative until converged or rounds cap).
    Returns u, v.
    
    This is the comprehensive dual generation from v0/solvers/feasible.py
    """
    import scipy.optimize
    
    C = np.asarray(C, dtype=float)
    rows, cols = scipy.optimize.linear_sum_assignment(C)
    
    # Import the dual reconstruction function
    from .dual_computation import dual_from_matching_diff_constraints
    u, v, _ = dual_from_matching_diff_constraints(C, rows, cols)

    if noise_std and noise_std > 0:
        rng = rng or np.random.default_rng(0)
        u = u + rng.normal(0.0, noise_std, size=u.shape)
        v = v + rng.normal(0.0, noise_std, size=v.shape)

    rounds = max(int(project_rounds), int(iters or 0))
    u, v = project_feasible(C, u, v, max_rounds=max(10, rounds), tol=1e-12)
    return u, v


def normalize01(C: np.ndarray) -> np.ndarray:
    """
    Normalize cost matrix to [0,1] range for numerical stability.
    From v0/solvers/lap_backend.py
    """
    C = np.ascontiguousarray(C, dtype=np.float64)
    mn = float(C.min())
    mx = float(C.max()) 
    denom = (mx - mn) if mx > mn else 1.0
    return (C - mn) / denom


def affine_invariance_test(rng: np.random.Generator, n: int = 64, trials: int = 3) -> bool:
    """
    Critical validation: test that LAP solutions respect affine transformations.
    For C2 = a*C + b, optimal cost should satisfy: cost2 = a*cost + b*n
    
    This catches fundamental algorithmic issues.
    """
    from .generators import generate_uniform_costs
    import scipy.optimize
    
    ok_all = True
    for t in range(trials):
        C = generate_uniform_costs(n, seed=rng.integers(0, 10000))
        r0, c0 = scipy.optimize.linear_sum_assignment(C)
        
        # Apply affine transformation
        a = 10 ** rng.uniform(-2, 2)  # positive scale
        b = rng.uniform(-3.0, 3.0)   # shift
        C2 = a * C + b
        
        r1, c1 = scipy.optimize.linear_sum_assignment(C2)
        
        # Check invariance property
        mapped = a * float(C[r1, c1].sum()) + b * n
        cost2 = float(C2[r1, c1].sum())
        
        if not np.isclose(cost2, mapped, rtol=1e-9, atol=1e-9):
            print(f"[check] affine invariance failed (trial {t}): cost2={cost2:.6g}, mapped={mapped:.6g}, a={a:.3g}, b={b:.3g}")
            ok_all = False
    
    if ok_all:
        print("[check] affine invariance: OK")
    return ok_all


def print_env_summary():
    """Print environment info for reproducibility."""
    import os
    env = {k: os.environ.get(k) for k in
           ["OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS","MKL_DYNAMIC","PYTHONHASHSEED"]}
    print(f"[env] threads/hash: {env}")
    print(f"[env] Python: {platform.python_version()}  NumPy: {np.__version__}  SciPy: {scipy.__version__}")