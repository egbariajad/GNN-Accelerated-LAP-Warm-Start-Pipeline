"""
LAP Solver Module

Clean interfaces for LAP library's lapjv and lapjv_seeded algorithms.
"""

import numpy as np
import lap
from typing import Tuple


def _resolve_seeded_api():
    """Return a callable seeded LAP function if available."""

    # Newer wheels expose lapjv_seeded at the top level; older ones hide it in
    # the private _seeded_jv module. We try both so the benchmark works no
    # matter which build is installed.
    seeded = getattr(lap, "lapjv_seeded", None)
    if seeded is not None:
        return seeded

    try:
        from lap._seeded_jv import lapjv_seeded  # type: ignore

        return lapjv_seeded
    except Exception:  # pragma: no cover - extremely small surface
        return None


_LAPJV_SEEDED = _resolve_seeded_api()


class LAPSolver:
    """Wrapper for LAP library's unseeded lapjv algorithm."""
    
    def __init__(self):
        self.name = "LAP"
    
    def solve(self, C: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Solve LAP using unseeded lapjv.
        
        Args:
            C: Cost matrix
            
        Returns:
            rows, cols, cost: Row assignments, column assignments, total cost
        """
        C = np.asarray(C, dtype=np.float64)
        n = C.shape[0]
        
        # Call LAP solver
        cost_returned, x, y = lap.lapjv(C, extend_cost=False)
        
        # Extract assignment
        rows = np.arange(n, dtype=np.int64)
        cols = np.asarray(x, dtype=np.int64)
        
        # Compute cost from assignment for consistency
        cost = sum(C[i, cols[i]] for i in range(n) if cols[i] >= 0)
        
        return rows, cols, float(cost)
    
    def __call__(self, C: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """Allow using solver as callable."""
        return self.solve(C)


class SeededLAPSolver:
    """Wrapper for LAP library's seeded lapjv_seeded algorithm."""
    
    def __init__(self):
        self.name = "SeededLAP"
        if _LAPJV_SEEDED is None:
            raise ImportError(
                "lap.lapjv_seeded is not available. Install the custom LAP build "
                "(pip install -e LAP) or upgrade the lap package to a version that "
                "exposes the seeded API."
            )
    
    def solve(self, C: np.ndarray, u: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Solve LAP using seeded lapjv_seeded with dual potentials.
        
        Args:
            C: Cost matrix
            u: Row dual potentials
            v: Column dual potentials
            
        Returns:
            rows, cols, cost: Row assignments, column assignments, total cost
        """
        C = np.asarray(C, dtype=np.float64)
        u = np.asarray(u, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        n = C.shape[0]
        
        # Call seeded LAP solver
        rows, cols, cost = _LAPJV_SEEDED(C, u, v)
        
        return np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64), float(cost)
    
    def __call__(self, C: np.ndarray, u: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """Allow using solver as callable."""
        return self.solve(C, u, v)
