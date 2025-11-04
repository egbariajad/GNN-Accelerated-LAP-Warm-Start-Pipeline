"""
Warm-Start LAP Solver Module

Alternative warm-start approach using reduced cost matrices.
This can work with any LAP solver that doesn't support direct seeding.
"""

import numpy as np
import lap
import scipy.optimize
from typing import Tuple
from .advanced_dual import reduce_costs


class WarmStartLAPSolver:
    """
    Warm-start LAP solver using reduced cost matrix approach.
    
    Instead of direct seeding, this:
    1. Computes reduced costs C' = C - u⊤ - v  
    2. Solves LAP on C'
    3. Returns cost on original C
    
    Works with any LAP solver, not just seeded variants.
    """
    
    def __init__(self, use_lap=True):
        self.name = "WarmStartLAP"
        self.use_lap = use_lap  # True for lap.lapjv, False for scipy
    
    def solve(self, C: np.ndarray, u: np.ndarray, v: np.ndarray, 
             shift_nonneg: bool = True) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Solve LAP using warm-start via reduced costs.
        
        Args:
            C: Original cost matrix
            u, v: Dual potentials for warm-start
            shift_nonneg: Whether to shift C' to be non-negative
            
        Returns:
            rows, cols, cost: Assignment on original C
        """
        C = np.asarray(C, dtype=np.float64)
        u = np.asarray(u, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        n = C.shape[0]
        
        # Compute reduced costs
        Cprime = reduce_costs(C, u, v, shift_nonneg=shift_nonneg)
        
        # Solve on reduced costs
        if self.use_lap:
            cost_prime, x, y = lap.lapjv(Cprime, extend_cost=False)
            rows = np.arange(n, dtype=np.int64)
            cols = np.asarray(x, dtype=np.int64)
        else:
            rows, cols = scipy.optimize.linear_sum_assignment(Cprime)
        
        # Compute cost on ORIGINAL matrix
        cost_original = float(C[rows, cols].sum())
        
        return rows, cols, cost_original
    
    def __call__(self, C: np.ndarray, u: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """Allow using solver as callable."""
        return self.solve(C, u, v)