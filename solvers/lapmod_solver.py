"""
LAPMOD Sparse Solver Module

Wrapper for LAP library's lapmod (sparse LAP solver) algorithm.
Handles automatic conversion from dense matrices to sparse format and
manages lapmod-specific constraints and optimizations.
"""

import numpy as np
import lap
from lap.tests.test_utils import sparse_from_dense
from typing import Tuple, Optional
import warnings


class LAPMODSolver:
    """Wrapper for LAP library's sparse lapmod algorithm."""
    
    def __init__(self):
        self.name = "LAPMOD"
    
    def solve(self, C: np.ndarray, mask: Optional[np.ndarray] = None, 
             fp_version: int = lap.FP_DYNAMIC) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Solve LAP using sparse lapmod algorithm.
        
        Args:
            C: Cost matrix
            mask: Optional boolean mask for sparse problems
            fp_version: Find-path version for sparse solver
            
        Returns:
            rows, cols, cost: Row assignments, column assignments, total cost
        """
        C = np.asarray(C, dtype=np.float64)
        n = C.shape[0]
        
        # Prepare sparse cost matrix
        if mask is not None:
            # Use mask to determine finite elements
            finite_mask = mask & np.isfinite(C)
            sparse_cost = C.copy()
            sparse_cost[~finite_mask] = lap.LARGE
        else:
            sparse_cost = C.copy()
            
        # Check for lapmod constraint: all costs must be < 1,000,000
        max_finite_cost = np.max(sparse_cost[np.isfinite(sparse_cost)])
        scale_factor = 1.0
        
        if max_finite_cost >= 1000000:
            # Scale down costs to fit within constraint
            scale_factor = 999999.0 / max_finite_cost
            sparse_cost = sparse_cost * scale_factor
            
        # Convert to sparse representation
        n_sparse, cc, ii, kk = sparse_from_dense(sparse_cost)
        
        # Solve using lapmod
        cost, row_assignments, col_assignments = lap.lapmod(
            n_sparse, cc, ii, kk,
            fast=True,
            return_cost=True,
            fp_version=fp_version
        )
        
        # Scale cost back if we scaled down
        if scale_factor != 1.0:
            cost = cost / scale_factor
        
        rows = np.arange(n, dtype=np.int64)
        cols = np.asarray(row_assignments, dtype=np.int64)
        
        return rows, cols, float(cost)
    
    def __call__(self, C: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, float]:
        """Allow using solver as callable."""
        return self.solve(C, mask)