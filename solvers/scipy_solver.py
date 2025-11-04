"""
SciPy Solver Module

Clean interface for SciPy's linear_sum_assignment algorithm.
"""

import numpy as np
import scipy.optimize
from typing import Tuple


class SciPySolver:
    """Wrapper for SciPy's linear_sum_assignment algorithm."""
    
    def __init__(self):
        self.name = "SciPy"
    
    def solve(self, C: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Solve LAP using SciPy's linear_sum_assignment.
        
        Args:
            C: Cost matrix
            
        Returns:
            rows, cols, cost: Row assignments, column assignments, total cost
        """
        C = np.asarray(C, dtype=np.float64)
        rows, cols = scipy.optimize.linear_sum_assignment(C)
        cost = C[rows, cols].sum()
        return rows, cols, float(cost)
    
    def __call__(self, C: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """Allow using solver as callable."""
        return self.solve(C)