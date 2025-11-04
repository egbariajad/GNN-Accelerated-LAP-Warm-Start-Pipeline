"""
Verification Module

Correctness verification for LAP solvers to ensure they produce consistent results.
"""

import numpy as np
from .scipy_solver import SciPySolver
from .lap_solver import LAPSolver, SeededLAPSolver


def verify_solver_correctness(C: np.ndarray, u_oracle: np.ndarray, v_oracle: np.ndarray, 
                            tolerance: float = 1e-10) -> bool:
    """
    Verify that all three solvers produce the same optimal cost.
    
    Args:
        C: Cost matrix
        u_oracle: Oracle dual potentials (row)
        v_oracle: Oracle dual potentials (column)
        tolerance: Numerical tolerance for cost comparison
        
    Returns:
        True if all solvers agree within tolerance
    """
    
    try:
        # Initialize solvers
        scipy_solver = SciPySolver()
        lap_solver = LAPSolver()
        seeded_solver = SeededLAPSolver()
        
        # Solve with each method
        _, _, scipy_cost = scipy_solver.solve(C)
        _, _, lap_cost = lap_solver.solve(C)
        _, _, seeded_cost = seeded_solver.solve(C, u_oracle, v_oracle)
        
        # Check consistency (within numerical tolerance)
        costs = [scipy_cost, lap_cost, seeded_cost]
        cost_range = max(costs) - min(costs)
        
        return cost_range < tolerance
        
    except Exception as e:
        print(f"Verification failed: {e}")
        return False