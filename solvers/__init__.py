"""
LAP Solvers Module

This module provides unified interfaces for different Linear Assignment Problem solvers:
- SciPy linear_sum_assignment (baseline reference)
- LAP lapjv (high-performance unseeded)
- LAP lapjv_seeded (warm-started with dual potentials)
- WarmStart LAP (reduced cost approach for any solver)

Advanced techniques from v0 research:
- Dual feasibility projection
- Reduced cost computation  
- Affine invariance testing
- Comprehensive validation

The module emphasizes clean interfaces, consistent timing, and correctness verification.
"""

from pathlib import Path
import sys

# Prefer the in-repo LAP build when present (contains the custom seeded solver).
_LOCAL_LAP_ROOT = Path(__file__).resolve().parent.parent / "LAP"
if _LOCAL_LAP_ROOT.exists() and str(_LOCAL_LAP_ROOT) not in sys.path:
    sys.path.insert(0, str(_LOCAL_LAP_ROOT))

from .scipy_solver import SciPySolver
from .lap_solver import LAPSolver, SeededLAPSolver
from .lapmod_solver import LAPMODSolver
from .dual_computation import compute_oracle_duals, dual_from_matching_diff_constraints
from .verification import verify_solver_correctness
from .timing import time_solver_rigorous
from .warmstart_solver import WarmStartLAPSolver
from .generators import (
    generate_uniform_costs, 
    generate_near_diagonal_costs, 
    generate_sparse_costs,
    generate_metric_costs,
    generate_clustered_costs,
    generate_noisy_linear_costs,
    generate_worst_case_costs,
    generate_identity_like_costs,
    generate_hard_random_costs
)
from .advanced_dual import (
    project_feasible,
    reduce_costs,
    check_dual_and_match,
    check_dual_feasible,
    make_feasible_duals,
    normalize01,
    affine_invariance_test,
    print_env_summary
)
from .logging_system import BenchmarkLogger, list_experiments, load_experiment
from .seed_baselines import (
    seed_row_col_minima,
    seed_greedy_matching,
    seed_noisy_optimal,
)

__all__ = [
    'SciPySolver',
    'LAPSolver', 
    'SeededLAPSolver',
    'LAPMODSolver',
    'WarmStartLAPSolver',
    'compute_oracle_duals',
    'dual_from_matching_diff_constraints',
    'verify_solver_correctness',
    'time_solver_rigorous',
    'generate_uniform_costs',
    'generate_near_diagonal_costs', 
    'generate_sparse_costs',
    'generate_metric_costs',
    'generate_clustered_costs',
    'generate_noisy_linear_costs',
    'generate_worst_case_costs',
    'generate_identity_like_costs', 
    'generate_hard_random_costs',
    'project_feasible',
    'reduce_costs',
    'check_dual_and_match', 
    'check_dual_feasible',
    'make_feasible_duals', 
    'normalize01',
    'affine_invariance_test',
    'print_env_summary',
    'BenchmarkLogger',
    'list_experiments', 
    'load_experiment',
    'seed_row_col_minima',
    'seed_greedy_matching',
    'seed_noisy_optimal',
]
