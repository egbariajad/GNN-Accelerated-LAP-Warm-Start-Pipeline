"""
Timing Module

Rigorous timing methodology for LAP solver benchmarking.
Provides statistical significance through multiple runs and warmups.
"""

import time
import statistics
from typing import Dict, Callable


def time_solver_rigorous(solver_func: Callable, num_warmups: int = 5, num_repeats: int = 30) -> Dict[str, float]:
    """
    Time a solver with rigorous methodology:
    - Multiple warmups to stabilize timing
    - Many repeats for statistical significance
    - Reports median (robust to outliers)
    
    Args:
        solver_func: Function to time (should take no arguments)
        num_warmups: Number of warmup runs
        num_repeats: Number of timed runs
        
    Returns:
        Dictionary with timing statistics
    """
    
    # Warmup runs
    for _ in range(num_warmups):
        try:
            _ = solver_func()
        except Exception:
            pass
    
    # Timed runs
    times = []
    for _ in range(num_repeats):
        start = time.perf_counter()
        try:
            result = solver_func()
            end = time.perf_counter()
            times.append(end - start)
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    if len(times) == 0:
        return {'success': False, 'error': 'All runs failed'}
    
    return {
        'success': True,
        'median': statistics.median(times),
        'mean': statistics.mean(times),
        'std': statistics.stdev(times) if len(times) > 1 else 0.0,
        'min': min(times),
        'max': max(times),
        'num_samples': len(times)
    }