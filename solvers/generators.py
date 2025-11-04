"""
Problem Generators Module

Various cost matrix generators for benchmarking LAP solvers.
Includes uniform random, structured, and sparse problem types.
"""

import numpy as np
from typing import Tuple


def generate_uniform_costs(n: int, seed: int = 42) -> np.ndarray:
    """
    Generate synthetic uniform[0,1] cost matrix.
    
    Args:
        n: Matrix size
        seed: Random seed for reproducibility
        
    Returns:
        n x n cost matrix with uniform random costs
    """
    np.random.seed(seed)
    return np.random.uniform(0.0, 1.0, (n, n)).astype(np.float64)


def generate_near_diagonal_costs(n: int, noise_level: float = 0.1, seed: int = 42) -> np.ndarray:
    """
    Generate near-diagonal + noise costs for tracking/association scenarios.
    
    Args:
        n: Matrix size
        noise_level: Amount of noise to add
        seed: Random seed for reproducibility
        
    Returns:
        n x n cost matrix with diagonal preference plus noise
    """
    np.random.seed(seed)
    
    # Start with a diagonal preference matrix
    C = np.zeros((n, n), dtype=np.float64)
    
    # Add diagonal preference (lower costs near diagonal)
    for i in range(n):
        for j in range(n):
            distance_from_diagonal = abs(i - j)
            C[i, j] = 0.1 + 0.9 * (distance_from_diagonal / n)
    
    # Add noise
    noise = np.random.normal(0, noise_level, (n, n))
    C += noise
    
    # Ensure all costs are positive
    C = np.maximum(C, 0.001)
    
    return C.astype(np.float64)


def generate_sparse_costs(n: int, sparsity_ratio: float = 0.3, seed: int = 42) -> np.ndarray:
    """
    Generate sparse cost matrix by setting many elements to high cost.
    
    Args:
        n: Matrix size
        sparsity_ratio: Fraction of edges to keep (0.3 = 30% of edges allowed)
        seed: Random seed for reproducibility
        
    Returns:
        n x n cost matrix with many high-cost (effectively forbidden) edges
    """
    np.random.seed(seed)
    
    # Generate base costs
    C = np.random.uniform(0.1, 1.0, (n, n))
    
    # Create sparsity mask
    keep_mask = np.random.random((n, n)) < sparsity_ratio
    
    # Ensure each row and column has at least one allowed edge for feasibility
    for i in range(n):
        if not np.any(keep_mask[i, :]):
            j = np.random.randint(n)
            keep_mask[i, j] = True
    
    for j in range(n):
        if not np.any(keep_mask[:, j]):
            i = np.random.randint(n)
            keep_mask[i, j] = True
    
    # Set forbidden edges to high cost (effectively sparse)
    C = np.where(keep_mask, C, 100.0)
    
    return C.astype(np.float64)


def generate_metric_costs(n: int, seed: int = 42) -> np.ndarray:
    """Generate metric cost matrix based on geometric point distances."""
    np.random.seed(seed)

    # Generate random points in 2D
    points = np.random.uniform(0, 100, (n, 2))

    # Calculate pairwise distances
    C = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            C[i, j] = np.linalg.norm(points[i] - points[j])

    return C


def generate_clustered_costs(n: int, blocks: int = 4, noise: float = 0.1, seed: int = 42) -> np.ndarray:
    """Generate block-structured cost matrix with low in-cluster costs."""
    rng = np.random.default_rng(seed)
    C = rng.uniform(0.0, 1.0, size=(n, n))
    block_size = max(1, n // max(1, blocks))
    for b in range(blocks):
        i0 = b * block_size
        i1 = n if b == blocks - 1 else min(n, (b + 1) * block_size)
        C[i0:i1, i0:i1] -= 0.4
    C += noise * rng.normal(0.0, 1.0, size=(n, n))
    return np.maximum(C, 0.0).astype(np.float64)


def generate_noisy_linear_costs(n: int, rank: int = 1, noise: float = 0.1, seed: int = 42) -> np.ndarray:
    """Generate low-rank linear structure with additive Gaussian noise."""
    rng = np.random.default_rng(seed)
    u = rng.normal(size=(n, rank))
    v = rng.normal(size=(rank, n))
    base = u @ v
    perturb = rng.normal(scale=noise, size=(n, n))
    C = base + perturb
    C -= C.min()
    return C.astype(np.float64)


def generate_worst_case_costs(n: int) -> np.ndarray:
    """
    Generate worst-case cost matrix (anti-diagonal structure).
    From baseline_harness.py MatrixGenerator.
    """
    C = np.ones((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            C[i, j] = abs(i - (n - 1 - j)) + 1
    return C


def generate_identity_like_costs(n: int, diagonal_cost: float = 0.0, 
                               off_diagonal_cost: float = 1.0) -> np.ndarray:
    """
    Generate identity-like cost matrix (optimal assignment is identity).
    From baseline_harness.py MatrixGenerator.
    """
    C = np.full((n, n), off_diagonal_cost, dtype=np.float64)
    np.fill_diagonal(C, diagonal_cost)
    return C


def generate_hard_random_costs(n: int, cost_range: Tuple[float, float] = (0.0, 100.0), 
                              seed: int = 42) -> np.ndarray:
    """
    Generate hard random cost matrix with row and column potentials.
    From baseline_harness.py MatrixGenerator.
    """
    np.random.seed(seed)
    
    low, high = cost_range
    C = np.random.uniform(low, high, (n, n))
    
    # Make problem harder by adding row and column potentials
    for i in range(n):
        C[i, :] += np.random.uniform(0, (high - low) * 0.1)
    for j in range(n):
        C[:, j] += np.random.uniform(0, (high - low) * 0.1)
    
    return C.astype(np.float64)
