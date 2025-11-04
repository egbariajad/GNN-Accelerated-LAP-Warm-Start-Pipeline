#!/usr/bin/env python3
"""
Demo: Seeded JV function in lap package

This demonstrates the new lapjv_seeded() function that accepts 
initial dual potentials (u,v) and skips JV initialization phase.
"""

import numpy as np
import lap

def demo_seeded_jv():
    """Demonstrate the seeded JV functionality"""
    
    print("=== Seeded JV Demo ===\n")
    
    # Create a test cost matrix
    C = np.array([
        [4.0, 2.0, 8.0, 6.0],
        [6.0, 4.0, 1.0, 2.0], 
        [8.0, 6.0, 4.0, 3.0],
        [2.0, 8.0, 5.0, 7.0]
    ])
    
    print("Cost matrix C:")
    print(C)
    
    # Run standard LAPJV
    cost_regular, x_regular, y_regular = lap.lapjv(C)
    print(f"\nStandard LAPJV:")
    print(f"  Cost: {cost_regular}")
    print(f"  Row assignment: {x_regular}")
    print(f"  Col assignment: {y_regular}")
    
    # Create feasible seed potentials
    u_seed = np.zeros(4)                    # row potentials
    v_seed = np.array([2.0, 2.0, 1.0, 2.0])  # column potentials
    
    print(f"\nSeed potentials:")
    print(f"  u (row): {u_seed}")
    print(f"  v (col): {v_seed}")
    
    # Check feasibility
    reduced_costs = C - u_seed[:, None] - v_seed[None, :]
    print(f"\nReduced costs C - u - v (all should be >= 0):")
    print(reduced_costs)
    print(f"Min reduced cost: {reduced_costs.min():.6f}")
    
    # Run seeded LAPJV
    x_seeded, y_seeded, cost_seeded = lap.lapjv_seeded(C, u_seed, v_seed)
    print(f"\nSeeded LAPJV:")
    print(f"  Cost: {cost_seeded}")
    print(f"  Row assignment: {x_seeded}")
    print(f"  Col assignment: {y_seeded}")
    
    # Compare results
    if cost_seeded == cost_regular:
        print(f"\n✓ SUCCESS: Both methods found optimal cost {cost_regular}")
    else:
        print(f"\n✗ ERROR: Different costs - regular: {cost_regular}, seeded: {cost_seeded}")
    
    # Test infeasible seeds
    print(f"\n=== Testing infeasible seeds ===")
    u_bad = np.array([10.0, 10.0, 10.0, 10.0])  # too large
    v_bad = np.zeros(4)
    
    try:
        x_bad, y_bad, cost_bad = lap.lapjv_seeded(C, u_bad, v_bad)
        print("✗ ERROR: Should have detected infeasible seeds")
    except ValueError as e:
        print(f"✓ SUCCESS: Correctly detected infeasible seeds: {e}")

if __name__ == "__main__":
    demo_seeded_jv()