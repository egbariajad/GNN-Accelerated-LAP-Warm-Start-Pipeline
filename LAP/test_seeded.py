#!/usr/bin/env python3
"""Quick test of the seeded JV function"""
import numpy as np
import lap

def test_seeded_jv():
    # Create a simple 3x3 test case
    C = np.array([
        [4.0, 1.0, 3.0],
        [2.0, 0.0, 5.0],
        [3.0, 2.0, 2.0]
    ])
    
    print("Cost matrix C:")
    print(C)
    
    # Run regular LAPJV first to get optimal solution
    cost_regular, x_regular, y_regular = lap.lapjv(C)
    print(f"\nRegular LAPJV: cost={cost_regular}, x={x_regular}, y={y_regular}")
    
    # Create feasible seed potentials (zero is always feasible)
    u_seed = np.zeros(3)
    v_seed = np.zeros(3)
    
    print(f"\nSeed potentials: u={u_seed}, v={v_seed}")
    
    # Test seeded function
    try:
        x_seeded, y_seeded, cost_seeded = lap.lapjv_seeded(C, u_seed, v_seed)
        print(f"Seeded LAPJV: cost={cost_seeded}, x={x_seeded}, y={y_seeded}")
        
        # Verify feasibility
        reduced_costs = C - u_seed[:, None] - v_seed[None, :]
        print(f"\nReduced costs (should all be >= 0):")
        print(reduced_costs)
        print(f"Min reduced cost: {reduced_costs.min()}")
        
        if cost_seeded == cost_regular:
            print("\n✓ SUCCESS: Seeded and regular solutions have same optimal cost!")
        else:
            print(f"\n✗ WARNING: Different costs - regular: {cost_regular}, seeded: {cost_seeded}")
            
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        return False
    
    return True

if __name__ == "__main__":
    test_seeded_jv()