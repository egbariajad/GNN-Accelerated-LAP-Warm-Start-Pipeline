#!/usr/bin/env python3
"""
Utility script to regenerate LAP assignments from cost matrices.

This replaces the functionality that was stored in the raw/ data directory.
Usage: python regenerate_assignments.py <processed_data_file.h5>
"""

import sys
import h5py
import numpy as np
import scipy.optimize
from pathlib import Path


def regenerate_assignments(processed_file: str, output_file: str = None):
    """Regenerate assignments (rows, cols) from processed data file."""
    
    processed_path = Path(processed_file)
    if not processed_path.exists():
        raise FileNotFoundError(f"Processed data file not found: {processed_file}")
    
    if output_file is None:
        output_file = processed_path.parent / f"{processed_path.stem}_with_assignments.h5"
    
    print(f"Reading processed data from: {processed_file}")
    print(f"Will write assignments to: {output_file}")
    
    with h5py.File(processed_file, 'r') as f_in:
        n_problems = len(f_in['C'])
        print(f"Regenerating assignments for {n_problems} problems...")
        
        # Read all data
        costs = f_in['C'][:]
        n_values = f_in['n'][:]
        
        # Regenerate assignments
        all_rows = []
        all_cols = []
        
        for i in range(n_problems):
            n = int(n_values[i])
            cost_flat = costs[i]
            cost_matrix = cost_flat[:n*n].reshape(n, n)
            
            # Compute optimal assignment
            rows, cols = scipy.optimize.linear_sum_assignment(cost_matrix)
            
            # Pad to consistent size for HDF5 storage
            max_n = 4096  # Adjust based on your largest problem size
            padded_rows = np.full(max_n, -1, dtype=np.int32)
            padded_cols = np.full(max_n, -1, dtype=np.int32)
            padded_rows[:n] = rows
            padded_cols[:n] = cols
            
            all_rows.append(padded_rows)
            all_cols.append(padded_cols)
            
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{n_problems}")
        
        # Write to new file with assignments
        with h5py.File(output_file, 'w') as f_out:
            # Copy all existing data
            for key in f_in.keys():
                f_out.create_dataset(key, data=f_in[key][:])
            
            # Add regenerated assignments
            f_out.create_dataset('rows', data=np.array(all_rows))
            f_out.create_dataset('cols', data=np.array(all_cols))
    
    print(f"✅ Successfully regenerated assignments!")
    print(f"📁 Output file: {output_file}")
    return output_file


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python regenerate_assignments.py <processed_data_file.h5> [output_file.h5]")
        sys.exit(1)
    
    processed_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        result = regenerate_assignments(processed_file, output_file)
        print(f"✅ Done! Assignments saved to: {result}")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)