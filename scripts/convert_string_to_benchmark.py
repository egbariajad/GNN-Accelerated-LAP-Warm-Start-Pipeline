#!/usr/bin/env python3
"""
Convert STRING dataset to format compatible with gnn_benchmark.py

The gnn_benchmark expects test.h5 files with:
  - C: cost matrices (flattened)
  - u: ground truth dual variables  
  - v: ground truth dual variables
  - n: problem sizes

We'll compute ground truth duals using SciPy's solution.
"""

import sys
from pathlib import Path
import numpy as np
import h5py
from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from solvers import SciPySolver

def convert_string_to_benchmark_format(
    input_file: str,
    output_file: str
):
    """Convert STRING dataset to gnn_benchmark compatible format."""
    
    print(f"Converting {input_file} to gnn_benchmark format...")
    
    # Load STRING data
    instances = []
    with h5py.File(input_file, 'r') as f:
        for key in tqdm(sorted(f.keys()), desc="Loading STRING data"):
            grp = f[key]
            C = grp['cost_matrix'][:]
            n = grp.attrs['n']
            instances.append((C, n))
    
    print(f"Loaded {len(instances)} instances")
    
    # Compute ground truth duals using SciPy
    print("Computing ground truth duals with SciPy...")
    solver = SciPySolver()
    
    processed = []
    for C, n in tqdm(instances, desc="Computing duals"):
        # Solve optimally
        rows, cols, cost = solver.solve(C)
        
        # Compute dual variables from optimal assignment
        # For LAP, the duals satisfy: C[i,j] >= u[i] + v[j] with equality for assigned pairs
        u = np.zeros(n, dtype=np.float64)
        v = np.zeros(n, dtype=np.float64)
        
        # Simple dual recovery: set u[i] = min_j(C[i,j] - v[j])
        # and v[j] = C[assigned_i, j] - u[assigned_i] for assigned pairs
        for i in range(n):
            j = cols[i]
            v[j] = C[i, j] - u[i]
        
        # Refine u values
        for i in range(n):
            u[i] = np.min(C[i, :] - v)
        
        processed.append({
            'C': C.flatten(),
            'u': u,
            'v': v,
            'n': n,
            'cost_optimal': cost
        })
    
    # Save in gnn_benchmark format
    print(f"Saving to {output_file}...")
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(output_file, 'w') as f:
        # Find max size for padding
        max_n = max(inst['n'] for inst in processed)
        num_instances = len(processed)
        
        # Pre-allocate padded arrays
        C_padded = np.zeros((num_instances, max_n * max_n), dtype=np.float64)
        u_padded = np.zeros((num_instances, max_n), dtype=np.float64)
        v_padded = np.zeros((num_instances, max_n), dtype=np.float64)
        n_array = np.array([inst['n'] for inst in processed])
        
        # Fill with data (padded with zeros)
        for i, inst in enumerate(processed):
            n = inst['n']
            C_padded[i, :n*n] = inst['C']
            u_padded[i, :n] = inst['u']
            v_padded[i, :n] = inst['v']
        
        f.create_dataset('C', data=C_padded, compression='gzip', compression_opts=4)
        f.create_dataset('u', data=u_padded, compression='gzip', compression_opts=4)
        f.create_dataset('v', data=v_padded, compression='gzip', compression_opts=4)
        f.create_dataset('n', data=n_array)
        
        # Store optimal costs as metadata
        costs = [inst['cost_optimal'] for inst in processed]
        f.attrs['optimal_costs'] = costs
    
    print(f"✓ Saved {len(processed)} instances to {output_file}")
    print(f"  Size range: {n_array.min()}-{n_array.max()}")
    print(f"  File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    # Convert the STRING test data
    input_file = "data/string_dataset/processed/test.h5"
    output_dir = "data/generated/processed/string"
    
    # Determine size category based on instances
    with h5py.File(input_file, 'r') as f:
        sizes = [f[key].attrs['n'] for key in f.keys()]
        min_size = min(sizes)
        max_size = max(sizes)
    
    print(f"STRING dataset size range: {min_size}-{max_size}")
    
    # Choose appropriate output directory
    if max_size <= 1024:
        output_subdir = "small"
    elif max_size <= 1536:
        output_subdir = "mid_1536"
    elif max_size <= 2048:
        output_subdir = "mid_2048"
    else:
        output_subdir = "large"
    
    output_file = f"{output_dir}/{output_subdir}/full/test.h5"
    
    print(f"\nOutput: {output_file}")
    convert_string_to_benchmark_format(input_file, output_file)
    
    print("\n" + "="*80)
    print("✓ Conversion complete!")
    print("="*80)
    print(f"\nNow you can test with gnn_benchmark.py:")
    print(f"  python scripts/gnn_benchmark.py \\")
    print(f"    --models progressive_clean_best.pt one_gnn_mid1536_full_clean_h192L4.pt \\")
    print(f"    --sizes {min_size} {max_size} \\")
    print(f"    --max-instances 20")
    print("="*80)
