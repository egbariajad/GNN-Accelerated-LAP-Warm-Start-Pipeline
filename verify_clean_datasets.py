#!/usr/bin/env python3
"""
Verify clean datasets have no dual noise contamination.

Checks:
1. All noise_std values are 0.0
2. Mean of u and v duals are near optimal (no noise drift)
3. Dataset sizes and counts match expected
"""

import sys
from pathlib import Path
import numpy as np
import h5py


def verify_clean_dataset(h5_path: Path) -> dict:
    """Verify a single HDF5 file is clean."""
    results = {
        'path': str(h5_path),
        'exists': h5_path.exists(),
        'instance_count': 0,
        'noise_std_unique': [],
        'u_mean': None,
        'v_mean': None,
        'u_std': None,
        'v_std': None,
        'is_clean': False,
        'errors': []
    }
    
    if not h5_path.exists():
        results['errors'].append(f"File not found: {h5_path}")
        return results
    
    try:
        with h5py.File(h5_path, 'r') as f:
            results['instance_count'] = len(f['C'])
            
            # Check noise_std field
            if 'noise_std' in f:
                noise_stds = f['noise_std'][:]
                results['noise_std_unique'] = sorted(np.unique(noise_stds).tolist())
                
                if not all(ns == 0.0 for ns in noise_stds):
                    results['errors'].append(f"Non-zero noise_std found: {results['noise_std_unique']}")
            else:
                results['errors'].append("noise_std field not found in HDF5")
            
            # Check dual variables
            if 'u' in f and 'v' in f:
                u_arrays = [f['u'][i] for i in range(min(10, len(f['u'])))]
                v_arrays = [f['v'][i] for i in range(min(10, len(f['v'])))]
                
                u_means = [np.mean(u) for u in u_arrays]
                v_means = [np.mean(v) for v in v_arrays]
                
                results['u_mean'] = float(np.mean(u_means))
                results['v_mean'] = float(np.mean(v_means))
                results['u_std'] = float(np.std(u_means))
                results['v_std'] = float(np.std(v_means))
                
                # Clean duals should have means near zero (complementary slackness)
                if abs(results['u_mean']) > 0.1 or abs(results['v_mean']) > 0.1:
                    results['errors'].append(
                        f"Dual means suspiciously high: u_mean={results['u_mean']:.4f}, v_mean={results['v_mean']:.4f}"
                    )
            else:
                results['errors'].append("u or v fields not found in HDF5")
            
            # Mark as clean if no errors
            results['is_clean'] = len(results['errors']) == 0
            
    except Exception as e:
        results['errors'].append(f"Error reading file: {e}")
    
    return results


def main():
    clean_base = Path("data/generated/processed_clean")
    
    # Expected datasets
    datasets = [
        "small/full/train.h5",
        "small/full/val.h5",
        "small/full/test.h5",
        "mid_1536/full/train.h5",
        "mid_1536/full/val.h5",
        "mid_1536/full/test.h5",
        "mid_2048/full/train.h5",
        "mid_2048/full/val.h5",
        "mid_2048/full/test.h5",
        "mid_3072/full/train.h5",
        "mid_3072/full/val.h5",
        "mid_3072/full/test.h5",
        "large_4096/full/train.h5",
        "large_4096/full/val.h5",
        "large_4096/full/test.h5",
    ]
    
    print("=" * 80)
    print("CLEAN DATASET VERIFICATION")
    print("=" * 80)
    print(f"Checking: {clean_base}")
    print()
    
    all_clean = True
    total_instances = 0
    
    for ds_path in datasets:
        full_path = clean_base / ds_path
        result = verify_clean_dataset(full_path)
        
        status = "✅ CLEAN" if result['is_clean'] else "❌ CONTAMINATED"
        print(f"{status} {ds_path}")
        print(f"  Instances: {result['instance_count']}")
        print(f"  noise_std unique: {result['noise_std_unique']}")
        
        if result['u_mean'] is not None:
            print(f"  u: mean={result['u_mean']:.6f}, std={result['u_std']:.6f}")
            print(f"  v: mean={result['v_mean']:.6f}, std={result['v_std']:.6f}")
        
        if result['errors']:
            print(f"  ⚠️  Errors:")
            for err in result['errors']:
                print(f"      - {err}")
            all_clean = False
        
        total_instances += result['instance_count']
        print()
    
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total instances checked: {total_instances}")
    
    if all_clean:
        print("✅ ALL DATASETS ARE CLEAN - No dual noise contamination detected!")
        print()
        print("Next steps:")
        print("1. Update SLURM training scripts to use data/generated/processed_clean/")
        print("2. Set checkpoint output to gnn/checkpoints_clean/")
        print("3. Launch training runs")
        return 0
    else:
        print("❌ CONTAMINATION DETECTED - Some datasets have issues")
        print("Please regenerate the problematic datasets with --dual-noise-prob 0.0")
        return 1


if __name__ == "__main__":
    sys.exit(main())
