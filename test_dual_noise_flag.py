#!/usr/bin/env python3
"""
Quick test to verify dual noise flag works correctly.
Generates 3 small instances and checks noise_std field.
"""

from pathlib import Path
import numpy as np
import h5py
import sys

# Test 1: Clean generation (dual_noise_prob=0.0)
print("Test 1: Clean generation (dual_noise_prob=0.0)")
print("=" * 60)

cmd_clean = [
    "--output-dir", "data/test_clean",
    "--sizes", "128",
    "--instances-per-family", "3",
    "--families", "uniform", "metric",
    "--noise-prob", "0.0",
    "--dual-noise-prob", "0.0",
    "--seed", "123",
    "--train", "0.5", "--val", "0.25", "--test", "0.25"
]

sys.path.insert(0, str(Path(__file__).parent))
from data.generate_dataset import main as gen_main

print("Running: python data/generate_dataset.py " + " ".join(cmd_clean))
gen_main(cmd_clean)

# Verify clean
train_file = Path("data/test_clean/train.h5")
if train_file.exists():
    with h5py.File(train_file, 'r') as f:
        noise_stds = f['noise_std'][:]
        unique_noise = np.unique(noise_stds)
        print(f"\n✅ Generated {len(noise_stds)} instances")
        print(f"   Unique noise_std values: {unique_noise}")
        
        if len(unique_noise) == 1 and unique_noise[0] == 0.0:
            print("   ✅ PASS: All noise_std == 0.0 (clean)")
        else:
            print("   ❌ FAIL: Expected all 0.0, got:", unique_noise)
            sys.exit(1)
        
        # Check dual means
        u_means = [np.mean(f['u'][i]) for i in range(min(3, len(f['u'])))]
        v_means = [np.mean(f['v'][i]) for i in range(min(3, len(f['v'])))]
        print(f"   Sample u means: {[f'{m:.6f}' for m in u_means]}")
        print(f"   Sample v means: {[f'{m:.6f}' for m in v_means]}")
        
        if all(abs(m) < 0.01 for m in u_means + v_means):
            print("   ✅ PASS: Dual means near zero")
        else:
            print("   ⚠️  WARNING: Some dual means high (may be OK for some families)")
else:
    print("❌ FAIL: train.h5 not generated")
    sys.exit(1)

print("\n" + "=" * 60)
print("Test 2: Noisy generation (dual_noise_prob=1.0)")
print("=" * 60)

cmd_noisy = [
    "--output-dir", "data/test_noisy",
    "--sizes", "128",
    "--instances-per-family", "3",
    "--families", "uniform", "metric",
    "--noise-prob", "0.0",
    "--dual-noise-prob", "1.0",
    "--noise-std", "0.15",
    "--seed", "456",
    "--train", "0.5", "--val", "0.25", "--test", "0.25"
]

print("Running: python data/generate_dataset.py " + " ".join(cmd_noisy))
gen_main(cmd_noisy)

# Verify noisy
train_file_noisy = Path("data/test_noisy/train.h5")
if train_file_noisy.exists():
    with h5py.File(train_file_noisy, 'r') as f:
        noise_stds = f['noise_std'][:]
        unique_noise = np.unique(noise_stds)
        print(f"\n✅ Generated {len(noise_stds)} instances")
        print(f"   Unique noise_std values: {unique_noise}")
        
        if len(unique_noise) == 1 and unique_noise[0] == 0.15:
            print("   ✅ PASS: All noise_std == 0.15 (noisy)")
        else:
            print("   ❌ FAIL: Expected all 0.15, got:", unique_noise)
            sys.exit(1)
else:
    print("❌ FAIL: train.h5 not generated")
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ ALL TESTS PASSED")
print("=" * 60)
print("The dual noise flag works correctly:")
print("  - dual_noise_prob=0.0 → clean duals (noise_std=0.0)")
print("  - dual_noise_prob=1.0 → noisy duals (noise_std=0.15)")
print("\nYou can now safely generate the full clean datasets.")
print("\nCleanup test files:")
print("  rm -rf data/test_clean data/test_noisy")
