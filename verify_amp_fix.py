#!/usr/bin/env python3
"""
Verify that AMP is disabled in the training script.
This prevents FP16 overflow issues that caused NaN training loss.
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

def check_amp_status():
    """Read training script and verify AMP is disabled."""
    
    train_file = Path(__file__).parent / "gnn" / "train_one_gnn.py"
    
    with open(train_file) as f:
        content = f.read()
    
    # Check for the disabled AMP settings
    if 'scaler = GradScaler(enabled=False)' in content:
        print("✅ GradScaler correctly disabled (enabled=False)")
    else:
        print("❌ WARNING: GradScaler not explicitly disabled")
        return False
    
    if 'use_amp = False' in content:
        print("✅ AMP correctly disabled (use_amp = False)")
    else:
        print("❌ WARNING: use_amp not explicitly set to False")
        return False
    
    # Check for safer big constant
    if "big = batch.cost.new_tensor(1e6)" in content:
        print("✅ Safer 'big' constant used (1e6 instead of 1e9)")
    else:
        print("⚠️  Note: 'big' constant may still be 1e9")
    
    print("\n" + "="*60)
    print("AMP DISABLED - Training will use FP32 for stability")
    print("This prevents FP16 overflow that caused NaN loss issues")
    print("="*60)
    
    return True

if __name__ == "__main__":
    success = check_amp_status()
    sys.exit(0 if success else 1)
