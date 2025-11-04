#!/usr/bin/env python3
"""
Create speedup breakdown by size graph comparing GNN+LAP components vs cold LAP baseline.
Shows median speedup values for each size category.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import glob
from datetime import datetime

def load_benchmark_data():
    """Load data from comprehensive and large-scale benchmarks."""
    results_dir = Path('/home/projects/nssl-prj10106/results')
    
    # Find most recent CSV files
    comprehensive_files = sorted(glob.glob(str(results_dir / 'comprehensive_benchmark' / 'comprehensive_benchmark_*.csv')))
    large_scale_files = sorted(glob.glob(str(results_dir / 'large_scale_benchmark' / 'large_scale_benchmark_*.csv')))
    
    if not comprehensive_files or not large_scale_files:
        raise FileNotFoundError("Could not find benchmark CSV files")
    
    comprehensive_file = comprehensive_files[-1]
    large_scale_file = large_scale_files[-1]
    
    print("\n" + "="*80)
    print("Creating Speedup Breakdown by Size (vs Cold LAP)")
    print("="*80)
    print(f"Comprehensive data: {Path(comprehensive_file).name}")
    print(f"Large-scale data:   {Path(large_scale_file).name}")
    print("="*80 + "\n")
    
    # Load data
    df_comp = pd.read_csv(comprehensive_file)
    df_large = pd.read_csv(large_scale_file)
    
    # Standardize column names
    if 'type' in df_large.columns:
        df_large = df_large.rename(columns={'type': 'family'})
    
    # Combine datasets
    df = pd.concat([df_comp, df_large], ignore_index=True)
    
    # Calculate missing speedup values from times
    mask_lap = df['speedup_vs_lap'].isna() & df['lap_time'].notna() & df['seeded_time'].notna()
    df.loc[mask_lap, 'speedup_vs_lap'] = df.loc[mask_lap, 'lap_time'] / df.loc[mask_lap, 'seeded_time']
    
    # Filter for sparse and uniform only
    df = df[df['family'].isin(['sparse', 'uniform'])].copy()
    
    print(f"Loaded comprehensive: {len(df_comp)} instances")
    print(f"Loaded large-scale: {len(df_large)} instances")
    print(f"Combined total: {len(df)} instances (sparse + uniform only)\n")
    
    return df

def create_speedup_breakdown_plot(df, output_path=None):
    """
    Create bar chart showing speedup breakdown by size category.
    Y-axis: Speedup vs cold LAP
    X-axis: Size categories
    Bars: GNN Inference speedup contribution, Seeded LAP speedup
    """
    
    # Get unique sizes, sorted
    sizes = sorted(df['size'].unique())
    
    # Calculate median values for each size
    size_labels = []
    lap_baseline_speedups = []  # Total speedup vs LAP (for reference line)
    gnn_contributions = []  # How much faster we are due to GNN inference
    seeded_lap_speedups = []  # Speedup of seeded LAP vs cold LAP
    
    for size in sizes:
        size_data = df[df['size'] == size]
        
        # Get median times
        median_lap_time = size_data['lap_time'].median()
        median_gnn_time = size_data['gnn_time'].median()
        median_seeded_time = size_data['seeded_time'].median()
        median_total_time = size_data['total_time'].median()
        
        # Calculate speedup vs cold LAP: lap_time / total_time
        # total_time = gnn_time + seeded_time
        total_speedup = median_lap_time / median_total_time if median_total_time > 0 else 0
        
        # Calculate component speedups relative to cold LAP time
        # GNN contribution: time saved by not having full LAP overhead on that portion
        # Seeded LAP: speedup of seeded LAP vs cold LAP in terms of that component
        
        # For visualization: show as stacked components
        # Bottom bar: seeded_time speedup component = lap_time / (gnn_time + seeded_time) - contribution from GNN
        # Top bar: GNN speedup contribution
        
        # Simpler approach: show the speedup of seeded_time vs lap_time directly
        seeded_speedup = median_lap_time / median_seeded_time if median_seeded_time > 0 else 0
        
        # Format size labels
        if size >= 1000:
            size_label = f"{size//1000}K" if size % 1000 == 0 else f"{size/1000:.1f}K"
        else:
            size_label = str(size)
        
        size_labels.append(size_label)
        lap_baseline_speedups.append(total_speedup)
        seeded_lap_speedups.append(seeded_speedup)
        
        # GNN is overhead, not speedup - for this chart we'll show:
        # 1. Seeded LAP speedup (how much faster seeded LAP is vs cold LAP)
        # 2. Total speedup (including GNN overhead)
        
    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 7))
    
    x = np.arange(len(size_labels))
    width = 0.6
    
    # Create bars
    # We'll show speedup vs cold LAP broken down by components
    # The approach: 
    # - Cold LAP time = baseline (100%)
    # - GNN+LAP total time = smaller (shows as speedup)
    # - Show GNN time as "overhead" and Seeded time as "savings"
    
    # Better visualization: Show three bars per size
    # 1. Cold LAP time (baseline = 1.0 speedup, for reference)
    # 2. GNN+LAP total speedup
    # 3. Seeded LAP only speedup (without GNN overhead)
    
    width = 0.25
    
    # Recalculate for better visualization
    cold_lap_baseline = [1.0] * len(sizes)  # Cold LAP is baseline
    
    bars1 = ax.bar(x - width, cold_lap_baseline, width, 
                   label='Cold LAP (baseline)', 
                   color='#d4a5a5', edgecolor='black', linewidth=1.5)
    
    bars2 = ax.bar(x, lap_baseline_speedups, width,
                   label='GNN+LAP (total)', 
                   color='#90caf9', edgecolor='black', linewidth=1.5)
    
    bars3 = ax.bar(x + width, seeded_lap_speedups, width,
                   label='Seeded LAP only', 
                   color='#66bb6a', edgecolor='black', linewidth=1.5)
    
    # Styling
    ax.set_xlabel('Problem Size', fontsize=14, fontweight='bold')
    ax.set_ylabel('Speedup vs Cold LAP', fontsize=14, fontweight='bold')
    ax.set_title('Speedup Breakdown by Size (vs Cold LAP Baseline)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(size_labels, fontsize=12)
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    
    # Add horizontal line at speedup=1
    ax.axhline(y=1.0, color='gray', linestyle=':', linewidth=2, alpha=0.7)
    
    # Add value labels on bars
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            if height > 0.1:  # Only label significant bars
                ax.annotate(f'{height:.2f}×',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom',
                           fontsize=9, fontweight='bold')
    
    autolabel(bars2)
    autolabel(bars3)
    
    plt.tight_layout()
    
    # Save figure
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {output_path}")
    
    plt.close()
    
    # Print statistics
    print("\nSpeedup Breakdown by Size:")
    print("-" * 80)
    print(f"{'Size':<10} {'Total Speedup':<20} {'Seeded LAP Only':<20}")
    print("-" * 80)
    for i, size_label in enumerate(size_labels):
        print(f"{size_label:<10} {lap_baseline_speedups[i]:<20.3f}× {seeded_lap_speedups[i]:<20.3f}×")
    print("-" * 80)

def main():
    """Main execution function."""
    # Load data
    df = load_benchmark_data()
    
    # Create output directory
    output_dir = Path('/home/projects/nssl-prj10106/results/custom_visualizations')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create plot
    print("\nGenerating speedup breakdown by size...\n")
    
    output = output_dir / f'speedup_breakdown_by_size_{timestamp}.png'
    create_speedup_breakdown_plot(df, output_path=output)
    
    print("\n" + "="*80)
    print("✅ Speedup breakdown visualization created successfully!")
    print("="*80)
    print(f"\nOutput: {output}")
    print("="*80 + "\n")

if __name__ == '__main__':
    main()
