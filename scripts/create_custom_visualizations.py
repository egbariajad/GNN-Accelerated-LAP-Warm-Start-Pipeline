#!/usr/bin/env python3
"""
Create Custom Visualizations from Benchmark Results

Generates publication-quality visualizations from existing CSV benchmark data:
1. Baseline (SciPy) vs GNN+Seeded - execution time breakdown
2. Baseline (LAP unseeded) vs GNN+Seeded - execution time breakdown  
3. Problem size vs speedup vs LAP
4. Speedup distribution vs size
5. Combined with large-scale results
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from datetime import datetime

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_all_results():
    """Load all benchmark results from CSV files."""
    
    results_dir = project_root / "results"
    
    # Load comprehensive benchmark (real data)
    comp_csv = list((results_dir / "comprehensive_benchmark").glob("comprehensive_benchmark_*.csv"))
    if comp_csv:
        comp_csv = sorted(comp_csv)[-1]  # Get most recent
        print(f"Loading comprehensive benchmark: {comp_csv.name}")
        df_real = pd.read_csv(comp_csv)
    else:
        print("⚠️  No comprehensive benchmark results found")
        df_real = pd.DataFrame()
    
    # Load large-scale benchmark (synthetic data)
    large_csv = list((results_dir / "large_scale_benchmark").glob("large_scale_benchmark_*.csv"))
    if large_csv:
        large_csv = sorted(large_csv)[-1]  # Get most recent
        print(f"Loading large-scale benchmark: {large_csv.name}")
        df_large = pd.read_csv(large_csv)
        df_large['dataset'] = 'synthetic_' + df_large['size'].astype(str)
    else:
        print("⚠️  No large-scale benchmark results found")
        df_large = pd.DataFrame()
    
    # Combine both
    if not df_real.empty and not df_large.empty:
        df_combined = pd.concat([df_real, df_large], ignore_index=True)
    elif not df_real.empty:
        df_combined = df_real
    elif not df_large.empty:
        df_combined = df_large
    else:
        print("❌ No benchmark data found!")
        return None
    
    print(f"Total instances: {len(df_combined)}")
    print(f"Size range: {df_combined['size'].min()} - {df_combined['size'].max()}")
    
    return df_combined


def create_custom_visualizations(df, output_dir):
    """Create all custom visualizations."""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Filter successful results
    df = df[df['scipy_time'].notna() & df['gnn_time'].notna() & df['seeded_time'].notna()].copy()
    
    if len(df) == 0:
        print("❌ No valid data to visualize!")
        return
    
    print(f"\nGenerating visualizations from {len(df)} successful instances...")
    
    # Compute total time
    df['total_time'] = df['gnn_time'] + df['seeded_time']
    df['speedup_vs_scipy'] = df['scipy_time'] / df['total_time']
    df['speedup_vs_lap'] = df['lap_time'] / df['total_time']
    
    # Create main figure
    fig = plt.figure(figsize=(24, 14))
    
    # ===== 1. Baseline (SciPy) vs GNN+Seeded - Execution Time Breakdown =====
    ax1 = plt.subplot(2, 3, 1)
    
    # Select representative instances across sizes (up to 25)
    sizes = sorted(df['size'].unique())
    n_show = min(25, len(df))
    show_indices = np.linspace(0, len(df)-1, n_show, dtype=int)
    
    scipy_times = (df.iloc[show_indices]['scipy_time'] * 1000).values
    gnn_times = (df.iloc[show_indices]['gnn_time'] * 1000).values
    seeded_times = (df.iloc[show_indices]['seeded_time'] * 1000).values
    
    x = np.arange(n_show)
    width = 0.8
    
    # Stacked bars for GNN+LAP
    ax1.bar(x, gnn_times, width, label='GNN Inference', color='orange', alpha=0.8)
    ax1.bar(x, seeded_times, width, bottom=gnn_times, label='Seeded LAP', color='steelblue', alpha=0.8)
    
    # SciPy baseline as horizontal line markers
    ax1.scatter(x, scipy_times, color='purple', marker='_', s=300, linewidths=4, 
               label='SciPy (baseline)', zorder=3)
    
    ax1.set_xlabel('Instance', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Time (ms)', fontsize=14, fontweight='bold')
    ax1.set_title('1. SciPy vs GNN+Seeded LAP', fontsize=16, fontweight='bold')
    ax1.legend(fontsize=12, loc='upper left')
    ax1.grid(axis='y', alpha=0.3)
    ax1.tick_params(labelsize=11)
    
    # ===== 2. Baseline (LAP unseeded) vs GNN+Seeded =====
    ax2 = plt.subplot(2, 3, 2)
    
    lap_times = (df.iloc[show_indices]['lap_time'] * 1000).values
    
    # Stacked bars for GNN+LAP
    ax2.bar(x, gnn_times, width, label='GNN Inference', color='orange', alpha=0.8)
    ax2.bar(x, seeded_times, width, bottom=gnn_times, label='Seeded LAP', color='steelblue', alpha=0.8)
    
    # LAP baseline as horizontal line markers
    ax2.scatter(x, lap_times, color='darkgreen', marker='_', s=300, linewidths=4,
               label='LAP unseeded (baseline)', zorder=3)
    
    ax2.set_xlabel('Instance', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Time (ms)', fontsize=14, fontweight='bold')
    ax2.set_title('2. LAP (unseeded) vs GNN+Seeded LAP', fontsize=16, fontweight='bold')
    ax2.legend(fontsize=12, loc='upper left')
    ax2.grid(axis='y', alpha=0.3)
    ax2.tick_params(labelsize=11)
    
    # ===== 3. Problem Size vs Speedup vs LAP =====
    ax3 = plt.subplot(2, 3, 3)
    
    # Separate by type if available
    if 'family' in df.columns:
        for family, color, marker in [('sparse', 'forestgreen', 'o'), ('uniform', 'coral', 's')]:
            family_df = df[df['family'].str.lower() == family]
            if not family_df.empty:
                ax3.scatter(family_df['size'], family_df['speedup_vs_lap'], 
                           s=100, alpha=0.7, edgecolors='black', linewidth=0.5,
                           color=color, marker=marker, label=family.capitalize())
    else:
        ax3.scatter(df['size'], df['speedup_vs_lap'], 
                   s=100, alpha=0.7, edgecolors='black', linewidth=0.5, color='steelblue')
    
    # Add trend line
    z = np.polyfit(df['size'], df['speedup_vs_lap'], 1)
    p = np.poly1d(z)
    x_trend = np.linspace(df['size'].min(), df['size'].max(), 100)
    ax3.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2, label='Trend')
    
    ax3.axhline(1.0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
    ax3.set_xlabel('Problem Size (n)', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Speedup vs LAP', fontsize=14, fontweight='bold')
    ax3.set_title('3. Speedup vs LAP by Problem Size', fontsize=16, fontweight='bold')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(labelsize=11)
    
    # ===== 4. Speedup Distribution vs Size =====
    ax4 = plt.subplot(2, 3, 4)
    
    # Group by size and create box plots
    size_groups = []
    size_labels = []
    for size in sorted(df['size'].unique()):
        size_data = df[df['size'] == size]['speedup_vs_lap'].values
        if len(size_data) > 0:
            size_groups.append(size_data)
            size_labels.append(str(size))
    
    bp = ax4.boxplot(size_groups, labels=size_labels, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)
    
    ax4.axhline(1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax4.set_xlabel('Problem Size (n)', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Speedup vs LAP', fontsize=14, fontweight='bold')
    ax4.set_title('4. Speedup Distribution by Size', fontsize=16, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    ax4.tick_params(labelsize=11)
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # ===== 5. Combined Real + Large-Scale: Speedup vs SciPy =====
    ax5 = plt.subplot(2, 3, 5)
    
    # Mark real vs synthetic
    df['source'] = df['dataset'].apply(lambda x: 'Synthetic' if 'synthetic' in str(x) else 'Real Data')
    
    for source, color, marker in [('Real Data', 'blue', 'o'), ('Synthetic', 'red', '^')]:
        source_df = df[df['source'] == source]
        if not source_df.empty and 'speedup_vs_scipy' in source_df.columns and source_df['speedup_vs_scipy'].notna().any():
            ax5.scatter(source_df['size'], source_df['speedup_vs_scipy'],
                       s=100, alpha=0.7, edgecolors='black', linewidth=0.5,
                       color=color, marker=marker, label=source)
    
    ax5.axhline(1.0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
    ax5.set_xlabel('Problem Size (n)', fontsize=14, fontweight='bold')
    ax5.set_ylabel('Speedup vs SciPy', fontsize=14, fontweight='bold')
    ax5.set_title('5. Real + Large-Scale: Speedup vs SciPy', fontsize=16, fontweight='bold')
    ax5.legend(fontsize=11)
    ax5.grid(True, alpha=0.3)
    ax5.set_xscale('log')
    ax5.tick_params(labelsize=11)
    
    # ===== 6. Combined Real + Large-Scale: Speedup vs LAP =====
    ax6 = plt.subplot(2, 3, 6)
    
    for source, color, marker in [('Real Data', 'blue', 'o'), ('Synthetic', 'red', '^')]:
        source_df = df[df['source'] == source]
        if not source_df.empty:
            ax6.scatter(source_df['size'], source_df['speedup_vs_lap'],
                       s=100, alpha=0.7, edgecolors='black', linewidth=0.5,
                       color=color, marker=marker, label=source)
    
    ax6.axhline(1.0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
    ax6.set_xlabel('Problem Size (n)', fontsize=14, fontweight='bold')
    ax6.set_ylabel('Speedup vs LAP', fontsize=14, fontweight='bold')
    ax6.set_title('6. Real + Large-Scale: Speedup vs LAP', fontsize=16, fontweight='bold')
    ax6.legend(fontsize=11)
    ax6.grid(True, alpha=0.3)
    ax6.set_xscale('log')
    ax6.tick_params(labelsize=11)
    
    plt.tight_layout()
    
    # Save figure
    output_path = output_dir / f"custom_visualization_{timestamp}.png"
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Custom visualization saved: {output_path}")
    
    # Print summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"\nTotal instances: {len(df)}")
    print(f"Size range: {df['size'].min()} - {df['size'].max()}")
    
    print(f"\nSpeedup vs SciPy:")
    if 'speedup_vs_scipy' in df.columns and df['speedup_vs_scipy'].notna().any():
        valid_scipy = df['speedup_vs_scipy'].dropna()
        print(f"  Mean:   {valid_scipy.mean():.2f}×")
        print(f"  Median: {valid_scipy.median():.2f}×")
        print(f"  Range:  {valid_scipy.min():.2f}× - {valid_scipy.max():.2f}×")
    else:
        print("  Not available")
    
    print(f"\nSpeedup vs LAP:")
    print(f"  Mean:   {df['speedup_vs_lap'].mean():.2f}×")
    print(f"  Median: {df['speedup_vs_lap'].median():.2f}×")
    print(f"  Range:  {df['speedup_vs_lap'].min():.2f}× - {df['speedup_vs_lap'].max():.2f}×")
    
    if 'source' in df.columns:
        print(f"\nBy Source:")
        for source in df['source'].unique():
            source_df = df[df['source'] == source]
            print(f"  {source}: {len(source_df)} instances, "
                  f"speedup vs LAP: {source_df['speedup_vs_lap'].mean():.2f}×")
    
    return output_path


def main():
    # Load all results
    df = load_all_results()
    
    if df is None or len(df) == 0:
        print("❌ No data to visualize!")
        return
    
    # Create output directory
    output_dir = project_root / "results" / "custom_visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate visualizations
    output_path = create_custom_visualizations(df, output_dir)
    
    print(f"\n{'='*80}")
    print("VISUALIZATION COMPLETE!")
    print("="*80)
    print(f"\nOutput: {output_path}")
    print("\nVisualization includes:")
    print("  1. SciPy vs GNN+Seeded LAP (time breakdown)")
    print("  2. LAP (unseeded) vs GNN+Seeded LAP (time breakdown)")
    print("  3. Problem size vs speedup vs LAP")
    print("  4. Speedup distribution by size (box plots)")
    print("  5. Real + Large-scale: speedup vs SciPy")
    print("  6. Real + Large-scale: speedup vs LAP")


if __name__ == "__main__":
    main()
