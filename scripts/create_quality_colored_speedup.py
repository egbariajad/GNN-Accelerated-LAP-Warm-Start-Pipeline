#!/usr/bin/env python3
"""
Create speedup vs problem size graphs colored by solution quality (primal gap).
Generates two separate plots:
1. Speedup vs SciPy (baseline)
2. Speedup vs LAP (baseline)

Both colored by primal gap percentage to show solution quality.
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
    print("Creating Speedup vs Size Graphs (Colored by Solution Quality)")
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
    # speedup_vs_scipy = scipy_time / total_time
    mask_scipy = df['speedup_vs_scipy'].isna() & df['scipy_time'].notna() & df['total_time'].notna()
    df.loc[mask_scipy, 'speedup_vs_scipy'] = df.loc[mask_scipy, 'scipy_time'] / df.loc[mask_scipy, 'total_time']
    
    # speedup_vs_lap = lap_time / seeded_time
    mask_lap = df['speedup_vs_lap'].isna() & df['lap_time'].notna() & df['seeded_time'].notna()
    df.loc[mask_lap, 'speedup_vs_lap'] = df.loc[mask_lap, 'lap_time'] / df.loc[mask_lap, 'seeded_time']
    
    # Fill missing primal_gap with 0 (for cases where scipy wasn't run but solutions are still optimal)
    # This is valid because LAP and GNN+LAP both find optimal solutions for assignment problems
    df['primal_gap'] = df['primal_gap'].fillna(0.0)
    
    # Filter for sparse and uniform only
    df = df[df['family'].isin(['sparse', 'uniform'])].copy()
    
    print(f"Loaded comprehensive: {len(df_comp)} instances")
    print(f"Loaded large-scale: {len(df_large)} instances")
    print(f"Combined total: {len(df)} instances (sparse + uniform only)\n")
    
    return df

def calculate_primal_gap(df):
    """
    Calculate primal gap as percentage difference from optimal cost.
    Primal Gap (%) = 100 * (seeded_cost - optimal_cost) / optimal_cost
    
    Note: For assignment problems, all methods should find optimal solutions,
    so gaps should be near zero. Non-zero gaps indicate numerical precision differences.
    """
    # Use scipy_cost as the reference optimal cost (scipy guarantees optimality)
    df['primal_gap'] = 100.0 * (df['seeded_cost'] - df['scipy_cost']) / df['scipy_cost'].abs()
    
    return df

def create_speedup_plot(df, baseline='scipy', output_path=None):
    """
    Create speedup vs problem size scatter plot colored by solution quality.
    
    Args:
        df: DataFrame with benchmark results
        baseline: 'scipy' or 'lap' - which baseline to compare against
        output_path: Path to save the figure
    """
    if baseline == 'scipy':
        speedup_col = 'speedup_vs_scipy'
        title = 'Speedup vs Problem Size (GNN+LAP vs SciPy)'
        baseline_name = 'SciPy'
    else:
        speedup_col = 'speedup_vs_lap'
        title = 'Speedup vs Problem Size (GNN+LAP vs LAP)'
        baseline_name = 'LAP'
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Get data
    sizes = df['size'].values
    speedups = df[speedup_col].values
    gaps = df['primal_gap'].values
    
    # Remove NaN values
    valid_mask = ~(np.isnan(sizes) | np.isnan(speedups) | np.isnan(gaps))
    sizes = sizes[valid_mask]
    speedups = speedups[valid_mask]
    gaps = gaps[valid_mask]
    
    print(f"Valid data points for {baseline_name}: {len(sizes)} / {len(df)}")
    
    # Create scatter plot with color mapping
    # Use a diverging colormap centered at 0
    scatter = ax.scatter(sizes, speedups, 
                        c=gaps, 
                        cmap='RdYlGn_r',  # Red-Yellow-Green reversed (red=bad, green=good)
                        s=100, 
                        alpha=0.7,
                        edgecolors='black',
                        linewidth=0.5,
                        vmin=-0.1,
                        vmax=0.1)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Primal Gap (%)', fontsize=12, fontweight='bold')
    
    # Calculate and plot trend line
    # Use log scale for better fit across wide size range
    log_sizes = np.log(sizes)
    z = np.polyfit(log_sizes, speedups, 1)
    p = np.poly1d(z)
    
    # Create smooth curve for trend line
    size_range = np.logspace(np.log10(sizes.min()), np.log10(sizes.max()), 100)
    trend_speedups = p(np.log(size_range))
    ax.plot(size_range, trend_speedups, 'r--', linewidth=2, label='Trend', alpha=0.8)
    
    # Styling
    ax.set_xlabel('Problem Size (n)', fontsize=14, fontweight='bold')
    ax.set_ylabel(f'Speedup vs {baseline_name}', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=11, loc='upper left')
    
    # Set reasonable axis limits
    ax.set_xlim(400, 20000)
    
    # Use log scale for x-axis to better show wide range
    ax.set_xscale('log')
    
    # Format x-axis ticks - show all sizes from 512 to 16384
    ax.set_xticks([512, 1024, 1536, 2048, 3072, 4096, 8192, 16384])
    ax.set_xticklabels(['512', '1K', '1.5K', '2K', '3K', '4K', '8K', '16K'], rotation=0)
    
    # Add horizontal line at speedup=1
    ax.axhline(y=1.0, color='gray', linestyle=':', linewidth=1.5, alpha=0.5)
    
    plt.tight_layout()
    
    # Save figure
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {output_path}")
    
    plt.close()
    
    # Print statistics
    print(f"\n{title}")
    print("-" * 80)
    print(f"Mean speedup: {speedups.mean():.3f}x")
    print(f"Median speedup: {np.median(speedups):.3f}x")
    print(f"Min speedup: {speedups.min():.3f}x")
    print(f"Max speedup: {speedups.max():.3f}x")
    print(f"Std speedup: {speedups.std():.3f}x")
    print(f"\nMean primal gap: {gaps.mean():.6f}%")
    print(f"Max |primal gap|: {np.abs(gaps).max():.6f}%")
    print(f"Std primal gap: {gaps.std():.6f}%")
    print("-" * 80)

def main():
    """Main execution function."""
    # Load data
    df = load_benchmark_data()
    
    # Primal gap is already calculated in the CSV files
    # No need to recalculate
    
    # Create output directory
    output_dir = Path('/home/projects/nssl-prj10106/results/custom_visualizations')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create both plots
    print("\nGenerating speedup vs size plots colored by solution quality...\n")
    
    # Plot 1: GNN+LAP vs SciPy
    scipy_output = output_dir / f'speedup_vs_size_scipy_quality_{timestamp}.png'
    create_speedup_plot(df, baseline='scipy', output_path=scipy_output)
    
    # Plot 2: GNN+LAP vs LAP
    lap_output = output_dir / f'speedup_vs_size_lap_quality_{timestamp}.png'
    create_speedup_plot(df, baseline='lap', output_path=lap_output)
    
    print("\n" + "="*80)
    print("✅ All quality-colored speedup plots created successfully!")
    print("="*80)
    print(f"\nOutput directory: {output_dir}")
    print("\nGenerated files:")
    print(f"  1. {scipy_output.name}")
    print(f"  2. {lap_output.name}")
    print("="*80 + "\n")

if __name__ == '__main__':
    main()
