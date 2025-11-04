#!/usr/bin/env python3
"""
Create STRING-style benchmark visualizations from comprehensive and large-scale results.

Generates:
1. Solver Comparison (SciPy, LAP, GNN+LAP bar chart)
2. GNN+LAP Speedup (horizontal bars for vs SciPy and vs LAP)
3. GNN+LAP Time Breakdown (stacked bar: GNN + LAP)
4. Summary Statistics Table
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_benchmark_data(comp_csv: Path, large_csv: Path = None):
    """Load and combine comprehensive and large-scale benchmark data."""
    
    # Load comprehensive benchmark (real data)
    df_comp = pd.read_csv(comp_csv)
    print(f"Loaded comprehensive: {len(df_comp)} instances")
    
    # Load large-scale if available
    if large_csv and large_csv.exists():
        df_large = pd.read_csv(large_csv)
        print(f"Loaded large-scale: {len(df_large)} instances")
        
        # Rename 'type' to 'family' in large-scale to match
        if 'type' in df_large.columns:
            df_large = df_large.rename(columns={'type': 'family'})
        
        # Add dataset name for large-scale
        df_large['dataset'] = df_large['size'].apply(lambda x: f'synthetic_{x}')
        
        # Combine
        df = pd.concat([df_comp, df_large], ignore_index=True)
        print(f"Combined total: {len(df)} instances")
    else:
        df = df_comp
        print("Using only comprehensive data")
    
    return df


def create_string_style_visualization(df: pd.DataFrame, output_path: Path, title_suffix: str = ""):
    """Create STRING-style 4-panel visualization."""
    
    # Calculate overall statistics
    mean_scipy = df['scipy_time'].mean()
    mean_lap = df['lap_time'].mean()
    mean_gnn = df['gnn_time'].mean()
    mean_seeded = df['seeded_time'].mean()
    mean_total = df['total_time'].mean()
    
    # Speedups (only for rows with scipy_time)
    df_with_scipy = df[df['scipy_time'].notna()]
    mean_speedup_scipy = df_with_scipy['speedup_vs_scipy'].mean() if len(df_with_scipy) > 0 else np.nan
    
    # Speedup vs LAP (all rows have lap_time)
    mean_speedup_lap = df['speedup_vs_lap'].mean() if 'speedup_vs_lap' in df.columns else (mean_lap / mean_total)
    
    # Problem size info
    sizes = df['size'].unique()
    min_size = df['size'].min()
    max_size = df['size'].max()
    n_instances = len(df)
    
    # Density
    mean_density = df['density'].mean()
    
    # GNN overhead
    gnn_overhead_pct = 100 * mean_gnn / mean_total
    lap_improvement = mean_lap - mean_seeded
    
    # Primal gap (only where available)
    df_with_gap = df[df['primal_gap'].notna()]
    mean_gap = df_with_gap['primal_gap'].mean() if len(df_with_gap) > 0 else 0.0
    
    # Create figure with specific layout matching STRING benchmark
    fig = plt.figure(figsize=(18, 10))
    
    # Title
    if title_suffix:
        fig.suptitle(f'Benchmark Results - {title_suffix}\n'
                    f'Size Range: {min_size}×{min_size} to {max_size}×{max_size}, '
                    f'Instances: {n_instances}, Mean Density: {mean_density:.2%}',
                    fontsize=16, fontweight='bold', y=0.98)
    else:
        fig.suptitle(f'Comprehensive Benchmark (n={min_size} to {max_size}, {n_instances} instances, density={mean_density:.2%})',
                    fontsize=16, fontweight='bold', y=0.98)
    
    # ===== Panel 1: Solver Comparison (Top Left) =====
    ax1 = plt.subplot(2, 2, 1)
    
    solvers = ['SciPy', 'LAP\n(Pure)', 'GNN+LAP\n(Seeded)']
    times = [mean_scipy, mean_lap, mean_total]
    colors = ['#ff6b6b', '#4ecdc4', '#95e1d3']
    
    bars = ax1.bar(solvers, times, color=colors, edgecolor='black', linewidth=1.5, alpha=0.85)
    
    # Add value labels on bars
    for bar, time in zip(bars, times):
        height = bar.get_height()
        if not np.isnan(height):
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{time:.2f}s', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax1.set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax1.set_title('Solver Comparison', fontsize=13, fontweight='bold', pad=10)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_axisbelow(True)
    
    # ===== Panel 2: GNN+LAP Speedup (Top Right) =====
    ax2 = plt.subplot(2, 2, 2)
    
    # Create horizontal bars with baseline at 1.0x
    speedup_labels = ['vs LAP', 'vs SciPy']
    speedup_values = [mean_speedup_lap, mean_speedup_scipy]
    speedup_colors = ['#4ecdc4', '#ff9999']
    
    y_pos = np.arange(len(speedup_labels))
    
    # Draw bars
    bars = ax2.barh(y_pos, speedup_values, color=speedup_colors, 
                    edgecolor='black', linewidth=1.5, alpha=0.8, height=0.6)
    
    # Add speedup text on bars
    for i, (bar, val) in enumerate(zip(bars, speedup_values)):
        if not np.isnan(val):
            ax2.text(val - 0.05, bar.get_y() + bar.get_height()/2,
                    f'{val:.2f}×', ha='right', va='center', 
                    fontsize=12, fontweight='bold', color='white' if val > 0.5 else 'black')
    
    # Baseline line at 1.0x
    ax2.axvline(1.0, color='red', linestyle='--', linewidth=2, label='Baseline (1.0×)')
    
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(speedup_labels, fontsize=11)
    ax2.set_xlabel('Speedup (×)', fontsize=12, fontweight='bold')
    ax2.set_title('GNN+LAP Speedup', fontsize=13, fontweight='bold', pad=10)
    ax2.legend(loc='lower right', fontsize=10)
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    ax2.set_xlim(0, max(speedup_values) * 1.15)
    ax2.set_axisbelow(True)
    
    # ===== Panel 3: GNN+LAP Time Breakdown (Bottom Left) =====
    ax3 = plt.subplot(2, 2, 3)
    
    components = ['GNN\nPrediction', 'LAP\nSolve']
    comp_times = [mean_gnn, mean_seeded]
    comp_colors = ['#ffa726', '#42a5f5']
    
    # Stacked bar
    bottom = 0
    bars = []
    for i, (comp, time, color) in enumerate(zip(components, comp_times, comp_colors)):
        bar = ax3.bar(['Pipeline'], [time], bottom=bottom, 
                     color=color, edgecolor='black', linewidth=1.5, 
                     alpha=0.85, label=comp, width=0.5)
        bars.append(bar)
        
        # Add time and percentage labels
        pct = 100 * time / mean_total
        ax3.text(0, bottom + time/2, 
                f'{time:.2f}s\n({pct:.1f}%)',
                ha='center', va='center', fontsize=11, fontweight='bold')
        
        bottom += time
    
    ax3.set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax3.set_title('GNN+LAP Time Breakdown', fontsize=13, fontweight='bold', pad=10)
    ax3.legend(loc='upper right', fontsize=10)
    ax3.set_ylim(0, mean_total * 1.15)
    ax3.set_xlim(-0.5, 0.5)
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    ax3.set_axisbelow(True)
    ax3.set_xticks([])
    
    # ===== Panel 4: Summary Statistics Table (Bottom Right) =====
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis('off')
    
    # Create table data
    table_data = [
        ['Metric', 'Summary Statistics'],
        ['', ''],
        ['Matrix Size Range', f'{min_size}×{min_size} to {max_size}×{max_size}'],
        ['Number of Instances', f'{n_instances}'],
        ['Mean Density', f'{mean_density:.2%}'],
        ['', ''],
        ['SciPy Time', f'{mean_scipy:.2f}s' if not np.isnan(mean_scipy) else 'N/A'],
        ['LAP Time', f'{mean_lap:.2f}s'],
        ['GNN+LAP Time', f'{mean_total:.2f}s'],
        ['', ''],
        ['Speedup vs SciPy', f'{mean_speedup_scipy:.2f}×' if not np.isnan(mean_speedup_scipy) else 'N/A'],
        ['Speedup vs LAP', f'{mean_speedup_lap:.2f}×'],
        ['', ''],
        ['GNN Overhead', f'{mean_gnn:.2f}s ({gnn_overhead_pct:.1f}%)'],
        ['LAP Improvement', f'{lap_improvement:.2f}s'],
        ['Mean Primal Gap', f'{mean_gap:.4f}%' if not np.isnan(mean_gap) else '0.00%'],
    ]
    
    # Create table
    table = ax4.table(cellText=table_data, cellLoc='left',
                     bbox=[0.05, 0.05, 0.9, 0.9],
                     colWidths=[0.5, 0.5])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    
    # Style the table
    for i, row in enumerate(table_data):
        for j in range(2):
            cell = table[(i, j)]
            
            # Header row
            if i == 0:
                cell.set_facecolor('#e0e0e0')
                cell.set_text_props(weight='bold', fontsize=11)
                cell.set_height(0.08)
            # Empty separator rows
            elif row[0] == '':
                cell.set_facecolor('#f5f5f5')
                cell.set_height(0.03)
            # Data rows
            else:
                if j == 0:  # Metric names
                    cell.set_text_props(weight='bold')
                cell.set_height(0.06)
            
            cell.set_edgecolor('black')
            cell.set_linewidth(0.5)
    
    ax4.set_title('Summary Statistics', fontsize=13, fontweight='bold', 
                 pad=20, loc='center')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"✅ STRING-style visualization saved: {output_path}")


def create_speedup_vs_size_plot(df: pd.DataFrame, output_path: Path):
    """Create speedup vs problem size scatter plot (like your second image)."""
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Get data with scipy_time available
    df_plot = df[df['scipy_time'].notna()].copy()
    
    if len(df_plot) == 0:
        print("⚠️ No data with SciPy times for speedup plot")
        return
    
    sizes = df_plot['size'].values
    speedups = df_plot['speedup_vs_scipy'].values
    gaps = df_plot['primal_gap'].values
    
    # Scatter plot colored by primal gap
    scatter = ax.scatter(sizes, speedups, c=gaps, cmap='RdYlGn_r', 
                        s=150, alpha=0.7, edgecolors='black', linewidth=1)
    
    # Add trend line
    z = np.polyfit(sizes, speedups, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(sizes.min(), sizes.max(), 100)
    ax.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2.5, label='Trend')
    
    # Baseline at 1.0x
    ax.axhline(1.0, color='gray', linestyle='-', linewidth=1.5, alpha=0.5, 
              label='Baseline (1.0×)')
    
    ax.set_xlabel('Problem Size (n)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Speedup vs SciPy', fontsize=14, fontweight='bold')
    ax.set_title('Speedup vs Problem Size (colored by solution quality)', 
                fontsize=16, fontweight='bold', pad=15)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(labelsize=11)
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Primal Gap (%)', fontsize=12, fontweight='bold')
    cbar.ax.tick_params(labelsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Speedup vs size plot saved: {output_path}")


def create_speedup_distribution_by_size(df: pd.DataFrame, output_path: Path):
    """Create speedup distribution grouped by problem size."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Group by size
    sizes = sorted(df['size'].unique())
    
    # === Subplot 1: Speedup vs SciPy by size ===
    df_scipy = df[df['scipy_time'].notna()]
    
    if len(df_scipy) > 0:
        size_labels = []
        speedup_data_scipy = []
        
        for size in sizes:
            size_data = df_scipy[df_scipy['size'] == size]
            if len(size_data) > 0:
                size_labels.append(f'{size}')
                speedup_data_scipy.append(size_data['speedup_vs_scipy'].values)
        
        bp1 = ax1.boxplot(speedup_data_scipy, labels=size_labels, patch_artist=True,
                         showmeans=True, meanline=True)
        
        # Color boxes
        for patch in bp1['boxes']:
            patch.set_facecolor('#ff9999')
            patch.set_alpha(0.7)
        
        ax1.axhline(1.0, color='red', linestyle='--', linewidth=2, alpha=0.6, 
                   label='Baseline (1.0×)')
        ax1.set_xlabel('Problem Size (n)', fontsize=13, fontweight='bold')
        ax1.set_ylabel('Speedup vs SciPy', fontsize=13, fontweight='bold')
        ax1.set_title('Speedup Distribution vs SciPy', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(axis='y', alpha=0.3)
        ax1.tick_params(labelsize=10)
    
    # === Subplot 2: Speedup vs LAP by size ===
    size_labels = []
    speedup_data_lap = []
    
    for size in sizes:
        size_data = df[df['size'] == size]
        if len(size_data) > 0 and 'speedup_vs_lap' in df.columns:
            speedups = size_data['speedup_vs_lap'].dropna()
            if len(speedups) > 0:
                size_labels.append(f'{size}')
                speedup_data_lap.append(speedups.values)
    
    if speedup_data_lap:
        bp2 = ax2.boxplot(speedup_data_lap, labels=size_labels, patch_artist=True,
                         showmeans=True, meanline=True)
        
        # Color boxes
        for patch in bp2['boxes']:
            patch.set_facecolor('#4ecdc4')
            patch.set_alpha(0.7)
        
        ax2.axhline(1.0, color='red', linestyle='--', linewidth=2, alpha=0.6, 
                   label='Baseline (1.0×)')
        ax2.set_xlabel('Problem Size (n)', fontsize=13, fontweight='bold')
        ax2.set_ylabel('Speedup vs LAP', fontsize=13, fontweight='bold')
        ax2.set_title('Speedup Distribution vs LAP', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(axis='y', alpha=0.3)
        ax2.tick_params(labelsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Speedup distribution by size saved: {output_path}")


def main():
    # Define paths
    results_dir = project_root / "results"
    comp_dir = results_dir / "comprehensive_benchmark"
    large_dir = results_dir / "large_scale_benchmark"
    
    # Find most recent CSV files
    comp_csvs = sorted(comp_dir.glob("comprehensive_benchmark_*.csv"))
    large_csvs = sorted(large_dir.glob("large_scale_benchmark_*.csv"))
    
    if not comp_csvs:
        print("❌ No comprehensive benchmark CSV files found!")
        return
    
    comp_csv = comp_csvs[-1]
    large_csv = large_csvs[-1] if large_csvs else None
    
    print(f"\n{'='*80}")
    print("Creating STRING-style Visualizations")
    print(f"{'='*80}")
    print(f"Comprehensive data: {comp_csv.name}")
    if large_csv:
        print(f"Large-scale data:   {large_csv.name}")
    print(f"{'='*80}\n")
    
    # Load data
    df = load_benchmark_data(comp_csv, large_csv)
    
    # Create output directory
    viz_dir = results_dir / "custom_visualizations"
    viz_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # 1. STRING-style 4-panel (like first image)
    viz1_path = viz_dir / f"string_style_benchmark_{timestamp}.png"
    create_string_style_visualization(df, viz1_path, 
                                     title_suffix="Sparse & Uniform (512-16384)")
    
    # 2. Speedup vs Size scatter (like second image)
    viz2_path = viz_dir / f"speedup_vs_size_{timestamp}.png"
    create_speedup_vs_size_plot(df, viz2_path)
    
    # 3. Speedup distribution by size (box plots)
    viz3_path = viz_dir / f"speedup_distribution_by_size_{timestamp}.png"
    create_speedup_distribution_by_size(df, viz3_path)
    
    print(f"\n{'='*80}")
    print("✅ All visualizations created successfully!")
    print(f"{'='*80}")
    print(f"\nOutput directory: {viz_dir}")
    print(f"\nGenerated files:")
    print(f"  1. {viz1_path.name}")
    print(f"  2. {viz2_path.name}")
    print(f"  3. {viz3_path.name}")
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
