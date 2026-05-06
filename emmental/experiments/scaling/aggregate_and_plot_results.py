#!/usr/bin/env python3
"""
Aggregate results from scaling experiments and generate plots.
"""

import os
import json
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300


def load_experiment_results(results_dir):
    """Load all experiment results from JSON files"""
    results = []
    pattern = os.path.join(results_dir, "results_*genes.json")
    
    for filepath in glob.glob(pattern):
        try:
            with open(filepath, 'r') as f:
                result = json.load(f)
                results.append(result)
        except Exception as e:
            print(f"Warning: Could not load {filepath}: {e}")
    
    if not results:
        raise ValueError(f"No results found in {results_dir}. Make sure experiments have completed.")
    
    return pd.DataFrame(results).sort_values('num_genes')


def plot_scaling_results(df, output_dir):
    """Generate plots for num_genes vs memory and num_genes vs time"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Memory vs Number of Genes (linear scale)
    ax1 = axes[0, 0]
    ax1.scatter(df['num_genes'], df['peak_memory'], alpha=0.7, s=100, color='steelblue')
    ax1.plot(df['num_genes'], df['peak_memory'], '--', alpha=0.5, color='steelblue')
    ax1.set_xlabel('Number of Genes', fontsize=12)
    ax1.set_ylabel('Peak Memory (GB)', fontsize=12)
    ax1.set_title('Peak Memory Usage vs Number of Genes', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Add trend line
    if len(df) > 1:
        z = np.polyfit(df['num_genes'], df['peak_memory'], 1)
        p = np.poly1d(z)
        ax1.plot(df['num_genes'], p(df['num_genes']), "r--", alpha=0.8, label=f'Trend: y={z[0]:.4f}x+{z[1]:.2f}')
        ax1.legend()
    
    # Plot 2: Time vs Number of Genes (linear scale)
    ax2 = axes[0, 1]
    ax2.scatter(df['num_genes'], df['total_time'] / 60, alpha=0.7, s=100, color='coral')
    ax2.plot(df['num_genes'], df['total_time'] / 60, '--', alpha=0.5, color='coral')
    ax2.set_xlabel('Number of Genes', fontsize=12)
    ax2.set_ylabel('Total Time (minutes)', fontsize=12)
    ax2.set_title('Total Training Time vs Number of Genes', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add trend line
    if len(df) > 1:
        z = np.polyfit(df['num_genes'], df['total_time'] / 60, 1)
        p = np.poly1d(z)
        ax2.plot(df['num_genes'], p(df['total_time'] / 60), "r--", alpha=0.8, label=f'Trend: y={z[0]:.4f}x+{z[1]:.2f}')
        ax2.legend()
    
    # Plot 3: Memory vs Number of Genes (log scale)
    ax3 = axes[1, 0]
    ax3.scatter(df['num_genes'], df['peak_memory'], alpha=0.7, s=100, color='steelblue')
    ax3.plot(df['num_genes'], df['peak_memory'], '--', alpha=0.5, color='steelblue')
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.set_xlabel('Number of Genes (log scale)', fontsize=12)
    ax3.set_ylabel('Peak Memory (GB, log scale)', fontsize=12)
    ax3.set_title('Peak Memory Usage vs Number of Genes (Log-Log)', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Time per Epoch vs Number of Genes
    ax4 = axes[1, 1]
    ax4.scatter(df['num_genes'], df['avg_time_per_epoch'], alpha=0.7, s=100, color='mediumseagreen')
    ax4.errorbar(df['num_genes'], df['avg_time_per_epoch'], 
                 yerr=df['std_time_per_epoch'], 
                 fmt='none', alpha=0.5, color='mediumseagreen')
    ax4.plot(df['num_genes'], df['avg_time_per_epoch'], '--', alpha=0.5, color='mediumseagreen')
    ax4.set_xlabel('Number of Genes', fontsize=12)
    ax4.set_ylabel('Time per Epoch (seconds)', fontsize=12)
    ax4.set_title('Average Time per Epoch vs Number of Genes', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_file = os.path.join(output_dir, 'scaling_plots.png')
    plt.savefig(output_file, bbox_inches='tight')
    print(f"Plots saved to: {output_file}")
    
    # Also create a separate detailed figure
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
    
    # Variants vs Genes
    ax = axes2[0]
    ax.scatter(df['num_genes'], df['num_variants'], alpha=0.7, s=100, color='purple')
    ax.plot(df['num_genes'], df['num_variants'], '--', alpha=0.5, color='purple')
    ax.set_xlabel('Number of Genes', fontsize=12)
    ax.set_ylabel('Total Number of Variants', fontsize=12)
    ax.set_title('Variants vs Genes', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Memory efficiency (memory per gene)
    ax = axes2[1]
    memory_per_gene = df['peak_memory'] / df['num_genes']
    ax.scatter(df['num_genes'], memory_per_gene, alpha=0.7, s=100, color='orange')
    ax.set_xlabel('Number of Genes', fontsize=12)
    ax.set_ylabel('Memory per Gene (GB)', fontsize=12)
    ax.set_title('Memory Efficiency', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Time efficiency (time per gene)
    ax = axes2[2]
    time_per_gene = (df['total_time'] / 60) / df['num_genes']
    ax.scatter(df['num_genes'], time_per_gene, alpha=0.7, s=100, color='teal')
    ax.set_xlabel('Number of Genes', fontsize=12)
    ax.set_ylabel('Time per Gene (minutes)', fontsize=12)
    ax.set_title('Time Efficiency', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file2 = os.path.join(output_dir, 'scaling_plots_detailed.png')
    plt.savefig(output_file2, bbox_inches='tight')
    print(f"Detailed plots saved to: {output_file2}")
    
    plt.close('all')


def print_summary_table(df):
    """Print a formatted summary table"""
    print("\n" + "=" * 100)
    print("SCALING EXPERIMENT SUMMARY")
    print("=" * 100)
    
    summary_cols = ['num_genes', 'num_variants', 'peak_memory', 'total_time', 'avg_time_per_epoch']
    summary_df = df[summary_cols].copy()
    summary_df['total_time'] = summary_df['total_time'] / 60  # Convert to minutes
    summary_df.columns = ['Genes', 'Variants', 'Peak Memory (GB)', 'Total Time (min)', 'Time/Epoch (s)']
    
    print(summary_df.to_string(index=False, float_format='%.2f'))
    print("=" * 100)
    
    # Print scaling relationships
    if len(df) > 1:
        print("\nSCALING RELATIONSHIPS:")
        print("-" * 100)
        
        # Memory scaling
        mem_slope = np.polyfit(df['num_genes'], df['peak_memory'], 1)[0]
        print(f"Memory scaling: ~{mem_slope:.4f} GB per gene")
        
        # Time scaling
        time_slope = np.polyfit(df['num_genes'], df['total_time'] / 60, 1)[0]
        print(f"Time scaling: ~{time_slope:.4f} minutes per gene")
        
        # Variants scaling
        var_slope = np.polyfit(df['num_genes'], df['num_variants'], 1)[0]
        print(f"Variants scaling: ~{var_slope:.1f} variants per gene")
        print("-" * 100)


def main():
    parser = argparse.ArgumentParser(description="Aggregate and plot scaling experiment results")
    parser.add_argument('--results_dir', type=str, default='experiments',
                       help='Directory containing experiment results')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Directory to save plots (default: same as results_dir)')
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = args.results_dir
    
    # Load results
    print(f"Loading results from {args.results_dir}...")
    df = load_experiment_results(args.results_dir)
    print(f"Loaded {len(df)} experiments")
    
    # Save aggregated results as CSV
    csv_file = os.path.join(args.output_dir, 'aggregated_results.csv')
    df.to_csv(csv_file, index=False)
    print(f"Aggregated results saved to: {csv_file}")
    
    # Print summary
    print_summary_table(df)
    
    # Generate plots
    print("\nGenerating plots...")
    plot_scaling_results(df, args.output_dir)
    
    print("\nDone!")


if __name__ == '__main__':
    main()

