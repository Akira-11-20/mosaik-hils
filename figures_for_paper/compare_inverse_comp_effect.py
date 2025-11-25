#!/usr/bin/env python3
"""
Compare inverse compensation effect by analyzing both datasets:
- 16_tau_noise: Without inverse compensation (tau=50,100,150,200ms, noise=0,5,10,15ms)
- 17_noise_inverse_heatmap: With inverse compensation (tau=100ms, noise=0,5,10,15ms)

Shows the improvement provided by inverse compensation at tau=100ms.
"""

import os
import re
from collections import defaultdict
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from plot_config import save_figure_both_sizes

sns.set_style("whitegrid")
plt.rcParams["font.size"] = 11


def parse_directory_name(dirname):
    """Parse directory name to extract parameters."""
    if "baseline_rt" in dirname:
        return None, None

    tau_match = re.search(r"tau(\d+)ms", dirname)
    tau = float(tau_match.group(1)) if tau_match else None

    noise_match = re.search(r"noise(\d+)ms", dirname)
    noise = float(noise_match.group(1)) if noise_match else 0.0

    return tau, noise


def load_position_data(h5_path):
    """Load position data from HDF5 file."""
    with h5py.File(h5_path, "r") as f:
        position = f["EnvSim-0_Spacecraft1DOF_0"]["position"][:]
        time = f["time"]["time_s"][:]
    return time, position


def calculate_rmse(baseline_pos, test_pos):
    """Calculate RMSE between baseline and test position."""
    min_len = min(len(baseline_pos), len(test_pos))
    baseline_pos = baseline_pos[:min_len]
    test_pos = test_pos[:min_len]
    return np.sqrt(np.mean((baseline_pos - test_pos) ** 2))


def collect_data(base_dir, target_tau=None):
    """
    Collect RMSE data from a directory.
    If target_tau is specified, only collect data for that tau value.
    Returns: dict[noise] -> [rmse_values]
    """
    # Find baseline
    baseline_dirs = [d for d in os.listdir(base_dir) if "baseline_rt" in d]
    if not baseline_dirs:
        print(f"Warning: No baseline in {base_dir}")
        return None, None

    baseline_h5 = base_dir / baseline_dirs[0] / "hils_data.h5"
    baseline_time, baseline_position = load_position_data(baseline_h5)

    # Collect results
    results = defaultdict(list)
    all_dirs = sorted([d for d in os.listdir(base_dir) if os.path.isdir(base_dir / d) and "baseline_rt" not in d])

    for dirname in all_dirs:
        h5_path = base_dir / dirname / "hils_data.h5"
        if not h5_path.exists():
            continue

        tau, noise = parse_directory_name(dirname)
        if tau is None:
            continue

        # Filter by target_tau if specified
        if target_tau is not None and tau != target_tau:
            continue

        try:
            time, position = load_position_data(h5_path)
            rmse = calculate_rmse(baseline_position, position)
            results[noise].append(rmse)
        except Exception as e:
            print(f"Error processing {dirname}: {e}")
            continue

    return results, baseline_position


def main():
    # Directories
    dir_no_comp = Path("/home/akira/mosaik-hils/figures_for_paper/16_tau_noise")
    dir_with_comp = Path("/home/akira/mosaik-hils/figures_for_paper/17_noise_inverse_heatmap")

    print("Collecting data without inverse compensation (tau=100ms)...")
    results_no_comp, baseline_pos = collect_data(dir_no_comp, target_tau=100.0)

    print("Collecting data with inverse compensation (tau=100ms)...")
    results_with_comp, _ = collect_data(dir_with_comp, target_tau=None)  # All data is tau=100ms

    if results_no_comp is None or results_with_comp is None:
        print("Error: Could not load data!")
        return

    # Get common noise levels
    noise_levels = sorted(set(results_no_comp.keys()) & set(results_with_comp.keys()))
    print(f"\nComparing noise levels: {noise_levels}")

    # Calculate statistics
    stats_no_comp = {}
    stats_with_comp = {}

    for noise in noise_levels:
        stats_no_comp[noise] = {
            "mean": np.mean(results_no_comp[noise]),
            "std": np.std(results_no_comp[noise]),
            "n": len(results_no_comp[noise]),
        }
        stats_with_comp[noise] = {
            "mean": np.mean(results_with_comp[noise]),
            "std": np.std(results_with_comp[noise]),
            "n": len(results_with_comp[noise]),
        }

    # Print comparison
    print("\n" + "="*70)
    print("INVERSE COMPENSATION EFFECTIVENESS (τ = 100ms)")
    print("="*70)
    print(f"{'Noise':>8} | {'No Comp (mm)':>20} | {'With Comp (mm)':>20} | {'Improvement':>12}")
    print("-"*70)

    for noise in noise_levels:
        no_comp = stats_no_comp[noise]["mean"] * 1000
        with_comp = stats_with_comp[noise]["mean"] * 1000
        improvement = ((no_comp - with_comp) / no_comp) * 100

        print(f"{noise:>6.0f} ms | {no_comp:>8.3f} ± {stats_no_comp[noise]['std']*1000:>6.3f} | "
              f"{with_comp:>8.3f} ± {stats_with_comp[noise]['std']*1000:>6.3f} | "
              f"{improvement:>9.1f} %")

    print("="*70)

    # ========== Create Comparison Plots ==========

    # Plot 1: Side-by-side comparison
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: Bar chart
    ax1 = axes[0]
    x = np.arange(len(noise_levels))
    width = 0.35

    no_comp_means = [stats_no_comp[n]["mean"] * 1000 for n in noise_levels]
    no_comp_stds = [stats_no_comp[n]["std"] * 1000 for n in noise_levels]
    with_comp_means = [stats_with_comp[n]["mean"] * 1000 for n in noise_levels]
    with_comp_stds = [stats_with_comp[n]["std"] * 1000 for n in noise_levels]

    bars1 = ax1.bar(x - width/2, no_comp_means, width, yerr=no_comp_stds,
                    label='Without Compensation', color='coral', alpha=0.8, capsize=5)
    bars2 = ax1.bar(x + width/2, with_comp_means, width, yerr=with_comp_stds,
                    label='With Inverse Compensation', color='skyblue', alpha=0.8, capsize=5)

    ax1.set_xlabel('Plant Noise Std Dev [ms]', fontsize=12, fontweight='bold')
    ax1.set_ylabel('RMSE from Baseline [mm]', fontsize=12, fontweight='bold')
    ax1.set_title('Effect of Inverse Compensation on Position Tracking\n(τ = 100ms)',
                 fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{int(n)}' for n in noise_levels])
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}',
                    ha='center', va='bottom', fontsize=9)

    # Right: Improvement percentage
    ax2 = axes[1]
    improvements = [((stats_no_comp[n]["mean"] - stats_with_comp[n]["mean"]) /
                     stats_no_comp[n]["mean"] * 100) for n in noise_levels]

    bars3 = ax2.bar(x, improvements, color='green', alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax2.set_xlabel('Plant Noise Std Dev [ms]', fontsize=12, fontweight='bold')
    ax2.set_ylabel('RMSE Reduction [%]', fontsize=12, fontweight='bold')
    ax2.set_title('Inverse Compensation Improvement\n(Positive = Better)',
                 fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{int(n)}' for n in noise_levels])
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar in bars3:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom' if height > 0 else 'top', fontsize=10, fontweight='bold')

    plt.tight_layout()
    output_path1 = Path("/home/akira/mosaik-hils/figures_for_paper/inverse_comp_comparison.png")
    save_figure_both_sizes(plt, output_path1.parent, base_name=output_path1.stem)
    print(f"\nComparison plot saved to: {output_path1}")

    # Plot 2: Box plots comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    for idx, noise in enumerate(noise_levels):
        ax = axes[idx // 2, idx % 2]

        data_to_plot = [
            np.array(results_no_comp[noise]) * 1000,
            np.array(results_with_comp[noise]) * 1000
        ]

        bp = ax.boxplot(data_to_plot, labels=['Without\nCompensation', 'With Inverse\nCompensation'],
                       patch_artist=True, widths=0.6)

        # Color the boxes
        bp['boxes'][0].set_facecolor('coral')
        bp['boxes'][1].set_facecolor('skyblue')
        bp['boxes'][0].set_alpha(0.7)
        bp['boxes'][1].set_alpha(0.7)

        ax.set_ylabel('RMSE from Baseline [mm]', fontsize=11, fontweight='bold')
        ax.set_title(f'Noise = {int(noise)} ms  (n={stats_no_comp[noise]["n"]}, {stats_with_comp[noise]["n"]} runs)',
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # Add mean markers
        ax.plot(1, stats_no_comp[noise]["mean"] * 1000, 'r*', markersize=12, label='Mean')
        ax.plot(2, stats_with_comp[noise]["mean"] * 1000, 'r*', markersize=12)
        ax.legend(fontsize=9)

    plt.suptitle('Monte Carlo Distribution Comparison (τ = 100ms)', fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()

    output_path2 = Path("/home/akira/mosaik-hils/figures_for_paper/inverse_comp_distributions.png")
    save_figure_both_sizes(plt, output_path2.parent, base_name=output_path2.stem)
    print(f"Distribution comparison saved to: {output_path2}")

    # Plot 3: Trend lines
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.errorbar(noise_levels, no_comp_means, yerr=no_comp_stds,
               marker='o', markersize=10, linewidth=2.5, capsize=6, capthick=2,
               label='Without Compensation', color='coral')
    ax.errorbar(noise_levels, with_comp_means, yerr=with_comp_stds,
               marker='s', markersize=10, linewidth=2.5, capsize=6, capthick=2,
               label='With Inverse Compensation', color='skyblue')

    ax.set_xlabel('Plant Noise Std Dev [ms]', fontsize=13, fontweight='bold')
    ax.set_ylabel('RMSE from Baseline [mm]', fontsize=13, fontweight='bold')
    ax.set_title('Position Tracking Performance vs Plant Noise\n(τ = 100ms, PID Control)',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(True, alpha=0.3)

    # Add text annotations showing improvement
    for i, noise in enumerate(noise_levels):
        mid_y = (no_comp_means[i] + with_comp_means[i]) / 2
        improvement = improvements[i]
        ax.annotate(f'↓ {improvement:.0f}%',
                   xy=(noise, mid_y), xytext=(5, 0), textcoords='offset points',
                   fontsize=10, fontweight='bold', color='green',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='green', alpha=0.8))

    plt.tight_layout()
    output_path3 = Path("/home/akira/mosaik-hils/figures_for_paper/inverse_comp_trend.png")
    save_figure_both_sizes(plt, output_path3.parent, base_name=output_path3.stem)
    print(f"Trend plot saved to: {output_path3}")

    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70)

    plt.show()


if __name__ == "__main__":
    main()
