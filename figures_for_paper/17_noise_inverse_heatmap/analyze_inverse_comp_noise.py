#!/usr/bin/env python3
"""
Analyze Monte Carlo simulation results for inverse compensation with plant noise.
Calculates RMSE and other performance metrics compared to baseline RT.

This dataset has:
- Fixed tau = 100ms
- Variable noise levels (0, 5, 10, 15 ms)
- Inverse compensation enabled (post-plant position)
- Multiple Monte Carlo runs per condition
"""

import os
import re
from collections import defaultdict
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sys

# Add parent directory to path for plot_config
sys.path.insert(0, str(Path(__file__).parent.parent))
from plot_config import save_figure_both_sizes

# Set style
sns.set_style("white")
plt.rcParams["font.size"] = 12
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["axes.grid"] = True


def parse_directory_name(dirname):
    """
    Parse directory name to extract tau and noise parameters.

    Examples:
    - 20251121-163619_baseline_rt -> tau=None, noise=None (baseline)
    - 20251121-163655_delay0ms_post_comp_tau100ms_linear -> tau=100.0, noise=0.0
    - 20251121-163731_delay0ms_post_comp_tau100ms_linear_noise5ms -> tau=100.0, noise=5.0
    """
    # Baseline case
    if "baseline_rt" in dirname:
        return None, None

    # Extract tau (time constant in ms)
    tau_match = re.search(r"tau(\d+)ms", dirname)
    tau = float(tau_match.group(1)) if tau_match else None

    # Extract noise (in ms)
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
    # Ensure same length
    min_len = min(len(baseline_pos), len(test_pos))
    baseline_pos = baseline_pos[:min_len]
    test_pos = test_pos[:min_len]

    # Calculate RMSE
    rmse = np.sqrt(np.mean((baseline_pos - test_pos) ** 2))
    return rmse


def calculate_settling_time(time, position, target=5.0, threshold=0.05):
    """
    Calculate settling time (time to reach and stay within threshold of target).

    Args:
        time: Time array
        position: Position array
        target: Target position
        threshold: Percentage threshold (0.05 = 5%)

    Returns:
        Settling time in seconds, or None if never settles
    """
    tolerance = target * threshold

    # Find where position is within tolerance
    within_tolerance = np.abs(position - target) <= tolerance

    # Find the last time it leaves the tolerance band
    if not np.any(within_tolerance):
        return None

    # Find the last time it exits the band
    last_exit_idx = 0
    for i in range(len(within_tolerance) - 1):
        if within_tolerance[i] and not within_tolerance[i + 1]:
            last_exit_idx = i + 1

    # If it settles, return the time when it last enters and stays
    for i in range(last_exit_idx, len(within_tolerance)):
        if np.all(within_tolerance[i:]):
            return time[i]

    return None


def calculate_max_overshoot(position, target=5.0):
    """Calculate maximum overshoot as percentage of target."""
    max_pos = np.max(position)
    overshoot = ((max_pos - target) / target) * 100
    return max(0, overshoot)  # Return 0 if no overshoot


def main():
    base_dir = Path("/home/akira/mosaik-hils/figures_for_paper/17_noise_inverse_heatmap")

    # Find baseline directory
    baseline_dirs = [d for d in os.listdir(base_dir) if "baseline_rt" in d]
    if not baseline_dirs:
        print("Error: No baseline_rt directory found!")
        return

    baseline_dir = baseline_dirs[0]
    print(f"Using baseline: {baseline_dir}")

    # Load baseline data
    baseline_h5 = base_dir / baseline_dir / "hils_data.h5"
    baseline_time, baseline_position = load_position_data(baseline_h5)
    print(f"Baseline position shape: {baseline_position.shape}")

    # Collect all simulation results grouped by noise level
    results = defaultdict(list)  # noise -> [dict with metrics]

    # Process all directories
    all_dirs = sorted([d for d in os.listdir(base_dir) if os.path.isdir(base_dir / d) and "baseline_rt" not in d])

    print(f"\nProcessing {len(all_dirs)} simulation directories...")

    for dirname in all_dirs:
        h5_path = base_dir / dirname / "hils_data.h5"
        if not h5_path.exists():
            print(f"Warning: {dirname} has no hils_data.h5, skipping")
            continue

        # Parse parameters
        tau, noise = parse_directory_name(dirname)

        if tau is None:
            print(f"Warning: Could not parse parameters from {dirname}, skipping")
            continue

        # Load position data
        try:
            time, position = load_position_data(h5_path)

            # Calculate metrics
            rmse = calculate_rmse(baseline_position, position)
            settling_time = calculate_settling_time(time, position)
            overshoot = calculate_max_overshoot(position)

            # Store results
            metrics = {
                "rmse": rmse,
                "settling_time": settling_time,
                "overshoot": overshoot,
                "tau": tau,
                "noise": noise,
            }
            results[noise].append(metrics)

        except Exception as e:
            print(f"Error processing {dirname}: {e}")
            continue

    # Get unique noise values
    all_noises = sorted(results.keys())

    print(f"\nFound {len(all_noises)} unique noise values: {all_noises}")
    print("\nNumber of Monte Carlo runs per noise level:")
    for noise, metrics_list in sorted(results.items()):
        print(f"  noise={noise}ms: {len(metrics_list)} runs")

    # Calculate statistics for each noise level
    noise_stats = {}
    for noise in all_noises:
        metrics_list = results[noise]

        rmse_values = [m["rmse"] for m in metrics_list]
        settling_times = [m["settling_time"] for m in metrics_list if m["settling_time"] is not None]
        overshoot_values = [m["overshoot"] for m in metrics_list]

        noise_stats[noise] = {
            "rmse_mean": np.mean(rmse_values),
            "rmse_std": np.std(rmse_values),
            "settling_mean": np.mean(settling_times) if settling_times else None,
            "settling_std": np.std(settling_times) if settling_times else None,
            "overshoot_mean": np.mean(overshoot_values),
            "overshoot_std": np.std(overshoot_values),
            "n_runs": len(metrics_list),
        }

    # Print summary statistics
    print("\n=== Summary Statistics ===")
    print(f"Total conditions: {len(all_noises)}")
    print(f"Noise range: {min(all_noises)}-{max(all_noises)} ms")

    print("\n--- RMSE from Baseline ---")
    for noise in all_noises:
        stats = noise_stats[noise]
        print(f"Noise {noise:2.0f}ms: RMSE = {stats['rmse_mean']:.6f} ± {stats['rmse_std']:.6f} m  (n={stats['n_runs']})")

    print("\n--- Settling Time (5% threshold) ---")
    for noise in all_noises:
        stats = noise_stats[noise]
        if stats['settling_mean'] is not None:
            print(f"Noise {noise:2.0f}ms: T_s = {stats['settling_mean']:.4f} ± {stats['settling_std']:.4f} s")
        else:
            print(f"Noise {noise:2.0f}ms: Never settles")

    print("\n--- Maximum Overshoot ---")
    for noise in all_noises:
        stats = noise_stats[noise]
        print(f"Noise {noise:2.0f}ms: Overshoot = {stats['overshoot_mean']:.2f} ± {stats['overshoot_std']:.2f} %")

    # ========== Create Plots ==========

    # Plot 1: RMSE vs Noise Level
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1a: RMSE
    ax1 = axes[0]
    noises = list(all_noises)
    rmse_means = [noise_stats[n]["rmse_mean"] for n in noises]
    rmse_stds = [noise_stats[n]["rmse_std"] for n in noises]

    ax1.errorbar(noises, rmse_means, yerr=rmse_stds, marker='o', markersize=8,
                 linewidth=2, capsize=5, capthick=2)
    ax1.set_xlabel("Plant Noise Std Dev [ms]", fontsize=12, fontweight="bold")
    ax1.set_ylabel("RMSE from Baseline [m]", fontsize=12, fontweight="bold")
    ax1.set_title("Position RMSE vs Noise Level\n(with Inverse Compensation)",
                  fontsize=13, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    # Plot 1b: Settling Time
    ax2 = axes[1]
    settling_means = [noise_stats[n]["settling_mean"] for n in noises if noise_stats[n]["settling_mean"] is not None]
    settling_stds = [noise_stats[n]["settling_std"] for n in noises if noise_stats[n]["settling_mean"] is not None]
    settling_noises = [n for n in noises if noise_stats[n]["settling_mean"] is not None]

    if settling_means:
        ax2.errorbar(settling_noises, settling_means, yerr=settling_stds,
                     marker='s', markersize=8, linewidth=2, capsize=5, capthick=2, color='green')
        ax2.set_xlabel("Plant Noise Std Dev [ms]", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Settling Time [s]", fontsize=12, fontweight="bold")
        ax2.set_title("Settling Time vs Noise Level\n(5% threshold)",
                      fontsize=13, fontweight="bold")
        ax2.grid(True, alpha=0.3)

    # Plot 1c: Overshoot
    ax3 = axes[2]
    overshoot_means = [noise_stats[n]["overshoot_mean"] for n in noises]
    overshoot_stds = [noise_stats[n]["overshoot_std"] for n in noises]

    ax3.errorbar(noises, overshoot_means, yerr=overshoot_stds,
                 marker='^', markersize=8, linewidth=2, capsize=5, capthick=2, color='red')
    ax3.set_xlabel("Plant Noise Std Dev [ms]", fontsize=12, fontweight="bold")
    ax3.set_ylabel("Maximum Overshoot [%]", fontsize=12, fontweight="bold")
    ax3.set_title("Overshoot vs Noise Level", fontsize=13, fontweight="bold")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path1 = base_dir / "inverse_comp_noise_metrics.png"
    save_figure_both_sizes(plt, Path("."), base_name="output_path1")
    print(f"\nMetrics plot saved to: {output_path1}")

    # Plot 2: Box plots for Monte Carlo distribution
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Box plot 1: RMSE distribution
    ax1 = axes[0]
    rmse_data = [np.array([m["rmse"] for m in results[n]]) for n in noises]
    positions = np.arange(len(noises))
    bp1 = ax1.boxplot(rmse_data, positions=positions, widths=0.6, patch_artist=True)
    for patch in bp1['boxes']:
        patch.set_facecolor('lightblue')
    ax1.set_xticks(positions)
    ax1.set_xticklabels([f"{int(n)}" for n in noises])
    ax1.set_xlabel("Plant Noise Std Dev [ms]", fontsize=12, fontweight="bold")
    ax1.set_ylabel("RMSE [m]", fontsize=12, fontweight="bold")
    ax1.set_title("RMSE Distribution (Monte Carlo)", fontsize=13, fontweight="bold")
    ax1.grid(True, alpha=0.3, axis='y')

    # Box plot 2: Settling time distribution
    ax2 = axes[1]
    settling_data = []
    settling_labels = []
    for n in noises:
        st_values = [m["settling_time"] for m in results[n] if m["settling_time"] is not None]
        if st_values:
            settling_data.append(st_values)
            settling_labels.append(f"{int(n)}")

    if settling_data:
        positions = np.arange(len(settling_data))
        bp2 = ax2.boxplot(settling_data, positions=positions, widths=0.6, patch_artist=True)
        for patch in bp2['boxes']:
            patch.set_facecolor('lightgreen')
        ax2.set_xticks(positions)
        ax2.set_xticklabels(settling_labels)
        ax2.set_xlabel("Plant Noise Std Dev [ms]", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Settling Time [s]", fontsize=12, fontweight="bold")
        ax2.set_title("Settling Time Distribution", fontsize=13, fontweight="bold")
        ax2.grid(True, alpha=0.3, axis='y')

    # Box plot 3: Overshoot distribution
    ax3 = axes[2]
    overshoot_data = [np.array([m["overshoot"] for m in results[n]]) for n in noises]
    bp3 = ax3.boxplot(overshoot_data, positions=positions, widths=0.6, patch_artist=True)
    for patch in bp3['boxes']:
        patch.set_facecolor('lightcoral')
    ax3.set_xticks(positions)
    ax3.set_xticklabels([f"{int(n)}" for n in noises])
    ax3.set_xlabel("Plant Noise Std Dev [ms]", fontsize=12, fontweight="bold")
    ax3.set_ylabel("Overshoot [%]", fontsize=12, fontweight="bold")
    ax3.set_title("Overshoot Distribution", fontsize=13, fontweight="bold")
    ax3.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path2 = base_dir / "inverse_comp_noise_distributions.png"
    save_figure_both_sizes(plt, Path("."), base_name="output_path2")
    print(f"Distribution plot saved to: {output_path2}")

    # Plot 3: Create a simple bar chart for comparison
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(noises))
    width = 0.35

    # Normalize metrics for comparison (0-1 scale)
    rmse_norm = np.array(rmse_means) / max(rmse_means) if max(rmse_means) > 0 else np.zeros_like(rmse_means)

    rects1 = ax.bar(x, rmse_means, width, yerr=rmse_stds, label='RMSE [m]',
                    capsize=5, alpha=0.8, color='steelblue')

    ax.set_xlabel("Plant Noise Std Dev [ms]", fontsize=12, fontweight="bold")
    ax.set_ylabel("RMSE from Baseline [m]", fontsize=12, fontweight="bold")
    ax.set_title("Impact of Plant Noise on Control Performance\n(with Inverse Compensation, τ=100ms)",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(n)}" for n in noises])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for rect in rects1:
        height = rect.get_height()
        ax.annotate(f'{height:.5f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    output_path3 = base_dir / "inverse_comp_noise_bar_chart.png"
    save_figure_both_sizes(plt, Path("."), base_name="output_path3")
    print(f"Bar chart saved to: {output_path3}")

    plt.show()


if __name__ == "__main__":
    main()
