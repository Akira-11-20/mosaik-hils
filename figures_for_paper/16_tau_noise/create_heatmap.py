#!/usr/bin/env python3
"""
Create heatmap of position RMSE from baseline RT vs plant tau and noise level.
Averages across multiple Monte Carlo runs with the same conditions.
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
plt.rcParams["figure.figsize"] = (10, 8)
plt.rcParams["axes.grid"] = False


def parse_directory_name(dirname):
    """
    Parse directory name to extract tau and noise parameters.

    Examples:
    - 20251121-124055_baseline_rt -> tau=None, noise=None (baseline)
    - 20251121-124128_delay0ms_post_comp_tau50ms_linear -> tau=50.0, noise=0.0
    - 20251121-124205_delay0ms_post_comp_tau50ms_linear_noise5ms -> tau=50.0, noise=5.0
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


def main():
    base_dir = Path("/home/akira/mosaik-hils/figures_for_paper/16_tau_noise")

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

    # Collect all simulation results grouped by (tau, noise)
    results = defaultdict(list)  # (tau, noise) -> [rmse1, rmse2, ...]

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

            # Calculate RMSE
            rmse = calculate_rmse(baseline_position, position)

            # Store result
            results[(tau, noise)].append(rmse)

        except Exception as e:
            print(f"Error processing {dirname}: {e}")
            continue

    # Prepare data for heatmap
    # Get unique tau and noise values
    all_taus = sorted(set(tau for tau, _ in results.keys()))
    all_noises = sorted(set(noise for _, noise in results.keys()))

    print(f"\nFound {len(all_taus)} unique tau values: {all_taus}")
    print(f"Found {len(all_noises)} unique noise values: {all_noises}")
    print("\nNumber of Monte Carlo runs per condition:")
    for (tau, noise), rmse_list in sorted(results.items()):
        print(f"  tau={tau}ms, noise={noise}ms: {len(rmse_list)} runs")

    # Create heatmap data matrix (averaged across Monte Carlo runs)
    heatmap_data = np.zeros((len(all_noises), len(all_taus)))
    heatmap_std = np.zeros((len(all_noises), len(all_taus)))  # Standard deviation

    for i, noise in enumerate(all_noises):
        for j, tau in enumerate(all_taus):
            if (tau, noise) in results:
                rmse_list = results[(tau, noise)]
                heatmap_data[i, j] = np.mean(rmse_list)
                heatmap_std[i, j] = np.std(rmse_list) if len(rmse_list) > 1 else 0.0
            else:
                heatmap_data[i, j] = np.nan
                heatmap_std[i, j] = np.nan

    # Create figure with two subplots (mean and std)
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Plot 1: Mean RMSE
    ax1 = axes[0]
    im1 = ax1.imshow(heatmap_data, cmap="YlOrRd", aspect="auto", origin="lower")

    # Set ticks and labels
    ax1.set_xticks(range(len(all_taus)))
    ax1.set_yticks(range(len(all_noises)))
    ax1.set_xticklabels([f"{int(tau)}" for tau in all_taus])
    ax1.set_yticklabels([f"{int(noise)}" for noise in all_noises])

    ax1.set_xlabel("Plant Time Constant τ [ms]", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Plant Noise Std Dev [ms]", fontsize=14, fontweight="bold")
    ax1.set_title(
        "Mean Position RMSE from Baseline RT\n(Averaged across Monte Carlo runs)", fontsize=14, fontweight="bold"
    )

    # Add colorbar
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label("RMSE [m]", fontsize=12, fontweight="bold")

    # Add text annotations
    for i in range(len(all_noises)):
        for j in range(len(all_taus)):
            if not np.isnan(heatmap_data[i, j]):
                text = ax1.text(
                    j,
                    i,
                    f"{heatmap_data[i, j]:.5f}",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=11,
                    fontweight="bold",
                )

    # Plot 2: Standard deviation
    ax2 = axes[1]
    im2 = ax2.imshow(heatmap_std, cmap="Purples", aspect="auto", origin="lower")

    ax2.set_xticks(range(len(all_taus)))
    ax2.set_yticks(range(len(all_noises)))
    ax2.set_xticklabels([f"{int(tau)}" for tau in all_taus])
    ax2.set_yticklabels([f"{int(noise)}" for noise in all_noises])

    ax2.set_xlabel("Plant Time Constant τ [ms]", fontsize=14, fontweight="bold")
    ax2.set_ylabel("Plant Noise Std Dev [ms]", fontsize=14, fontweight="bold")
    ax2.set_title("Standard Deviation of RMSE\n(Across Monte Carlo runs)", fontsize=14, fontweight="bold")

    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label("Std Dev [m]", fontsize=12, fontweight="bold")

    # Add text annotations
    for i in range(len(all_noises)):
        for j in range(len(all_taus)):
            if not np.isnan(heatmap_std[i, j]):
                text = ax2.text(
                    j,
                    i,
                    f"{heatmap_std[i, j]:.6f}",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=11,
                    fontweight="bold",
                )

    plt.tight_layout()

    # Save figure
    output_path = base_dir / "tau_noise_heatmap.png"
    save_figure_both_sizes(plt, Path("."), base_name="output_path")
    print(f"\nHeatmap saved to: {output_path}")

    # Create a single combined heatmap showing mean ± std
    fig2, ax = plt.subplots(figsize=(12, 8))

    im = ax.imshow(heatmap_data, cmap="YlOrRd", aspect="auto", origin="lower")

    ax.set_xticks(range(len(all_taus)))
    ax.set_yticks(range(len(all_noises)))
    ax.set_xticklabels([f"{int(tau)}" for tau in all_taus])
    ax.set_yticklabels([f"{int(noise)}" for noise in all_noises])

    ax.set_xlabel("Plant Time Constant τ [ms]", fontsize=14, fontweight="bold")
    ax.set_ylabel("Plant Noise Std Dev [ms]", fontsize=14, fontweight="bold")
    ax.set_title(
        "Position RMSE from Baseline RT\n(Mean ± Std Dev across Monte Carlo runs)", fontsize=14, fontweight="bold"
    )

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Mean RMSE [m]", fontsize=12, fontweight="bold")

    # Add text annotations with mean ± std
    for i in range(len(all_noises)):
        for j in range(len(all_taus)):
            if not np.isnan(heatmap_data[i, j]):
                mean_val = heatmap_data[i, j]
                std_val = heatmap_std[i, j]
                text = ax.text(
                    j,
                    i,
                    f"{mean_val:.5f}\n±{std_val:.6f}",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=10,
                    fontweight="bold",
                )

    plt.tight_layout()

    output_path2 = base_dir / "tau_noise_heatmap_combined.png"
    save_figure_both_sizes(plt, Path("."), base_name="output_path2")
    print(f"Combined heatmap saved to: {output_path2}")

    # Print summary statistics
    print("\n=== Summary Statistics ===")
    print(f"Total conditions: {len(results)}")
    print(f"Tau range: {min(all_taus)}-{max(all_taus)} ms")
    print(f"Noise range: {min(all_noises)}-{max(all_noises)} ms")
    print(f"RMSE range: {np.nanmin(heatmap_data):.6f} - {np.nanmax(heatmap_data):.6f} m")
    print(f"Average RMSE: {np.nanmean(heatmap_data):.6f} m")
    print(f"Average std dev: {np.nanmean(heatmap_std):.6f} m")

    # Find worst case
    worst_idx = np.unravel_index(np.nanargmax(heatmap_data), heatmap_data.shape)
    worst_noise = all_noises[worst_idx[0]]
    worst_tau = all_taus[worst_idx[1]]
    worst_rmse = heatmap_data[worst_idx]
    print(f"\nWorst case: tau={worst_tau}ms, noise={worst_noise}ms, RMSE={worst_rmse:.6f}m")

    # Find best case
    best_idx = np.unravel_index(np.nanargmin(heatmap_data), heatmap_data.shape)
    best_noise = all_noises[best_idx[0]]
    best_tau = all_taus[best_idx[1]]
    best_rmse = heatmap_data[best_idx]
    print(f"Best case: tau={best_tau}ms, noise={best_noise}ms, RMSE={best_rmse:.6f}m")

    plt.show()


if __name__ == "__main__":
    main()
