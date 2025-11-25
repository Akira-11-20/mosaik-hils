#!/usr/bin/env python3
"""
Plot position traces comparing baseline RT with different tau and noise conditions.
"""

import os
import re
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import sys

# Add parent directory to path for plot_config
sys.path.insert(0, str(Path(__file__).parent.parent))
from plot_config import save_figure_both_sizes

# Set style
plt.rcParams["font.size"] = 11
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3


def parse_directory_name(dirname):
    """Parse directory name to extract tau and noise parameters."""
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
    """Load position and time data from HDF5 file."""
    with h5py.File(h5_path, "r") as f:
        position = f["EnvSim-0_Spacecraft1DOF_0"]["position"][:]
        time = f["time"]["time_s"][:]
    return time, position


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
    print(f"Baseline: time shape={baseline_time.shape}, position shape={baseline_position.shape}")

    # Collect all simulation results
    all_dirs = sorted([d for d in os.listdir(base_dir) if os.path.isdir(base_dir / d) and "baseline_rt" not in d])

    # Select representative conditions for plotting
    # We'll plot: one run per (tau, noise) combination
    conditions_to_plot = [
        (50.0, 0.0),
        (50.0, 15.0),
        (100.0, 0.0),
        (100.0, 15.0),
        (150.0, 0.0),
        (150.0, 15.0),
        (200.0, 0.0),
        (200.0, 15.0),
    ]

    # Find first run for each condition
    condition_dirs = {}
    for dirname in all_dirs:
        tau, noise = parse_directory_name(dirname)
        if tau is not None and (tau, noise) in conditions_to_plot:
            if (tau, noise) not in condition_dirs:
                condition_dirs[(tau, noise)] = dirname

    print(f"\nFound {len(condition_dirs)} conditions to plot")

    # Create figure with subplots (2 rows x 4 columns)
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for idx, (tau, noise) in enumerate(conditions_to_plot):
        ax = axes[idx]

        if (tau, noise) in condition_dirs:
            dirname = condition_dirs[(tau, noise)]
            h5_path = base_dir / dirname / "hils_data.h5"

            try:
                time, position = load_position_data(h5_path)

                # Plot baseline
                ax.plot(baseline_time, baseline_position, "k-", linewidth=2, label="Baseline RT", alpha=0.7)

                # Plot test condition
                ax.plot(time, position, "r-", linewidth=1.5, label=f"τ={tau:.0f}ms, noise={noise:.0f}ms", alpha=0.8)

                # Calculate RMSE
                min_len = min(len(baseline_position), len(position))
                rmse = np.sqrt(np.mean((baseline_position[:min_len] - position[:min_len]) ** 2))

                ax.set_title(f"τ={tau:.0f}ms, Noise={noise:.0f}ms (RMSE={rmse:.4f}m)", fontweight="bold", fontsize=12)
                ax.set_xlabel("Time [s]", fontsize=10)
                ax.set_ylabel("Position [m]", fontsize=10)
                ax.legend(loc="best", fontsize=9)
                ax.grid(True, alpha=0.3)

                # Set y-axis limits for consistency
                ax.set_ylim([-0.5, 6.0])

            except Exception as e:
                print(f"Error plotting {dirname}: {e}")
                ax.text(
                    0.5,
                    0.5,
                    f"Error loading\nτ={tau}ms, noise={noise}ms",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
        else:
            ax.text(0.5, 0.5, f"No data\nτ={tau}ms, noise={noise}ms", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"τ={tau:.0f}ms, Noise={noise:.0f}ms (No Data)", fontweight="bold")

    plt.suptitle(
        "Position Trajectories: Baseline RT vs Different Plant Dynamics", fontsize=16, fontweight="bold", y=0.995
    )
    plt.tight_layout()

    # Save figure
    output_path = base_dir / "position_traces_comparison.png"
    save_figure_both_sizes(plt, Path("."), base_name="output_path")
    print(f"\nPosition traces saved to: {output_path}")

    # Create a zoomed-in version for better detail (first 0.5 seconds)
    fig2, axes2 = plt.subplots(2, 4, figsize=(20, 10))
    axes2 = axes2.flatten()

    for idx, (tau, noise) in enumerate(conditions_to_plot):
        ax = axes2[idx]

        if (tau, noise) in condition_dirs:
            dirname = condition_dirs[(tau, noise)]
            h5_path = base_dir / dirname / "hils_data.h5"

            try:
                time, position = load_position_data(h5_path)

                # Find indices for zoom window (0 to 0.5s)
                zoom_mask = time <= 0.5
                zoom_time = time[zoom_mask]
                zoom_position = position[zoom_mask]
                zoom_baseline_time = baseline_time[baseline_time <= 0.5]
                zoom_baseline_position = baseline_position[baseline_time <= 0.5]

                # Plot baseline
                ax.plot(zoom_baseline_time, zoom_baseline_position, "k-", linewidth=2, label="Baseline RT", alpha=0.7)

                # Plot test condition
                ax.plot(
                    zoom_time,
                    zoom_position,
                    "r-",
                    linewidth=1.5,
                    label=f"τ={tau:.0f}ms, noise={noise:.0f}ms",
                    alpha=0.8,
                )

                # Calculate RMSE
                min_len = min(len(baseline_position), len(position))
                rmse = np.sqrt(np.mean((baseline_position[:min_len] - position[:min_len]) ** 2))

                ax.set_title(f"τ={tau:.0f}ms, Noise={noise:.0f}ms (RMSE={rmse:.4f}m)", fontweight="bold", fontsize=12)
                ax.set_xlabel("Time [s]", fontsize=10)
                ax.set_ylabel("Position [m]", fontsize=10)
                ax.legend(loc="best", fontsize=9)
                ax.grid(True, alpha=0.3)
                ax.set_xlim([0, 0.5])

            except Exception as e:
                print(f"Error plotting {dirname}: {e}")

    plt.suptitle(
        "Position Trajectories (Zoomed: 0-0.5s): Baseline RT vs Different Plant Dynamics",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()

    output_path2 = base_dir / "position_traces_comparison_zoom.png"
    save_figure_both_sizes(plt, Path("."), base_name="output_path2")
    print(f"Zoomed position traces saved to: {output_path2}")

    plt.show()


if __name__ == "__main__":
    main()
