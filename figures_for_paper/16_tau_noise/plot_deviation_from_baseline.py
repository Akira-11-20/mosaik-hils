#!/usr/bin/env python3
"""
Plot position deviation from baseline RT for different tau and noise conditions.
"""

import os
import re
from collections import defaultdict
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
    if "baseline_rt" in dirname:
        return None, None

    tau_match = re.search(r"tau(\d+)ms", dirname)
    tau = float(tau_match.group(1)) if tau_match else None

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

    # Collect all runs grouped by condition
    condition_runs = defaultdict(list)  # (tau, noise) -> [(dirname, time, position), ...]

    all_dirs = sorted([d for d in os.listdir(base_dir) if os.path.isdir(base_dir / d) and "baseline_rt" not in d])

    print(f"\nProcessing {len(all_dirs)} directories...")

    for dirname in all_dirs:
        h5_path = base_dir / dirname / "hils_data.h5"
        if not h5_path.exists():
            continue

        tau, noise = parse_directory_name(dirname)
        if tau is None:
            continue

        try:
            time, position = load_position_data(h5_path)
            condition_runs[(tau, noise)].append((dirname, time, position))
        except Exception as e:
            print(f"Error loading {dirname}: {e}")

    # Select conditions to plot
    conditions_to_plot = [
        (50.0, 0.0),
        (50.0, 5.0),
        (50.0, 10.0),
        (50.0, 15.0),
        (100.0, 0.0),
        (100.0, 5.0),
        (100.0, 10.0),
        (100.0, 15.0),
        (150.0, 0.0),
        (150.0, 5.0),
        (150.0, 10.0),
        (150.0, 15.0),
        (200.0, 0.0),
        (200.0, 5.0),
        (200.0, 10.0),
        (200.0, 15.0),
    ]

    # Create figure with subplots (4 rows x 4 columns)
    fig, axes = plt.subplots(4, 4, figsize=(20, 20))
    axes = axes.flatten()

    for idx, (tau, noise) in enumerate(conditions_to_plot):
        ax = axes[idx]

        if (tau, noise) not in condition_runs:
            ax.text(0.5, 0.5, f"No data\nτ={tau}ms, noise={noise}ms", ha="center", va="center", transform=ax.transAxes)
            continue

        runs = condition_runs[(tau, noise)]
        print(f"τ={tau}ms, noise={noise}ms: {len(runs)} runs")

        # Plot all runs as deviations from baseline
        colors = plt.cm.tab10(np.linspace(0, 1, len(runs)))
        all_deviations = []

        for i, (dirname, time, position) in enumerate(runs):
            # Calculate deviation from baseline
            min_len = min(len(baseline_position), len(position))
            deviation = position[:min_len] - baseline_position[:min_len]
            time_aligned = time[:min_len]

            ax.plot(time_aligned, deviation * 1000, color=colors[i], linewidth=0.8, alpha=0.6)
            all_deviations.append(deviation)

        # Calculate mean and std of deviations
        all_deviations = np.array(all_deviations)
        mean_deviation = np.mean(all_deviations, axis=0)
        std_deviation = np.std(all_deviations, axis=0)

        # Plot mean deviation
        ax.plot(time_aligned, mean_deviation * 1000, "k-", linewidth=2.5, label=f"Mean (n={len(runs)})", zorder=10)

        # Plot ±1 std band
        ax.fill_between(
            time_aligned,
            (mean_deviation - std_deviation) * 1000,
            (mean_deviation + std_deviation) * 1000,
            color="gray",
            alpha=0.3,
            label="±1 σ",
        )

        # Plot zero line
        ax.axhline(y=0, color="r", linestyle="--", linewidth=1.5, alpha=0.7, label="Baseline")

        # Calculate RMSE
        rmse = np.sqrt(np.mean(mean_deviation**2))
        max_deviation = np.max(np.abs(all_deviations))

        ax.set_xlabel("Time [s]", fontsize=10)
        ax.set_ylabel("Position Deviation [mm]", fontsize=10)
        ax.set_title(
            f"τ={tau:.0f}ms, Noise={noise:.0f}ms\n"
            + f"RMSE={rmse * 1000:.2f}mm, Max|dev|={max_deviation * 1000:.2f}mm",
            fontweight="bold",
            fontsize=10,
        )
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)

        # Set reasonable y-axis limits
        if noise == 0.0:
            ax.set_ylim([-0.5, 0.5])
        else:
            y_max = max(20, max_deviation * 1000 * 1.2)
            ax.set_ylim([-y_max, y_max])

    plt.suptitle(
        "Position Deviation from Baseline RT (mm)\nAll Monte Carlo Runs", fontsize=16, fontweight="bold", y=0.995
    )
    plt.tight_layout()

    output_path = base_dir / "deviation_from_baseline_all_runs.png"
    save_figure_both_sizes(plt, Path("."), base_name="output_path")
    print(f"\nDeviation plot saved to: {output_path}")

    # Create zoomed version (0-0.5s)
    fig2, axes2 = plt.subplots(4, 4, figsize=(20, 20))
    axes2 = axes2.flatten()

    for idx, (tau, noise) in enumerate(conditions_to_plot):
        ax = axes2[idx]

        if (tau, noise) not in condition_runs:
            continue

        runs = condition_runs[(tau, noise)]
        colors = plt.cm.tab10(np.linspace(0, 1, len(runs)))
        all_deviations = []

        for i, (dirname, time, position) in enumerate(runs):
            # Calculate deviation from baseline
            min_len = min(len(baseline_position), len(position))
            deviation = position[:min_len] - baseline_position[:min_len]
            time_aligned = time[:min_len]

            # Zoom to first 0.5s
            mask = time_aligned <= 0.5
            ax.plot(time_aligned[mask], deviation[mask] * 1000, color=colors[i], linewidth=0.8, alpha=0.6)
            all_deviations.append(deviation)

        # Calculate mean and std of deviations
        all_deviations = np.array(all_deviations)
        mean_deviation = np.mean(all_deviations, axis=0)
        std_deviation = np.std(all_deviations, axis=0)

        mask = time_aligned <= 0.5
        ax.plot(
            time_aligned[mask],
            mean_deviation[mask] * 1000,
            "k-",
            linewidth=2.5,
            label=f"Mean (n={len(runs)})",
            zorder=10,
        )

        ax.fill_between(
            time_aligned[mask],
            (mean_deviation[mask] - std_deviation[mask]) * 1000,
            (mean_deviation[mask] + std_deviation[mask]) * 1000,
            color="gray",
            alpha=0.3,
            label="±1 σ",
        )

        ax.axhline(y=0, color="r", linestyle="--", linewidth=1.5, alpha=0.7, label="Baseline")

        rmse = np.sqrt(np.mean(mean_deviation**2))
        max_deviation = np.max(np.abs(all_deviations))

        ax.set_xlabel("Time [s]", fontsize=10)
        ax.set_ylabel("Position Deviation [mm]", fontsize=10)
        ax.set_title(
            f"τ={tau:.0f}ms, Noise={noise:.0f}ms\n"
            + f"RMSE={rmse * 1000:.2f}mm, Max|dev|={max_deviation * 1000:.2f}mm",
            fontweight="bold",
            fontsize=10,
        )
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 0.5])

        if noise == 0.0:
            ax.set_ylim([-0.5, 0.5])
        else:
            y_max = max(20, max_deviation * 1000 * 1.2)
            ax.set_ylim([-y_max, y_max])

    plt.suptitle(
        "Position Deviation from Baseline RT (mm) - Zoomed 0-0.5s\nAll Monte Carlo Runs",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()

    output_path2 = base_dir / "deviation_from_baseline_all_runs_zoom.png"
    save_figure_both_sizes(plt, Path("."), base_name="output_path2")
    print(f"Zoomed deviation plot saved to: {output_path2}")

    plt.show()


if __name__ == "__main__":
    main()
