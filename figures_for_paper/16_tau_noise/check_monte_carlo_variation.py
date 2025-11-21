#!/usr/bin/env python3
"""
Check if Monte Carlo runs show actual variation for the same conditions.
"""

import os
import re
from collections import defaultdict
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


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

    # Collect all runs grouped by condition
    condition_runs = defaultdict(list)  # (tau, noise) -> [(dirname, time, position), ...]

    all_dirs = sorted([d for d in os.listdir(base_dir) if os.path.isdir(base_dir / d) and "baseline_rt" not in d])

    print(f"Processing {len(all_dirs)} directories...")

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

    # Select a few conditions to visualize
    conditions_to_check = [
        (50.0, 0.0),  # No noise
        (50.0, 15.0),  # High noise
        (100.0, 0.0),  # No noise
        (100.0, 15.0),  # High noise
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for idx, (tau, noise) in enumerate(conditions_to_check):
        ax = axes[idx]

        if (tau, noise) not in condition_runs:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue

        runs = condition_runs[(tau, noise)]
        print(f"\nCondition: τ={tau}ms, noise={noise}ms")
        print(f"  Number of runs: {len(runs)}")

        # Plot all runs for this condition
        colors = plt.cm.tab10(np.linspace(0, 1, len(runs)))

        positions_at_t1 = []
        positions_at_t2 = []
        all_positions = []

        for i, (dirname, time, position) in enumerate(runs):
            ax.plot(time, position, color=colors[i], linewidth=1, alpha=0.7, label=f"Run {i + 1}")
            all_positions.append(position)

            # Sample positions at specific times
            idx_t1 = np.argmin(np.abs(time - 1.0))  # at t=1.0s
            idx_t2 = np.argmin(np.abs(time - 2.0))  # at t=2.0s
            positions_at_t1.append(position[idx_t1])
            positions_at_t2.append(position[idx_t2])

        # Calculate statistics
        all_positions = np.array(all_positions)
        mean_position = np.mean(all_positions, axis=0)
        std_position = np.std(all_positions, axis=0)
        max_std = np.max(std_position)
        mean_std = np.mean(std_position)

        # Plot mean
        ax.plot(runs[0][1], mean_position, "k--", linewidth=2.5, label=f"Mean (max std={max_std:.6f}m)", zorder=10)

        print(f"  Position at t=1.0s: mean={np.mean(positions_at_t1):.6f}m, std={np.std(positions_at_t1):.6f}m")
        print(f"  Position at t=2.0s: mean={np.mean(positions_at_t2):.6f}m, std={np.std(positions_at_t2):.6f}m")
        print(f"  Max std across all time: {max_std:.6f}m")
        print(f"  Mean std across all time: {mean_std:.6f}m")

        # Check if all runs are identical
        if len(runs) > 1:
            first_pos = runs[0][2]
            all_identical = True
            for _, _, position in runs[1:]:
                min_len = min(len(first_pos), len(position))
                if not np.allclose(first_pos[:min_len], position[:min_len], atol=1e-10):
                    all_identical = False
                    break

            if all_identical:
                print("  WARNING: All runs are IDENTICAL! No variation detected.")
            else:
                print("  Runs show variation (not identical)")

        ax.set_xlabel("Time [s]", fontsize=11)
        ax.set_ylabel("Position [m]", fontsize=11)
        ax.set_title(
            f"τ={tau:.0f}ms, Noise={noise:.0f}ms ({len(runs)} Monte Carlo runs)\n"
            + f"Max Std={max_std:.6f}m, Mean Std={mean_std:.6f}m",
            fontweight="bold",
            fontsize=11,
        )
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8, ncol=2)
        ax.set_ylim([-0.5, 6.0])

    plt.suptitle("Monte Carlo Variation Check: Multiple Runs per Condition", fontsize=14, fontweight="bold")
    plt.tight_layout()

    output_path = base_dir / "monte_carlo_variation_check.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nVariation check plot saved to: {output_path}")

    # Create zoomed version focusing on differences
    fig2, axes2 = plt.subplots(2, 2, figsize=(16, 12))
    axes2 = axes2.flatten()

    for idx, (tau, noise) in enumerate(conditions_to_check):
        ax = axes2[idx]

        if (tau, noise) not in condition_runs:
            continue

        runs = condition_runs[(tau, noise)]
        colors = plt.cm.tab10(np.linspace(0, 1, len(runs)))

        # Plot first 0.5 seconds
        for i, (dirname, time, position) in enumerate(runs):
            mask = time <= 0.5
            ax.plot(time[mask], position[mask], color=colors[i], linewidth=1.5, alpha=0.7, label=f"Run {i + 1}")

        all_positions = np.array([pos for _, _, pos in runs])
        mean_position = np.mean(all_positions, axis=0)
        max_std = np.max(np.std(all_positions, axis=0))

        mask = runs[0][1] <= 0.5
        ax.plot(runs[0][1][mask], mean_position[mask], "k--", linewidth=2.5, label="Mean", zorder=10)

        ax.set_xlabel("Time [s]", fontsize=11)
        ax.set_ylabel("Position [m]", fontsize=11)
        ax.set_title(
            f"τ={tau:.0f}ms, Noise={noise:.0f}ms (Zoomed 0-0.5s)\n" + f"Max Std={max_std:.6f}m",
            fontweight="bold",
            fontsize=11,
        )
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8, ncol=2)
        ax.set_xlim([0, 0.5])

    plt.suptitle("Monte Carlo Variation Check (Zoomed): Multiple Runs per Condition", fontsize=14, fontweight="bold")
    plt.tight_layout()

    output_path2 = base_dir / "monte_carlo_variation_check_zoom.png"
    plt.savefig(output_path2, dpi=300, bbox_inches="tight")
    print(f"Zoomed variation check plot saved to: {output_path2}")

    plt.show()


if __name__ == "__main__":
    main()
