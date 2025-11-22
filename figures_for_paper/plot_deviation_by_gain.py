#!/usr/bin/env python3
"""
Deviation from baseline plotted by gain (alpha).
Compare gain α=9 vs α=10 with different colors.
"""

import os
import re
from collections import defaultdict
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

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


def collect_dataset(base_dir, target_tau=100.0):
    """Collect data from a directory for specified tau value."""
    # Find baseline
    baseline_dirs = [d for d in os.listdir(base_dir) if "baseline_rt" in d]
    if not baseline_dirs:
        print(f"Warning: No baseline in {base_dir}")
        return None, None, None

    baseline_h5 = base_dir / baseline_dirs[0] / "hils_data.h5"
    baseline_time, baseline_position = load_position_data(baseline_h5)

    # Get gain value
    import json
    all_dirs = [d for d in os.listdir(base_dir)
                if os.path.isdir(base_dir / d) and "baseline_rt" not in d]

    # Find a directory with target_tau to get the gain
    gain = None
    for dirname in all_dirs:
        tau, noise = parse_directory_name(dirname)
        if tau == target_tau:
            config_path = base_dir / dirname / "simulation_config.json"
            with open(config_path, 'r') as f:
                config = json.load(f)
                gain = config["inverse_compensation"]["gain"]
            break

    if gain is None:
        print(f"Warning: Could not find gain for tau={target_tau} in {base_dir}")
        return None, None, None

    print(f"\nProcessing {base_dir.name}:")
    print(f"  Gain: α={gain}, τ={target_tau}ms")

    # Collect results
    results = defaultdict(list)

    for dirname in all_dirs:
        h5_path = base_dir / dirname / "hils_data.h5"
        if not h5_path.exists():
            continue

        tau, noise = parse_directory_name(dirname)
        if tau is None or tau != target_tau:
            continue

        try:
            time, position = load_position_data(h5_path)
            deviation = position - baseline_position[:len(position)]
            results[noise].append({
                "time": time,
                "deviation": deviation,
            })
        except Exception as e:
            print(f"  Error processing {dirname}: {e}")
            continue

    print(f"  Collected {sum(len(results[n]) for n in results)} simulations")
    print(f"  Noise levels: {sorted(results.keys())}")

    return results, baseline_position, gain


def main():
    # Directories
    dir_gain9 = Path("/home/akira/mosaik-hils/figures_for_paper/17_noise_inverse_heatmap")
    dir_gain10 = Path("/home/akira/mosaik-hils/figures_for_paper/16_tau_noise")

    print("="*70)
    print("Deviation from Baseline by Gain")
    print("="*70)

    # Collect data
    results_gain9, baseline9, gain9 = collect_dataset(dir_gain9, target_tau=100.0)
    results_gain10, baseline10, gain10 = collect_dataset(dir_gain10, target_tau=100.0)

    if results_gain9 is None or results_gain10 is None:
        print("Error: Could not load data!")
        return

    # Get common noise levels
    noise_levels = sorted(set(results_gain9.keys()) & set(results_gain10.keys()))
    print(f"\nCommon noise levels: {noise_levels} ms")

    # Color scheme
    colors = {
        gain9: '#FF6B6B',   # Red for gain=9
        gain10: '#4ECDC4',  # Teal for gain=10
    }

    # Create figure: 2x2 subplots for each noise level
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    axes = axes.flatten()

    for idx, noise in enumerate(noise_levels):
        ax = axes[idx]

        # Plot for gain 9
        for run_data in results_gain9[noise]:
            time = run_data["time"]
            deviation = run_data["deviation"] * 1000  # Convert to mm
            ax.plot(time, deviation, color=colors[gain9], alpha=0.15, linewidth=0.8)

        # Calculate mean and std for gain 9
        all_deviations_9 = np.array([r["deviation"] for r in results_gain9[noise]])
        mean_dev_9 = np.mean(all_deviations_9, axis=0)
        std_dev_9 = np.std(all_deviations_9, axis=0)
        time_9 = results_gain9[noise][0]["time"]

        ax.plot(time_9, mean_dev_9 * 1000, color=colors[gain9], linewidth=3,
                label=f'α={gain9} (mean)', zorder=100)
        ax.fill_between(time_9, (mean_dev_9 - std_dev_9) * 1000,
                        (mean_dev_9 + std_dev_9) * 1000,
                        alpha=0.3, color=colors[gain9], label=f'α={gain9} (±1σ)', zorder=50)

        # Plot for gain 10
        for run_data in results_gain10[noise]:
            time = run_data["time"]
            deviation = run_data["deviation"] * 1000  # Convert to mm
            ax.plot(time, deviation, color=colors[gain10], alpha=0.15, linewidth=0.8)

        # Calculate mean and std for gain 10
        all_deviations_10 = np.array([r["deviation"] for r in results_gain10[noise]])
        mean_dev_10 = np.mean(all_deviations_10, axis=0)
        std_dev_10 = np.std(all_deviations_10, axis=0)
        time_10 = results_gain10[noise][0]["time"]

        ax.plot(time_10, mean_dev_10 * 1000, color=colors[gain10], linewidth=3,
                label=f'α={gain10} (mean)', zorder=100)
        ax.fill_between(time_10, (mean_dev_10 - std_dev_10) * 1000,
                        (mean_dev_10 + std_dev_10) * 1000,
                        alpha=0.3, color=colors[gain10], label=f'α={gain10} (±1σ)', zorder=50)

        # Zero line
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.7, zorder=10)

        # Labels and title
        ax.set_xlabel('Time [s]', fontsize=12, fontweight='bold')
        ax.set_ylabel('Deviation from Baseline [mm]', fontsize=12, fontweight='bold')

        # Calculate RMSE for title
        rmse_9 = np.sqrt(np.mean(all_deviations_9**2))
        rmse_10 = np.sqrt(np.mean(all_deviations_10**2))

        ax.set_title(f'Noise = {int(noise)} ms\n'
                    f'RMSE: α={gain9}→{rmse_9*1000:.2f}mm, α={gain10}→{rmse_10*1000:.2f}mm',
                    fontsize=13, fontweight='bold')

        ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 5])

    plt.suptitle('Deviation from Baseline RT by Compensation Gain (τ=100ms)',
                fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout()

    output_path = Path("/home/akira/mosaik-hils/figures_for_paper/deviation_by_gain_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved: {output_path}")

    # Create zoomed version (first 2 seconds)
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    axes = axes.flatten()

    for idx, noise in enumerate(noise_levels):
        ax = axes[idx]

        # Plot for gain 9
        for run_data in results_gain9[noise]:
            time = run_data["time"]
            deviation = run_data["deviation"] * 1000
            mask = time <= 2.0
            ax.plot(time[mask], deviation[mask], color=colors[gain9], alpha=0.15, linewidth=0.8)

        # Mean and std for gain 9
        all_deviations_9 = np.array([r["deviation"] for r in results_gain9[noise]])
        mean_dev_9 = np.mean(all_deviations_9, axis=0)
        std_dev_9 = np.std(all_deviations_9, axis=0)
        time_9 = results_gain9[noise][0]["time"]
        mask_9 = time_9 <= 2.0

        ax.plot(time_9[mask_9], mean_dev_9[mask_9] * 1000, color=colors[gain9],
                linewidth=3, label=f'α={gain9} (mean)', zorder=100)
        ax.fill_between(time_9[mask_9],
                        (mean_dev_9[mask_9] - std_dev_9[mask_9]) * 1000,
                        (mean_dev_9[mask_9] + std_dev_9[mask_9]) * 1000,
                        alpha=0.3, color=colors[gain9], label=f'α={gain9} (±1σ)', zorder=50)

        # Plot for gain 10
        for run_data in results_gain10[noise]:
            time = run_data["time"]
            deviation = run_data["deviation"] * 1000
            mask = time <= 2.0
            ax.plot(time[mask], deviation[mask], color=colors[gain10], alpha=0.15, linewidth=0.8)

        # Mean and std for gain 10
        all_deviations_10 = np.array([r["deviation"] for r in results_gain10[noise]])
        mean_dev_10 = np.mean(all_deviations_10, axis=0)
        std_dev_10 = np.std(all_deviations_10, axis=0)
        time_10 = results_gain10[noise][0]["time"]
        mask_10 = time_10 <= 2.0

        ax.plot(time_10[mask_10], mean_dev_10[mask_10] * 1000, color=colors[gain10],
                linewidth=3, label=f'α={gain10} (mean)', zorder=100)
        ax.fill_between(time_10[mask_10],
                        (mean_dev_10[mask_10] - std_dev_10[mask_10]) * 1000,
                        (mean_dev_10[mask_10] + std_dev_10[mask_10]) * 1000,
                        alpha=0.3, color=colors[gain10], label=f'α={gain10} (±1σ)', zorder=50)

        ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.7, zorder=10)

        ax.set_xlabel('Time [s]', fontsize=12, fontweight='bold')
        ax.set_ylabel('Deviation from Baseline [mm]', fontsize=12, fontweight='bold')

        rmse_9 = np.sqrt(np.mean(all_deviations_9**2))
        rmse_10 = np.sqrt(np.mean(all_deviations_10**2))

        ax.set_title(f'Noise = {int(noise)} ms (Transient Phase)\n'
                    f'RMSE: α={gain9}→{rmse_9*1000:.2f}mm, α={gain10}→{rmse_10*1000:.2f}mm',
                    fontsize=13, fontweight='bold')

        ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 2])

    plt.suptitle('Deviation from Baseline RT by Gain (τ=100ms) - Transient Phase',
                fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout()

    output_path_zoom = Path("/home/akira/mosaik-hils/figures_for_paper/deviation_by_gain_comparison_zoom.png")
    plt.savefig(output_path_zoom, dpi=300, bbox_inches='tight')
    print(f"Zoomed figure saved: {output_path_zoom}")

    # Print summary statistics
    print("\n" + "="*70)
    print("Summary: Mean Absolute Deviation")
    print("="*70)
    print(f"{'Noise':>8} | {'Gain α={gain9} (mm)':>20} | {'Gain α={gain10} (mm)':>20}")
    print("-"*70)

    for noise in noise_levels:
        all_dev_9 = np.array([r["deviation"] for r in results_gain9[noise]])
        all_dev_10 = np.array([r["deviation"] for r in results_gain10[noise]])

        mad_9 = np.mean(np.abs(all_dev_9)) * 1000
        mad_10 = np.mean(np.abs(all_dev_10)) * 1000

        print(f"{noise:>6.0f} ms | {mad_9:>20.3f} | {mad_10:>20.3f}")

    print("="*70)

    plt.show()


if __name__ == "__main__":
    main()
