#!/usr/bin/env python3
"""
Deviation from baseline for all gain values (α=8, 9, 10, 11, 12).
Transient phase (0-2s) with color-coded gains.
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


def collect_all_gains(base_dir, target_tau=100.0):
    """Collect data for all gain values at specified tau."""
    # Find baseline
    baseline_dirs = [d for d in os.listdir(base_dir) if "baseline_rt" in d]
    if not baseline_dirs:
        print(f"Error: No baseline in {base_dir}")
        return None, None

    baseline_h5 = base_dir / baseline_dirs[0] / "hils_data.h5"
    baseline_time, baseline_position = load_position_data(baseline_h5)

    # Collect results by gain and noise
    import json
    results = defaultdict(lambda: defaultdict(list))  # results[gain][noise] = [data_list]

    all_dirs = [d for d in os.listdir(base_dir)
                if os.path.isdir(base_dir / d) and "baseline_rt" not in d]

    for dirname in all_dirs:
        h5_path = base_dir / dirname / "hils_data.h5"
        config_path = base_dir / dirname / "simulation_config.json"

        if not h5_path.exists() or not config_path.exists():
            continue

        # Get gain from config
        with open(config_path, 'r') as f:
            config = json.load(f)
            gain = config["inverse_compensation"]["gain"]
            tau = config["plant"]["time_constant_s"] * 1000

        if tau != target_tau:
            continue

        # Get noise from directory name
        tau_parsed, noise = parse_directory_name(dirname)
        if tau_parsed is None:
            continue

        try:
            time, position = load_position_data(h5_path)
            deviation = position - baseline_position[:len(position)]
            results[gain][noise].append({
                "time": time,
                "deviation": deviation,
            })
        except Exception as e:
            print(f"  Error processing {dirname}: {e}")
            continue

    # Print summary
    print(f"\nData collected from {base_dir.name}:")
    print(f"  τ = {target_tau}ms")
    all_gains = sorted(results.keys())
    print(f"  Gains found: {all_gains}")
    for gain in all_gains:
        total = sum(len(results[gain][n]) for n in results[gain])
        print(f"    α={gain}: {total} simulations across {len(results[gain])} noise levels")

    return results, baseline_position


def main():
    base_dir = Path("/home/akira/mosaik-hils/figures_for_paper/17_noise_inverse_heatmap")

    print("="*70)
    print("Deviation from Baseline - All Compensation Gains")
    print("="*70)

    # Collect data for all gains
    results, baseline = collect_all_gains(base_dir, target_tau=100.0)

    if results is None:
        print("Error: Could not load data!")
        return

    all_gains = sorted(results.keys())
    all_noises = sorted(set(noise for gain in all_gains for noise in results[gain].keys()))

    print(f"\nGains: {all_gains}")
    print(f"Noise levels: {all_noises} ms")

    # Color scheme - gradient from blue (low gain) to red (high gain)
    colors = {
        8:  '#3498DB',  # Blue
        9:  '#52B788',  # Green-blue
        10: '#F39C12',  # Orange
        11: '#E67E22',  # Dark orange
        12: '#E74C3C',  # Red
    }

    # Line styles for distinction
    linestyles = {
        8:  '-',
        9:  '-',
        10: '-',
        11: '-',
        12: '-',
    }

    # Create figure: 4x1 subplots for each noise level (zoomed to 0-2s), vertically stacked
    fig, axes = plt.subplots(4, 1, figsize=(14, 20))

    # Labels for subplots
    subplot_labels = ['(a)', '(b)', '(c)', '(d)']

    for idx, noise in enumerate(all_noises):
        ax = axes[idx]

        # Plot for each gain
        for gain in all_gains:
            if noise not in results[gain]:
                continue

            # Plot individual runs (very transparent)
            for run_data in results[gain][noise]:
                time = run_data["time"]
                deviation = run_data["deviation"] * 1000  # Convert to mm
                mask = time <= 2.0
                ax.plot(time[mask], deviation[mask],
                       color=colors[gain], alpha=0.08, linewidth=0.6)

            # Calculate mean and std
            all_deviations = np.array([r["deviation"] for r in results[gain][noise]])
            mean_dev = np.mean(all_deviations, axis=0)
            std_dev = np.std(all_deviations, axis=0)
            time_arr = results[gain][noise][0]["time"]
            mask = time_arr <= 2.0

            # Plot mean line (thick)
            ax.plot(time_arr[mask], mean_dev[mask] * 1000,
                   color=colors[gain], linewidth=3.5,
                   linestyle=linestyles[gain],
                   label=f'α={gain}', zorder=100)

            # Plot std band
            ax.fill_between(time_arr[mask],
                           (mean_dev[mask] - std_dev[mask]) * 1000,
                           (mean_dev[mask] + std_dev[mask]) * 1000,
                           alpha=0.25, color=colors[gain], zorder=50)

        # Zero line
        ax.axhline(y=0, color='black', linestyle='--', linewidth=2, alpha=0.7, zorder=10)

        # Labels and title
        ax.set_xlabel('Time [s]', fontsize=13, fontweight='bold')
        ax.set_ylabel('Deviation from Baseline [mm]', fontsize=13, fontweight='bold')

        # Title with subplot label
        ax.set_title(f'{subplot_labels[idx]} Noise = {int(noise)} ms',
                    fontsize=16, fontweight='bold')

        # Legend with larger font
        ax.legend(loc='upper right', fontsize=12, framealpha=0.95,
                 ncol=1, columnspacing=0.8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 2])

    plt.tight_layout()

    output_path = base_dir / "deviation_all_gains_comparison.png"
    save_figure_both_sizes(plt, output_path.parent, base_name=output_path.stem)
    print(f"\nFigure saved: {output_path}")

    # Create summary statistics table
    print("\n" + "="*90)
    print("Performance Metrics by Gain")
    print("="*90)
    print(f"{'Noise':>8} | {'Metric':>10} | " +
          " | ".join([f"α={g:>2}" for g in all_gains]))
    print("-"*90)

    for noise in all_noises:
        # RMSE
        rmse_vals = []
        for gain in all_gains:
            if noise in results[gain]:
                all_dev = np.array([r["deviation"] for r in results[gain][noise]])
                rmse = np.sqrt(np.mean(all_dev**2)) * 1000
                rmse_vals.append(f"{rmse:>8.2f}")
            else:
                rmse_vals.append("    N/A")

        print(f"{noise:>6.0f} ms | {'RMSE [mm]':>10} | " + " | ".join(rmse_vals))

        # MAE
        mae_vals = []
        for gain in all_gains:
            if noise in results[gain]:
                all_dev = np.array([r["deviation"] for r in results[gain][noise]])
                mae = np.mean(np.abs(all_dev)) * 1000
                mae_vals.append(f"{mae:>8.2f}")
            else:
                mae_vals.append("    N/A")

        print(f"{'':>8} | {'MAE [mm]':>10} | " + " | ".join(mae_vals))
        print("-"*90)

    # Find optimal gain for each noise level
    print("\n" + "="*90)
    print("Optimal Gain by Noise Level (based on RMSE)")
    print("="*90)
    for noise in all_noises:
        rmse_by_gain = {}
        for gain in all_gains:
            if noise in results[gain]:
                all_dev = np.array([r["deviation"] for r in results[gain][noise]])
                rmse = np.sqrt(np.mean(all_dev**2)) * 1000
                rmse_by_gain[gain] = rmse

        if rmse_by_gain:
            optimal_gain = min(rmse_by_gain, key=rmse_by_gain.get)
            optimal_rmse = rmse_by_gain[optimal_gain]
            print(f"Noise {int(noise):>2}ms: α={optimal_gain:>2} (RMSE={optimal_rmse:.2f}mm)")

    print("="*90)

    plt.show()


if __name__ == "__main__":
    main()
