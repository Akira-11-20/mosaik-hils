#!/usr/bin/env python3
"""
Comprehensive analysis of inverse compensation with plant noise.
Creates position traces, deviation plots, and statistical analysis.
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
sns.set_style("whitegrid")
plt.rcParams["font.size"] = 11
plt.rcParams["figure.figsize"] = (14, 10)


def parse_directory_name(dirname):
    """Parse directory name to extract parameters."""
    if "baseline_rt" in dirname:
        return None, None

    tau_match = re.search(r"tau(\d+)ms", dirname)
    tau = float(tau_match.group(1)) if tau_match else None

    noise_match = re.search(r"noise(\d+)ms", dirname)
    noise = float(noise_match.group(1)) if noise_match else 0.0

    return tau, noise


def load_data(h5_path):
    """Load all relevant data from HDF5 file."""
    with h5py.File(h5_path, "r") as f:
        data = {
            "time": f["time"]["time_s"][:],
            "position": f["EnvSim-0_Spacecraft1DOF_0"]["position"][:],
            "velocity": f["EnvSim-0_Spacecraft1DOF_0"]["velocity"][:],
            "force": f["EnvSim-0_Spacecraft1DOF_0"]["force"][:],
        }

        # Try to load controller data
        try:
            data["error"] = f["ControllerSim-0_PIDController_0"]["error"][:]
        except KeyError:
            data["error"] = None

    return data


def calculate_iae(error, time):
    """Calculate Integral Absolute Error."""
    if error is None:
        return None
    dt = np.mean(np.diff(time))
    return np.sum(np.abs(error)) * dt


def calculate_ise(error, time):
    """Calculate Integral Squared Error."""
    if error is None:
        return None
    dt = np.mean(np.diff(time))
    return np.sum(error**2) * dt


def main():
    base_dir = Path("/home/akira/mosaik-hils/figures_for_paper/17_noise_inverse_heatmap")

    # Find baseline
    baseline_dirs = [d for d in os.listdir(base_dir) if "baseline_rt" in d]
    if not baseline_dirs:
        print("Error: No baseline_rt directory found!")
        return

    baseline_dir = baseline_dirs[0]
    print(f"Using baseline: {baseline_dir}")

    # Load baseline data
    baseline_h5 = base_dir / baseline_dir / "hils_data.h5"
    baseline_data = load_data(baseline_h5)
    print(f"Baseline loaded: {len(baseline_data['time'])} time points")

    # Collect results by noise level
    results = defaultdict(list)

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
            data = load_data(h5_path)

            # Calculate metrics
            rmse = np.sqrt(np.mean((data["position"] - baseline_data["position"])**2))
            mae = np.mean(np.abs(data["position"] - baseline_data["position"]))
            max_dev = np.max(np.abs(data["position"] - baseline_data["position"]))

            iae = calculate_iae(data["error"], data["time"])
            ise = calculate_ise(data["error"], data["time"])

            results[noise].append({
                "data": data,
                "rmse": rmse,
                "mae": mae,
                "max_dev": max_dev,
                "iae": iae,
                "ise": ise,
            })

        except Exception as e:
            print(f"Error processing {dirname}: {e}")
            continue

    all_noises = sorted(results.keys())
    print(f"\nFound {len(all_noises)} noise levels: {all_noises}")

    # ========== Plot 1: Position Traces Comparison ==========
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    colors = ['blue', 'green', 'orange', 'red']

    for idx, noise in enumerate(all_noises):
        ax = axes[idx // 2, idx % 2]

        # Plot baseline
        ax.plot(baseline_data["time"], baseline_data["position"],
                'k--', linewidth=2, label='Baseline RT', alpha=0.7)

        # Plot first 5 Monte Carlo runs
        for i, result in enumerate(results[noise][:5]):
            alpha = 0.5 if i > 0 else 0.8
            label = f'Run {i+1}' if i == 0 else None
            ax.plot(result["data"]["time"], result["data"]["position"],
                   color=colors[idx], alpha=alpha, linewidth=1.5, label=label)

        # Target line
        ax.axhline(y=5.0, color='gray', linestyle=':', linewidth=1, alpha=0.5, label='Target')

        ax.set_xlabel('Time [s]', fontsize=11, fontweight='bold')
        ax.set_ylabel('Position [m]', fontsize=11, fontweight='bold')
        ax.set_title(f'Noise = {int(noise)} ms ({len(results[noise])} runs)',
                    fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 5])

    plt.tight_layout()
    output_path1 = base_dir / "position_traces_by_noise.png"
    save_figure_both_sizes(plt, output_path1.parent, base_name=output_path1.stem)
    print(f"\nPosition traces saved to: {output_path1}")

    # ========== Plot 2: Deviation from Baseline ==========
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    for idx, noise in enumerate(all_noises):
        ax = axes[idx // 2, idx % 2]

        # Plot deviation for all runs
        for result in results[noise]:
            deviation = result["data"]["position"] - baseline_data["position"]
            ax.plot(result["data"]["time"], deviation * 1000,  # Convert to mm
                   color=colors[idx], alpha=0.3, linewidth=0.5)

        # Calculate and plot mean deviation
        all_deviations = np.array([r["data"]["position"] - baseline_data["position"]
                                   for r in results[noise]])
        mean_dev = np.mean(all_deviations, axis=0)
        std_dev = np.std(all_deviations, axis=0)

        time = results[noise][0]["data"]["time"]
        ax.plot(time, mean_dev * 1000, 'k-', linewidth=2, label='Mean')
        ax.fill_between(time, (mean_dev - std_dev) * 1000, (mean_dev + std_dev) * 1000,
                        alpha=0.3, color='gray', label='±1 σ')

        ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.set_xlabel('Time [s]', fontsize=11, fontweight='bold')
        ax.set_ylabel('Deviation from Baseline [mm]', fontsize=11, fontweight='bold')
        ax.set_title(f'Noise = {int(noise)} ms (RMSE = {np.mean([r["rmse"] for r in results[noise]]):.4f} m)',
                    fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 5])

    plt.tight_layout()
    output_path2 = base_dir / "deviation_from_baseline_by_noise.png"
    save_figure_both_sizes(plt, output_path2.parent, base_name=output_path2.stem)
    print(f"Deviation plot saved to: {output_path2}")

    # ========== Plot 3: Statistical Comparison ==========
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # Plot 3a: RMSE
    ax1 = fig.add_subplot(gs[0, 0])
    rmse_data = [[r["rmse"] for r in results[n]] for n in all_noises]
    bp1 = ax1.boxplot(rmse_data, labels=[f'{int(n)}' for n in all_noises], patch_artist=True)
    for patch, color in zip(bp1['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax1.set_xlabel('Noise Level [ms]', fontweight='bold')
    ax1.set_ylabel('RMSE [m]', fontweight='bold')
    ax1.set_title('Root Mean Square Error', fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')

    # Plot 3b: MAE
    ax2 = fig.add_subplot(gs[0, 1])
    mae_data = [[r["mae"] for r in results[n]] for n in all_noises]
    bp2 = ax2.boxplot(mae_data, labels=[f'{int(n)}' for n in all_noises], patch_artist=True)
    for patch, color in zip(bp2['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax2.set_xlabel('Noise Level [ms]', fontweight='bold')
    ax2.set_ylabel('MAE [m]', fontweight='bold')
    ax2.set_title('Mean Absolute Error', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    # Plot 3c: Max Deviation
    ax3 = fig.add_subplot(gs[1, 0])
    maxdev_data = [[r["max_dev"] for r in results[n]] for n in all_noises]
    bp3 = ax3.boxplot(maxdev_data, labels=[f'{int(n)}' for n in all_noises], patch_artist=True)
    for patch, color in zip(bp3['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax3.set_xlabel('Noise Level [ms]', fontweight='bold')
    ax3.set_ylabel('Max Deviation [m]', fontweight='bold')
    ax3.set_title('Maximum Deviation from Baseline', fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')

    # Plot 3d: IAE
    ax4 = fig.add_subplot(gs[1, 1])
    iae_data = [[r["iae"] for r in results[n] if r["iae"] is not None] for n in all_noises]
    if any(iae_data):
        bp4 = ax4.boxplot(iae_data, labels=[f'{int(n)}' for n in all_noises], patch_artist=True)
        for patch, color in zip(bp4['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax4.set_xlabel('Noise Level [ms]', fontweight='bold')
        ax4.set_ylabel('IAE [m·s]', fontweight='bold')
        ax4.set_title('Integral Absolute Error', fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')

    # Plot 3e: Trend comparison (mean values)
    ax5 = fig.add_subplot(gs[2, :])

    rmse_means = [np.mean([r["rmse"] for r in results[n]]) for n in all_noises]
    mae_means = [np.mean([r["mae"] for r in results[n]]) for n in all_noises]
    maxdev_means = [np.mean([r["max_dev"] for r in results[n]]) for n in all_noises]

    x = np.array(all_noises)
    ax5.plot(x, np.array(rmse_means) * 1000, 'o-', linewidth=2, markersize=8, label='RMSE', color='blue')
    ax5.plot(x, np.array(mae_means) * 1000, 's-', linewidth=2, markersize=8, label='MAE', color='green')
    ax5.plot(x, np.array(maxdev_means) * 1000, '^-', linewidth=2, markersize=8, label='Max Dev', color='red')

    ax5.set_xlabel('Plant Noise Std Dev [ms]', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Error Metrics [mm]', fontsize=12, fontweight='bold')
    ax5.set_title('Mean Error Metrics vs Noise Level (τ = 100ms, Inverse Comp Enabled)',
                 fontsize=13, fontweight='bold')
    ax5.legend(fontsize=11, loc='upper left')
    ax5.grid(True, alpha=0.3)

    save_figure_both_sizes(plt, base_dir, base_name="statistical_comparison")
    print(f"Statistical comparison saved to: {base_dir / 'statistical_comparison.png'}")

    # ========== Print Numerical Summary ==========
    print("\n" + "="*60)
    print("NUMERICAL SUMMARY")
    print("="*60)

    for noise in all_noises:
        print(f"\n--- Noise Level: {int(noise)} ms ---")
        print(f"Number of runs: {len(results[noise])}")

        rmse_vals = [r["rmse"] for r in results[noise]]
        mae_vals = [r["mae"] for r in results[noise]]
        maxdev_vals = [r["max_dev"] for r in results[noise]]

        print(f"RMSE:     {np.mean(rmse_vals)*1000:.3f} ± {np.std(rmse_vals)*1000:.3f} mm")
        print(f"MAE:      {np.mean(mae_vals)*1000:.3f} ± {np.std(mae_vals)*1000:.3f} mm")
        print(f"Max Dev:  {np.mean(maxdev_vals)*1000:.3f} ± {np.std(maxdev_vals)*1000:.3f} mm")

        if results[noise][0]["iae"] is not None:
            iae_vals = [r["iae"] for r in results[noise]]
            ise_vals = [r["ise"] for r in results[noise]]
            print(f"IAE:      {np.mean(iae_vals):.4f} ± {np.std(iae_vals):.4f} m·s")
            print(f"ISE:      {np.mean(ise_vals):.4f} ± {np.std(ise_vals):.4f} m²·s")

    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)

    plt.show()


if __name__ == "__main__":
    main()
