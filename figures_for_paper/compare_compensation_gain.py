#!/usr/bin/env python3
"""
逆補償ゲイン(α)による性能比較分析
Compare inverse compensation performance for different gain values.

Compares:
- 16_tau_noise: gain=10, tau=100ms, noise=0,5,10,15ms
- 17_noise_inverse_heatmap: gain=8, tau=100ms, noise=0,5,10,15ms
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
plt.rcParams['font.family'] = 'DejaVu Sans'


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
        velocity = f["EnvSim-0_Spacecraft1DOF_0"]["velocity"][:]
    return time, position, velocity


def calculate_metrics(baseline_pos, test_pos, time):
    """Calculate various performance metrics."""
    min_len = min(len(baseline_pos), len(test_pos))
    baseline_pos = baseline_pos[:min_len]
    test_pos = test_pos[:min_len]
    time = time[:min_len]

    # RMSE
    rmse = np.sqrt(np.mean((baseline_pos - test_pos) ** 2))

    # MAE
    mae = np.mean(np.abs(baseline_pos - test_pos))

    # Max deviation
    max_dev = np.max(np.abs(baseline_pos - test_pos))

    # Time-weighted error (後半を重視)
    weights = np.linspace(0.5, 1.5, len(baseline_pos))
    weighted_error = np.sqrt(np.average((baseline_pos - test_pos)**2, weights=weights))

    return {
        "rmse": rmse,
        "mae": mae,
        "max_dev": max_dev,
        "weighted_error": weighted_error,
    }


def collect_dataset(base_dir, target_tau=100.0):
    """
    Collect data from a directory for specified tau value.
    Returns: (results_dict, baseline_position, gain_value)
    """
    # Find baseline
    baseline_dirs = [d for d in os.listdir(base_dir) if "baseline_rt" in d]
    if not baseline_dirs:
        print(f"Warning: No baseline in {base_dir}")
        return None, None, None

    baseline_h5 = base_dir / baseline_dirs[0] / "hils_data.h5"
    baseline_time, baseline_position, _ = load_position_data(baseline_h5)

    # Get gain value from simulation config
    all_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(base_dir / d) and "baseline_rt" not in d]
    if not all_dirs:
        return None, None, None

    # Read gain from first simulation config
    import json
    config_path = base_dir / all_dirs[0] / "simulation_config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
        gain = config["inverse_compensation"]["gain"]

    print(f"\nProcessing {base_dir.name}:")
    print(f"  Inverse compensation gain: {gain}")

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
            time, position, velocity = load_position_data(h5_path)
            metrics = calculate_metrics(baseline_position, position, time)
            results[noise].append(metrics)
        except Exception as e:
            print(f"  Error processing {dirname}: {e}")
            continue

    print(f"  Collected {sum(len(results[n]) for n in results)} simulations")
    print(f"  Noise levels: {sorted(results.keys())}")

    return results, baseline_position, gain


def main():
    # Directories to compare
    dir_gain8 = Path("/home/akira/mosaik-hils/figures_for_paper/17_noise_inverse_heatmap")
    dir_gain10 = Path("/home/akira/mosaik-hils/figures_for_paper/16_tau_noise")

    print("="*70)
    print("逆補償ゲイン比較分析 (Inverse Compensation Gain Comparison)")
    print("="*70)

    # Collect data
    results_gain8, baseline8, gain8 = collect_dataset(dir_gain8, target_tau=100.0)
    results_gain10, baseline10, gain10 = collect_dataset(dir_gain10, target_tau=100.0)

    if results_gain8 is None or results_gain10 is None:
        print("Error: Could not load data!")
        return

    # Get common noise levels
    noise_levels = sorted(set(results_gain8.keys()) & set(results_gain10.keys()))
    print(f"\n共通のノイズレベル: {noise_levels} ms")

    # Calculate statistics
    stats = {
        gain8: {},
        gain10: {}
    }

    for gain, results in [(gain8, results_gain8), (gain10, results_gain10)]:
        for noise in noise_levels:
            metrics_list = results[noise]
            stats[gain][noise] = {
                "rmse_mean": np.mean([m["rmse"] for m in metrics_list]) * 1000,
                "rmse_std": np.std([m["rmse"] for m in metrics_list]) * 1000,
                "mae_mean": np.mean([m["mae"] for m in metrics_list]) * 1000,
                "mae_std": np.std([m["mae"] for m in metrics_list]) * 1000,
                "max_dev_mean": np.mean([m["max_dev"] for m in metrics_list]) * 1000,
                "max_dev_std": np.std([m["max_dev"] for m in metrics_list]) * 1000,
                "n": len(metrics_list),
            }

    # Print comparison table
    print("\n" + "="*90)
    print(f"{'Noise':>8} | {'Gain=8 RMSE (mm)':>25} | {'Gain=10 RMSE (mm)':>25} | {'差分':>12}")
    print("-"*90)

    for noise in noise_levels:
        rmse8 = stats[gain8][noise]["rmse_mean"]
        std8 = stats[gain8][noise]["rmse_std"]
        rmse10 = stats[gain10][noise]["rmse_mean"]
        std10 = stats[gain10][noise]["rmse_std"]
        diff = rmse8 - rmse10
        diff_pct = (diff / rmse10 * 100) if rmse10 > 0 else 0

        print(f"{noise:>6.0f} ms | {rmse8:>10.3f} ± {std8:>8.3f} | {rmse10:>10.3f} ± {std10:>8.3f} | "
              f"{diff:>+6.3f} ({diff_pct:>+5.1f}%)")

    print("="*90)

    # ========== Create Comparison Plots ==========

    # Figure 1: Main comparison (3 subplots)
    fig = plt.figure(figsize=(18, 5))

    colors = {gain8: '#FF6B6B', gain10: '#4ECDC4'}  # Red for gain=8, Teal for gain=10
    labels = {gain8: f'Gain α={gain8}', gain10: f'Gain α={gain10}'}

    # Plot 1: RMSE comparison
    ax1 = plt.subplot(131)
    for gain, results in [(gain8, results_gain8), (gain10, results_gain10)]:
        rmse_means = [stats[gain][n]["rmse_mean"] for n in noise_levels]
        rmse_stds = [stats[gain][n]["rmse_std"] for n in noise_levels]

        ax1.errorbar(noise_levels, rmse_means, yerr=rmse_stds,
                    marker='o', markersize=10, linewidth=2.5, capsize=6, capthick=2,
                    label=labels[gain], color=colors[gain], alpha=0.9)

    ax1.set_xlabel('Plant Noise Std Dev [ms]', fontsize=12, fontweight='bold')
    ax1.set_ylabel('RMSE from Baseline [mm]', fontsize=12, fontweight='bold')
    ax1.set_title('Position Tracking Error\n(τ=100ms, PID Control)', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=11, loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)

    # Plot 2: MAE comparison
    ax2 = plt.subplot(132)
    for gain, results in [(gain8, results_gain8), (gain10, results_gain10)]:
        mae_means = [stats[gain][n]["mae_mean"] for n in noise_levels]
        mae_stds = [stats[gain][n]["mae_std"] for n in noise_levels]

        ax2.errorbar(noise_levels, mae_means, yerr=mae_stds,
                    marker='s', markersize=10, linewidth=2.5, capsize=6, capthick=2,
                    label=labels[gain], color=colors[gain], alpha=0.9)

    ax2.set_xlabel('Plant Noise Std Dev [ms]', fontsize=12, fontweight='bold')
    ax2.set_ylabel('MAE from Baseline [mm]', fontsize=12, fontweight='bold')
    ax2.set_title('Mean Absolute Error', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=11, loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)

    # Plot 3: Max Deviation comparison
    ax3 = plt.subplot(133)
    for gain, results in [(gain8, results_gain8), (gain10, results_gain10)]:
        maxdev_means = [stats[gain][n]["max_dev_mean"] for n in noise_levels]
        maxdev_stds = [stats[gain][n]["max_dev_std"] for n in noise_levels]

        ax3.errorbar(noise_levels, maxdev_means, yerr=maxdev_stds,
                    marker='^', markersize=10, linewidth=2.5, capsize=6, capthick=2,
                    label=labels[gain], color=colors[gain], alpha=0.9)

    ax3.set_xlabel('Plant Noise Std Dev [ms]', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Max Deviation [mm]', fontsize=12, fontweight='bold')
    ax3.set_title('Maximum Position Error', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=11, loc='upper left')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(bottom=0)

    plt.suptitle('逆補償ゲインの影響 (Inverse Compensation Gain Effect)',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    output_path1 = Path("/home/akira/mosaik-hils/figures_for_paper/gain_comparison_metrics.png")
    save_figure_both_sizes(plt, output_path1.parent, base_name=output_path1.stem)
    print(f"\nメトリクス比較図を保存: {output_path1}")

    # Figure 2: Box plot comparison for each noise level
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    for idx, noise in enumerate(noise_levels):
        ax = axes[idx // 2, idx % 2]

        data_to_plot = [
            np.array([m["rmse"] * 1000 for m in results_gain8[noise]]),
            np.array([m["rmse"] * 1000 for m in results_gain10[noise]]),
        ]

        bp = ax.boxplot(data_to_plot, tick_labels=[f'α={gain8}', f'α={gain10}'],
                       patch_artist=True, widths=0.6)

        # Color the boxes
        bp['boxes'][0].set_facecolor(colors[gain8])
        bp['boxes'][1].set_facecolor(colors[gain10])
        bp['boxes'][0].set_alpha(0.7)
        bp['boxes'][1].set_alpha(0.7)

        ax.set_ylabel('RMSE from Baseline [mm]', fontsize=11, fontweight='bold')
        ax.set_title(f'Noise = {int(noise)} ms  (n={stats[gain8][noise]["n"]}, {stats[gain10][noise]["n"]})',
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # Add mean markers
        ax.plot(1, stats[gain8][noise]["rmse_mean"], 'k*', markersize=15, label='Mean', zorder=10)
        ax.plot(2, stats[gain10][noise]["rmse_mean"], 'k*', markersize=15, zorder=10)
        ax.legend(fontsize=9)

        # Add improvement annotation
        improvement = stats[gain10][noise]["rmse_mean"] - stats[gain8][noise]["rmse_mean"]
        improvement_pct = (improvement / stats[gain8][noise]["rmse_mean"]) * 100 if stats[gain8][noise]["rmse_mean"] > 0 else 0

        if improvement > 0:
            ax.text(1.5, max(data_to_plot[0].max(), data_to_plot[1].max()) * 0.95,
                   f'α=10の方が\n{improvement:.2f}mm良い\n({improvement_pct:+.1f}%)',
                   ha='center', va='top', fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))
        else:
            ax.text(1.5, max(data_to_plot[0].max(), data_to_plot[1].max()) * 0.95,
                   f'α=8の方が\n{-improvement:.2f}mm良い\n({-improvement_pct:+.1f}%)',
                   ha='center', va='top', fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcoral', alpha=0.8))

    plt.suptitle('ゲイン別モンテカルロ分布比較 (Monte Carlo Distributions by Gain)',
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()

    output_path2 = Path("/home/akira/mosaik-hils/figures_for_paper/gain_comparison_distributions.png")
    save_figure_both_sizes(plt, output_path2.parent, base_name=output_path2.stem)
    print(f"分布比較図を保存: {output_path2}")

    # Figure 3: Performance difference (gain10 - gain8)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: Absolute difference
    ax1 = axes[0]
    rmse_diff = [stats[gain10][n]["rmse_mean"] - stats[gain8][n]["rmse_mean"] for n in noise_levels]
    mae_diff = [stats[gain10][n]["mae_mean"] - stats[gain8][n]["mae_mean"] for n in noise_levels]
    maxdev_diff = [stats[gain10][n]["max_dev_mean"] - stats[gain8][n]["max_dev_mean"] for n in noise_levels]

    x = np.arange(len(noise_levels))
    width = 0.25

    bars1 = ax1.bar(x - width, rmse_diff, width, label='RMSE', color='steelblue', alpha=0.8)
    bars2 = ax1.bar(x, mae_diff, width, label='MAE', color='orange', alpha=0.8)
    bars3 = ax1.bar(x + width, maxdev_diff, width, label='Max Dev', color='green', alpha=0.8)

    ax1.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
    ax1.set_xlabel('Plant Noise Std Dev [ms]', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Performance Difference [mm]\n(Positive = α=10 better)', fontsize=12, fontweight='bold')
    ax1.set_title('絶対差 (α=10 - α=8)', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{int(n)}' for n in noise_levels])
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')

    # Right: Percentage difference
    ax2 = axes[1]
    rmse_pct = [(stats[gain10][n]["rmse_mean"] - stats[gain8][n]["rmse_mean"]) / stats[gain8][n]["rmse_mean"] * 100
                for n in noise_levels]
    mae_pct = [(stats[gain10][n]["mae_mean"] - stats[gain8][n]["mae_mean"]) / stats[gain8][n]["mae_mean"] * 100
               for n in noise_levels]
    maxdev_pct = [(stats[gain10][n]["max_dev_mean"] - stats[gain8][n]["max_dev_mean"]) / stats[gain8][n]["max_dev_mean"] * 100
                  for n in noise_levels]

    bars4 = ax2.bar(x - width, rmse_pct, width, label='RMSE', color='steelblue', alpha=0.8)
    bars5 = ax2.bar(x, mae_pct, width, label='MAE', color='orange', alpha=0.8)
    bars6 = ax2.bar(x + width, maxdev_pct, width, label='Max Dev', color='green', alpha=0.8)

    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
    ax2.set_xlabel('Plant Noise Std Dev [ms]', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Performance Difference [%]\n(Positive = α=10 better)', fontsize=12, fontweight='bold')
    ax2.set_title('相対差 (α=10 vs α=8)', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{int(n)}' for n in noise_levels])
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.suptitle('ゲイン差の性能への影響 (Performance Impact of Gain Difference)',
                fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()

    output_path3 = Path("/home/akira/mosaik-hils/figures_for_paper/gain_comparison_difference.png")
    save_figure_both_sizes(plt, output_path3.parent, base_name=output_path3.stem)
    print(f"性能差図を保存: {output_path3}")

    # Print final summary
    print("\n" + "="*70)
    print("結論 (Conclusions)")
    print("="*70)

    avg_rmse_gain8 = np.mean([stats[gain8][n]["rmse_mean"] for n in noise_levels])
    avg_rmse_gain10 = np.mean([stats[gain10][n]["rmse_mean"] for n in noise_levels])
    overall_improvement = ((avg_rmse_gain10 - avg_rmse_gain8) / avg_rmse_gain8 * 100)

    print(f"\n平均RMSE:")
    print(f"  Gain α=8:  {avg_rmse_gain8:.3f} mm")
    print(f"  Gain α=10: {avg_rmse_gain10:.3f} mm")
    print(f"  差分: {avg_rmse_gain10 - avg_rmse_gain8:+.3f} mm ({overall_improvement:+.1f}%)")

    if overall_improvement > 0:
        print(f"\n→ Gain α=10の方が全体的に性能が良い")
    else:
        print(f"\n→ Gain α=8の方が全体的に性能が良い")

    print("\nノイズ依存性:")
    for noise in noise_levels:
        diff = stats[gain10][noise]["rmse_mean"] - stats[gain8][noise]["rmse_mean"]
        print(f"  Noise {int(noise):2d}ms: {diff:+.3f} mm")

    print("="*70)

    plt.show()


if __name__ == "__main__":
    main()
