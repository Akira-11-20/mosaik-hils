#!/usr/bin/env python3
"""
Compare linear tau model sweep results against RT baseline.

This script:
1. Loads RT baseline and multiple linear tau model scenarios
2. Computes error metrics (RMSE, MAE, Max Error)
3. Generates comparison plots (all tau values in one plot)
4. Saves summary statistics to text and CSV files
"""

import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path to import plot_config
sys.path.insert(0, str(Path(__file__).parent.parent))
from plot_config import (
    BASELINE_DEVIATION_STYLE,
    BASELINE_STYLE,
    FIGURE_SETTINGS,
    FONT_SETTINGS,
    GRID_SETTINGS,
    get_scenario_style,
)


def load_hdf5_data(file_path):
    """Load simulation data from HDF5 file."""
    data = {}
    with h5py.File(file_path, "r") as f:
        # Load time data
        data["time_s"] = f["time"]["time_s"][:]
        data["time_ms"] = f["time"]["time_ms"][:]

        # Find environment simulator group
        env_group = None
        for key in f.keys():
            if "Spacecraft1DOF" in key:
                env_group = key
                break

        if env_group:
            data["position"] = f[env_group]["position"][:]
            data["velocity"] = f[env_group]["velocity"][:]
            data["acceleration"] = f[env_group]["acceleration"][:]

        # Find plant simulator group
        plant_group = None
        for key in f.keys():
            if "ThrustStand" in key:
                plant_group = key
                break

        if plant_group:
            data["measured_thrust"] = f[plant_group]["measured_thrust"][:]

    return data


def compute_error_metrics(data, baseline):
    """Compute RMSE, MAE, and Max Error for position and velocity."""
    # Ensure data lengths match
    min_len = min(len(data["position"]), len(baseline["position"]))

    pos_data = data["position"][:min_len]
    pos_baseline = baseline["position"][:min_len]
    vel_data = data["velocity"][:min_len]
    vel_baseline = baseline["velocity"][:min_len]

    # Position errors
    pos_error = pos_data - pos_baseline
    pos_rmse = np.sqrt(np.mean(pos_error**2))
    pos_mae = np.mean(np.abs(pos_error))
    pos_max_error = np.max(np.abs(pos_error))

    # Velocity errors
    vel_error = vel_data - vel_baseline
    vel_rmse = np.sqrt(np.mean(vel_error**2))
    vel_mae = np.mean(np.abs(vel_error))
    vel_max_error = np.max(np.abs(vel_error))

    return {
        "pos_rmse": pos_rmse,
        "pos_mae": pos_mae,
        "pos_max_error": pos_max_error,
        "vel_rmse": vel_rmse,
        "vel_mae": vel_mae,
        "vel_max_error": vel_max_error,
    }


def plot_comparison(scenarios, baseline_data, output_file, title_suffix=""):
    """Create 4-row comparison plot."""
    fig, axes = plt.subplots(4, 1, figsize=(12, 16))

    # Row 1: Position Trajectory
    ax = axes[0]
    # Draw baseline first (underneath)
    if baseline_data is not None:
        ax.plot(
            baseline_data["time_s"],
            baseline_data["position"],
            **BASELINE_STYLE,
        )
    for i, scenario in enumerate(scenarios):
        style = get_scenario_style(i, label=scenario["label"])
        ax.plot(scenario["data"]["time_s"], scenario["data"]["position"], **style)
    ax.set_xlabel("Time [s]", fontsize=FONT_SETTINGS["label_size"], fontweight=FONT_SETTINGS["label_weight"])
    ax.set_ylabel("Position [m]", fontsize=FONT_SETTINGS["label_size"], fontweight=FONT_SETTINGS["label_weight"])
    ax.set_title(
        f"(a) Position Trajectory Comparison{title_suffix}",
        fontsize=FONT_SETTINGS["title_size"],
        fontweight=FONT_SETTINGS["title_weight"],
    )
    ax.legend(fontsize=FONT_SETTINGS["legend_size"])
    ax.grid(True, alpha=GRID_SETTINGS["alpha"])

    # Row 2: Position Deviation from Baseline
    ax = axes[1]
    # Draw baseline from (0,0) first (underneath)
    if baseline_data is not None:
        max_time = baseline_data["time_s"][-1]
        ax.plot([0, max_time], [0, 0], **BASELINE_DEVIATION_STYLE)
    for i, scenario in enumerate(scenarios):
        style = get_scenario_style(i, label=scenario["label"])
        min_len = min(len(scenario["data"]["position"]), len(baseline_data["position"]))
        deviation = scenario["data"]["position"][:min_len] - baseline_data["position"][:min_len]
        ax.plot(scenario["data"]["time_s"][:min_len], deviation, **style)
    ax.set_xlabel("Time [s]", fontsize=FONT_SETTINGS["label_size"], fontweight=FONT_SETTINGS["label_weight"])
    ax.set_ylabel(
        "Position Deviation [m]", fontsize=FONT_SETTINGS["label_size"], fontweight=FONT_SETTINGS["label_weight"]
    )
    ax.set_title(
        f"(b) Position Deviation from RT Baseline{title_suffix}",
        fontsize=FONT_SETTINGS["title_size"],
        fontweight=FONT_SETTINGS["title_weight"],
    )
    ax.legend(fontsize=FONT_SETTINGS["legend_size"], loc="upper right")
    ax.grid(True, alpha=GRID_SETTINGS["alpha"])

    # Row 3: Velocity Trajectory
    ax = axes[2]
    # Draw baseline first (underneath)
    if baseline_data is not None:
        ax.plot(
            baseline_data["time_s"],
            baseline_data["velocity"],
            **BASELINE_STYLE,
        )
    for i, scenario in enumerate(scenarios):
        style = get_scenario_style(i, label=scenario["label"])
        ax.plot(scenario["data"]["time_s"], scenario["data"]["velocity"], **style)
    ax.set_xlabel("Time [s]", fontsize=FONT_SETTINGS["label_size"], fontweight=FONT_SETTINGS["label_weight"])
    ax.set_ylabel("Velocity [m/s]", fontsize=FONT_SETTINGS["label_size"], fontweight=FONT_SETTINGS["label_weight"])
    ax.set_title(
        f"(c) Velocity Trajectory Comparison{title_suffix}",
        fontsize=FONT_SETTINGS["title_size"],
        fontweight=FONT_SETTINGS["title_weight"],
    )
    ax.legend(fontsize=FONT_SETTINGS["legend_size"])
    ax.grid(True, alpha=GRID_SETTINGS["alpha"])

    # Row 4: Velocity Deviation from Baseline
    ax = axes[3]
    # Draw baseline from (0,0) first (underneath)
    if baseline_data is not None:
        max_time = baseline_data["time_s"][-1]
        ax.plot([0, max_time], [0, 0], **BASELINE_DEVIATION_STYLE)
    for i, scenario in enumerate(scenarios):
        style = get_scenario_style(i, label=scenario["label"])
        min_len = min(len(scenario["data"]["velocity"]), len(baseline_data["velocity"]))
        deviation = scenario["data"]["velocity"][:min_len] - baseline_data["velocity"][:min_len]
        ax.plot(scenario["data"]["time_s"][:min_len], deviation, **style)
    ax.set_xlabel("Time [s]", fontsize=FONT_SETTINGS["label_size"], fontweight=FONT_SETTINGS["label_weight"])
    ax.set_ylabel(
        "Velocity Deviation [m/s]", fontsize=FONT_SETTINGS["label_size"], fontweight=FONT_SETTINGS["label_weight"]
    )
    ax.set_title(
        f"(d) Velocity Deviation from RT Baseline{title_suffix}",
        fontsize=FONT_SETTINGS["title_size"],
        fontweight=FONT_SETTINGS["title_weight"],
    )
    ax.legend(fontsize=FONT_SETTINGS["legend_size"], loc="upper right")
    ax.grid(True, alpha=GRID_SETTINGS["alpha"])

    plt.tight_layout()
    plt.savefig(output_file, **FIGURE_SETTINGS)
    plt.close()
    print(f"Saved plot: {output_file}")


def main():
    # Define base directory
    base_dir = Path(__file__).parent

    # Define baseline and scenario directories
    baseline_dir = base_dir / "20251120-145747_baseline_rt"
    scenario_dirs = [
        {"dir": base_dir / "20251120-145820_delay0ms_nocomp_tau50ms_linear", "tau": 50.0},
        {"dir": base_dir / "20251120-145854_delay0ms_nocomp_tau100ms_linear", "tau": 100.0},
        {"dir": base_dir / "20251120-145927_delay0ms_nocomp_tau150ms_linear", "tau": 150.0},
        {"dir": base_dir / "20251120-150001_delay0ms_nocomp_tau200ms_linear", "tau": 200.0},
    ]

    # Load baseline data
    print(f"Loading baseline from: {baseline_dir}")
    baseline_data = load_hdf5_data(baseline_dir / "hils_data.h5")
    print(f"  Final position: {baseline_data['position'][-1]:.6f} m")
    print(f"  Final velocity: {baseline_data['velocity'][-1]:.6f} m/s")

    # Load scenario data and compute metrics
    scenarios = []
    for scenario_info in scenario_dirs:
        dir_path = scenario_info["dir"]
        tau = scenario_info["tau"]

        print(f"\nLoading tau={tau}ms from: {dir_path.name}")
        data = load_hdf5_data(dir_path / "hils_data.h5")
        metrics = compute_error_metrics(data, baseline_data)

        scenarios.append(
            {
                "dir": dir_path,
                "tau": tau,
                "data": data,
                "metrics": metrics,
                "label": f"τ = {tau}ms (Linear)",
            }
        )

        print(f"  Position RMSE: {metrics['pos_rmse']:.6f} m")
        print(f"  Position MAE:  {metrics['pos_mae']:.6f} m")
        print(f"  Position Max Error: {metrics['pos_max_error']:.6f} m")
        print(f"  Velocity RMSE: {metrics['vel_rmse']:.6f} m/s")
        print(f"  Velocity MAE:  {metrics['vel_mae']:.6f} m/s")
        print(f"  Velocity Max Error: {metrics['vel_max_error']:.6f} m/s")

    # Generate comparison plot (all scenarios in one plot)
    print("\nGenerating comparison plot...")

    plot_comparison(
        scenarios,
        baseline_data,
        base_dir / "linear_tau_sweep_comparison.png",
        "",
    )

    # Save summary statistics
    print("\nSaving summary statistics...")
    summary_file = base_dir / "comparison_summary.txt"
    with open(summary_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("LINEAR TAU MODEL SWEEP COMPARISON: SUMMARY STATISTICS\n")
        f.write("=" * 80 + "\n\n")

        f.write("RT Baseline:\n")
        f.write(f"  Directory: {baseline_dir.name}\n")
        f.write(f"  Final Position: {baseline_data['position'][-1]:.6f} m\n")
        f.write(f"  Final Velocity: {baseline_data['velocity'][-1]:.6f} m/s\n")
        f.write("-" * 80 + "\n\n")

        for scenario in scenarios:
            f.write(f"Plant Tau = {scenario['tau']}ms (Linear Model):\n")
            f.write(f"  Directory: {scenario['dir'].name}\n")
            f.write("  Position vs RT:\n")
            f.write(f"    RMSE: {scenario['metrics']['pos_rmse']:.6f} m\n")
            f.write(f"    MAE:  {scenario['metrics']['pos_mae']:.6f} m\n")
            f.write(f"    Max Deviation: {scenario['metrics']['pos_max_error']:.6f} m\n")
            f.write("  Velocity vs RT:\n")
            f.write(f"    RMSE: {scenario['metrics']['vel_rmse']:.6f} m/s\n")
            f.write(f"    MAE:  {scenario['metrics']['vel_mae']:.6f} m/s\n")
            f.write(f"    Max Deviation: {scenario['metrics']['vel_max_error']:.6f} m/s\n")
            f.write("\n")

        f.write("=" * 80 + "\n")

    print(f"Saved summary: {summary_file}")

    # Save metrics to CSV
    csv_file = base_dir / "comparison_metrics.csv"
    with open(csv_file, "w") as f:
        f.write("Plant_Tau[ms],Pos_RMSE[m],Pos_MAE[m],Pos_MaxErr[m],Vel_RMSE[m/s],Vel_MAE[m/s],Vel_MaxErr[m/s]\n")
        for scenario in scenarios:
            f.write(
                f"{scenario['tau']:.0f},"
                f"{scenario['metrics']['pos_rmse']:.6f},"
                f"{scenario['metrics']['pos_mae']:.6f},"
                f"{scenario['metrics']['pos_max_error']:.6f},"
                f"{scenario['metrics']['vel_rmse']:.6f},"
                f"{scenario['metrics']['vel_mae']:.6f},"
                f"{scenario['metrics']['vel_max_error']:.6f}\n"
            )

    print(f"Saved CSV: {csv_file}")
    print("\nDone!")


if __name__ == "__main__":
    main()
