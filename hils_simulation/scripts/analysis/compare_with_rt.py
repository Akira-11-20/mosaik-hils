"""
Compare latest simulations with RT baseline using error metrics

This script compares the most recent N simulations against an RT baseline:
- RMSE (Root Mean Square Error)
- MAE (Mean Absolute Error)
- Max Error
- Settling time comparison
- Overshoot comparison

Usage:
    uv run python scripts/analysis/compare_with_rt.py
    uv run python scripts/analysis/compare_with_rt.py --rt-dir results/20251107-143226 --num-sims 5
"""

import argparse
import json
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def load_simulation_data(h5_file_path: Path):
    """Load simulation data from HDF5 file"""
    with h5py.File(h5_file_path, "r") as f:
        data = {}

        # Check if using new group-based structure or old steps structure
        if "steps" in f.keys():
            # Old structure
            steps = f["steps"]

            # Get time
            time_cols = [col for col in steps.keys() if col.startswith("time_")]
            if time_cols:
                data["time"] = steps[time_cols[0]][:]

            # Get position
            pos_cols = [col for col in steps.keys() if "position_" in col and "Env" in col]
            if pos_cols:
                data["position"] = steps[pos_cols[0]][:]

            # Get velocity
            vel_cols = [col for col in steps.keys() if "velocity_" in col and "Env" in col]
            if vel_cols:
                data["velocity"] = steps[vel_cols[0]][:]

            # Get thrust/force
            thrust_cols = [col for col in steps.keys() if "measured_thrust_" in col and "Plant" in col]
            if thrust_cols:
                data["thrust"] = steps[thrust_cols[0]][:]

            # Get control command (if available)
            cmd_cols = [col for col in steps.keys() if "command_" in col and "Controller" in col]
            if cmd_cols:
                data["command"] = steps[cmd_cols[0]][:]
        else:
            # New group-based structure
            # Get time
            if "time" in f:
                time_group = f["time"]
                if "time_s" in time_group:
                    data["time"] = time_group["time_s"][:]
                elif "time_ms" in time_group:
                    data["time"] = time_group["time_ms"][:] / 1000.0  # Convert to seconds

            # Find Env group
            env_groups = [k for k in f.keys() if "Env" in k and "Spacecraft" in k]
            if env_groups:
                env = f[env_groups[0]]
                if "position" in env:
                    data["position"] = env["position"][:]
                if "velocity" in env:
                    data["velocity"] = env["velocity"][:]
                if "acceleration" in env:
                    data["acceleration"] = env["acceleration"][:]
                if "force" in env:
                    data["force"] = env["force"][:]

            # Find Plant group
            plant_groups = [k for k in f.keys() if "Plant" in k and "ThrustStand" in k]
            if plant_groups:
                plant = f[plant_groups[0]]
                if "measured_thrust" in plant:
                    data["thrust"] = plant["measured_thrust"][:]

            # Find Controller group
            ctrl_groups = [k for k in f.keys() if "Controller" in k and "PID" in k]
            if ctrl_groups:
                ctrl = f[ctrl_groups[0]]
                if "command" in ctrl:
                    data["command"] = ctrl["command"][:]

    return data


def load_config(result_dir: Path):
    """Load simulation configuration"""
    config_file = result_dir / "simulation_config.json"
    if config_file.exists():
        with open(config_file) as f:
            return json.load(f)
    return {}


def calculate_error_metrics(reference: np.ndarray, test: np.ndarray):
    """Calculate error metrics between reference and test signals"""
    # Ensure same length
    min_len = min(len(reference), len(test))
    ref = reference[:min_len]
    tst = test[:min_len]

    # Calculate errors
    error = tst - ref
    rmse = np.sqrt(np.mean(error**2))
    mae = np.mean(np.abs(error))
    max_error = np.max(np.abs(error))

    # Calculate percentage errors (avoid division by zero)
    ref_range = np.max(ref) - np.min(ref)
    if ref_range > 0:
        rmse_percent = (rmse / ref_range) * 100
        mae_percent = (mae / ref_range) * 100
        max_error_percent = (max_error / ref_range) * 100
    else:
        rmse_percent = mae_percent = max_error_percent = 0.0

    return {
        "rmse": rmse,
        "mae": mae,
        "max_error": max_error,
        "rmse_percent": rmse_percent,
        "mae_percent": mae_percent,
        "max_error_percent": max_error_percent,
        "error_signal": error,
    }


def calculate_settling_time(time: np.ndarray, position: np.ndarray, target: float, threshold: float = 0.02):
    """Calculate settling time (time to reach within threshold of target)"""
    target_range = target * threshold
    within_range = np.abs(position - target) <= target_range

    if not np.any(within_range):
        return None

    # Find first time it enters and stays within range
    settled_idx = np.where(within_range)[0][0]

    # Check if it stays settled (no excursions > threshold after this point)
    if np.all(within_range[settled_idx:]):
        return time[settled_idx]

    # If it leaves the range, find the last entry point
    for i in range(len(within_range) - 1, -1, -1):
        if within_range[i] and np.all(within_range[i:]):
            return time[i]

    return None


def calculate_overshoot(position: np.ndarray, target: float):
    """Calculate maximum overshoot percentage"""
    max_pos = np.max(position)
    if max_pos > target:
        overshoot = ((max_pos - target) / target) * 100
        return overshoot
    return 0.0


def find_latest_simulations(results_dir: Path, num_sims: int = 5, exclude_rt: bool = True):
    """Find the latest N simulation directories"""
    sim_dirs = sorted(
        [d for d in results_dir.iterdir() if d.is_dir()],
        key=lambda x: x.name,
        reverse=True,
    )

    if exclude_rt:
        # Filter out RT simulations (no _inverse_comp suffix and not the baseline)
        sim_dirs = [d for d in sim_dirs if "_inverse_comp" in d.name or "visualizations" not in d.name]

    return sim_dirs[:num_sims]


def create_heatmap(sim_configs, metrics_list, sim_labels, output_dir: Path):
    """Create heatmap of RMSE vs time constant and noise"""
    # Extract parameters from configs and collect all metrics for each (tau, noise) combination
    from collections import defaultdict

    data_points = []
    # Dictionary to group metrics by (tau, noise) combination
    metrics_by_params = defaultdict(lambda: {"rmse": [], "mae": [], "max_error": []})

    for config, metrics, label in zip(sim_configs, metrics_list, sim_labels):
        if not config:
            continue

        # Skip baseline configurations
        if "baseline" in label.lower():
            continue

        plant_config = config.get("plant", {})
        # Try both seconds and milliseconds formats
        tau = plant_config.get("time_constant") or plant_config.get("time_constant_s")
        noise = plant_config.get("time_constant_noise") or plant_config.get("time_constant_noise_s")

        # Convert from seconds to milliseconds if needed
        if tau is not None and tau < 1.0:  # Likely in seconds
            tau = tau * 1000.0
        if noise is not None and noise < 1.0:  # Likely in seconds
            noise = noise * 1000.0

        if tau is not None and noise is not None and "position" in metrics:
            rmse = metrics["position"]["rmse"]
            data_points.append(
                {
                    "tau": tau,
                    "noise": noise,
                    "rmse": rmse,
                    "label": label,
                }
            )

            # Collect all metrics for averaging
            param_key = (tau, noise)
            metrics_by_params[param_key]["rmse"].append(rmse)
            if "mae" in metrics["position"]:
                metrics_by_params[param_key]["mae"].append(metrics["position"]["mae"])
            if "max_error" in metrics["position"]:
                metrics_by_params[param_key]["max_error"].append(metrics["position"]["max_error"])

    if len(data_points) < 2:
        print("Not enough data points for heatmap (need tau and noise values)")
        return

    # Convert to structured data
    tau_values = sorted(set(d["tau"] for d in data_points))
    noise_values = sorted(set(d["noise"] for d in data_points))

    if len(tau_values) < 2 or len(noise_values) < 2:
        print(f"Insufficient parameter variation for heatmap (tau: {len(tau_values)}, noise: {len(noise_values)})")
        return

    # Create 2D grid and compute averages for each parameter combination
    rmse_grid = np.full((len(noise_values), len(tau_values)), np.nan)

    for (tau, noise), metric_values in metrics_by_params.items():
        tau_idx = tau_values.index(tau)
        noise_idx = noise_values.index(noise)
        # Average all RMSE values for this parameter combination
        rmse_grid[noise_idx, tau_idx] = np.mean(metric_values["rmse"])

        # Print info about averaging if multiple values exist
        if len(metric_values["rmse"]) > 1:
            print(
                f"  τ={tau:.0f}ms, noise={noise:.0f}ms: Averaging {len(metric_values['rmse'])} RMSE values ({metric_values['rmse']})"
            )
            print(f"    Average RMSE: {np.mean(metric_values['rmse']):.6f}")

    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(rmse_grid, cmap="YlOrRd", aspect="auto", origin="lower")

    # Set ticks and labels
    ax.set_xticks(np.arange(len(tau_values)))
    ax.set_yticks(np.arange(len(noise_values)))
    ax.set_xticklabels([f"{tau:.0f}" for tau in tau_values], fontsize=11)
    ax.set_yticklabels([f"{noise:.0f}" for noise in noise_values], fontsize=11)

    ax.set_xlabel("Plant Time Constant τ [ms]", fontsize=13, fontweight="bold")
    ax.set_ylabel("Plant Time Constant Noise [ms]", fontsize=13, fontweight="bold")
    ax.set_title("Position RMSE Heatmap\n(vs RT Baseline)", fontsize=15, fontweight="bold")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("RMSE [m]", fontsize=12, fontweight="bold")

    # Annotate cells with values
    for i in range(len(noise_values)):
        for j in range(len(tau_values)):
            if not np.isnan(rmse_grid[i, j]):
                text_color = "white" if rmse_grid[i, j] > (np.nanmax(rmse_grid) / 2) else "black"
                ax.text(
                    j,
                    i,
                    f"{rmse_grid[i, j]:.4f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=10,
                    fontweight="bold",
                )

    plt.tight_layout()
    plt.savefig(output_dir / "rmse_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Heatmap saved: {output_dir / 'rmse_heatmap.png'}")

    # Also create heatmap for MAE and Max Error if available
    for metric_name in ["mae", "max_error"]:
        metric_grid = np.full((len(noise_values), len(tau_values)), np.nan)

        # Use averaged metrics from metrics_by_params dictionary
        for (tau, noise), metric_values in metrics_by_params.items():
            if len(metric_values[metric_name]) > 0:
                tau_idx = tau_values.index(tau)
                noise_idx = noise_values.index(noise)
                # Average all values for this parameter combination
                metric_grid[noise_idx, tau_idx] = np.mean(metric_values[metric_name])

                # Print info about averaging if multiple values exist
                if len(metric_values[metric_name]) > 1:
                    print(
                        f"  τ={tau:.0f}ms, noise={noise:.0f}ms: Averaging {len(metric_values[metric_name])} {metric_name} values"
                    )
                    print(f"    Average {metric_name}: {np.mean(metric_values[metric_name]):.6f}")

        # Skip if no valid data
        if np.all(np.isnan(metric_grid)):
            continue

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(metric_grid, cmap="YlOrRd", aspect="auto", origin="lower")

        ax.set_xticks(np.arange(len(tau_values)))
        ax.set_yticks(np.arange(len(noise_values)))
        ax.set_xticklabels([f"{tau:.0f}" for tau in tau_values], fontsize=11)
        ax.set_yticklabels([f"{noise:.0f}" for noise in noise_values], fontsize=11)

        ax.set_xlabel("Plant Time Constant τ [ms]", fontsize=13, fontweight="bold")
        ax.set_ylabel("Plant Time Constant Noise [ms]", fontsize=13, fontweight="bold")

        title_map = {"mae": "Mean Absolute Error (MAE)", "max_error": "Maximum Error"}
        ax.set_title(
            f"Position {title_map.get(metric_name, metric_name.upper())} Heatmap\n(vs RT Baseline)",
            fontsize=15,
            fontweight="bold",
        )

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(f"{title_map.get(metric_name, metric_name.upper())} [m]", fontsize=12, fontweight="bold")

        for i in range(len(noise_values)):
            for j in range(len(tau_values)):
                if not np.isnan(metric_grid[i, j]):
                    text_color = "white" if metric_grid[i, j] > (np.nanmax(metric_grid) / 2) else "black"
                    ax.text(
                        j,
                        i,
                        f"{metric_grid[i, j]:.4f}",
                        ha="center",
                        va="center",
                        color=text_color,
                        fontsize=10,
                        fontweight="bold",
                    )

        plt.tight_layout()
        filename = f"{metric_name}_heatmap.png"
        plt.savefig(output_dir / filename, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Heatmap saved: {output_dir / filename}")


def create_comparison_plots(rt_data, sim_data_list, sim_labels, metrics_list, sim_configs, output_dir: Path):
    """Create comprehensive comparison plots"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Combined Position and Velocity comparison (4 subplots)
    fig, axes = plt.subplots(4, 1, figsize=(14, 16))

    # Use distinct colors for different simulation types
    # Define a palette with clearly distinguishable colors
    color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                     '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    colors = [color_palette[i % len(color_palette)] for i in range(len(sim_data_list))]

    # Subplot 1: Position trajectories
    ax = axes[0]
    ax.plot(rt_data["time"], rt_data["position"], "k-", linewidth=2, label="RT Baseline", alpha=0.8)
    for i, (data, label, color) in enumerate(zip(sim_data_list, sim_labels, colors)):
        min_len = min(len(rt_data["time"]), len(data["time"]))
        ax.plot(data["time"][:min_len], data["position"][:min_len], color=color, linewidth=1.5, label=label, alpha=0.7)

    ax.set_xlabel("Time [s]", fontsize=12)
    ax.set_ylabel("Position [m]", fontsize=12)
    ax.set_title("Position Trajectories: RT Baseline vs Simulations", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    # Subplot 2: Position error signals
    ax = axes[1]
    for i, (metrics, label, color) in enumerate(zip(metrics_list, sim_labels, colors)):
        min_len = min(len(rt_data["time"]), len(metrics["position"]["error_signal"]))
        ax.plot(
            rt_data["time"][:min_len],
            metrics["position"]["error_signal"][:min_len],
            color=color,
            linewidth=1.5,
            label=label,
            alpha=0.7,
        )

    ax.axhline(y=0, color="k", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlabel("Time [s]", fontsize=12)
    ax.set_ylabel("Position Error [m]", fontsize=12)
    ax.set_title("Position Error vs RT Baseline", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    # Subplot 3: Velocity trajectories
    ax = axes[2]
    ax.plot(rt_data["time"], rt_data["velocity"], "k-", linewidth=2, label="RT Baseline", alpha=0.8)
    for i, (data, label, color) in enumerate(zip(sim_data_list, sim_labels, colors)):
        min_len = min(len(rt_data["time"]), len(data["time"]))
        ax.plot(data["time"][:min_len], data["velocity"][:min_len], color=color, linewidth=1.5, label=label, alpha=0.7)

    ax.set_xlabel("Time [s]", fontsize=12)
    ax.set_ylabel("Velocity [m/s]", fontsize=12)
    ax.set_title("Velocity Trajectories: RT Baseline vs Simulations", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    # Subplot 4: Velocity error signals
    ax = axes[3]
    for i, (metrics, label, color) in enumerate(zip(metrics_list, sim_labels, colors)):
        if "velocity" in metrics:
            min_len = min(len(rt_data["time"]), len(metrics["velocity"]["error_signal"]))
            ax.plot(
                rt_data["time"][:min_len],
                metrics["velocity"]["error_signal"][:min_len],
                color=color,
                linewidth=1.5,
                label=label,
                alpha=0.7,
            )

    ax.axhline(y=0, color="k", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlabel("Time [s]", fontsize=12)
    ax.set_ylabel("Velocity Error [m/s]", fontsize=12)
    ax.set_title("Velocity Error vs RT Baseline", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / "trajectory_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Error metrics bar chart
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    x = np.arange(len(sim_labels))
    width = 0.6

    # RMSE
    rmse_values = [m["position"]["rmse"] for m in metrics_list]
    axes[0].bar(x, rmse_values, width, color=colors, alpha=0.7)
    axes[0].set_xlabel("Simulation", fontsize=11)
    axes[0].set_ylabel("RMSE [m]", fontsize=11)
    axes[0].set_title("Root Mean Square Error", fontsize=12, fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"Sim {i + 1}" for i in range(len(sim_labels))], rotation=45, ha="right")
    axes[0].grid(True, alpha=0.3, axis="y")

    # Add values on bars
    for i, v in enumerate(rmse_values):
        axes[0].text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=9)

    # MAE
    mae_values = [m["position"]["mae"] for m in metrics_list]
    axes[1].bar(x, mae_values, width, color=colors, alpha=0.7)
    axes[1].set_xlabel("Simulation", fontsize=11)
    axes[1].set_ylabel("MAE [m]", fontsize=11)
    axes[1].set_title("Mean Absolute Error", fontsize=12, fontweight="bold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"Sim {i + 1}" for i in range(len(sim_labels))], rotation=45, ha="right")
    axes[1].grid(True, alpha=0.3, axis="y")

    for i, v in enumerate(mae_values):
        axes[1].text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=9)

    # Max Error
    max_err_values = [m["position"]["max_error"] for m in metrics_list]
    axes[2].bar(x, max_err_values, width, color=colors, alpha=0.7)
    axes[2].set_xlabel("Simulation", fontsize=11)
    axes[2].set_ylabel("Max Error [m]", fontsize=11)
    axes[2].set_title("Maximum Error", fontsize=12, fontweight="bold")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([f"Sim {i + 1}" for i in range(len(sim_labels))], rotation=45, ha="right")
    axes[2].grid(True, alpha=0.3, axis="y")

    for i, v in enumerate(max_err_values):
        axes[2].text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / "error_metrics_bar.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 3. Create heatmap if we have parameter sweep data
    create_heatmap(sim_configs, metrics_list, sim_labels, output_dir)

    print(f"Comparison plots saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Compare latest simulations with RT baseline")
    parser.add_argument(
        "--rt-dir",
        type=str,
        default="results/20251107-143226",
        help="RT baseline directory (default: results/20251107-143226)",
    )
    parser.add_argument(
        "--num-sims",
        type=int,
        default=5,
        help="Number of latest simulations to compare (default: 5). Ignored if --result-dirs is specified.",
    )
    parser.add_argument(
        "--result-dirs",
        type=str,
        nargs="+",
        help="Specific result directories to compare (overrides --num-sims)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/comparison_with_rt",
        help="Output directory for comparison results (default: results/comparison_with_rt)",
    )

    args = parser.parse_args()

    # Setup paths
    base_dir = Path(__file__).parent.parent.parent
    rt_dir = base_dir / args.rt_dir
    results_dir = base_dir / "results"
    output_dir = base_dir / args.output_dir

    # Load RT baseline
    print("=" * 70)
    print("Loading RT Baseline Data")
    print("=" * 70)
    rt_h5 = rt_dir / "hils_data.h5"

    if not rt_h5.exists():
        print(f"Error: RT baseline file not found: {rt_h5}")
        return

    rt_data = load_simulation_data(rt_h5)
    rt_config = load_config(rt_dir)

    print(f"RT baseline: {rt_dir.name}")
    print(f"  Data points: {len(rt_data.get('time', []))}")
    print(f"  Duration: {rt_data['time'][-1]:.3f} s" if "time" in rt_data else "N/A")

    # Find simulations to compare
    print("\n" + "=" * 70)
    if args.result_dirs:
        print(f"Using Specified {len(args.result_dirs)} Simulation(s)")
        print("=" * 70)
        sim_dirs = [base_dir / d if not Path(d).is_absolute() else Path(d) for d in args.result_dirs]

        # Filter out RT baseline from the list
        sim_dirs = [d for d in sim_dirs if str(rt_dir) not in str(d)]

        print(f"Comparing {len(sim_dirs)} simulation(s):")
        for i, sim_dir in enumerate(sim_dirs, 1):
            print(f"  {i}. {sim_dir.name if sim_dir.exists() else sim_dir}")
            if not sim_dir.exists():
                print("     Warning: Directory does not exist!")

        # Filter to only existing directories
        sim_dirs = [d for d in sim_dirs if d.exists() and d.is_dir()]

        if not sim_dirs:
            print("Error: No valid simulation directories found")
            return
    else:
        print(f"Finding Latest {args.num_sims} Simulations")
        print("=" * 70)

        sim_dirs = find_latest_simulations(results_dir, args.num_sims, exclude_rt=True)

        if not sim_dirs:
            print(f"Error: No simulation directories found in {results_dir}")
            return

        print(f"Found {len(sim_dirs)} simulations:")
        for i, sim_dir in enumerate(sim_dirs, 1):
            print(f"  {i}. {sim_dir.name}")

    # Load simulation data
    print("\n" + "=" * 70)
    print("Loading Simulation Data")
    print("=" * 70)

    sim_data_list = []
    sim_labels = []
    sim_configs = []

    for sim_dir in sim_dirs:
        h5_file = sim_dir / "hils_data.h5"
        if not h5_file.exists():
            print(f"Warning: {sim_dir.name} has no hils_data.h5, skipping...")
            continue

        data = load_simulation_data(h5_file)
        config = load_config(sim_dir)

        sim_data_list.append(data)
        sim_labels.append(sim_dir.name)
        sim_configs.append(config)

        print(f"Loaded: {sim_dir.name}")
        print(f"  Data points: {len(data.get('time', []))}")

    # Calculate error metrics
    print("\n" + "=" * 70)
    print("Calculating Error Metrics")
    print("=" * 70)

    metrics_list = []
    results_table = []

    target_position = rt_config.get("control", {}).get("target_position", 5.0)

    for i, (data, label, config) in enumerate(zip(sim_data_list, sim_labels, sim_configs), 1):
        print(f"\nSimulation {i}: {label}")

        metrics = {}

        # Position metrics
        if "position" in data and "position" in rt_data:
            pos_metrics = calculate_error_metrics(rt_data["position"], data["position"])
            metrics["position"] = pos_metrics

            print(f"  Position RMSE: {pos_metrics['rmse']:.6f} m ({pos_metrics['rmse_percent']:.2f}%)")
            print(f"  Position MAE:  {pos_metrics['mae']:.6f} m ({pos_metrics['mae_percent']:.2f}%)")
            print(f"  Position Max Error: {pos_metrics['max_error']:.6f} m ({pos_metrics['max_error_percent']:.2f}%)")

        # Velocity metrics
        if "velocity" in data and "velocity" in rt_data:
            vel_metrics = calculate_error_metrics(rt_data["velocity"], data["velocity"])
            metrics["velocity"] = vel_metrics

            print(f"  Velocity RMSE: {vel_metrics['rmse']:.6f} m/s")
            print(f"  Velocity MAE:  {vel_metrics['mae']:.6f} m/s")

        # Settling time
        if "position" in data and "time" in data:
            settling_time = calculate_settling_time(data["time"], data["position"], target_position)
            rt_settling_time = calculate_settling_time(rt_data["time"], rt_data["position"], target_position)

            if settling_time is not None:
                print(f"  Settling time: {settling_time:.3f} s")
                if rt_settling_time is not None:
                    delta_settling = settling_time - rt_settling_time
                    print(f"    vs RT: {delta_settling:+.3f} s")
                    metrics["settling_time"] = settling_time
                    metrics["settling_time_delta"] = delta_settling

        # Overshoot
        if "position" in data:
            overshoot = calculate_overshoot(data["position"], target_position)
            rt_overshoot = calculate_overshoot(rt_data["position"], target_position)

            print(f"  Overshoot: {overshoot:.2f}%")
            print(f"    vs RT: {overshoot - rt_overshoot:+.2f}%")
            metrics["overshoot"] = overshoot
            metrics["overshoot_delta"] = overshoot - rt_overshoot

        # Configuration details
        if config:
            plant_tau = config.get("plant", {}).get("time_constant")
            plant_noise = config.get("plant", {}).get("time_constant_noise")
            comp_gain = config.get("inverse_compensation", {}).get("gain")

            print(f"  Config: τ={plant_tau}ms, noise={plant_noise}ms, gain={comp_gain}")

        metrics_list.append(metrics)

        # Build results table row
        row = {
            "Simulation": label,
            "RMSE_pos": pos_metrics["rmse"] if "position" in metrics else None,
            "MAE_pos": pos_metrics["mae"] if "position" in metrics else None,
            "Max_Error": pos_metrics["max_error"] if "position" in metrics else None,
            "RMSE_vel": vel_metrics["rmse"] if "velocity" in metrics else None,
            "Settling_Time": metrics.get("settling_time"),
            "Overshoot": metrics.get("overshoot"),
        }
        results_table.append(row)

    # Create summary table
    print("\n" + "=" * 70)
    print("Summary Table")
    print("=" * 70)

    df = pd.DataFrame(results_table)
    print(df.to_string(index=False))

    # Save summary to CSV
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_file = output_dir / "comparison_summary.csv"
    df.to_csv(csv_file, index=False)
    print(f"\nSummary saved to: {csv_file}")

    # Save detailed metrics to JSON
    json_file = output_dir / "detailed_metrics.json"
    detailed_results = {
        "rt_baseline": {
            "directory": str(rt_dir),
            "config": rt_config,
        },
        "simulations": [
            {
                "directory": str(sim_dir),
                "label": label,
                "config": config,
                "metrics": {
                    k: {
                        mk: float(mv) if isinstance(mv, (int, float, np.number)) else None
                        for mk, mv in v.items()
                        if mk != "error_signal"
                    }
                    for k, v in metrics.items()
                    if isinstance(v, dict)
                },
            }
            for sim_dir, label, config, metrics in zip(sim_dirs, sim_labels, sim_configs, metrics_list)
        ],
    }

    with open(json_file, "w") as f:
        json.dump(detailed_results, f, indent=2)

    print(f"Detailed metrics saved to: {json_file}")

    # Create comparison plots
    print("\n" + "=" * 70)
    print("Creating Comparison Plots")
    print("=" * 70)

    create_comparison_plots(rt_data, sim_data_list, sim_labels, metrics_list, sim_configs, output_dir)

    print("\n" + "=" * 70)
    print("Comparison Complete!")
    print("=" * 70)
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
