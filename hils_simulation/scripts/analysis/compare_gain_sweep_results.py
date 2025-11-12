#!/usr/bin/env python3
"""
Compare inverse compensation gain (alpha) sweep results with baseline.

This script compares inverse compensation gain sweep simulation results
against a baseline case. Analyzes position, velocity, and thrust differences.
"""

import argparse
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


def extract_gain_from_config(config_file: Path) -> float:
    """Extract inverse compensation gain from simulation config JSON."""
    try:
        with open(config_file) as f:
            config = json.load(f)
            gain = config.get("inverse_compensation", {}).get("gain")
            return float(gain) if gain is not None else None
    except Exception as e:
        print(f"Warning: Could not read config file {config_file}: {e}")
        return None


def load_simulation_data(h5_file: Path) -> dict:
    """Load simulation data from HDF5 file."""
    data = {}

    with h5py.File(h5_file, "r") as f:
        # Load time
        data["time"] = f["time"]["time_s"][:]

        # Load environment data
        env_group = f["EnvSim-0_Spacecraft1DOF_0"]
        data["position"] = env_group["position"][:]
        data["velocity"] = env_group["velocity"][:]
        data["thrust"] = env_group["force"][:]  # Actual thrust applied to spacecraft

    return data


def calculate_metrics(test_data: dict, baseline_data: dict) -> dict:
    """Calculate comparison metrics."""
    metrics = {}

    min_len = min(len(test_data["time"]), len(baseline_data["time"]))

    for key in ["position", "velocity", "thrust"]:
        test_val = test_data[key][:min_len]
        baseline_val = baseline_data[key][:min_len]

        diff = test_val - baseline_val

        metrics[key] = {
            "rmse": np.sqrt(np.mean(diff**2)),
            "mae": np.mean(np.abs(diff)),
            "max_error": np.max(np.abs(diff)),
            "mean_diff": np.mean(diff),
            "std_diff": np.std(diff),
        }

    return metrics


def plot_gain_sweep_comparison(baseline_data: dict, sweep_data: dict, output_dir: Path):
    """Generate comparison plots for all gain sweep cases."""

    # Sort by gain
    sorted_cases = sorted(sweep_data.items(), key=lambda x: x[1]["gain"])

    # Create figure with 6 subplots (vertical layout)
    fig, axes = plt.subplots(6, 1, figsize=(12, 24))

    # Define colors for different gain cases
    colors = ["#e74c3c", "#f39c12", "#f1c40f", "#2ecc71", "#3498db", "#9b59b6", "#e91e63"]

    # Get time array
    time = baseline_data["time"]

    # Position comparison
    ax = axes[0]
    ax.plot(time, baseline_data["position"], "k-", label="Baseline (gain=11.0)", linewidth=2.0)
    for idx, (case_name, case_data) in enumerate(sorted_cases):
        min_len = min(len(time), len(case_data["data"]["time"]))
        gain = case_data["gain"]
        color = colors[idx % len(colors)]
        ax.plot(
            time[:min_len],
            case_data["data"]["position"][:min_len],
            "--",
            color=color,
            label=f"α={gain:.1f}",
            linewidth=1.5,
        )
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Position [m]")
    ax.set_title("Position Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Position error
    ax = axes[1]
    for idx, (case_name, case_data) in enumerate(sorted_cases):
        min_len = min(len(time), len(case_data["data"]["time"]))
        gain = case_data["gain"]
        color = colors[idx % len(colors)]
        pos_diff = case_data["data"]["position"][:min_len] - baseline_data["position"][:min_len]
        ax.plot(time[:min_len], pos_diff, "-", color=color, label=f"α={gain:.1f}", linewidth=1.5)
    ax.axhline(y=0, color="k", linestyle="--", alpha=0.3)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Position Error [m]")
    ax.set_title("Position Error (Test - Baseline)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Velocity comparison
    ax = axes[2]
    ax.plot(time, baseline_data["velocity"], "k-", label="Baseline (gain=11.0)", linewidth=2.0)
    for idx, (case_name, case_data) in enumerate(sorted_cases):
        min_len = min(len(time), len(case_data["data"]["time"]))
        gain = case_data["gain"]
        color = colors[idx % len(colors)]
        ax.plot(
            time[:min_len],
            case_data["data"]["velocity"][:min_len],
            "--",
            color=color,
            label=f"α={gain:.1f}",
            linewidth=1.5,
        )
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Velocity [m/s]")
    ax.set_title("Velocity Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Velocity error
    ax = axes[3]
    for idx, (case_name, case_data) in enumerate(sorted_cases):
        min_len = min(len(time), len(case_data["data"]["time"]))
        gain = case_data["gain"]
        color = colors[idx % len(colors)]
        vel_diff = case_data["data"]["velocity"][:min_len] - baseline_data["velocity"][:min_len]
        ax.plot(time[:min_len], vel_diff, "-", color=color, label=f"α={gain:.1f}", linewidth=1.5)
    ax.axhline(y=0, color="k", linestyle="--", alpha=0.3)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Velocity Error [m/s]")
    ax.set_title("Velocity Error (Test - Baseline)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Thrust comparison
    ax = axes[4]
    ax.plot(time, baseline_data["thrust"], "k-", label="Baseline (gain=11.0)", linewidth=2.0)
    for idx, (case_name, case_data) in enumerate(sorted_cases):
        min_len = min(len(time), len(case_data["data"]["time"]))
        gain = case_data["gain"]
        color = colors[idx % len(colors)]
        ax.plot(
            time[:min_len],
            case_data["data"]["thrust"][:min_len],
            "--",
            color=color,
            label=f"α={gain:.1f}",
            linewidth=1.5,
        )
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Thrust [N]")
    ax.set_title("Thrust Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Thrust error
    ax = axes[5]
    for idx, (case_name, case_data) in enumerate(sorted_cases):
        min_len = min(len(time), len(case_data["data"]["time"]))
        gain = case_data["gain"]
        color = colors[idx % len(colors)]
        thrust_diff = case_data["data"]["thrust"][:min_len] - baseline_data["thrust"][:min_len]
        ax.plot(time[:min_len], thrust_diff, "-", color=color, label=f"α={gain:.1f}", linewidth=1.5)
    ax.axhline(y=0, color="k", linestyle="--", alpha=0.3)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Thrust Error [N]")
    ax.set_title("Thrust Error (Test - Baseline)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_file = output_dir / "gain_sweep_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Gain sweep comparison plot saved to: {output_file}")
    plt.close()


def plot_metrics_summary(all_metrics: dict, output_dir: Path):
    """Plot summary of metrics across all gain values."""

    # Sort by gain
    sorted_cases = sorted(all_metrics.items(), key=lambda x: x[1]["gain"])

    gains = [item[1]["gain"] for item in sorted_cases]

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))

    metrics_to_plot = [
        ("position", "rmse", "Position RMSE"),
        ("position", "mae", "Position MAE"),
        ("position", "max_error", "Position Max Error"),
        ("velocity", "rmse", "Velocity RMSE"),
        ("velocity", "mae", "Velocity MAE"),
        ("velocity", "max_error", "Velocity Max Error"),
        ("thrust", "rmse", "Thrust RMSE"),
        ("thrust", "mae", "Thrust MAE"),
        ("thrust", "max_error", "Thrust Max Error"),
    ]

    for idx, (var, metric_key, title) in enumerate(metrics_to_plot):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]

        values = [item[1]["metrics"][var][metric_key] for item in sorted_cases]

        ax.plot(gains, values, "bo-", linewidth=2, markersize=8)
        ax.set_xlabel("Inverse Compensation Gain (α)")
        ax.set_ylabel("Error")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        # Add value labels
        for x, y in zip(gains, values):
            ax.text(x, y, f"{y:.2e}", fontsize=8, ha="center", va="bottom")

    plt.tight_layout()

    output_file = output_dir / "gain_metrics_summary.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Metrics summary plot saved to: {output_file}")
    plt.close()


def save_summary_report(baseline_data: dict, all_metrics: dict, output_dir: Path):
    """Save comprehensive summary report."""

    # Sort by gain
    sorted_cases = sorted(all_metrics.items(), key=lambda x: x[1]["gain"])

    # Save JSON
    json_file = output_dir / "gain_sweep_comparison.json"
    json_data = {
        "baseline": {
            "data_points": len(baseline_data["time"]),
        },
        "cases": {},
    }

    for case_name, case_info in sorted_cases:
        json_data["cases"][case_name] = {"gain": case_info["gain"], "metrics": case_info["metrics"]}

    with open(json_file, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"JSON summary saved to: {json_file}")

    # Save text report
    txt_file = output_dir / "gain_sweep_comparison.txt"
    with open(txt_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("Inverse Compensation Gain Sweep Results Comparison Summary\n")
        f.write("=" * 80 + "\n\n")

        f.write("Baseline: gain=11.0 (default)\n")
        f.write(f"Data points: {len(baseline_data['time'])}\n\n")

        f.write("=" * 80 + "\n")
        f.write("Gain Cases Comparison\n")
        f.write("=" * 80 + "\n\n")

        for case_name, case_info in sorted_cases:
            f.write(f"\nCase: {case_name}\n")
            f.write("-" * 80 + "\n")
            f.write(f"Inverse Compensation Gain (α): {case_info['gain']:5.1f}\n\n")

            for var in ["position", "velocity", "thrust"]:
                f.write(f"{var.upper()}:\n")
                for metric_name, value in case_info["metrics"][var].items():
                    f.write(f"  {metric_name:15s}: {value:12.6e}\n")
                f.write("\n")

    print(f"Text summary saved to: {txt_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare inverse compensation gain sweep results against baseline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python compare_gain_sweep_results.py results/20251111-134052_sweep
        """,
    )

    parser.add_argument("sweep_dir", type=str, help="Path to sweep results directory")

    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir)
    if not sweep_dir.exists():
        print(f"Error: Sweep directory not found: {sweep_dir}")
        return

    # Find baseline and test cases
    baseline_dir = None
    test_dirs = []

    for subdir in sorted(sweep_dir.iterdir()):
        if not subdir.is_dir():
            continue

        if "baseline" in subdir.name.lower():
            baseline_dir = subdir
        else:
            test_dirs.append(subdir)

    if baseline_dir is None:
        print("Error: No baseline directory found")
        return

    if not test_dirs:
        print("Error: No test case directories found")
        return

    print("\n" + "=" * 80)
    print("Loading data...")
    print("=" * 80)

    # Load baseline
    baseline_file = baseline_dir / "hils_data.h5"
    baseline_config = baseline_dir / "simulation_config.json"

    if not baseline_file.exists():
        print(f"Error: Baseline HDF5 file not found: {baseline_file}")
        return

    baseline_data = load_simulation_data(baseline_file)
    baseline_gain = extract_gain_from_config(baseline_config)

    print(f"Baseline loaded: {baseline_dir.name}")
    print(f"  Data points: {len(baseline_data['time'])}")
    print(f"  Baseline gain: {baseline_gain}")

    # Load test cases
    sweep_data = {}
    all_metrics = {}

    for test_dir in test_dirs:
        test_file = test_dir / "hils_data.h5"
        test_config = test_dir / "simulation_config.json"

        if not test_file.exists():
            print(f"Warning: Skipping {test_dir.name} (no HDF5 file)")
            continue

        test_data = load_simulation_data(test_file)
        gain = extract_gain_from_config(test_config)

        if gain is None:
            print(f"Warning: Could not extract gain from {test_dir.name}")
            continue

        sweep_data[test_dir.name] = {
            "data": test_data,
            "gain": gain,
        }

        # Calculate metrics
        metrics = calculate_metrics(test_data, baseline_data)
        all_metrics[test_dir.name] = {
            "gain": gain,
            "metrics": metrics,
        }

        print(f"Loaded: {test_dir.name}")
        print(f"  Gain (α): {gain}")

    print(f"\nTotal test cases loaded: {len(sweep_data)}")

    # Create output directory in sweep directory
    output_dir = sweep_dir / "comparison_analysis"
    output_dir.mkdir(exist_ok=True)

    print("\n" + "=" * 80)
    print("Calculating metrics...")
    print("=" * 80)

    # Print metrics summary
    sorted_cases = sorted(all_metrics.items(), key=lambda x: x[1]["gain"])

    for case_name, case_info in sorted_cases:
        print(f"\n{case_name}:")
        print(f"  Gain (α): {case_info['gain']}")
        for var in ["position", "velocity", "thrust"]:
            rmse = case_info["metrics"][var]["rmse"]
            mae = case_info["metrics"][var]["mae"]
            print(f"  {var}: RMSE={rmse:.6e}, MAE={mae:.6e}")

    print("\n" + "=" * 80)
    print("Generating plots...")
    print("=" * 80)

    plot_gain_sweep_comparison(baseline_data, sweep_data, output_dir)
    plot_metrics_summary(all_metrics, output_dir)

    print("\n" + "=" * 80)
    print("Saving summary report...")
    print("=" * 80)

    save_summary_report(baseline_data, all_metrics, output_dir)

    print("\n" + "=" * 80)
    print("Comparison complete!")
    print("=" * 80)
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
