#!/usr/bin/env python3
"""
Compare mosaik and Pure Python simulation results.

This script compares position, velocity, and thrust data between
mosaik (with communication delays) and Pure Python (no Mosaik overhead) scenarios.
"""

import argparse
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


def load_mosaik_data(mosaik_file: Path) -> dict:
    """Load data from mosaik simulation HDF5 file."""
    data = {}

    with h5py.File(mosaik_file, "r") as f:
        # Load time
        data["time"] = f["time"]["time_s"][:]

        # Load environment data (position, velocity)
        env_group = f["EnvSim-0_Spacecraft1DOF_0"]
        data["position"] = env_group["position"][:]
        data["velocity"] = env_group["velocity"][:]

        # Load plant data (thrust)
        plant_group = f["PlantSim-0_ThrustStand_0"]
        data["thrust"] = plant_group["measured_thrust"][:]

        # Load controller data (for comparison)
        ctrl_group = f["ControllerSim-0_PIDController_0"]
        data["error"] = ctrl_group["error"][:]

    return data


def load_pure_data(pure_file: Path) -> dict:
    """Load data from Pure Python simulation HDF5 file."""
    data = {}

    with h5py.File(pure_file, "r") as f:
        # Load time
        data["time"] = f["data"]["time_s"][:]

        # Load environment data
        data["position"] = f["data"]["position_Spacecraft"][:]
        data["velocity"] = f["data"]["velocity_Spacecraft"][:]

        # Load plant data (thrust)
        data["thrust"] = f["data"]["measured_thrust_Plant"][:]

        # Load controller data
        data["error"] = f["data"]["error_Controller"][:]

    return data


def calculate_metrics(mosaik_data: dict, pure_data: dict) -> dict:
    """Calculate comparison metrics between mosaik and Pure Python."""
    metrics = {}

    # Ensure same time length for comparison
    min_len = min(len(mosaik_data["time"]), len(pure_data["time"]))

    for key in ["position", "velocity"]:
        mosaik_val = mosaik_data[key][:min_len]
        pure_val = pure_data[key][:min_len]

        diff = mosaik_val - pure_val

        metrics[key] = {
            "rmse": np.sqrt(np.mean(diff**2)),
            "mae": np.mean(np.abs(diff)),
            "max_error": np.max(np.abs(diff)),
            "mean_diff": np.mean(diff),
            "std_diff": np.std(diff),
        }

    return metrics


def plot_comparison(mosaik_data: dict, pure_data: dict, output_dir: Path):
    """Generate comparison plots."""

    # Ensure same time length
    min_len = min(len(mosaik_data["time"]), len(pure_data["time"]))
    time_mosaik = mosaik_data["time"][:min_len]
    time_pure = pure_data["time"][:min_len]

    fig, axes = plt.subplots(4, 1, figsize=(12, 14))
    # fig.suptitle('mosaik vs Pure Python Comparison', fontsize=16, fontweight='bold')

    # Position comparison
    ax = axes[0]
    ax.plot(time_mosaik, mosaik_data["position"][:min_len], "b-", label="mosaik", linewidth=1.5)
    ax.plot(time_pure, pure_data["position"][:min_len], "r--", label="Pure Python", linewidth=1.5)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Position [m]")
    ax.set_title("Position Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Position error
    ax = axes[1]
    pos_diff = mosaik_data["position"][:min_len] - pure_data["position"][:min_len]
    ax.plot(time_mosaik, pos_diff, "g-", linewidth=1.5)
    ax.axhline(y=0, color="k", linestyle="--", alpha=0.3)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Position Error [m]")
    ax.set_title("Position Error (mosaik - Pure)")
    ax.grid(True, alpha=0.3)

    # Velocity comparison
    ax = axes[2]
    ax.plot(time_mosaik, mosaik_data["velocity"][:min_len], "b-", label="mosaik", linewidth=1.5)
    ax.plot(time_pure, pure_data["velocity"][:min_len], "r--", label="Pure Python", linewidth=1.5)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Velocity [m/s]")
    ax.set_title("Velocity Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Velocity error
    ax = axes[3]
    vel_diff = mosaik_data["velocity"][:min_len] - pure_data["velocity"][:min_len]
    ax.plot(time_mosaik, vel_diff, "g-", linewidth=1.5)
    ax.axhline(y=0, color="k", linestyle="--", alpha=0.3)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Velocity Error [m/s]")
    ax.set_title("Velocity Error (mosaik - Pure)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    output_file = output_dir / "mosaik_vs_pure_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Comparison plot saved to: {output_file}")
    plt.close()


def plot_error_metrics(metrics: dict, output_dir: Path):
    """Plot error metrics bar chart."""

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    # fig.suptitle('Error Metrics: mosaik vs Pure Python', fontsize=14, fontweight='bold')

    variables = ["position", "velocity"]
    metric_names = ["RMSE", "MAE", "Max Error"]
    metric_keys = ["rmse", "mae", "max_error"]

    for idx, (metric_name, metric_key) in enumerate(zip(metric_names, metric_keys)):
        ax = axes[idx]
        values = [metrics[var][metric_key] for var in variables]

        bars = ax.bar(variables, values, color=["#3498db", "#e74c3c"])
        ax.set_ylabel("Error")
        ax.set_title(metric_name)
        ax.grid(True, alpha=0.3, axis="y")

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, height, f"{height:.4e}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()

    # Save plot
    output_file = output_dir / "error_metrics_bar.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Error metrics plot saved to: {output_file}")
    plt.close()


def save_metrics_to_file(metrics: dict, output_dir: Path):
    """Save metrics to JSON and text files."""

    # Save as JSON
    json_file = output_dir / "comparison_metrics.json"
    with open(json_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {json_file}")

    # Save as formatted text
    txt_file = output_dir / "comparison_metrics.txt"
    with open(txt_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("mosaik vs Pure Python Comparison Metrics\n")
        f.write("=" * 80 + "\n\n")

        for var in ["position", "velocity"]:
            f.write(f"{var.upper()}\n")
            f.write("-" * 40 + "\n")
            for metric_name, value in metrics[var].items():
                f.write(f"  {metric_name:15s}: {value:12.6e}\n")
            f.write("\n")

    print(f"Metrics summary saved to: {txt_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare mosaik and Pure Python simulation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare specific runs
  python compare_mosaik_pure.py \\
    --mosaik results/20251110-140751/mosaik_data.h5 \\
    --pure results_pure/20251110-140745/mosaik_data.h5

  # Auto-detect latest runs
  python compare_mosaik_pure.py --auto
        """,
    )

    parser.add_argument("--mosaik", type=str, help="Path to mosaik HDF5 file")
    parser.add_argument("--pure", type=str, help="Path to Pure Python HDF5 file")
    parser.add_argument("--auto", action="store_true", help="Auto-detect latest mosaik and Pure Python results")
    parser.add_argument(
        "--output", type=str, default="results/mosaik_pure_comparison", help="Output directory for comparison results"
    )

    args = parser.parse_args()

    # Determine input files
    if args.auto:
        # Find latest mosaik and Pure Python results
        base_dir = Path(__file__).parent.parent.parent
        mosaik_dirs = sorted((base_dir / "results").glob("2025*"), reverse=True)
        pure_dirs = sorted((base_dir / "results_pure").glob("2025*"), reverse=True)

        if not mosaik_dirs or not pure_dirs:
            print("Error: Could not find mosaik or Pure Python results directories")
            return

        mosaik_file = mosaik_dirs[0] / "mosaik_data.h5"
        pure_file = pure_dirs[0] / "mosaik_data.h5"

        print(f"Auto-detected mosaik: {mosaik_dirs[0].name}")
        print(f"Auto-detected Pure: {pure_dirs[0].name}")
    else:
        if not args.mosaik or not args.pure:
            parser.print_help()
            print("\nError: Either provide --mosaik and --pure, or use --auto")
            return

        mosaik_file = Path(args.mosaik)
        pure_file = Path(args.pure)

    # Check files exist
    if not mosaik_file.exists():
        print(f"Error: mosaik file not found: {mosaik_file}")
        return

    if not pure_file.exists():
        print(f"Error: Pure Python file not found: {pure_file}")
        return

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("Loading data...")
    print("=" * 80)

    # Load data
    mosaik_data = load_mosaik_data(mosaik_file)
    pure_data = load_pure_data(pure_file)

    print(f"mosaik data points: {len(mosaik_data['time'])}")
    print(f"Pure data points: {len(pure_data['time'])}")

    # Calculate metrics
    print("\n" + "=" * 80)
    print("Calculating metrics...")
    print("=" * 80)

    metrics = calculate_metrics(mosaik_data, pure_data)

    # Print metrics to console
    print("\nComparison Metrics:")
    print("-" * 80)
    for var in ["position", "velocity"]:
        print(f"\n{var.upper()}:")
        for metric_name, value in metrics[var].items():
            print(f"  {metric_name:15s}: {value:12.6e}")

    # Save metrics
    save_metrics_to_file(metrics, output_dir)

    # Generate plots
    print("\n" + "=" * 80)
    print("Generating plots...")
    print("=" * 80)

    plot_comparison(mosaik_data, pure_data, output_dir)
    plot_error_metrics(metrics, output_dir)

    print("\n" + "=" * 80)
    print("Comparison complete!")
    print("=" * 80)
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
