"""
Compare simulation results from delay-tau sweep experiments.

This script analyzes sweep results where both cmd_delay and tau (time constant)
are varied, comparing performance with inverse compensation.

Expected directory structure:
    results/YYYYMMDD-HHMMSS_sweep/
        ├── YYYYMMDD-HHMMSS_baseline_rt/
        ├── YYYYMMDD-HHMMSS_cmd5ms_sense0ms_comp_tau100ms/
        ├── YYYYMMDD-HHMMSS_cmd10ms_sense0ms_comp_tau100ms/
        └── ...
"""

import argparse
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def load_simulation_data(h5_file_path: Path) -> dict:
    """Load simulation data from HDF5 file."""
    with h5py.File(h5_file_path, "r") as f:
        # Load time
        time = f["time"]["time_s"][:]

        # Find spacecraft entity (support both old and new naming)
        spacecraft_keys = [k for k in f.keys() if "Spacecraft" in k]
        if not spacecraft_keys:
            raise ValueError(f"No spacecraft entity found in {h5_file_path}")
        spacecraft_key = spacecraft_keys[0]

        # Load spacecraft state
        position = f[spacecraft_key]["position"][:]
        velocity = f[spacecraft_key]["velocity"][:]

        # Load plant data
        plant_keys = [k for k in f.keys() if "ThrustStand" in k or "Plant" in k]
        if plant_keys:
            plant_key = plant_keys[0]
            measured_thrust = f[plant_key]["measured_thrust"][:]
            actual_thrust = f[plant_key]["actual_thrust"][:]
            time_constant = f[plant_key]["time_constant"][:]
        else:
            measured_thrust = None
            actual_thrust = None
            time_constant = None

        # Load controller data
        controller_keys = [k for k in f.keys() if "PIDController" in k or "Controller" in k]
        if controller_keys:
            controller_key = controller_keys[0]
            error = f[controller_key]["error"][:]
        else:
            error = None

        # Load inverse compensator data (if exists)
        # Support both old naming (InverseCompensator) and new naming (InverseCompSim)
        inversecomp_keys = [k for k in f.keys() if "InverseComp" in k or "cmd_compensator" in k]
        if inversecomp_keys:
            inversecomp_key = inversecomp_keys[0]
            try:
                # Try to load compensated output
                if "compensated_output" in f[inversecomp_key]:
                    comp_output = f[inversecomp_key]["compensated_output"][:]
                else:
                    comp_output = None

                # Try to load current_gain
                if "current_gain" in f[inversecomp_key]:
                    comp_gain = f[inversecomp_key]["current_gain"][:]
                else:
                    comp_gain = None

                # Try to load current_tau
                if "current_tau" in f[inversecomp_key]:
                    comp_tau = f[inversecomp_key]["current_tau"][:]
                else:
                    comp_tau = None
            except KeyError as e:
                print(f"⚠️  Warning: Failed to load inverse compensator data: {e}")
                comp_output = None
                comp_gain = None
                comp_tau = None
        else:
            comp_output = None
            comp_gain = None
            comp_tau = None

    return {
        "time": time,
        "position": position,
        "velocity": velocity,
        "measured_thrust": measured_thrust,
        "actual_thrust": actual_thrust,
        "time_constant": time_constant,
        "error": error,
        "comp_output": comp_output,
        "comp_gain": comp_gain,
        "comp_tau": comp_tau,
    }


def load_config(config_file: Path) -> dict:
    """Load simulation configuration from JSON file."""
    with open(config_file, "r") as f:
        return json.load(f)


def parse_sweep_dir_name(dir_name: str) -> dict:
    """
    Parse sweep directory name to extract parameters.

    Examples:
        "20251111-183844_cmd5ms_sense0ms_comp_tau100ms" -> {
            "cmd_delay": 5.0,
            "sense_delay": 0.0,
            "tau": 100.0,
            "has_comp": True
        }
        "20251111-183809_baseline_rt" -> {
            "cmd_delay": 0.0,
            "sense_delay": 0.0,
            "tau": 0.0,
            "has_comp": False
        }
    """
    import re

    if "baseline" in dir_name or "_rt" in dir_name:
        return {
            "cmd_delay": 0.0,
            "sense_delay": 0.0,
            "tau": 0.0,
            "has_comp": False,
            "is_baseline": True,
        }

    params = {
        "cmd_delay": 0.0,
        "sense_delay": 0.0,
        "tau": 0.0,
        "has_comp": False,
        "is_baseline": False,
    }

    # Extract cmd delay
    cmd_match = re.search(r"cmd(\d+(?:\.\d+)?)ms", dir_name)
    if cmd_match:
        params["cmd_delay"] = float(cmd_match.group(1))

    # Extract sense delay
    sense_match = re.search(r"sense(\d+(?:\.\d+)?)ms", dir_name)
    if sense_match:
        params["sense_delay"] = float(sense_match.group(1))

    # Extract tau
    tau_match = re.search(r"tau(\d+(?:\.\d+)?)ms", dir_name)
    if tau_match:
        params["tau"] = float(tau_match.group(1))

    # Check if compensation is enabled
    params["has_comp"] = "comp" in dir_name

    return params


def calculate_metrics(data: dict, config: dict, target_position: float = 5.0) -> dict:
    """Calculate performance metrics."""
    time = data["time"]
    position = data["position"]
    velocity = data["velocity"]
    error = data["error"]

    # Calculate RMSE (position)
    rmse = np.sqrt(np.mean((position - target_position) ** 2))

    # Calculate MAE (position)
    mae = np.mean(np.abs(position - target_position))

    # Calculate max error
    max_error = np.max(np.abs(position - target_position))

    # Calculate settling time (within 2% of target)
    settling_threshold = 0.02 * target_position
    settled_indices = np.where(np.abs(position - target_position) <= settling_threshold)[0]
    if len(settled_indices) > 0:
        settling_time = time[settled_indices[0]]
    else:
        settling_time = time[-1]  # Never settled

    # Calculate overshoot
    overshoot = (np.max(position) - target_position) / target_position * 100.0

    # Calculate steady-state error (last 20% of simulation)
    steady_start_idx = int(0.8 * len(position))
    steady_state_error = np.mean(np.abs(position[steady_start_idx:] - target_position))

    # Average tau (if available)
    avg_tau = np.mean(data["time_constant"]) if data["time_constant"] is not None else None

    # Average comp_tau (if available)
    avg_comp_tau = np.mean(data["comp_tau"]) if data["comp_tau"] is not None else None

    # Average comp_gain (if available)
    avg_comp_gain = np.mean(data["comp_gain"]) if data["comp_gain"] is not None else None

    return {
        "rmse": rmse,
        "mae": mae,
        "max_error": max_error,
        "settling_time": settling_time,
        "overshoot": overshoot,
        "steady_state_error": steady_state_error,
        "avg_tau": avg_tau,
        "avg_comp_tau": avg_comp_tau,
        "avg_comp_gain": avg_comp_gain,
    }


def analyze_sweep(sweep_dir: Path, output_dir: Path):
    """Analyze delay-tau sweep results."""
    print(f"\n📊 Analyzing sweep: {sweep_dir.name}")
    print("=" * 80)

    # Find all subdirectories
    subdirs = [d for d in sweep_dir.iterdir() if d.is_dir()]
    print(f"Found {len(subdirs)} simulation directories")

    # Load baseline (RT) data
    baseline_dir = None
    for subdir in subdirs:
        params = parse_sweep_dir_name(subdir.name)
        if params.get("is_baseline", False):
            baseline_dir = subdir
            break

    if baseline_dir is None:
        print("⚠️  Warning: No baseline RT simulation found")
        baseline_data = None
        baseline_config = None
    else:
        print(f"📍 Baseline: {baseline_dir.name}")
        h5_file = baseline_dir / "hils_data.h5"
        config_file = baseline_dir / "simulation_config.json"
        baseline_data = load_simulation_data(h5_file)
        baseline_config = load_config(config_file)

    # Analyze all simulations
    results = []
    for subdir in subdirs:
        h5_file = subdir / "hils_data.h5"
        config_file = subdir / "simulation_config.json"

        if not h5_file.exists() or not config_file.exists():
            print(f"⚠️  Skipping {subdir.name}: missing files")
            continue

        print(f"  Processing: {subdir.name}")

        # Load data and config
        data = load_simulation_data(h5_file)
        config = load_config(config_file)

        # Parse parameters from directory name
        params = parse_sweep_dir_name(subdir.name)

        # Also extract from config (more accurate)
        cmd_delay = config["communication"]["cmd_delay_s"] * 1000  # to ms
        sense_delay = config["communication"]["sense_delay_s"] * 1000  # to ms
        plant_tau = config["plant"]["time_constant_s"] * 1000  # to ms
        comp_enabled = config["inverse_compensation"]["enabled"]
        comp_base_tau = config["inverse_compensation"]["base_tau_ms"]

        # Calculate metrics
        target_pos = config["control"]["target_position_m"]
        metrics = calculate_metrics(data, config, target_position=target_pos)

        # Store results
        result = {
            "dir_name": subdir.name,
            "cmd_delay_ms": cmd_delay,
            "sense_delay_ms": sense_delay,
            "plant_tau_ms": plant_tau,
            "comp_enabled": comp_enabled,
            "comp_base_tau_ms": comp_base_tau,
            "is_baseline": params.get("is_baseline", False),
            **metrics,
        }
        results.append(result)

    # Create DataFrame
    df = pd.DataFrame(results)

    # Sort by cmd_delay and plant_tau
    df = df.sort_values(["cmd_delay_ms", "plant_tau_ms"])

    # Save summary CSV
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "delay_tau_sweep_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Summary saved to: {csv_path}")

    # Save detailed metrics JSON
    json_path = output_dir / "delay_tau_sweep_metrics.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"✅ Detailed metrics saved to: {json_path}")

    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(
        df[
            [
                "cmd_delay_ms",
                "plant_tau_ms",
                "comp_enabled",
                "avg_tau",
                "avg_comp_tau",
                "rmse",
                "mae",
                "settling_time",
            ]
        ].to_string(index=False)
    )

    # Generate plots
    print("\n" + "=" * 80)
    print("GENERATING PLOTS")
    print("=" * 80)

    # Filter only compensation-enabled runs (exclude baseline)
    df_comp = df[(df["comp_enabled"] == True) & (df["is_baseline"] == False)].copy()
    df_baseline = df[df["is_baseline"] == True]

    if len(df_comp) == 0:
        print("⚠️  No compensation runs found, skipping plots")
        return df

    # 1. Heatmap: RMSE vs cmd_delay and tau
    plot_heatmap(
        df_comp,
        "cmd_delay_ms",
        "plant_tau_ms",
        "rmse",
        "RMSE vs CMD Delay and Plant Tau",
        output_dir / "heatmap_rmse_delay_tau.png",
    )

    # 2. Heatmap: MAE vs cmd_delay and tau
    plot_heatmap(
        df_comp,
        "cmd_delay_ms",
        "plant_tau_ms",
        "mae",
        "MAE vs CMD Delay and Plant Tau",
        output_dir / "heatmap_mae_delay_tau.png",
    )

    # 3. Heatmap: Settling Time vs cmd_delay and tau
    plot_heatmap(
        df_comp,
        "cmd_delay_ms",
        "plant_tau_ms",
        "settling_time",
        "Settling Time vs CMD Delay and Plant Tau",
        output_dir / "heatmap_settling_time_delay_tau.png",
    )

    # 4. Line plot: RMSE vs cmd_delay (for each tau)
    plot_line_by_delay(
        df_comp, "rmse", "RMSE vs CMD Delay", output_dir / "line_rmse_vs_delay.png"
    )

    # 5. Line plot: RMSE vs tau (for each cmd_delay)
    plot_line_by_tau(
        df_comp, "rmse", "RMSE vs Plant Tau", output_dir / "line_rmse_vs_tau.png"
    )

    # 6. Tau comparison plot (plant_tau vs comp_tau)
    if "avg_tau" in df_comp.columns and "avg_comp_tau" in df_comp.columns:
        plot_tau_comparison(df_comp, output_dir / "tau_comparison.png")

    # 7. Position trajectories comparison (sample)
    plot_trajectory_comparison(
        sweep_dir, df_comp, baseline_dir, output_dir / "trajectory_comparison.png"
    )

    print(f"\n✅ All plots saved to: {output_dir}")

    return df


def plot_heatmap(df, x_col, y_col, z_col, title, output_path):
    """Generate heatmap plot."""
    # Pivot data for heatmap
    pivot = df.pivot_table(index=y_col, columns=x_col, values=z_col)

    # Sort index in descending order so larger tau values appear at the top
    pivot = pivot.sort_index(ascending=False)

    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlOrRd", cbar_kws={"label": z_col})
    plt.title(title)
    plt.xlabel("CMD Delay [ms]")
    plt.ylabel("Plant Tau [ms]")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Saved: {output_path.name}")


def plot_line_by_delay(df, metric_col, title, output_path):
    """Plot metric vs cmd_delay for each tau value."""
    plt.figure(figsize=(10, 6))

    # Group by plant_tau
    for tau, group in df.groupby("plant_tau_ms"):
        plt.plot(
            group["cmd_delay_ms"], group[metric_col], marker="o", label=f"τ={tau:.0f}ms"
        )

    plt.xlabel("CMD Delay [ms]")
    plt.ylabel(metric_col)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Saved: {output_path.name}")


def plot_line_by_tau(df, metric_col, title, output_path):
    """Plot metric vs tau for each cmd_delay value."""
    plt.figure(figsize=(10, 6))

    # Group by cmd_delay
    for delay, group in df.groupby("cmd_delay_ms"):
        plt.plot(
            group["plant_tau_ms"], group[metric_col], marker="o", label=f"delay={delay:.0f}ms"
        )

    plt.xlabel("Plant Tau [ms]")
    plt.ylabel(metric_col)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Saved: {output_path.name}")


def plot_tau_comparison(df, output_path):
    """Plot plant_tau vs comp_tau comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Scatter plot: plant_tau vs comp_tau
    ax = axes[0]
    for delay, group in df.groupby("cmd_delay_ms"):
        ax.scatter(
            group["avg_tau"],
            group["avg_comp_tau"],
            label=f"delay={delay:.0f}ms",
            alpha=0.7,
            s=100,
        )

    # Add diagonal line (perfect match)
    min_tau = min(df["avg_tau"].min(), df["avg_comp_tau"].min())
    max_tau = max(df["avg_tau"].max(), df["avg_comp_tau"].max())
    ax.plot([min_tau, max_tau], [min_tau, max_tau], "k--", alpha=0.5, label="Perfect match")

    ax.set_xlabel("Plant Tau [ms]")
    ax.set_ylabel("Inverse Comp Tau [ms]")
    ax.set_title("Tau Comparison: Plant vs Inverse Compensator")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Bar plot: tau difference
    ax = axes[1]
    df_sorted = df.sort_values("cmd_delay_ms")
    tau_diff = df_sorted["avg_comp_tau"] - df_sorted["avg_tau"]
    x_labels = [f"{row['cmd_delay_ms']:.0f}ms\nτ={row['plant_tau_ms']:.0f}" for _, row in df_sorted.iterrows()]

    colors = ["green" if abs(d) < 5 else "orange" if abs(d) < 10 else "red" for d in tau_diff]
    ax.bar(range(len(tau_diff)), tau_diff, color=colors, alpha=0.7)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Simulation Config")
    ax.set_ylabel("Tau Difference (Comp - Plant) [ms]")
    ax.set_title("Tau Mismatch: Inverse Comp - Plant")
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Saved: {output_path.name}")


def plot_trajectory_comparison(sweep_dir, df_comp, baseline_dir, output_path):
    """Plot position trajectories for selected simulations."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Load baseline
    if baseline_dir is not None:
        baseline_data = load_simulation_data(baseline_dir / "hils_data.h5")
        axes[0].plot(
            baseline_data["time"],
            baseline_data["position"],
            label="Baseline (RT, no delay)",
            linewidth=2,
            color="black",
        )
        axes[1].plot(
            baseline_data["time"],
            baseline_data["velocity"],
            label="Baseline (RT, no delay)",
            linewidth=2,
            color="black",
        )

    # Select all unique delays
    selected_delays = sorted(df_comp["cmd_delay_ms"].unique())

    colors = plt.cm.tab10(np.linspace(0, 1, len(selected_delays)))

    for i, delay in enumerate(selected_delays):
        subset = df_comp[df_comp["cmd_delay_ms"] == delay]
        for _, row in subset.iterrows():
            sim_dir = sweep_dir / row["dir_name"]
            data = load_simulation_data(sim_dir / "hils_data.h5")

            label = f"delay={delay:.0f}ms, τ={row['plant_tau_ms']:.0f}ms"
            axes[0].plot(data["time"], data["position"], label=label, alpha=0.7, color=colors[i])
            axes[1].plot(data["time"], data["velocity"], label=label, alpha=0.7, color=colors[i])

    # Target line
    if baseline_dir is not None:
        config = load_config(baseline_dir / "simulation_config.json")
        target = config["control"]["target_position_m"]
        axes[0].axhline(target, color="red", linestyle="--", linewidth=1, label="Target")

    axes[0].set_xlabel("Time [s]")
    axes[0].set_ylabel("Position [m]")
    axes[0].set_title("Position Trajectories Comparison")
    axes[0].legend(loc="best", fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Velocity [m/s]")
    axes[1].set_title("Velocity Trajectories Comparison")
    axes[1].legend(loc="best", fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Saved: {output_path.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare delay-tau sweep simulation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Analyze a sweep directory
    python compare_delay_tau_sweep.py results/20251111-183809_sweep

    # Specify custom output directory
    python compare_delay_tau_sweep.py results/20251111-183809_sweep --output-dir my_analysis
        """,
    )

    parser.add_argument(
        "sweep_dir",
        type=str,
        help="Path to sweep directory (e.g., results/20251111-183809_sweep)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for analysis results (default: results/comparison_delay_tau_sweep)",
    )

    args = parser.parse_args()

    # Convert to Path
    sweep_dir = Path(args.sweep_dir)

    if not sweep_dir.exists():
        print(f"❌ Error: Sweep directory not found: {sweep_dir}")
        return

    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path("results") / "comparison_delay_tau_sweep"

    # Run analysis
    df = analyze_sweep(sweep_dir, output_dir)

    print("\n" + "=" * 80)
    print("✅ ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"Results saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - delay_tau_sweep_summary.csv")
    print("  - delay_tau_sweep_metrics.json")
    print("  - heatmap_rmse_delay_tau.png")
    print("  - heatmap_mae_delay_tau.png")
    print("  - heatmap_settling_time_delay_tau.png")
    print("  - line_rmse_vs_delay.png")
    print("  - line_rmse_vs_tau.png")
    print("  - tau_comparison.png")
    print("  - trajectory_comparison.png")


if __name__ == "__main__":
    main()
