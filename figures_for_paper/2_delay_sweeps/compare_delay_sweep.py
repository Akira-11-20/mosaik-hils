"""
Compare delay sweep results with RT baseline.

This script creates comparison plots for delay sweep results, showing:
- Position and velocity trajectories
- Deviation from RT baseline
- Error metrics (RMSE, MAE, Max Error)

The RT baseline is shown as a thick black line as reference.
"""

import json
import re
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import common plot configuration
sys.path.insert(0, str(Path(__file__).parent.parent))
from plot_config import (
    BASELINE_DEVIATION_STYLE,
    BASELINE_STYLE,
    COLOR_PALETTE,
    FIGURE_SETTINGS,
    FONT_SETTINGS,
    GRID_SETTINGS,
    SCENARIO_STYLE,
    save_figure_both_sizes,
)


def load_simulation_data(result_dir: Path):
    """Load HDF5 data and config from a result directory"""
    h5_file = result_dir / "hils_data.h5"
    config_file = result_dir / "simulation_config.json"

    if not h5_file.exists():
        raise FileNotFoundError(f"HDF5 file not found: {h5_file}")

    # Load HDF5 data
    data = {}
    with h5py.File(h5_file, "r") as f:
        # Load time
        if "time" in f:
            time_group = f["time"]
            if "time_s" in time_group:
                data["time_s"] = time_group["time_s"][:]
            elif "time_ms" in time_group:
                data["time_s"] = time_group["time_ms"][:] / 1000.0

        # Load environment (spacecraft) data
        env_group = None
        for key in f.keys():
            if "EnvSim" in key or ("Env" in key and "Spacecraft" in key):
                env_group = f[key]
                break

        if env_group:
            data["position"] = env_group["position"][:]
            data["velocity"] = env_group["velocity"][:]
            if "force" in env_group:
                data["force"] = env_group["force"][:]
            if "acceleration" in env_group:
                data["acceleration"] = env_group["acceleration"][:]

        # Load plant data
        plant_group = None
        for key in f.keys():
            if "PlantSim" in key or ("Plant" in key and "ThrustStand" in key):
                plant_group = f[key]
                break

        if plant_group:
            if "time_constant" in plant_group:
                data["plant_tau"] = plant_group["time_constant"][:]
            if "measured_thrust" in plant_group:
                data["measured_thrust"] = plant_group["measured_thrust"][:]
            if "actual_thrust" in plant_group:
                data["actual_thrust"] = plant_group["actual_thrust"][:]

        # Load controller data
        ctrl_group = None
        for key in f.keys():
            if "ControllerSim" in key or ("Controller" in key and "PID" in key):
                ctrl_group = f[key]
                break

        if ctrl_group:
            if "error" in ctrl_group:
                data["error"] = ctrl_group["error"][:]

    # Load config
    config = {}
    if config_file.exists():
        with open(config_file) as f:
            config = json.load(f)

    return data, config


def parse_dir_name(dir_name: str):
    """Parse directory name to extract cmd_delay"""
    if "baseline" in dir_name.lower() or "_rt" in dir_name.lower():
        return 0.0, True

    cmd_match = re.search(r"cmd(\d+(?:\.\d+)?)ms", dir_name)
    if cmd_match:
        return float(cmd_match.group(1)), False
    return None, False


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

    return {
        "rmse": rmse,
        "mae": mae,
        "max_error": max_error,
        "error_signal": error,
    }


def create_comparison_plots(sweep_dir: Path):
    """Create comparison plots for delay sweep"""
    print(f"\n📊 Creating comparison plots for: {sweep_dir.name}")
    print("=" * 80)

    # Find all subdirectories
    subdirs = sorted([d for d in sweep_dir.iterdir() if d.is_dir()])
    print(f"Found {len(subdirs)} subdirectories")

    # Load all data
    scenarios = []
    baseline_data = None
    baseline_config = None
    baseline_dir = None

    for subdir in subdirs:
        dir_name = subdir.name
        cmd_delay, is_baseline = parse_dir_name(dir_name)

        if cmd_delay is None:
            print(f"⚠️  Skipping {dir_name}: could not parse delay")
            continue

        try:
            data, config = load_simulation_data(subdir)

            if is_baseline:
                baseline_data = data
                baseline_config = config
                baseline_dir = subdir
                print(f"✅ Loaded baseline: {dir_name}")
            else:
                scenarios.append(
                    {
                        "name": dir_name,
                        "cmd_delay": cmd_delay,
                        "data": data,
                        "config": config,
                        "dir": subdir,
                    }
                )
                print(f"✅ Loaded: {dir_name} (cmd_delay={cmd_delay}ms)")

        except Exception as e:
            print(f"❌ Error loading {dir_name}: {e}")

    # Sort by cmd_delay
    scenarios = sorted(scenarios, key=lambda x: x["cmd_delay"])

    if not scenarios:
        print("❌ No valid scenarios found")
        return

    if baseline_data is None:
        print("⚠️  Warning: No baseline found")

    print(f"\n✅ Loaded {len(scenarios)} scenarios")

    # Create output directory (save in sweep_dir itself)
    output_dir = sweep_dir

    # Create plots
    print("\n📈 Generating plots...")

    # Split scenarios: exclude 200ms for main plot
    scenarios_low_delay = [s for s in scenarios if s["cmd_delay"] < 200.0]
    scenarios_high_delay = [s for s in scenarios if s["cmd_delay"] >= 200.0]

    # Create main comparison plot (excluding 200ms)
    if scenarios_low_delay:
        print(f"\n📈 Creating main comparison plot (delays: {[s['cmd_delay'] for s in scenarios_low_delay]}ms)...")
        plot_comprehensive_comparison(
            scenarios_low_delay, baseline_data, baseline_config, baseline_dir, output_dir, suffix="_low_delay"
        )

    # Create separate plot for high delays (200ms+)
    if scenarios_high_delay:
        print(
            f"\n📈 Creating high-delay comparison plot (delays: {[s['cmd_delay'] for s in scenarios_high_delay]}ms)..."
        )
        plot_comprehensive_comparison(
            scenarios_high_delay, baseline_data, baseline_config, baseline_dir, output_dir, suffix="_high_delay"
        )

    # Print summary statistics (all scenarios)
    print_summary_statistics(scenarios, baseline_data, baseline_config, baseline_dir, output_dir)

    print(f"\n✅ All plots saved to: {output_dir}")


def plot_comprehensive_comparison(scenarios, baseline_data, baseline_config, baseline_dir, output_dir, suffix=""):
    """Create comprehensive comparison plot with multiple panels

    Args:
        scenarios: List of scenario dictionaries
        baseline_data: RT baseline data
        baseline_config: RT baseline config
        baseline_dir: RT baseline directory
        output_dir: Output directory for plots
        suffix: Filename suffix (e.g., "_low_delay" or "_high_delay")
    """

    # Create figure with 4 rows x 1 column (vertical layout)
    fig, axes = plt.subplots(4, 1, figsize=(14, 16))

    # Use color palette from plot_config
    scenario_colors = [COLOR_PALETTE[i % len(COLOR_PALETTE)] for i in range(len(scenarios))]

    # === Row 1: Position Trajectory ===
    ax = axes[0]

    # Plot baseline first (will be drawn underneath)
    if baseline_data is not None:
        ax.plot(
            baseline_data["time_s"],
            baseline_data["position"],
            **BASELINE_STYLE,
        )

    # Plot scenarios last (they will be drawn on top)
    for i, scenario in enumerate(scenarios):
        data = scenario["data"]
        cmd_delay = scenario["cmd_delay"]
        ax.plot(
            data["time_s"],
            data["position"],
            label=f"CMD Delay {cmd_delay:.0f}ms",
            color=scenario_colors[i],
            **SCENARIO_STYLE,
        )

    ax.set_xlabel("Time [s]", fontsize=FONT_SETTINGS["label_size"], fontweight=FONT_SETTINGS["label_weight"])
    ax.set_ylabel("Position [m]", fontsize=FONT_SETTINGS["label_size"], fontweight=FONT_SETTINGS["label_weight"])
    ax.set_title(
        "(a) Position Trajectories", fontsize=FONT_SETTINGS["title_size"], fontweight=FONT_SETTINGS["title_weight"]
    )
    ax.legend(fontsize=FONT_SETTINGS["legend_size"], loc="best")
    ax.grid(True, **GRID_SETTINGS)

    # === Row 2: Position Deviation from RT ===
    ax = axes[1]
    if baseline_data is not None:
        rt_position = baseline_data["position"]

        # Zero reference line (baseline) - plot as a dashed line from (0,0)
        max_time = baseline_data["time_s"][-1]
        ax.plot([0, max_time], [0, 0], **BASELINE_DEVIATION_STYLE)

        # Plot scenario deviations
        for i, scenario in enumerate(scenarios):
            data = scenario["data"]
            cmd_delay = scenario["cmd_delay"]
            position_diff = data["position"] - rt_position
            ax.plot(
                data["time_s"],
                position_diff,
                label=f"CMD Delay {cmd_delay:.0f}ms",
                color=scenario_colors[i],
                **SCENARIO_STYLE,
            )

    ax.set_xlabel("Time [s]", fontsize=FONT_SETTINGS["label_size"], fontweight=FONT_SETTINGS["label_weight"])
    ax.set_ylabel("Position Error [m]", fontsize=FONT_SETTINGS["label_size"], fontweight=FONT_SETTINGS["label_weight"])
    ax.set_title(
        "(b) Position Deviation from RT Baseline",
        fontsize=FONT_SETTINGS["title_size"],
        fontweight=FONT_SETTINGS["title_weight"],
    )
    ax.legend(fontsize=FONT_SETTINGS["legend_size"], loc="best")
    ax.grid(True, **GRID_SETTINGS)

    # === Row 3: Velocity Trajectory ===
    ax = axes[2]

    # Plot baseline first (will be drawn underneath)
    if baseline_data is not None:
        ax.plot(
            baseline_data["time_s"],
            baseline_data["velocity"],
            **BASELINE_STYLE,
        )

    # Plot scenarios last (they will be drawn on top)
    for i, scenario in enumerate(scenarios):
        data = scenario["data"]
        cmd_delay = scenario["cmd_delay"]
        ax.plot(
            data["time_s"],
            data["velocity"],
            label=f"CMD Delay {cmd_delay:.0f}ms",
            color=scenario_colors[i],
            **SCENARIO_STYLE,
        )

    ax.set_xlabel("Time [s]", fontsize=FONT_SETTINGS["label_size"], fontweight=FONT_SETTINGS["label_weight"])
    ax.set_ylabel("Velocity [m/s]", fontsize=FONT_SETTINGS["label_size"], fontweight=FONT_SETTINGS["label_weight"])
    ax.set_title(
        "(c) Velocity Trajectories", fontsize=FONT_SETTINGS["title_size"], fontweight=FONT_SETTINGS["title_weight"]
    )
    ax.legend(fontsize=FONT_SETTINGS["legend_size"], loc="best")
    ax.grid(True, **GRID_SETTINGS)

    # === Row 4: Velocity Deviation from RT ===
    ax = axes[3]
    if baseline_data is not None:
        rt_velocity = baseline_data["velocity"]

        # Zero reference line (baseline) - plot as a dashed line from (0,0)
        max_time = baseline_data["time_s"][-1]
        ax.plot([0, max_time], [0, 0], **BASELINE_DEVIATION_STYLE)

        # Plot scenario deviations
        for i, scenario in enumerate(scenarios):
            data = scenario["data"]
            cmd_delay = scenario["cmd_delay"]
            velocity_diff = data["velocity"] - rt_velocity
            ax.plot(
                data["time_s"],
                velocity_diff,
                label=f"CMD Delay {cmd_delay:.0f}ms",
                color=scenario_colors[i],
                **SCENARIO_STYLE,
            )

    ax.set_xlabel("Time [s]", fontsize=FONT_SETTINGS["label_size"], fontweight=FONT_SETTINGS["label_weight"])
    ax.set_ylabel(
        "Velocity Error [m/s]", fontsize=FONT_SETTINGS["label_size"], fontweight=FONT_SETTINGS["label_weight"]
    )
    ax.set_title(
        "(d) Velocity Deviation from RT Baseline",
        fontsize=FONT_SETTINGS["title_size"],
        fontweight=FONT_SETTINGS["title_weight"],
    )
    ax.legend(fontsize=FONT_SETTINGS["legend_size"], loc="best")
    ax.grid(True, **GRID_SETTINGS)

    plt.tight_layout()

    # Save figure with suffix
    output_file = output_dir / f"delay_sweep_comparison{suffix}.png"
    save_figure_both_sizes(plt, output_file.parent, base_name=output_file.stem)
    plt.close()
    print(f"  ✅ Saved: {output_file.name}")


def print_summary_statistics(scenarios, baseline_data, baseline_config, baseline_dir, output_dir):
    """Print summary statistics with deviation from RT baseline and save to file"""
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS: DEVIATION FROM RT BASELINE")
    print("=" * 80)

    # Prepare output for both console and file
    summary_lines = []
    summary_lines.append("=" * 80)
    summary_lines.append("DELAY SWEEP COMPARISON: SUMMARY STATISTICS")
    summary_lines.append("=" * 80)

    if baseline_data is None:
        msg = "\n⚠️  No RT baseline available"
        print(msg)
        summary_lines.append(msg)
        return

    rt_position = baseline_data["position"]
    rt_velocity = baseline_data["velocity"]

    baseline_info = [
        "\nRT Baseline:",
        f"  Directory: {baseline_dir.name if baseline_dir else 'N/A'}",
        f"  Final Position: {rt_position[-1]:.6f} m",
        f"  Final Velocity: {rt_velocity[-1]:.6f} m/s",
        "-" * 80,
    ]

    for line in baseline_info:
        print(line)
        summary_lines.append(line)

    # Prepare data for CSV-style summary
    csv_data = []
    csv_headers = [
        "CMD_Delay[ms]",
        "Pos_RMSE[m]",
        "Pos_MAE[m]",
        "Pos_MaxErr[m]",
        "Vel_RMSE[m/s]",
        "Vel_MAE[m/s]",
        "Vel_MaxErr[m/s]",
    ]

    # Calculate metrics for each scenario
    for scenario in scenarios:
        data = scenario["data"]
        cmd_delay = scenario["cmd_delay"]

        # Position metrics
        pos_metrics = calculate_error_metrics(rt_position, data["position"])

        # Velocity metrics
        vel_metrics = calculate_error_metrics(rt_velocity, data["velocity"])

        # Store for CSV
        csv_data.append(
            [
                f"{cmd_delay:.0f}",
                f"{pos_metrics['rmse']:.6f}",
                f"{pos_metrics['mae']:.6f}",
                f"{pos_metrics['max_error']:.6f}",
                f"{vel_metrics['rmse']:.6f}",
                f"{vel_metrics['mae']:.6f}",
                f"{vel_metrics['max_error']:.6f}",
            ]
        )

        scenario_info = [
            f"\nCMD Delay = {cmd_delay:.0f}ms:",
            f"  Directory: {scenario['name']}",
            "  Position vs RT:",
            f"    RMSE: {pos_metrics['rmse']:.6f} m",
            f"    MAE:  {pos_metrics['mae']:.6f} m",
            f"    Max Deviation: {pos_metrics['max_error']:.6f} m",
            "  Velocity vs RT:",
            f"    RMSE: {vel_metrics['rmse']:.6f} m/s",
            f"    MAE:  {vel_metrics['mae']:.6f} m/s",
            f"    Max Deviation: {vel_metrics['max_error']:.6f} m/s",
        ]

        for line in scenario_info:
            print(line)
            summary_lines.append(line)

    summary_lines.append("\n" + "=" * 80)
    print("\n" + "=" * 80)

    # Save summary to text file
    summary_file = output_dir / "comparison_summary.txt"
    with open(summary_file, "w") as f:
        f.write("\n".join(summary_lines))
    print(f"  ✅ Summary saved to: {summary_file.name}")

    # Save CSV summary
    csv_file = output_dir / "comparison_metrics.csv"
    with open(csv_file, "w") as f:
        f.write(",".join(csv_headers) + "\n")
        for row in csv_data:
            f.write(",".join(row) + "\n")
    print(f"  ✅ CSV metrics saved to: {csv_file.name}")


if __name__ == "__main__":
    # Get script directory (delay_sweeps directory)
    script_dir = Path(__file__).parent

    # Check if we're in the correct directory
    if not script_dir.name == "delay_sweeps":
        print(f"⚠️  Warning: Script should be in 'delay_sweeps' directory, but found: {script_dir.name}")

    # Use current directory as sweep directory
    sweep_dir = script_dir

    if not sweep_dir.exists():
        print(f"❌ Error: Directory not found: {sweep_dir}")
        sys.exit(1)

    create_comparison_plots(sweep_dir)
