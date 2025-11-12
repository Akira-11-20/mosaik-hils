"""
Compare delay-tau sweep results with simple plots.

This script creates simple comparison plots for sweep results with varying
cmd_delay and plant tau values.
"""

import json
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


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
        data["time_s"] = f["time"]["time_s"][:]

        # Load environment (spacecraft) data
        env_group = None
        for key in f.keys():
            if "EnvSim" in key:
                env_group = f[key]
                break

        if env_group:
            data["position"] = env_group["position"][:]
            data["velocity"] = env_group["velocity"][:]
            data["force"] = env_group["force"][:]

        # Load plant data (tau)
        plant_group = None
        for key in f.keys():
            if "PlantSim" in key:
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
            if "ControllerSim" in key:
                ctrl_group = f[key]
                break

        if ctrl_group:
            data["error"] = ctrl_group["error"][:]

        # Load inverse compensator data (if exists)
        comp_group = None
        for key in f.keys():
            if "InverseCompSim" in key or "InverseCompensatorSim" in key:
                comp_group = f[key]
                break

        if comp_group:
            if "current_gain" in comp_group:
                data["comp_gain"] = comp_group["current_gain"][:]
            if "current_tau" in comp_group:
                data["comp_tau"] = comp_group["current_tau"][:]
            if "input_thrust" in comp_group:
                data["comp_input"] = comp_group["input_thrust"][:]
            if "output_thrust" in comp_group:
                data["comp_output"] = comp_group["output_thrust"][:]

    # Load config
    config = {}
    if config_file.exists():
        with open(config_file, "r") as f:
            config = json.load(f)

    return data, config


def parse_dir_name(dir_name: str):
    """Parse directory name to extract cmd_delay"""
    import re

    if "baseline" in dir_name or "_rt" in dir_name:
        return 0.0, True

    cmd_match = re.search(r"cmd(\d+(?:\.\d+)?)ms", dir_name)
    if cmd_match:
        return float(cmd_match.group(1)), False
    return None, False


def create_comparison_plots(sweep_dir: Path):
    """Create comparison plots for delay-tau sweep"""
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

    print(f"\n✅ Loaded {len(scenarios)} scenarios")

    # Create output directory
    output_dir = sweep_dir / "analysis"
    output_dir.mkdir(exist_ok=True)

    # Create plots
    print("\n📈 Generating plots...")

    # Create multi-panel plot
    plot_comprehensive_comparison(
        scenarios, baseline_data, baseline_config, baseline_dir, output_dir
    )

    print(f"\n✅ All plots saved to: {output_dir}")


def plot_comprehensive_comparison(
    scenarios, baseline_data, baseline_config, baseline_dir, output_dir
):
    """Create comprehensive comparison plot with multiple panels"""

    # Create figure with 4 rows x 1 column (vertical layout)
    fig, axes = plt.subplots(4, 1, figsize=(12, 16))
    fig.suptitle(
        "Delay-Tau Sweep Comparison: Effect of CMD Delay",
        fontsize=16,
        fontweight="bold",
    )

    # Color map for different delays
    scenario_colors = plt.cm.viridis(np.linspace(0, 1, len(scenarios)))

    # === Row 1: Position Trajectory ===
    ax = axes[0]
    if baseline_data is not None:
        ax.plot(
            baseline_data["time_s"],
            baseline_data["position"],
            label="RT Baseline",
            color="black",
            linewidth=2.5,
            linestyle="--",
            alpha=0.9,
        )

    for i, scenario in enumerate(scenarios):
        data = scenario["data"]
        cmd_delay = scenario["cmd_delay"]
        ax.plot(
            data["time_s"],
            data["position"],
            label=f"CMD Delay {cmd_delay:.0f}ms",
            linewidth=1.5,
            color=scenario_colors[i],
        )

    ax.set_xlabel("Time [s]", fontsize=11)
    ax.set_ylabel("Position [m]", fontsize=11)
    ax.set_title("Position Trajectory Comparison", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # === Row 2: Position Deviation from RT ===
    ax = axes[1]
    if baseline_data is not None:
        rt_position = baseline_data["position"]

        for i, scenario in enumerate(scenarios):
            data = scenario["data"]
            cmd_delay = scenario["cmd_delay"]
            position_diff = data["position"] - rt_position
            ax.plot(
                data["time_s"],
                position_diff,
                label=f"CMD Delay {cmd_delay:.0f}ms",
                linewidth=1.5,
                color=scenario_colors[i],
            )

        ax.axhline(y=0, color="purple", linestyle="--", alpha=0.5, linewidth=2, label="RT Baseline (0)")

    ax.set_xlabel("Time [s]", fontsize=11)
    ax.set_ylabel("Position Deviation from RT [m]", fontsize=11)
    ax.set_title("Position Deviation from RT Baseline", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # === Row 3: Velocity Trajectory ===
    ax = axes[2]
    if baseline_data is not None:
        ax.plot(
            baseline_data["time_s"],
            baseline_data["velocity"],
            label="RT Baseline",
            color="black",
            linewidth=2.5,
            linestyle="--",
            alpha=0.9,
        )

    for i, scenario in enumerate(scenarios):
        data = scenario["data"]
        cmd_delay = scenario["cmd_delay"]
        ax.plot(
            data["time_s"],
            data["velocity"],
            label=f"CMD Delay {cmd_delay:.0f}ms",
            linewidth=1.5,
            color=scenario_colors[i],
        )

    ax.set_xlabel("Time [s]", fontsize=11)
    ax.set_ylabel("Velocity [m/s]", fontsize=11)
    ax.set_title("Velocity Trajectory Comparison", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # === Row 4: Velocity Deviation from RT ===
    ax = axes[3]
    if baseline_data is not None:
        rt_velocity = baseline_data["velocity"]

        for i, scenario in enumerate(scenarios):
            data = scenario["data"]
            cmd_delay = scenario["cmd_delay"]
            velocity_diff = data["velocity"] - rt_velocity
            ax.plot(
                data["time_s"],
                velocity_diff,
                label=f"CMD Delay {cmd_delay:.0f}ms",
                linewidth=1.5,
                color=scenario_colors[i],
            )

        ax.axhline(y=0, color="purple", linestyle="--", alpha=0.5, linewidth=2, label="RT Baseline (0)")

    ax.set_xlabel("Time [s]", fontsize=11)
    ax.set_ylabel("Velocity Deviation from RT [m/s]", fontsize=11)
    ax.set_title("Velocity Deviation from RT Baseline", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_file = output_dir / "comparison_with_rt_baseline.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Saved: {output_file.name}")

    # Print summary statistics
    print_summary_statistics_v2(scenarios, baseline_data, baseline_config, baseline_dir)


def print_summary_statistics_v2(scenarios, baseline_data, baseline_config, baseline_dir):
    """Print summary statistics with deviation from RT baseline"""
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS: DEVIATION FROM RT BASELINE")
    print("=" * 80)

    if baseline_data is None:
        print("\n⚠️  No RT baseline available")
        return

    rt_position = baseline_data["position"]
    rt_velocity = baseline_data["velocity"]

    print("\nRT Baseline:")
    if baseline_dir:
        print(f"  Directory: {baseline_dir.name}")
    print(f"  Final Position: {rt_position[-1]:.6f} m")
    print(f"  Final Velocity: {rt_velocity[-1]:.6f} m/s")

    print("\n" + "-" * 80)

    # Calculate RMSE and MAE for each scenario
    for scenario in scenarios:
        data = scenario["data"]
        cmd_delay = scenario["cmd_delay"]

        # Position metrics
        pos_diff = data["position"] - rt_position
        pos_rmse = np.sqrt(np.mean(pos_diff**2))
        pos_mae = np.mean(np.abs(pos_diff))
        pos_max = np.max(np.abs(pos_diff))

        # Velocity metrics
        vel_diff = data["velocity"] - rt_velocity
        vel_rmse = np.sqrt(np.mean(vel_diff**2))
        vel_mae = np.mean(np.abs(vel_diff))
        vel_max = np.max(np.abs(vel_diff))

        print(f"\nCMD Delay = {cmd_delay:.0f}ms:")
        print(f"  Directory: {scenario['name']}")
        print(f"  Position vs RT:")
        print(f"    RMSE: {pos_rmse:.6f} m")
        print(f"    MAE:  {pos_mae:.6f} m")
        print(f"    Max Deviation: {pos_max:.6f} m")
        print(f"  Velocity vs RT:")
        print(f"    RMSE: {vel_rmse:.6f} m/s")
        print(f"    MAE:  {vel_mae:.6f} m/s")
        print(f"    Max Deviation: {vel_max:.6f} m/s")

        # Tau statistics (if available)
        if "plant_tau" in data and data["plant_tau"] is not None:
            tau = data["plant_tau"]
            print(f"  Plant Tau:")
            print(f"    Mean: {np.mean(tau):.2f} ms, Std: {np.std(tau):.2f} ms")
            print(f"    Min: {np.min(tau):.2f} ms, Max: {np.max(tau):.2f} ms")

        if "comp_tau" in data and data["comp_tau"] is not None:
            tau = data["comp_tau"]
            print(f"  Comp Tau:")
            print(f"    Mean: {np.mean(tau):.2f} ms, Std: {np.std(tau):.2f} ms")
            print(f"    Min: {np.min(tau):.2f} ms, Max: {np.max(tau):.2f} ms")

    print("\n" + "=" * 80)


def plot_position_trajectories(scenarios, baseline_data, baseline_config, output_dir):
    """Plot position trajectories for all scenarios"""
    fig, ax = plt.subplots(figsize=(12, 6))

