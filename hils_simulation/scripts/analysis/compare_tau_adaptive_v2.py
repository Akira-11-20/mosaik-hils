"""
Compare tau (time constant) analysis for adaptive compensation sweep results
Focus on position, velocity, and thrust comparison with RT baseline

Analyzes scenarios:
- RT Baseline (no delay)
- No compensation (with delay)
- Fixed gain compensation
- Adaptive gain compensation
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

        # Load plant data (thrust and tau)
        plant_group = None
        for key in f.keys():
            if "PlantSim" in key:
                plant_group = f[key]
                break

        if plant_group:
            data["thrust"] = plant_group["measured_thrust"][:]
            if "time_constant" in plant_group:
                data["tau"] = plant_group["time_constant"][:]
            else:
                data["tau"] = None

        # Load controller data
        ctrl_group = None
        for key in f.keys():
            if "ControllerSim" in key:
                ctrl_group = f[key]
                break

        if ctrl_group:
            data["error"] = ctrl_group["error"][:]

    # Load config
    config = {}
    if config_file.exists():
        with open(config_file, "r") as f:
            config = json.load(f)

    return data, config


def analyze_tau_comparison(sweep_dir: Path):
    """Compare tau across scenarios with focus on position, velocity, thrust"""

    # Find all subdirectories (excluding analysis)
    subdirs = sorted([d for d in sweep_dir.iterdir() if d.is_dir() and d.name != "analysis"])

    print(f"Found {len(subdirs)} subdirectories (excluding 'analysis')")

    # Load data from all scenarios
    scenarios = {}
    rt_baseline = None

    for subdir in subdirs:
        name = subdir.name
        # Extract scenario type from name
        if "baseline_rt" in name or "rt_baseline" in name:
            scenario_type = "RT Baseline"
            print(f"Loading RT baseline: {name}")
            data, config = load_simulation_data(subdir)
            rt_baseline = {"data": data, "config": config, "dir": subdir}
            continue  # Process separately
        elif "nocomp" in name:
            scenario_type = "No Compensation"
        elif "fixed_comp" in name:
            scenario_type = "Fixed Gain"
        elif "adaptive_comp" in name:
            scenario_type = "Adaptive Gain"
        else:
            scenario_type = name

        print(f"Loading: {name} ({scenario_type})")
        data, config = load_simulation_data(subdir)
        scenarios[scenario_type] = {"data": data, "config": config, "dir": subdir}

    if not rt_baseline:
        print("Warning: No RT baseline found")

    if len(scenarios) < 1:
        print(f"Error: No HILS scenarios found")
        return

    # Create 2x2 comparison plots (position, velocity)
    fig, axes = plt.subplots(4, 1, figsize=(12, 16))
    fig.suptitle("Adaptive Compensation Comparison with RT Baseline", fontsize=16, fontweight="bold")

    colors = {
        "RT Baseline": "purple",
        "No Compensation": "red",
        "Fixed Gain": "blue",
        "Adaptive Gain": "green",
    }

    linestyles = {
        "RT Baseline": "--",
        "No Compensation": "-",
        "Fixed Gain": "-",
        "Adaptive Gain": "-",
    }

    linewidths = {
        "RT Baseline": 2.5,
        "No Compensation": 1.5,
        "Fixed Gain": 1.5,
        "Adaptive Gain": 1.5,
    }

    # === Row 1: Position ===

    # Plot 1a: Position Trajectory
    ax = axes[0]
    if rt_baseline:
        data = rt_baseline["data"]
        ax.plot(data["time_s"], data["position"],
                label="RT Baseline",
                color=colors["RT Baseline"],
                linewidth=linewidths["RT Baseline"],
                linestyle=linestyles["RT Baseline"],
                alpha=0.9)

    for scenario_type, info in scenarios.items():
        data = info["data"]
        ax.plot(data["time_s"], data["position"],
                label=scenario_type,
                color=colors[scenario_type],
                linewidth=linewidths[scenario_type],
                linestyle=linestyles[scenario_type])

    ax.set_xlabel("Time [s]", fontsize=11)
    ax.set_ylabel("Position [m]", fontsize=11)
    ax.set_title("Position Trajectory Comparison", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Plot 1b: Position Deviation from RT
    ax = axes[1]
    if rt_baseline:
        rt_position = rt_baseline["data"]["position"]

        for scenario_type, info in scenarios.items():
            data = info["data"]
            position_diff = data["position"] - rt_position
            ax.plot(data["time_s"], position_diff,
                    label=scenario_type,
                    color=colors[scenario_type],
                    linewidth=linewidths[scenario_type])

        ax.axhline(y=0, color='purple', linestyle='--', alpha=0.5, linewidth=2, label='RT Baseline (0)')

    ax.set_xlabel("Time [s]", fontsize=11)
    ax.set_ylabel("Position Deviation from RT [m]", fontsize=11)
    ax.set_title("Position Deviation from RT Baseline", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # === Row 2: Velocity ===

    # Plot 2a: Velocity Trajectory
    ax = axes[2]
    if rt_baseline:
        data = rt_baseline["data"]
        ax.plot(data["time_s"], data["velocity"],
                label="RT Baseline",
                color=colors["RT Baseline"],
                linewidth=linewidths["RT Baseline"],
                linestyle=linestyles["RT Baseline"],
                alpha=0.9)

    for scenario_type, info in scenarios.items():
        data = info["data"]
        ax.plot(data["time_s"], data["velocity"],
                label=scenario_type,
                color=colors[scenario_type],
                linewidth=linewidths[scenario_type],
                linestyle=linestyles[scenario_type])

    ax.set_xlabel("Time [s]", fontsize=11)
    ax.set_ylabel("Velocity [m/s]", fontsize=11)
    ax.set_title("Velocity Trajectory Comparison", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Plot 2b: Velocity Deviation from RT
    ax = axes[3]
    if rt_baseline:
        rt_velocity = rt_baseline["data"]["velocity"]

        for scenario_type, info in scenarios.items():
            data = info["data"]
            velocity_diff = data["velocity"] - rt_velocity
            ax.plot(data["time_s"], velocity_diff,
                    label=scenario_type,
                    color=colors[scenario_type],
                    linewidth=linewidths[scenario_type])

        ax.axhline(y=0, color='purple', linestyle='--', alpha=0.5, linewidth=2, label='RT Baseline (0)')

    ax.set_xlabel("Time [s]", fontsize=11)
    ax.set_ylabel("Velocity Deviation from RT [m/s]", fontsize=11)
    ax.set_title("Velocity Deviation from RT Baseline", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_dir = sweep_dir / "analysis"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "comparison_with_rt_baseline.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"\n✓ Saved plot: {output_file}")

    plt.show()

    # Print summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS: DEVIATION FROM RT BASELINE")
    print("=" * 80)

    if rt_baseline:
        rt_data = rt_baseline["data"]
        rt_position = rt_data["position"]
        rt_velocity = rt_data["velocity"]
        rt_thrust = rt_data["thrust"]

        print("\nRT Baseline:")
        print(f"  Directory: {rt_baseline['dir'].name}")
        print(f"  Final Position: {rt_position[-1]:.6f} m")
        print(f"  Final Velocity: {rt_velocity[-1]:.6f} m/s")

        print("\n" + "-" * 80)

        # Calculate RMSE and MAE for each scenario
        for scenario_type, info in scenarios.items():
            data = info["data"]

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

            print(f"\n{scenario_type}:")
            print(f"  Directory: {info['dir'].name}")
            print(f"  Position vs RT:")
            print(f"    RMSE: {pos_rmse:.6f} m")
            print(f"    MAE:  {pos_mae:.6f} m")
            print(f"    Max Deviation: {pos_max:.6f} m")
            print(f"  Velocity vs RT:")
            print(f"    RMSE: {vel_rmse:.6f} m/s")
            print(f"    MAE:  {vel_mae:.6f} m/s")
            print(f"    Max Deviation: {vel_max:.6f} m/s")

            # Tau statistics (if available)
            if data["tau"] is not None:
                tau = data["tau"]
                print(f"  Plant Tau:")
                print(f"    Mean: {np.mean(tau):.2f} ms, Std: {np.std(tau):.2f} ms")
                print(f"    Min: {np.min(tau):.2f} ms, Max: {np.max(tau):.2f} ms")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    import sys

    # Accept sweep directory as command line argument
    if len(sys.argv) > 1:
        sweep_dir = Path(sys.argv[1])
    else:
        sweep_dir = Path("/home/akira/mosaik-hils/hils_simulation/results/20251111-151534_adaptive_comp_sweep")

    if not sweep_dir.exists():
        print(f"Error: Directory not found: {sweep_dir}")
        sys.exit(1)

    print(f"Analyzing adaptive compensation sweep: {sweep_dir.name}")
    analyze_tau_comparison(sweep_dir)
