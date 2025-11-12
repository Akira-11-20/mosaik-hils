"""
Compare tau (time constant) analysis for adaptive compensation sweep results

Analyzes three scenarios:
1. No compensation
2. Fixed gain compensation
3. Adaptive gain compensation
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

        if plant_group and "time_constant" in plant_group:
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

    # Load config
    config = {}
    if config_file.exists():
        with open(config_file) as f:
            config = json.load(f)

    return data, config


def analyze_tau_comparison(sweep_dir: Path):
    """Compare tau across scenarios"""

    # Find all subdirectories (excluding RT baseline for main comparison)
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
            continue  # Skip RT baseline in main comparison
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

    if len(scenarios) < 2:
        print(f"Warning: Expected at least 2 HILS scenarios, found {len(scenarios)}")
        if rt_baseline:
            print("Note: RT baseline found but excluded from main comparison")

    # Create comparison plots
    num_plots = 4 if rt_baseline else 3
    fig, axes = plt.subplots(num_plots, 2, figsize=(14, num_plots * 4))
    fig.suptitle("Adaptive Compensation Comparison: Tau Analysis", fontsize=16, fontweight="bold")

    colors = {
        "No Compensation": "red",
        "Fixed Gain": "blue",
        "Adaptive Gain": "green",
        "RT Baseline": "purple",
    }

    # Plot 1: Position
    ax = axes[0, 0]
    # Plot RT baseline first (if exists) with thicker line
    if rt_baseline:
        data = rt_baseline["data"]
        ax.plot(
            data["time_s"],
            data["position"],
            label="RT Baseline",
            color=colors["RT Baseline"],
            linewidth=2.5,
            linestyle="--",
            alpha=0.8,
        )
    # Plot HILS scenarios
    for scenario_type, info in scenarios.items():
        data = info["data"]
        ax.plot(data["time_s"], data["position"], label=scenario_type, color=colors[scenario_type], linewidth=1.5)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Position [m]")
    ax.set_title("Position Trajectory")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Position Error
    ax = axes[0, 1]
    # Plot RT baseline error
    if rt_baseline:
        data = rt_baseline["data"]
        ax.plot(
            data["time_s"],
            data["error"],
            label="RT Baseline",
            color=colors["RT Baseline"],
            linewidth=2.5,
            linestyle="--",
            alpha=0.8,
        )
    # Plot HILS scenarios
    for scenario_type, info in scenarios.items():
        data = info["data"]
        ax.plot(data["time_s"], data["error"], label=scenario_type, color=colors[scenario_type], linewidth=1.5)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Position Error [m]")
    ax.set_title("Position Error")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Plant Time Constant (Tau)
    ax = axes[1, 0]
    for scenario_type, info in scenarios.items():
        data = info["data"]
        if data["tau"] is not None:
            ax.plot(data["time_s"], data["tau"], label=scenario_type, color=colors[scenario_type], linewidth=1.5)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Plant Tau [ms]")
    ax.set_title("Plant Time Constant (Tau)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Compensator Gain (if available)
    ax = axes[1, 1]
    has_gain_data = False
    for scenario_type, info in scenarios.items():
        data = info["data"]
        if "comp_gain" in data and data["comp_gain"] is not None:
            ax.plot(data["time_s"], data["comp_gain"], label=scenario_type, color=colors[scenario_type], linewidth=1.5)
            has_gain_data = True
    if has_gain_data:
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Compensation Gain")
        ax.set_title("Inverse Compensator Gain")
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No compensator gain data available", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Inverse Compensator Gain")

    # Plot 5: Force Applied
    ax = axes[2, 0]
    for scenario_type, info in scenarios.items():
        data = info["data"]
        ax.plot(data["time_s"], data["force"], label=scenario_type, color=colors[scenario_type], linewidth=1.5)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Force [N]")
    ax.set_title("Applied Force")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 6: Error Statistics Summary
    ax = axes[2, 1]
    scenario_names = []
    rmse_values = []
    mae_values = []

    # Add RT baseline to comparison
    if rt_baseline:
        rt_error = rt_baseline["data"]["error"]
        scenario_names.append("RT Baseline")
        rmse_values.append(np.sqrt(np.mean(rt_error**2)))
        mae_values.append(np.mean(np.abs(rt_error)))

    for scenario_type, info in scenarios.items():
        data = info["data"]
        error = data["error"]

        # Calculate metrics
        rmse = np.sqrt(np.mean(error**2))
        mae = np.mean(np.abs(error))

        scenario_names.append(scenario_type)
        rmse_values.append(rmse)
        mae_values.append(mae)

    x = np.arange(len(scenario_names))
    width = 0.35

    ax.bar(x - width / 2, rmse_values, width, label="RMSE", alpha=0.8)
    ax.bar(x + width / 2, mae_values, width, label="MAE", alpha=0.8)

    ax.set_ylabel("Error [m]")
    ax.set_title("Error Metrics Comparison (with RT Baseline)")
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_names, rotation=15, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Plot 7: Error Deviation from RT Baseline (if RT exists)
    if rt_baseline:
        ax = axes[3, 0]
        rt_position = rt_baseline["data"]["position"]

        for scenario_type, info in scenarios.items():
            data = info["data"]
            position_diff = data["position"] - rt_position
            ax.plot(data["time_s"], position_diff, label=scenario_type, color=colors[scenario_type], linewidth=1.5)

        ax.axhline(y=0, color="purple", linestyle="--", alpha=0.5, label="RT Baseline")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Position Deviation from RT [m]")
        ax.set_title("Position Deviation from RT Baseline")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 8: RMSE vs RT Baseline
        ax = axes[3, 1]
        scenario_names_hils = []
        rmse_vs_rt = []

        for scenario_type, info in scenarios.items():
            data = info["data"]
            position_diff = data["position"] - rt_position
            rmse = np.sqrt(np.mean(position_diff**2))

            scenario_names_hils.append(scenario_type)
            rmse_vs_rt.append(rmse)

        x = np.arange(len(scenario_names_hils))
        bars = ax.bar(x, rmse_vs_rt, alpha=0.8, color=[colors[s] for s in scenario_names_hils])

        ax.set_ylabel("RMSE vs RT Baseline [m]")
        ax.set_title("Position RMSE Compared to RT Baseline")
        ax.set_xticks(x)
        ax.set_xticklabels(scenario_names_hils, rotation=15, ha="right")
        ax.grid(True, alpha=0.3, axis="y")

        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, rmse_vs_rt)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, height, f"{val:.4f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()

    # Save figure
    output_dir = sweep_dir / "analysis"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "tau_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"\n✓ Saved plot: {output_file}")

    plt.show()

    # Print summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    # Print RT baseline first (if exists)
    if rt_baseline:
        print("\nRT Baseline:")
        print(f"  Directory: {rt_baseline['dir'].name}")
        data = rt_baseline["data"]
        error = data["error"]

        rmse = np.sqrt(np.mean(error**2))
        mae = np.mean(np.abs(error))
        max_error = np.max(np.abs(error))

        print(f"  RMSE: {rmse:.6f} m")
        print(f"  MAE:  {mae:.6f} m")
        print(f"  Max Error: {max_error:.6f} m")

        if data["tau"] is not None:
            tau = data["tau"]
            print(f"  Plant Tau - Mean: {np.mean(tau):.2f} ms, Std: {np.std(tau):.2f} ms")
            print(f"              Min: {np.min(tau):.2f} ms, Max: {np.max(tau):.2f} ms")

        print("\n" + "-" * 70)

    for scenario_type, info in scenarios.items():
        data = info["data"]
        config = info["config"]
        error = data["error"]

        print(f"\n{scenario_type}:")
        print(f"  Directory: {info['dir'].name}")

        # Error metrics
        rmse = np.sqrt(np.mean(error**2))
        mae = np.mean(np.abs(error))
        max_error = np.max(np.abs(error))

        print(f"  RMSE: {rmse:.6f} m")
        print(f"  MAE:  {mae:.6f} m")
        print(f"  Max Error: {max_error:.6f} m")

        # Tau statistics
        if data["tau"] is not None:
            tau = data["tau"]
            print(f"  Plant Tau - Mean: {np.mean(tau):.2f} ms, Std: {np.std(tau):.2f} ms")
            print(f"              Min: {np.min(tau):.2f} ms, Max: {np.max(tau):.2f} ms")

        # Compensator gain statistics (if available)
        if "comp_gain" in data and data["comp_gain"] is not None:
            gain = data["comp_gain"]
            print(f"  Comp Gain - Mean: {np.mean(gain):.2f}, Std: {np.std(gain):.2f}")
            print(f"              Min: {np.min(gain):.2f}, Max: {np.max(gain):.2f}")

        # Config info
        if "inverse_compensation" in config:
            comp_config = config["inverse_compensation"]
            print("  Compensation Config:")
            if "gain" in comp_config:
                print(f"    Gain: {comp_config['gain']}")
            if "tau_to_gain_ratio" in comp_config:
                print(f"    Tau-to-Gain Ratio: {comp_config['tau_to_gain_ratio']}")
            if "tau_model_type" in comp_config:
                print(f"    Tau Model Type: {comp_config['tau_model_type']}")

        # Print deviation from RT baseline (if exists)
        if rt_baseline:
            rt_position = rt_baseline["data"]["position"]
            position_diff = data["position"] - rt_position
            rmse_vs_rt = np.sqrt(np.mean(position_diff**2))
            mae_vs_rt = np.mean(np.abs(position_diff))
            max_diff_vs_rt = np.max(np.abs(position_diff))

            print("  Deviation from RT Baseline:")
            print(f"    RMSE: {rmse_vs_rt:.6f} m")
            print(f"    MAE:  {mae_vs_rt:.6f} m")
            print(f"    Max Deviation: {max_diff_vs_rt:.6f} m")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    import sys

    # Accept sweep directory as command line argument
    if len(sys.argv) > 1:
        sweep_dir = Path(sys.argv[1])
    else:
        sweep_dir = Path("/home/akira/mosaik-hils/hils_simulation/results/20251111-150245_adaptive_comp_sweep")

    if not sweep_dir.exists():
        print(f"Error: Directory not found: {sweep_dir}")
        sys.exit(1)

    print(f"Analyzing adaptive compensation sweep: {sweep_dir.name}")
    analyze_tau_comparison(sweep_dir)
