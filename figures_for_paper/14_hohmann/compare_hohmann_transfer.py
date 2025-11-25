"""
Compare Hohmann transfer scenarios: baseline vs inverse compensation.

This script creates comparison plots for Hohmann transfer orbital simulations:
- 3D orbital trajectories
- Position and velocity norms over time
- Orbital elements (semi-major axis, eccentricity, altitude)
- Deviation from baseline
- Error metrics (RMSE, MAE, Max Error)

The baseline (zero delay) is shown as a thick black line as reference.
"""

import json
import re
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path for plot_config
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


def load_orbital_data(result_dir: Path):
    """Load orbital HILS data from HDF5 file"""
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

        # Load orbital environment data
        env_group = None
        for key in f.keys():
            if "OrbitalEnvSim" in key or "OrbitalSpacecraft" in key:
                env_group = f[key]
                break

        if env_group:
            # Position (xyz)
            data["position_x"] = env_group["position_x"][:]
            data["position_y"] = env_group["position_y"][:]
            data["position_z"] = env_group["position_z"][:]
            data["position_norm"] = env_group["position_norm"][:]

            # Velocity (xyz)
            data["velocity_x"] = env_group["velocity_x"][:]
            data["velocity_y"] = env_group["velocity_y"][:]
            data["velocity_z"] = env_group["velocity_z"][:]
            data["velocity_norm"] = env_group["velocity_norm"][:]

            # Orbital elements
            data["semi_major_axis"] = env_group["semi_major_axis"][:]
            data["eccentricity"] = env_group["eccentricity"][:]
            data["altitude"] = env_group["altitude"][:]

            # Forces
            if "force_x" in env_group:
                data["force_x"] = env_group["force_x"][:]
                data["force_y"] = env_group["force_y"][:]
                data["force_z"] = env_group["force_z"][:]
                data["norm_force"] = env_group["norm_force"][:]

        # Load plant data (if exists)
        plant_group = None
        for key in f.keys():
            if "OrbitalPlantSim" in key or "OrbitalThrustStand" in key:
                plant_group = f[key]
                break

        if plant_group:
            if "measured_force_x" in plant_group:
                data["measured_force_x"] = plant_group["measured_force_x"][:]
                data["measured_force_y"] = plant_group["measured_force_y"][:]
                data["measured_force_z"] = plant_group["measured_force_z"][:]
            if "norm_measured_force" in plant_group:
                data["norm_measured_force"] = plant_group["norm_measured_force"][:]
            if "alpha" in plant_group:
                data["plant_tau"] = plant_group["alpha"][:]

        # Load controller data
        ctrl_group = None
        for key in f.keys():
            if "OrbitalControllerSim" in key or "OrbitalController" in key:
                ctrl_group = f[key]
                break

        if ctrl_group:
            if "thrust_command_x" in ctrl_group:
                data["thrust_cmd_x"] = ctrl_group["thrust_command_x"][:]
                data["thrust_cmd_y"] = ctrl_group["thrust_command_y"][:]
                data["thrust_cmd_z"] = ctrl_group["thrust_command_z"][:]

    # Load config
    config = {}
    if config_file.exists():
        with open(config_file) as f:
            config = json.load(f)

    return data, config


def parse_dir_name(dir_name: str):
    """Parse directory name to extract scenario parameters"""
    # Check for baseline
    if "baseline" in dir_name.lower():
        return {
            "tau": 0.0,
            "inv_comp": False,
            "gain": 0.0,
            "controller_type": "hohmann",
            "is_baseline": True,
        }

    # Parse parameters from directory name
    params = {
        "is_baseline": False,
    }

    # Extract tau (delay)
    tau_match = re.search(r"tau=(\d+(?:\.\d+)?)", dir_name)
    if tau_match:
        params["tau"] = float(tau_match.group(1))

    # Extract inverse compensation flag
    inv_comp_match = re.search(r"inv_comp=(True|False)", dir_name)
    if inv_comp_match:
        params["inv_comp"] = inv_comp_match.group(1) == "True"

    # Extract gain
    gain_match = re.search(r"gain=(\d+(?:\.\d+)?)", dir_name)
    if gain_match:
        params["gain"] = float(gain_match.group(1))

    # Extract controller type
    controller_match = re.search(r"controller_type=(\w+)", dir_name)
    if controller_match:
        params["controller_type"] = controller_match.group(1)

    return params


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


def plot_3d_trajectories(scenarios, baseline_data, output_dir):
    """Create 3D plot of orbital trajectories"""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Plot baseline
    if baseline_data is not None:
        ax.plot(
            baseline_data["position_x"],
            baseline_data["position_y"],
            baseline_data["position_z"],
            label="Baseline (τ=0)",
            color="black",
            linewidth=2.5,
            linestyle="--",
            alpha=0.8,
        )

    # Plot scenarios
    scenario_colors = [COLOR_PALETTE[i % len(COLOR_PALETTE)] for i in range(len(scenarios))]
    for i, scenario in enumerate(scenarios):
        data = scenario["data"]
        params = scenario["params"]

        # Create label
        if params["inv_comp"]:
            label = f"Inverse Comp (τ={params['tau']:.0f}ms, gain={params['gain']:.0f})"
        else:
            label = f"No Comp (τ={params['tau']:.0f}ms)"

        ax.plot(
            data["position_x"],
            data["position_y"],
            data["position_z"],
            label=label,
            color=scenario_colors[i],
            linewidth=1.8,
            alpha=0.7,
        )

    # Plot Earth (centered at origin)
    EARTH_RADIUS = 6371.0  # km
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x = EARTH_RADIUS * np.outer(np.cos(u), np.sin(v))
    y = EARTH_RADIUS * np.outer(np.sin(u), np.sin(v))
    z = EARTH_RADIUS * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color="blue", alpha=0.3)

    ax.set_xlabel("X [km]", fontsize=FONT_SETTINGS["label_size"], fontweight=FONT_SETTINGS["label_weight"])
    ax.set_ylabel("Y [km]", fontsize=FONT_SETTINGS["label_size"], fontweight=FONT_SETTINGS["label_weight"])
    ax.set_zlabel("Z [km]", fontsize=FONT_SETTINGS["label_size"], fontweight=FONT_SETTINGS["label_weight"])
    ax.set_title(
        "3D Orbital Trajectories (Hohmann Transfer)",
        fontsize=FONT_SETTINGS["title_size"],
        fontweight=FONT_SETTINGS["title_weight"],
    )
    ax.legend(fontsize=FONT_SETTINGS["legend_size"])

    # Set equal aspect ratio
    max_range = (
        np.array(
            [
                data["position_x"].max() - data["position_x"].min(),
                data["position_y"].max() - data["position_y"].min(),
                data["position_z"].max() - data["position_z"].min(),
            ]
        ).max()
        / 2.0
    )

    mid_x = (data["position_x"].max() + data["position_x"].min()) * 0.5
    mid_y = (data["position_y"].max() + data["position_y"].min()) * 0.5
    mid_z = (data["position_z"].max() + data["position_z"].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.tight_layout()
    output_file = output_dir / "hohmann_3d_trajectories.png"
    save_figure_both_sizes(plt, output_file.parent, base_name=output_file.stem)
    plt.close()
    print(f"  ✅ Saved: {output_file.name}")


def plot_thrust_comparison(scenarios, baseline_data, output_dir):
    """Create thrust comparison plots"""

    # Create figure with 2 rows x 1 column (vertical layout)
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    scenario_colors = [COLOR_PALETTE[i % len(COLOR_PALETTE)] for i in range(len(scenarios))]

    # === Row 1: Applied Force Norm (Full Time Range) ===
    ax = axes[0]
    if baseline_data is not None:
        ax.plot(
            baseline_data["time_s"],
            baseline_data["norm_force"],
            **BASELINE_STYLE,
        )

    for i, scenario in enumerate(scenarios):
        data = scenario["data"]
        params = scenario["params"]

        label = f"τ={params['tau']:.0f}s, Inv={params['inv_comp']}, gain={params['gain']:.0f}"
        ax.plot(
            data["time_s"],
            data["norm_force"],
            label=label,
            color=scenario_colors[i],
            **SCENARIO_STYLE,
        )

    ax.set_xlabel("Time [s]", fontsize=FONT_SETTINGS["label_size"], fontweight=FONT_SETTINGS["label_weight"])
    ax.set_ylabel("Applied Force [N]", fontsize=FONT_SETTINGS["label_size"], fontweight=FONT_SETTINGS["label_weight"])
    ax.set_title(
        "(a) Applied Force Magnitude (on Spacecraft)",
        fontsize=FONT_SETTINGS["title_size"],
        fontweight=FONT_SETTINGS["title_weight"],
    )
    ax.legend(fontsize=FONT_SETTINGS["legend_size"], loc="best")
    ax.grid(True, **GRID_SETTINGS)

    # === Row 2: Applied Force Deviation from Baseline ===
    ax = axes[1]
    if baseline_data is not None:
        max_time = baseline_data["time_s"][-1]
        ax.plot([0, max_time], [0, 0], **BASELINE_DEVIATION_STYLE)

        for i, scenario in enumerate(scenarios):
            data = scenario["data"]
            params = scenario["params"]
            force_diff = data["norm_force"] - baseline_data["norm_force"]

            label = f"τ={params['tau']:.0f}s, Inv={params['inv_comp']}, gain={params['gain']:.0f}"
            ax.plot(
                data["time_s"],
                force_diff,
                label=label,
                color=scenario_colors[i],
                **SCENARIO_STYLE,
            )

    ax.set_xlabel("Time [s]", fontsize=FONT_SETTINGS["label_size"], fontweight=FONT_SETTINGS["label_weight"])
    ax.set_ylabel("Force Error [N]", fontsize=FONT_SETTINGS["label_size"], fontweight=FONT_SETTINGS["label_weight"])
    ax.set_title(
        "(b) Applied Force Deviation from Baseline",
        fontsize=FONT_SETTINGS["title_size"],
        fontweight=FONT_SETTINGS["title_weight"],
    )
    ax.legend(fontsize=FONT_SETTINGS["legend_size"], loc="best")
    ax.grid(True, **GRID_SETTINGS)

    plt.tight_layout()
    output_file = output_dir / "hohmann_thrust_comparison.png"
    save_figure_both_sizes(plt, output_file.parent, base_name=output_file.stem)
    plt.close()
    print(f"  ✅ Saved: {output_file.name}")


def plot_comprehensive_comparison(scenarios, baseline_data, output_dir):
    """Create comprehensive comparison plots with multiple panels"""

    # Create figure with 5 rows x 1 column (vertical layout)
    fig, axes = plt.subplots(5, 1, figsize=(14, 20))

    scenario_colors = [COLOR_PALETTE[i % len(COLOR_PALETTE)] for i in range(len(scenarios))]

    # === Row 1: Altitude (Full Time Range) ===
    ax = axes[0]
    if baseline_data is not None:
        ax.plot(
            baseline_data["time_s"],
            baseline_data["altitude"] / 1000,  # Convert to km
            **BASELINE_STYLE,
        )

    for i, scenario in enumerate(scenarios):
        data = scenario["data"]
        params = scenario["params"]

        label = f"τ={params['tau']:.0f}s, Inv={params['inv_comp']}, gain={params['gain']:.0f}"
        ax.plot(
            data["time_s"],
            data["altitude"] / 1000,  # Convert to km
            label=label,
            color=scenario_colors[i],
            **SCENARIO_STYLE,
        )

    ax.set_xlabel("Time [s]", fontsize=FONT_SETTINGS["label_size"], fontweight=FONT_SETTINGS["label_weight"])
    ax.set_ylabel("Altitude [km]", fontsize=FONT_SETTINGS["label_size"], fontweight=FONT_SETTINGS["label_weight"])
    ax.set_title("(a) Altitude", fontsize=FONT_SETTINGS["title_size"], fontweight=FONT_SETTINGS["title_weight"])
    ax.legend(fontsize=FONT_SETTINGS["legend_size"], loc="best")
    ax.grid(True, **GRID_SETTINGS)

    # === Row 2: Altitude Deviation from Baseline (Full Time Range) ===
    ax = axes[1]
    if baseline_data is not None:
        max_time = baseline_data["time_s"][-1]
        ax.plot([0, max_time], [0, 0], **BASELINE_DEVIATION_STYLE)

        for i, scenario in enumerate(scenarios):
            data = scenario["data"]
            params = scenario["params"]
            alt_diff = (data["altitude"] - baseline_data["altitude"]) / 1000  # Convert to km

            label = f"τ={params['tau']:.0f}s, Inv={params['inv_comp']}, gain={params['gain']:.0f}"
            ax.plot(
                data["time_s"],
                alt_diff,
                label=label,
                color=scenario_colors[i],
                **SCENARIO_STYLE,
            )

    ax.set_xlabel("Time [s]", fontsize=FONT_SETTINGS["label_size"], fontweight=FONT_SETTINGS["label_weight"])
    ax.set_ylabel("Altitude Error [km]", fontsize=FONT_SETTINGS["label_size"], fontweight=FONT_SETTINGS["label_weight"])
    ax.set_title(
        "(b) Altitude Deviation from Baseline",
        fontsize=FONT_SETTINGS["title_size"],
        fontweight=FONT_SETTINGS["title_weight"],
    )
    ax.legend(fontsize=FONT_SETTINGS["legend_size"], loc="best")
    ax.grid(True, **GRID_SETTINGS)

    # === Row 3: Altitude Zoom (Time >= 4000s) ===
    ax = axes[2]
    if baseline_data is not None:
        # Find indices where time >= 4000
        time_mask = baseline_data["time_s"] >= 4000
        ax.plot(
            baseline_data["time_s"][time_mask],
            baseline_data["altitude"][time_mask] / 1000,  # Convert to km
            **BASELINE_STYLE,
        )

    for i, scenario in enumerate(scenarios):
        data = scenario["data"]
        params = scenario["params"]

        # Find indices where time >= 4000
        time_mask = data["time_s"] >= 4000

        label = f"τ={params['tau']:.0f}s, Inv={params['inv_comp']}, gain={params['gain']:.0f}"
        ax.plot(
            data["time_s"][time_mask],
            data["altitude"][time_mask] / 1000,  # Convert to km
            label=label,
            color=scenario_colors[i],
            **SCENARIO_STYLE,
        )

    ax.set_xlabel("Time [s]", fontsize=FONT_SETTINGS["label_size"], fontweight=FONT_SETTINGS["label_weight"])
    ax.set_ylabel("Altitude [km]", fontsize=FONT_SETTINGS["label_size"], fontweight=FONT_SETTINGS["label_weight"])
    ax.set_title(
        "(c) Altitude (Time ≥ 4000s, Zoomed)",
        fontsize=FONT_SETTINGS["title_size"],
        fontweight=FONT_SETTINGS["title_weight"],
    )
    ax.legend(fontsize=FONT_SETTINGS["legend_size"], loc="best")
    ax.grid(True, **GRID_SETTINGS)

    # === Row 4: Velocity Norm ===
    ax = axes[3]
    if baseline_data is not None:
        ax.plot(
            baseline_data["time_s"],
            baseline_data["velocity_norm"],
            **BASELINE_STYLE,
        )

    for i, scenario in enumerate(scenarios):
        data = scenario["data"]
        params = scenario["params"]

        label = f"τ={params['tau']:.0f}s, Inv={params['inv_comp']}, gain={params['gain']:.0f}"
        ax.plot(
            data["time_s"],
            data["velocity_norm"],
            label=label,
            color=scenario_colors[i],
            **SCENARIO_STYLE,
        )

    ax.set_xlabel("Time [s]", fontsize=FONT_SETTINGS["label_size"], fontweight=FONT_SETTINGS["label_weight"])
    ax.set_ylabel(
        "Velocity Norm [km/s]", fontsize=FONT_SETTINGS["label_size"], fontweight=FONT_SETTINGS["label_weight"]
    )
    ax.set_title("(d) Orbital Velocity", fontsize=FONT_SETTINGS["title_size"], fontweight=FONT_SETTINGS["title_weight"])
    ax.legend(fontsize=FONT_SETTINGS["legend_size"], loc="best")
    ax.grid(True, **GRID_SETTINGS)

    # === Row 5: Velocity Deviation from Baseline ===
    ax = axes[4]
    if baseline_data is not None:
        max_time = baseline_data["time_s"][-1]
        ax.plot([0, max_time], [0, 0], **BASELINE_DEVIATION_STYLE)

        for i, scenario in enumerate(scenarios):
            data = scenario["data"]
            params = scenario["params"]
            vel_diff = data["velocity_norm"] - baseline_data["velocity_norm"]

            label = f"τ={params['tau']:.0f}s, Inv={params['inv_comp']}, gain={params['gain']:.0f}"
            ax.plot(
                data["time_s"],
                vel_diff,
                label=label,
                color=scenario_colors[i],
                **SCENARIO_STYLE,
            )

    ax.set_xlabel("Time [s]", fontsize=FONT_SETTINGS["label_size"], fontweight=FONT_SETTINGS["label_weight"])
    ax.set_ylabel(
        "Velocity Error [km/s]", fontsize=FONT_SETTINGS["label_size"], fontweight=FONT_SETTINGS["label_weight"]
    )
    ax.set_title(
        "(e) Velocity Deviation from Baseline",
        fontsize=FONT_SETTINGS["title_size"],
        fontweight=FONT_SETTINGS["title_weight"],
    )
    ax.legend(fontsize=FONT_SETTINGS["legend_size"], loc="best")
    ax.grid(True, **GRID_SETTINGS)

    # # === Row 5: Semi-major Axis ===
    # ax = axes[4]
    # if baseline_data is not None:
    #     ax.plot(
    #         baseline_data["time_s"],
    #         baseline_data["semi_major_axis"],
    #         **BASELINE_STYLE,
    #     )

    # for i, scenario in enumerate(scenarios):
    #     data = scenario["data"]
    #     params = scenario["params"]

    #     label = f"τ={params['tau']:.0f}ms, Inv={params['inv_comp']}, gain={params['gain']:.0f}"
    #     ax.plot(
    #         data["time_s"],
    #         data["semi_major_axis"],
    #         label=label,
    #         color=scenario_colors[i],
    #         **SCENARIO_STYLE,
    #     )

    # ax.set_xlabel("Time [s]", fontsize=FONT_SETTINGS["label_size"], fontweight=FONT_SETTINGS["label_weight"])
    # ax.set_ylabel("Semi-major Axis [km]", fontsize=FONT_SETTINGS["label_size"], fontweight=FONT_SETTINGS["label_weight"])
    # ax.set_title("(e) Semi-major Axis (Hohmann Transfer)", fontsize=FONT_SETTINGS["title_size"], fontweight=FONT_SETTINGS["title_weight"])
    # ax.legend(fontsize=FONT_SETTINGS["legend_size"], loc='best')
    # ax.grid(True, **GRID_SETTINGS)

    # # === Row 6: Eccentricity ===
    # ax = axes[5]
    # if baseline_data is not None:
    #     ax.plot(
    #         baseline_data["time_s"],
    #         baseline_data["eccentricity"],
    #         **BASELINE_STYLE,
    #     )

    # for i, scenario in enumerate(scenarios):
    #     data = scenario["data"]
    #     params = scenario["params"]

    #     label = f"τ={params['tau']:.0f}ms, Inv={params['inv_comp']}, gain={params['gain']:.0f}"
    #     ax.plot(
    #         data["time_s"],
    #         data["eccentricity"],
    #         label=label,
    #         color=scenario_colors[i],
    #         **SCENARIO_STYLE,
    #     )

    # ax.set_xlabel("Time [s]", fontsize=FONT_SETTINGS["label_size"], fontweight=FONT_SETTINGS["label_weight"])
    # ax.set_ylabel("Eccentricity", fontsize=FONT_SETTINGS["label_size"], fontweight=FONT_SETTINGS["label_weight"])
    # ax.set_title("(f) Eccentricity", fontsize=FONT_SETTINGS["title_size"], fontweight=FONT_SETTINGS["title_weight"])
    # ax.legend(fontsize=FONT_SETTINGS["legend_size"], loc='best')
    # ax.grid(True, **GRID_SETTINGS)

    plt.tight_layout()
    output_file = output_dir / "hohmann_comparison.png"
    save_figure_both_sizes(plt, output_file.parent, base_name=output_file.stem)
    plt.close()
    print(f"  ✅ Saved: {output_file.name}")


def print_summary_statistics(scenarios, baseline_data, output_dir):
    """Print summary statistics and save to file"""
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS: HOHMANN TRANSFER COMPARISON")
    print("=" * 80)

    summary_lines = []
    summary_lines.append("=" * 80)
    summary_lines.append("HOHMANN TRANSFER COMPARISON: SUMMARY STATISTICS")
    summary_lines.append("=" * 80)

    if baseline_data is None:
        msg = "\n⚠️  No baseline available"
        print(msg)
        summary_lines.append(msg)
        return

    baseline_info = [
        "\nBaseline:",
        f"  Final Altitude: {baseline_data['altitude'][-1]:.6f} km",
        f"  Final Velocity Norm: {baseline_data['velocity_norm'][-1]:.6f} km/s",
        f"  Final Semi-major Axis: {baseline_data['semi_major_axis'][-1]:.6f} km",
        f"  Final Eccentricity: {baseline_data['eccentricity'][-1]:.6f}",
        "-" * 80,
    ]

    for line in baseline_info:
        print(line)
        summary_lines.append(line)

    # Prepare CSV data
    csv_data = []
    csv_headers = [
        "Tau[ms]",
        "InvComp",
        "Gain",
        "Alt_RMSE[km]",
        "Alt_MAE[km]",
        "Alt_MaxErr[km]",
        "Vel_RMSE[km/s]",
        "Vel_MAE[km/s]",
        "Vel_MaxErr[km/s]",
        "SMA_RMSE[km]",
        "Ecc_RMSE",
    ]

    for scenario in scenarios:
        data = scenario["data"]
        params = scenario["params"]

        # Calculate metrics
        alt_metrics = calculate_error_metrics(baseline_data["altitude"], data["altitude"])
        vel_metrics = calculate_error_metrics(baseline_data["velocity_norm"], data["velocity_norm"])
        sma_metrics = calculate_error_metrics(baseline_data["semi_major_axis"], data["semi_major_axis"])
        ecc_metrics = calculate_error_metrics(baseline_data["eccentricity"], data["eccentricity"])

        csv_data.append(
            [
                f"{params['tau']:.0f}",
                str(params["inv_comp"]),
                f"{params['gain']:.0f}",
                f"{alt_metrics['rmse']:.6f}",
                f"{alt_metrics['mae']:.6f}",
                f"{alt_metrics['max_error']:.6f}",
                f"{vel_metrics['rmse']:.6f}",
                f"{vel_metrics['mae']:.6f}",
                f"{vel_metrics['max_error']:.6f}",
                f"{sma_metrics['rmse']:.6f}",
                f"{ecc_metrics['rmse']:.6f}",
            ]
        )

        scenario_info = [
            f"\nScenario: τ={params['tau']:.0f}ms, Inv={params['inv_comp']}, gain={params['gain']:.0f}",
            f"  Directory: {scenario['name']}",
            "  Altitude vs Baseline:",
            f"    RMSE: {alt_metrics['rmse']:.6f} km",
            f"    MAE:  {alt_metrics['mae']:.6f} km",
            f"    Max Deviation: {alt_metrics['max_error']:.6f} km",
            "  Velocity Norm vs Baseline:",
            f"    RMSE: {vel_metrics['rmse']:.6f} km/s",
            f"    MAE:  {vel_metrics['mae']:.6f} km/s",
            f"    Max Deviation: {vel_metrics['max_error']:.6f} km/s",
            "  Orbital Elements:",
            f"    Semi-major Axis RMSE: {sma_metrics['rmse']:.6f} km",
            f"    Eccentricity RMSE: {ecc_metrics['rmse']:.6f}",
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


def main():
    # Get script directory
    script_dir = Path(__file__).parent

    print("=" * 80)
    print("Hohmann Transfer Comparison Analysis")
    print("=" * 80)

    # Find all subdirectories
    subdirs = sorted([d for d in script_dir.iterdir() if d.is_dir()])
    print(f"Found {len(subdirs)} subdirectories")

    # Load all data
    scenarios = []
    baseline_data = None

    for subdir in subdirs:
        dir_name = subdir.name
        params = parse_dir_name(dir_name)

        try:
            data, config = load_orbital_data(subdir)

            if params["is_baseline"]:
                baseline_data = data
                print(f"✅ Loaded baseline: {dir_name}")
            else:
                scenarios.append(
                    {
                        "name": dir_name,
                        "params": params,
                        "data": data,
                        "config": config,
                        "dir": subdir,
                    }
                )
                print(f"✅ Loaded: {dir_name}")

        except Exception as e:
            print(f"❌ Error loading {dir_name}: {e}")

    # Sort scenarios by tau, then by inverse compensation flag
    scenarios = sorted(scenarios, key=lambda x: (x["params"]["tau"], not x["params"]["inv_comp"]))

    if not scenarios:
        print("❌ No valid scenarios found")
        return

    if baseline_data is None:
        print("⚠️  Warning: No baseline found")

    print(f"\n✅ Loaded {len(scenarios)} scenarios")

    # Create output directory
    output_dir = script_dir

    # Create plots
    print("\n📈 Generating plots...")

    # 3D trajectory plot
    plot_3d_trajectories(scenarios, baseline_data, output_dir)

    # Thrust comparison plot
    plot_thrust_comparison(scenarios, baseline_data, output_dir)

    # Comprehensive comparison plot
    plot_comprehensive_comparison(scenarios, baseline_data, output_dir)

    # Print summary statistics
    print_summary_statistics(scenarios, baseline_data, output_dir)

    print(f"\n✅ All plots and summaries saved to: {output_dir}")


if __name__ == "__main__":
    main()
