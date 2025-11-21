"""
Compare Formation Flying scenarios: baseline vs inverse compensation.

This script creates comparison plots for formation flying orbital simulations:
- 3D relative position trajectories
- Relative distance over time
- Relative position components (X, Y, Z)
- Thrust magnitude comparison
- Deviation from baseline
- Error metrics (RMSE, MAE, Max Error)

The baseline (tau=0) is shown as a thick black line as reference.
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
try:
    from plot_config import (
        BASELINE_DEVIATION_STYLE,
        BASELINE_STYLE,
        COLOR_PALETTE,
        FIGURE_SETTINGS,
        FONT_SETTINGS,
        GRID_SETTINGS,
        SCENARIO_STYLE,
    )
except ImportError:
    # Fallback if plot_config doesn't exist
    COLOR_PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    BASELINE_STYLE = {"color": "black", "linewidth": 2.5, "linestyle": "--", "alpha": 0.8, "label": "Baseline (τ=0)"}
    BASELINE_DEVIATION_STYLE = {"color": "black", "linewidth": 1, "linestyle": "--", "alpha": 0.5, "label": "Baseline"}
    SCENARIO_STYLE = {"linewidth": 1.8, "alpha": 0.7}
    FONT_SETTINGS = {
        "label_size": 12,
        "label_weight": "normal",
        "title_size": 14,
        "title_weight": "bold",
        "legend_size": 10,
    }
    FIGURE_SETTINGS = {"dpi": 300, "bbox_inches": "tight"}
    GRID_SETTINGS = {"alpha": 0.3}


def load_formation_data(result_dir: Path):
    """Load formation flying HILS data from HDF5 file"""
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

        # Find Chaser and Target environment groups
        env_groups = [k for k in f.keys() if "OrbitalEnvSim" in k]
        if len(env_groups) < 2:
            raise ValueError(f"Expected 2 spacecraft, found {len(env_groups)}")

        # Typically, 0 is Chaser, 1 is Target
        chaser_group = f[env_groups[0]]
        target_group = f[env_groups[1]]

        # Chaser data
        data["chaser_pos_x"] = chaser_group["position_x"][:]
        data["chaser_pos_y"] = chaser_group["position_y"][:]
        data["chaser_pos_z"] = chaser_group["position_z"][:]
        data["chaser_vel_x"] = chaser_group["velocity_x"][:]
        data["chaser_vel_y"] = chaser_group["velocity_y"][:]
        data["chaser_vel_z"] = chaser_group["velocity_z"][:]
        data["chaser_altitude"] = chaser_group["altitude"][:]

        # Target data
        data["target_pos_x"] = target_group["position_x"][:]
        data["target_pos_y"] = target_group["position_y"][:]
        data["target_pos_z"] = target_group["position_z"][:]
        data["target_altitude"] = target_group["altitude"][:]

        # Calculate relative position and velocity
        data["rel_pos_x"] = data["chaser_pos_x"] - data["target_pos_x"]
        data["rel_pos_y"] = data["chaser_pos_y"] - data["target_pos_y"]
        data["rel_pos_z"] = data["chaser_pos_z"] - data["target_pos_z"]
        data["rel_distance"] = np.sqrt(data["rel_pos_x"] ** 2 + data["rel_pos_y"] ** 2 + data["rel_pos_z"] ** 2)

        # Load thrust data from environment (applied force)
        try:
            data["norm_force"] = chaser_group["norm_force"][:]
        except (IndexError, KeyError):
            data["norm_force"] = None

        # Load controller data
        try:
            ctrl_group_name = [k for k in f.keys() if "OrbitalControllerSim" in k][0]
            ctrl_group = f[ctrl_group_name]
            thrust_x = ctrl_group["thrust_command_x"][:]
            thrust_y = ctrl_group["thrust_command_y"][:]
            thrust_z = ctrl_group["thrust_command_z"][:]
            data["norm_thrust_command"] = np.sqrt(thrust_x**2 + thrust_y**2 + thrust_z**2)
        except (IndexError, KeyError):
            data["norm_thrust_command"] = None

        # Load plant data (if exists)
        try:
            plant_group_name = [k for k in f.keys() if "OrbitalPlantSim" in k][0]
            plant_group = f[plant_group_name]
            if "alpha" in plant_group:
                data["plant_tau"] = plant_group["alpha"][:]
        except (IndexError, KeyError):
            data["plant_tau"] = None

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
            "controller_type": "formation",
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


def plot_3d_relative_trajectories(scenarios, baseline_data, output_dir):
    """Create 3D plot of relative position trajectories"""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Plot baseline
    if baseline_data is not None:
        ax.plot(
            baseline_data["rel_pos_x"],
            baseline_data["rel_pos_y"],
            baseline_data["rel_pos_z"],
            label="Baseline (τ=0)",
            color="black",
            linewidth=2.5,
            linestyle="--",
            alpha=0.8,
        )
        # Initial position marker (baseline)
        ax.scatter(
            baseline_data["rel_pos_x"][0],
            baseline_data["rel_pos_y"][0],
            baseline_data["rel_pos_z"][0],
            s=150,
            marker="o",
            color="black",
            alpha=0.9,
            edgecolors="white",
            linewidths=1.5,
        )

    # Plot scenarios
    scenario_colors = [COLOR_PALETTE[i % len(COLOR_PALETTE)] for i in range(len(scenarios))]
    for i, scenario in enumerate(scenarios):
        data = scenario["data"]
        params = scenario["params"]

        # Create label
        if params["inv_comp"]:
            label = f"Inverse Comp (τ={params['tau']:.0f}s, gain={params['gain']:.0f})"
        else:
            label = f"No Comp (τ={params['tau']:.0f}s)"

        ax.plot(
            data["rel_pos_x"],
            data["rel_pos_y"],
            data["rel_pos_z"],
            label=label,
            color=scenario_colors[i],
            linewidth=1.8,
            alpha=0.7,
        )

        # Initial position marker
        ax.scatter(
            data["rel_pos_x"][0],
            data["rel_pos_y"][0],
            data["rel_pos_z"][0],
            s=150,
            marker="o",
            color=scenario_colors[i],
            alpha=0.9,
            edgecolors="black",
            linewidths=1.5,
        )

        # Final position marker
        ax.scatter(
            data["rel_pos_x"][-1],
            data["rel_pos_y"][-1],
            data["rel_pos_z"][-1],
            s=200,
            marker="^",
            color=scenario_colors[i],
            alpha=0.9,
            edgecolors="black",
            linewidths=1.5,
        )

    # Plot target position (origin)
    ax.scatter(
        0,
        0,
        0,
        color="gold",
        s=400,
        marker="*",
        label="Target (origin)",
        zorder=100,
        edgecolors="black",
        linewidths=2,
    )

    ax.set_xlabel("Relative X [m]", fontsize=FONT_SETTINGS["label_size"], fontweight=FONT_SETTINGS["label_weight"])
    ax.set_ylabel("Relative Y [m]", fontsize=FONT_SETTINGS["label_size"], fontweight=FONT_SETTINGS["label_weight"])
    ax.set_zlabel("Relative Z [m]", fontsize=FONT_SETTINGS["label_size"], fontweight=FONT_SETTINGS["label_weight"])
    ax.set_title(
        "3D Relative Position Trajectories (Formation Flying)",
        fontsize=FONT_SETTINGS["title_size"],
        fontweight=FONT_SETTINGS["title_weight"],
    )
    ax.legend(fontsize=FONT_SETTINGS["legend_size"], loc="upper left")
    ax.grid(True, alpha=0.3)

    # Set equal aspect ratio
    if baseline_data is not None:
        all_data = [baseline_data] + [s["data"] for s in scenarios]
    else:
        all_data = [s["data"] for s in scenarios]

    max_range = 0
    for d in all_data:
        max_range = max(
            max_range,
            np.max(np.abs(d["rel_pos_x"])),
            np.max(np.abs(d["rel_pos_y"])),
            np.max(np.abs(d["rel_pos_z"])),
        )

    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range, max_range)

    plt.tight_layout()
    output_file = output_dir / "formation_3d_relative.png"
    plt.savefig(output_file, **FIGURE_SETTINGS)
    plt.close()
    print(f"  ✅ Saved: {output_file.name}")


def plot_relative_distance_thrust(scenarios, baseline_data, output_dir):
    """Create relative distance and thrust comparison plots"""

    # Create figure with 5 rows x 1 column (added zoomed deviation plot)
    fig, axes = plt.subplots(5, 1, figsize=(14, 20))

    scenario_colors = [COLOR_PALETTE[i % len(COLOR_PALETTE)] for i in range(len(scenarios))]

    # === Row 1: Relative Distance ===
    ax = axes[0]
    if baseline_data is not None:
        ax.plot(
            baseline_data["time_s"] / 60.0,  # Convert to minutes
            baseline_data["rel_distance"],
            **BASELINE_STYLE,
        )

    for i, scenario in enumerate(scenarios):
        data = scenario["data"]
        params = scenario["params"]

        label = f"τ={params['tau']:.0f}s, Inv={params['inv_comp']}, gain={params['gain']:.0f}"
        ax.plot(
            data["time_s"] / 60.0,
            data["rel_distance"],
            label=label,
            color=scenario_colors[i],
            **SCENARIO_STYLE,
        )

    ax.set_xlabel("Time [min]", fontsize=FONT_SETTINGS["label_size"], fontweight=FONT_SETTINGS["label_weight"])
    ax.set_ylabel(
        "Relative Distance [m]", fontsize=FONT_SETTINGS["label_size"], fontweight=FONT_SETTINGS["label_weight"]
    )
    ax.set_title(
        "(a) Relative Distance (Chaser-Target)",
        fontsize=FONT_SETTINGS["title_size"],
        fontweight=FONT_SETTINGS["title_weight"],
    )
    ax.legend(fontsize=FONT_SETTINGS["legend_size"], loc="best")
    ax.grid(True, **GRID_SETTINGS)

    # === Row 2: Relative Distance Deviation from Baseline ===
    ax = axes[1]
    if baseline_data is not None:
        max_time = baseline_data["time_s"][-1] / 60.0
        ax.plot([0, max_time], [0, 0], **BASELINE_DEVIATION_STYLE)

        for i, scenario in enumerate(scenarios):
            data = scenario["data"]
            params = scenario["params"]
            dist_diff = data["rel_distance"] - baseline_data["rel_distance"]

            label = f"τ={params['tau']:.0f}s, Inv={params['inv_comp']}, gain={params['gain']:.0f}"
            ax.plot(
                data["time_s"] / 60.0,
                dist_diff,
                label=label,
                color=scenario_colors[i],
                **SCENARIO_STYLE,
            )

    ax.set_xlabel("Time [min]", fontsize=FONT_SETTINGS["label_size"], fontweight=FONT_SETTINGS["label_weight"])
    ax.set_ylabel("Distance Error [m]", fontsize=FONT_SETTINGS["label_size"], fontweight=FONT_SETTINGS["label_weight"])
    ax.set_title(
        "(b) Relative Distance Deviation from Baseline",
        fontsize=FONT_SETTINGS["title_size"],
        fontweight=FONT_SETTINGS["title_weight"],
    )
    ax.legend(fontsize=FONT_SETTINGS["legend_size"], loc="best")
    ax.grid(True, **GRID_SETTINGS)

    # === Row 3: Relative Distance Deviation from Baseline (Time >= 60 min) ===
    ax = axes[2]
    if baseline_data is not None:
        # Filter data for time >= 60 minutes
        time_min = baseline_data["time_s"] / 60.0
        time_mask = time_min >= 60.0

        if np.any(time_mask):
            time_filtered = time_min[time_mask]
            ax.plot([time_filtered[0], time_filtered[-1]], [0, 0], **BASELINE_DEVIATION_STYLE)

            for i, scenario in enumerate(scenarios):
                data = scenario["data"]
                params = scenario["params"]

                # Filter scenario data for time >= 60 minutes
                time_min_scenario = data["time_s"] / 60.0
                time_mask_scenario = time_min_scenario >= 60.0

                if np.any(time_mask_scenario):
                    dist_diff = data["rel_distance"] - baseline_data["rel_distance"]

                    label = f"τ={params['tau']:.0f}s, Inv={params['inv_comp']}, gain={params['gain']:.0f}"
                    ax.plot(
                        time_min_scenario[time_mask_scenario],
                        dist_diff[time_mask_scenario],
                        label=label,
                        color=scenario_colors[i],
                        **SCENARIO_STYLE,
                    )

    ax.set_xlabel("Time [min]", fontsize=FONT_SETTINGS["label_size"], fontweight=FONT_SETTINGS["label_weight"])
    ax.set_ylabel("Distance Error [m]", fontsize=FONT_SETTINGS["label_size"], fontweight=FONT_SETTINGS["label_weight"])
    ax.set_title(
        "(c) Relative Distance Deviation from Baseline (Time ≥ 60 min)",
        fontsize=FONT_SETTINGS["title_size"],
        fontweight=FONT_SETTINGS["title_weight"],
    )
    ax.legend(fontsize=FONT_SETTINGS["legend_size"], loc="best")
    ax.grid(True, **GRID_SETTINGS)

    # === Row 4: Thrust Magnitude ===
    ax = axes[3]
    if baseline_data is not None and baseline_data["norm_force"] is not None:
        ax.plot(
            baseline_data["time_s"] / 60.0,
            baseline_data["norm_force"],
            **BASELINE_STYLE,
        )

    for i, scenario in enumerate(scenarios):
        data = scenario["data"]
        params = scenario["params"]

        label = f"τ={params['tau']:.0f}s, Inv={params['inv_comp']}, gain={params['gain']:.0f}"
        if data["norm_force"] is not None:
            ax.plot(
                data["time_s"] / 60.0,
                data["norm_force"],
                label=label,
                color=scenario_colors[i],
                **SCENARIO_STYLE,
            )
        elif data["norm_thrust_command"] is not None:
            ax.plot(
                data["time_s"] / 60.0,
                data["norm_thrust_command"],
                label=label,
                color=scenario_colors[i],
                **SCENARIO_STYLE,
            )

    ax.set_xlabel("Time [min]", fontsize=FONT_SETTINGS["label_size"], fontweight=FONT_SETTINGS["label_weight"])
    ax.set_ylabel("Thrust [N]", fontsize=FONT_SETTINGS["label_size"], fontweight=FONT_SETTINGS["label_weight"])
    ax.set_title("(d) Thrust Magnitude", fontsize=FONT_SETTINGS["title_size"], fontweight=FONT_SETTINGS["title_weight"])
    ax.legend(fontsize=FONT_SETTINGS["legend_size"], loc="best")
    ax.grid(True, **GRID_SETTINGS)

    # === Row 5: Thrust Deviation from Baseline ===
    ax = axes[4]
    if baseline_data is not None and baseline_data["norm_force"] is not None:
        max_time = baseline_data["time_s"][-1] / 60.0
        ax.plot([0, max_time], [0, 0], **BASELINE_DEVIATION_STYLE)

        for i, scenario in enumerate(scenarios):
            data = scenario["data"]
            params = scenario["params"]

            if data["norm_force"] is not None:
                thrust_diff = data["norm_force"] - baseline_data["norm_force"]
                label = f"τ={params['tau']:.0f}s, Inv={params['inv_comp']}, gain={params['gain']:.0f}"
                ax.plot(
                    data["time_s"] / 60.0,
                    thrust_diff,
                    label=label,
                    color=scenario_colors[i],
                    **SCENARIO_STYLE,
                )

    ax.set_xlabel("Time [min]", fontsize=FONT_SETTINGS["label_size"], fontweight=FONT_SETTINGS["label_weight"])
    ax.set_ylabel("Thrust Error [N]", fontsize=FONT_SETTINGS["label_size"], fontweight=FONT_SETTINGS["label_weight"])
    ax.set_title(
        "(e) Thrust Deviation from Baseline",
        fontsize=FONT_SETTINGS["title_size"],
        fontweight=FONT_SETTINGS["title_weight"],
    )
    ax.legend(fontsize=FONT_SETTINGS["legend_size"], loc="best")
    ax.grid(True, **GRID_SETTINGS)

    plt.tight_layout()
    output_file = output_dir / "formation_distance_thrust.png"
    plt.savefig(output_file, **FIGURE_SETTINGS)
    plt.close()
    print(f"  ✅ Saved: {output_file.name}")


def plot_relative_trajectories_2d(scenarios, baseline_data, output_dir):
    """Create 2D plots of relative position trajectories in different planes"""

    # Create figure with 3 subplots (XY, XZ, YZ planes)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    scenario_colors = [COLOR_PALETTE[i % len(COLOR_PALETTE)] for i in range(len(scenarios))]

    planes = [
        ("rel_pos_x", "rel_pos_y", "Relative X [m]", "Relative Y [m]", "XY Plane (Orbital Plane)"),
        ("rel_pos_x", "rel_pos_z", "Relative X [m]", "Relative Z [m]", "XZ Plane"),
        ("rel_pos_y", "rel_pos_z", "Relative Y [m]", "Relative Z [m]", "YZ Plane"),
    ]

    for ax_idx, (x_key, y_key, x_label, y_label, title) in enumerate(planes):
        ax = axes[ax_idx]

        # Plot baseline
        if baseline_data is not None:
            ax.plot(
                baseline_data[x_key],
                baseline_data[y_key],
                color="black",
                linewidth=2.5,
                linestyle="--",
                alpha=0.8,
                label="Baseline (τ=0)",
            )
            # Initial position marker (baseline)
            ax.scatter(
                baseline_data[x_key][0],
                baseline_data[y_key][0],
                s=150,
                marker="o",
                color="black",
                alpha=0.9,
                edgecolors="white",
                linewidths=1.5,
                zorder=10,
            )

        # Plot scenarios
        for i, scenario in enumerate(scenarios):
            data = scenario["data"]
            params = scenario["params"]

            # Create label
            if params["inv_comp"]:
                label = f"Inv Comp (τ={params['tau']:.0f}ms, g={params['gain']:.0f})"
            else:
                label = f"No Comp (τ={params['tau']:.0f}ms)"

            ax.plot(
                data[x_key],
                data[y_key],
                label=label,
                color=scenario_colors[i],
                linewidth=1.8,
                alpha=0.7,
            )

            # Initial position marker
            ax.scatter(
                data[x_key][0],
                data[y_key][0],
                s=150,
                marker="o",
                color=scenario_colors[i],
                alpha=0.9,
                edgecolors="black",
                linewidths=1.5,
                zorder=10,
            )

            # Final position marker
            ax.scatter(
                data[x_key][-1],
                data[y_key][-1],
                s=200,
                marker="^",
                color=scenario_colors[i],
                alpha=0.9,
                edgecolors="black",
                linewidths=1.5,
                zorder=10,
            )

        # Plot target position (origin)
        ax.scatter(
            0,
            0,
            color="gold",
            s=400,
            marker="*",
            label="Target (origin)" if ax_idx == 0 else None,
            zorder=100,
            edgecolors="black",
            linewidths=2,
        )

        ax.set_xlabel(x_label, fontsize=FONT_SETTINGS["label_size"], fontweight=FONT_SETTINGS["label_weight"])
        ax.set_ylabel(y_label, fontsize=FONT_SETTINGS["label_size"], fontweight=FONT_SETTINGS["label_weight"])
        ax.set_title(
            f"({chr(97 + ax_idx)}) {title}",
            fontsize=FONT_SETTINGS["title_size"],
            fontweight=FONT_SETTINGS["title_weight"],
        )
        ax.grid(True, **GRID_SETTINGS)
        ax.set_aspect("equal", adjustable="box")

        if ax_idx == 0:
            ax.legend(fontsize=FONT_SETTINGS["legend_size"] - 1, loc="best")

    plt.tight_layout()
    output_file = output_dir / "formation_relative_2d_planes.png"
    plt.savefig(output_file, **FIGURE_SETTINGS)
    plt.close()
    print(f"  ✅ Saved: {output_file.name}")


def compute_rtn_frame(pos, vel):
    """
    Compute RTN (Radial-Tangential-Normal) frame transformation matrix

    Args:
        pos: Position vector [x, y, z] in ECI frame (km)
        vel: Velocity vector [vx, vy, vz] in ECI frame (km/s)

    Returns:
        R_eci_to_rtn: 3x3 transformation matrix from ECI to RTN
    """
    # Radial unit vector (R)
    r_hat = pos / np.linalg.norm(pos)

    # Normal unit vector (N) - perpendicular to orbital plane
    h = np.cross(pos, vel)  # Angular momentum vector
    n_hat = h / np.linalg.norm(h)

    # Tangential unit vector (T) - in orbital plane, perpendicular to radial
    t_hat = np.cross(n_hat, r_hat)

    # Construct transformation matrix (rows are RTN unit vectors in ECI frame)
    R_eci_to_rtn = np.array([r_hat, t_hat, n_hat])

    return R_eci_to_rtn


def transform_to_rtn(data):
    """
    Transform relative position from ECI to RTN frame for entire trajectory

    Args:
        data: Dictionary with chaser/target position and velocity data

    Returns:
        Dictionary with RTN relative positions (rel_pos_r, rel_pos_t, rel_pos_n)
    """
    n_points = len(data["time_s"])

    rel_pos_r = np.zeros(n_points)
    rel_pos_t = np.zeros(n_points)
    rel_pos_n = np.zeros(n_points)

    for i in range(n_points):
        # Target position and velocity (reference frame)
        target_pos = np.array([data["target_pos_x"][i], data["target_pos_y"][i], data["target_pos_z"][i]])

        # Use chaser velocity for RTN frame (or could use target velocity)
        # For formation flying, target velocity is more appropriate as reference
        target_vel = np.zeros(3)  # We don't have target velocity stored
        # Estimate velocity from position derivative (if needed)
        if i < n_points - 1:
            dt = data["time_s"][i + 1] - data["time_s"][i]
            target_vel = np.array(
                [
                    (data["target_pos_x"][i + 1] - data["target_pos_x"][i]) / dt,
                    (data["target_pos_y"][i + 1] - data["target_pos_y"][i]) / dt,
                    (data["target_pos_z"][i + 1] - data["target_pos_z"][i]) / dt,
                ]
            )
        elif i > 0:
            dt = data["time_s"][i] - data["time_s"][i - 1]
            target_vel = np.array(
                [
                    (data["target_pos_x"][i] - data["target_pos_x"][i - 1]) / dt,
                    (data["target_pos_y"][i] - data["target_pos_y"][i - 1]) / dt,
                    (data["target_pos_z"][i] - data["target_pos_z"][i - 1]) / dt,
                ]
            )

        # Get RTN transformation matrix
        if np.linalg.norm(target_pos) > 0 and np.linalg.norm(target_vel) > 0:
            R_eci_to_rtn = compute_rtn_frame(target_pos, target_vel)

            # Relative position in ECI
            rel_pos_eci = np.array([data["rel_pos_x"][i], data["rel_pos_y"][i], data["rel_pos_z"][i]])

            # Transform to RTN
            rel_pos_rtn = R_eci_to_rtn @ rel_pos_eci

            rel_pos_r[i] = rel_pos_rtn[0]
            rel_pos_t[i] = rel_pos_rtn[1]
            rel_pos_n[i] = rel_pos_rtn[2]

    data["rel_pos_r"] = rel_pos_r
    data["rel_pos_t"] = rel_pos_t
    data["rel_pos_n"] = rel_pos_n

    return data


def plot_relative_trajectories_rtn(scenarios, baseline_data, output_dir):
    """Create 2D plot of relative position trajectories in RT plane (Hill's frame)"""

    # Transform all data to RTN frame
    if baseline_data is not None:
        baseline_data = transform_to_rtn(baseline_data)

    for scenario in scenarios:
        scenario["data"] = transform_to_rtn(scenario["data"])

    # Create figure with single plot (RT plane only)
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    scenario_colors = [COLOR_PALETTE[i % len(COLOR_PALETTE)] for i in range(len(scenarios))]

    # Plot baseline
    if baseline_data is not None:
        ax.plot(
            baseline_data["rel_pos_r"],
            baseline_data["rel_pos_t"],
            color="black",
            linewidth=2.5,
            linestyle="--",
            alpha=0.8,
            label="Baseline (τ=0)",
        )
        # Initial position marker (baseline)
        ax.scatter(
            baseline_data["rel_pos_r"][0],
            baseline_data["rel_pos_t"][0],
            s=150,
            marker="o",
            color="black",
            alpha=0.9,
            edgecolors="white",
            linewidths=1.5,
            zorder=10,
        )

    # Plot scenarios
    for i, scenario in enumerate(scenarios):
        data = scenario["data"]
        params = scenario["params"]

        # Create label
        if params["inv_comp"]:
            label = f"Inv Comp (τ={params['tau']:.0f}s, gain={params['gain']:.0f})"
        else:
            label = f"No Comp (τ={params['tau']:.0f}s)"

        ax.plot(
            data["rel_pos_r"],
            data["rel_pos_t"],
            label=label,
            color=scenario_colors[i],
            linewidth=1.8,
            alpha=0.7,
        )

        # Initial position marker
        ax.scatter(
            data["rel_pos_r"][0],
            data["rel_pos_t"][0],
            s=150,
            marker="o",
            color=scenario_colors[i],
            alpha=0.9,
            edgecolors="black",
            linewidths=1.5,
            zorder=10,
        )

        # Final position marker
        ax.scatter(
            data["rel_pos_r"][-1],
            data["rel_pos_t"][-1],
            s=200,
            marker="^",
            color=scenario_colors[i],
            alpha=0.9,
            edgecolors="black",
            linewidths=1.5,
            zorder=10,
        )

    # Plot target position (origin)
    ax.scatter(
        0,
        0,
        color="gold",
        s=400,
        marker="*",
        label="Target (origin)",
        zorder=100,
        edgecolors="black",
        linewidths=2,
    )

    ax.set_xlabel("Radial (R) [m]", fontsize=FONT_SETTINGS["label_size"], fontweight=FONT_SETTINGS["label_weight"])
    ax.set_ylabel("Along-track (T) [m]", fontsize=FONT_SETTINGS["label_size"], fontweight=FONT_SETTINGS["label_weight"])
    ax.set_title(
        "RT Plane (In-plane) - Formation Flying Relative Motion",
        fontsize=FONT_SETTINGS["title_size"],
        fontweight=FONT_SETTINGS["title_weight"],
    )
    ax.grid(True, **GRID_SETTINGS)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(fontsize=FONT_SETTINGS["legend_size"], loc="best")

    plt.tight_layout()
    output_file = output_dir / "formation_relative_rt_plane.png"
    plt.savefig(output_file, **FIGURE_SETTINGS)
    plt.close()
    print(f"  ✅ Saved: {output_file.name}")


def plot_relative_trajectories_rtn_zoomed(scenarios, baseline_data, output_dir):
    """Create zoomed 2D plot of relative position trajectories in RT plane (25m x 25m)"""

    # Transform all data to RTN frame (if not already done)
    if baseline_data is not None and "rel_pos_r" not in baseline_data:
        baseline_data = transform_to_rtn(baseline_data)

    for scenario in scenarios:
        if "rel_pos_r" not in scenario["data"]:
            scenario["data"] = transform_to_rtn(scenario["data"])

    # Create figure with single plot (RT plane, zoomed to 25m x 25m)
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    scenario_colors = [COLOR_PALETTE[i % len(COLOR_PALETTE)] for i in range(len(scenarios))]

    # Plot baseline
    if baseline_data is not None:
        ax.plot(
            baseline_data["rel_pos_r"],
            baseline_data["rel_pos_t"],
            color="black",
            linewidth=2.5,
            linestyle="--",
            alpha=0.8,
            label="Baseline (τ=0)",
        )
        # Initial position marker (baseline)
        ax.scatter(
            baseline_data["rel_pos_r"][0],
            baseline_data["rel_pos_t"][0],
            s=200,
            marker="o",
            color="black",
            alpha=0.9,
            edgecolors="white",
            linewidths=2,
            zorder=10,
        )

    # Plot scenarios
    for i, scenario in enumerate(scenarios):
        data = scenario["data"]
        params = scenario["params"]

        # Create label
        if params["inv_comp"]:
            label = f"Inv Comp (τ={params['tau']:.0f}s, gain={params['gain']:.0f})"
        else:
            label = f"No Comp (τ={params['tau']:.0f}s)"

        ax.plot(
            data["rel_pos_r"],
            data["rel_pos_t"],
            label=label,
            color=scenario_colors[i],
            linewidth=2.0,
            alpha=0.7,
        )

        # Initial position marker
        ax.scatter(
            data["rel_pos_r"][0],
            data["rel_pos_t"][0],
            s=200,
            marker="o",
            color=scenario_colors[i],
            alpha=0.9,
            edgecolors="black",
            linewidths=2,
            zorder=10,
        )

        # Final position marker
        ax.scatter(
            data["rel_pos_r"][-1],
            data["rel_pos_t"][-1],
            s=250,
            marker="^",
            color=scenario_colors[i],
            alpha=0.9,
            edgecolors="black",
            linewidths=2,
            zorder=10,
        )

    # Plot target position (origin)
    ax.scatter(
        0,
        0,
        color="gold",
        s=500,
        marker="*",
        label="Target (origin)",
        zorder=100,
        edgecolors="black",
        linewidths=2,
    )

    # Set limits to 25m x 25m centered around origin
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)

    ax.set_xlabel("Radial (R) [m]", fontsize=FONT_SETTINGS["label_size"], fontweight=FONT_SETTINGS["label_weight"])
    ax.set_ylabel("Along-track (T) [m]", fontsize=FONT_SETTINGS["label_size"], fontweight=FONT_SETTINGS["label_weight"])
    ax.set_title(
        "RT Plane (Zoomed 25m × 25m) - Formation Flying Relative Motion",
        fontsize=FONT_SETTINGS["title_size"],
        fontweight=FONT_SETTINGS["title_weight"],
    )
    ax.grid(True, **GRID_SETTINGS)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(fontsize=FONT_SETTINGS["legend_size"], loc="best")

    plt.tight_layout()
    output_file = output_dir / "formation_relative_rt_plane_zoomed.png"
    plt.savefig(output_file, **FIGURE_SETTINGS)
    plt.close()
    print(f"  ✅ Saved: {output_file.name}")


def plot_altitude_comparison(scenarios, baseline_data, output_dir):
    """Create altitude comparison plots for Chaser and Target"""

    # Create figure with 2 rows x 1 column
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    scenario_colors = [COLOR_PALETTE[i % len(COLOR_PALETTE)] for i in range(len(scenarios))]

    # === Row 1: Chaser Altitude ===
    ax = axes[0]
    if baseline_data is not None:
        ax.plot(
            baseline_data["time_s"] / 60.0,
            baseline_data["chaser_altitude"] / 1000,  # Convert to km
            **BASELINE_STYLE,
        )

    for i, scenario in enumerate(scenarios):
        data = scenario["data"]
        params = scenario["params"]

        label = f"τ={params['tau']:.0f}s, Inv={params['inv_comp']}, gain={params['gain']:.0f}"
        ax.plot(
            data["time_s"] / 60.0,
            data["chaser_altitude"] / 1000,  # Convert to km
            label=label,
            color=scenario_colors[i],
            **SCENARIO_STYLE,
        )

    ax.set_xlabel("Time [min]", fontsize=FONT_SETTINGS["label_size"], fontweight=FONT_SETTINGS["label_weight"])
    ax.set_ylabel("Altitude [km]", fontsize=FONT_SETTINGS["label_size"], fontweight=FONT_SETTINGS["label_weight"])
    ax.set_title("(a) Chaser Altitude", fontsize=FONT_SETTINGS["title_size"], fontweight=FONT_SETTINGS["title_weight"])
    ax.legend(fontsize=FONT_SETTINGS["legend_size"], loc="best")
    ax.grid(True, **GRID_SETTINGS)

    # === Row 2: Altitude Deviation from Baseline ===
    ax = axes[1]
    if baseline_data is not None:
        max_time = baseline_data["time_s"][-1] / 60.0
        ax.plot([0, max_time], [0, 0], **BASELINE_DEVIATION_STYLE)

        for i, scenario in enumerate(scenarios):
            data = scenario["data"]
            params = scenario["params"]
            alt_diff = (data["chaser_altitude"] - baseline_data["chaser_altitude"]) / 1000  # Convert to km

            label = f"τ={params['tau']:.0f}s, Inv={params['inv_comp']}, gain={params['gain']:.0f}"
            ax.plot(
                data["time_s"] / 60.0,
                alt_diff,
                label=label,
                color=scenario_colors[i],
                **SCENARIO_STYLE,
            )

    ax.set_xlabel("Time [min]", fontsize=FONT_SETTINGS["label_size"], fontweight=FONT_SETTINGS["label_weight"])
    ax.set_ylabel("Altitude Error [km]", fontsize=FONT_SETTINGS["label_size"], fontweight=FONT_SETTINGS["label_weight"])
    ax.set_title(
        "(b) Chaser Altitude Deviation from Baseline",
        fontsize=FONT_SETTINGS["title_size"],
        fontweight=FONT_SETTINGS["title_weight"],
    )
    ax.legend(fontsize=FONT_SETTINGS["legend_size"], loc="best")
    ax.grid(True, **GRID_SETTINGS)

    plt.tight_layout()
    output_file = output_dir / "formation_altitude.png"
    plt.savefig(output_file, **FIGURE_SETTINGS)
    plt.close()
    print(f"  ✅ Saved: {output_file.name}")


def print_summary_statistics(scenarios, baseline_data, output_dir):
    """Print summary statistics and save to file"""
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS: FORMATION FLYING COMPARISON")
    print("=" * 80)

    summary_lines = []
    summary_lines.append("=" * 80)
    summary_lines.append("FORMATION FLYING COMPARISON: SUMMARY STATISTICS")
    summary_lines.append("=" * 80)

    if baseline_data is None:
        msg = "\n⚠️  No baseline available"
        print(msg)
        summary_lines.append(msg)
        return

    baseline_info = [
        "\nBaseline:",
        f"  Final Relative Distance: {baseline_data['rel_distance'][-1]:.6f} m",
        f"  Final Relative X: {baseline_data['rel_pos_x'][-1]:.6f} m",
        f"  Final Relative Y: {baseline_data['rel_pos_y'][-1]:.6f} m",
        f"  Final Relative Z: {baseline_data['rel_pos_z'][-1]:.6f} m",
        f"  Chaser Final Altitude: {baseline_data['chaser_altitude'][-1]:.6f} m",
        "-" * 80,
    ]

    for line in baseline_info:
        print(line)
        summary_lines.append(line)

    # Prepare CSV data
    csv_data = []
    csv_headers = [
        "Tau[s]",
        "InvComp",
        "Gain",
        "RelDist_RMSE[m]",
        "RelDist_MAE[m]",
        "RelDist_MaxErr[m]",
        "RelX_RMSE[m]",
        "RelY_RMSE[m]",
        "RelZ_RMSE[m]",
        "Thrust_RMSE[N]",
    ]

    for scenario in scenarios:
        data = scenario["data"]
        params = scenario["params"]

        # Calculate metrics
        dist_metrics = calculate_error_metrics(baseline_data["rel_distance"], data["rel_distance"])
        x_metrics = calculate_error_metrics(baseline_data["rel_pos_x"], data["rel_pos_x"])
        y_metrics = calculate_error_metrics(baseline_data["rel_pos_y"], data["rel_pos_y"])
        z_metrics = calculate_error_metrics(baseline_data["rel_pos_z"], data["rel_pos_z"])

        # Thrust metrics (if available)
        if data["norm_force"] is not None and baseline_data["norm_force"] is not None:
            thrust_metrics = calculate_error_metrics(baseline_data["norm_force"], data["norm_force"])
        else:
            thrust_metrics = {"rmse": 0.0}

        csv_data.append(
            [
                f"{params['tau']:.0f}",
                str(params["inv_comp"]),
                f"{params['gain']:.0f}",
                f"{dist_metrics['rmse']:.6f}",
                f"{dist_metrics['mae']:.6f}",
                f"{dist_metrics['max_error']:.6f}",
                f"{x_metrics['rmse']:.6f}",
                f"{y_metrics['rmse']:.6f}",
                f"{z_metrics['rmse']:.6f}",
                f"{thrust_metrics['rmse']:.6f}",
            ]
        )

        scenario_info = [
            f"\nScenario: τ={params['tau']:.0f}s, Inv={params['inv_comp']}, gain={params['gain']:.0f}",
            f"  Directory: {scenario['name']}",
            "  Relative Distance vs Baseline:",
            f"    RMSE: {dist_metrics['rmse']:.6f} m",
            f"    MAE:  {dist_metrics['mae']:.6f} m",
            f"    Max Deviation: {dist_metrics['max_error']:.6f} m",
            "  Relative Position Components RMSE:",
            f"    X: {x_metrics['rmse']:.6f} m",
            f"    Y: {y_metrics['rmse']:.6f} m",
            f"    Z: {z_metrics['rmse']:.6f} m",
            "  Thrust vs Baseline:",
            f"    RMSE: {thrust_metrics['rmse']:.6f} N",
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
    print("Formation Flying Comparison Analysis")
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
            data, config = load_formation_data(subdir)

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

    # 3D relative trajectory plot
    plot_3d_relative_trajectories(scenarios, baseline_data, output_dir)

    # 2D relative trajectory plots (XY, XZ, YZ planes) - ECI frame
    plot_relative_trajectories_2d(scenarios, baseline_data, output_dir)

    # 2D relative trajectory plots (RT plane) - RTN frame (Hill's frame)
    plot_relative_trajectories_rtn(scenarios, baseline_data, output_dir)

    # 2D relative trajectory plots (RT plane, zoomed) - RTN frame (25m x 25m)
    plot_relative_trajectories_rtn_zoomed(scenarios, baseline_data, output_dir)

    # Relative distance and thrust plots
    plot_relative_distance_thrust(scenarios, baseline_data, output_dir)

    # Altitude comparison
    plot_altitude_comparison(scenarios, baseline_data, output_dir)

    # Print summary statistics
    print_summary_statistics(scenarios, baseline_data, output_dir)

    print(f"\n✅ All plots and summaries saved to: {output_dir}")


if __name__ == "__main__":
    main()
