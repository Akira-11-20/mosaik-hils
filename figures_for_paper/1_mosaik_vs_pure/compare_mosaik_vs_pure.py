"""
Compare Mosaik-based simulation vs Pure Python simulation

This script compares two simulations (Mosaik and Pure Python) by:
- Plotting position and velocity trajectories side-by-side
- Computing and plotting the difference between them
- Calculating error metrics (RMSE, MAE, Max Error)

The script expects two timestamped directories in the current directory,
each containing a hils_data.h5 file.
"""

import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


def load_simulation_data(h5_file_path: Path):
    """Load simulation data from HDF5 file"""
    with h5py.File(h5_file_path, "r") as f:
        data = {}

        # Check if using Pure Python structure (has 'data' group)
        if "data" in f.keys():
            # Pure Python structure
            data_group = f["data"]

            # Get time
            if "time_s" in data_group:
                data["time"] = data_group["time_s"][:]
            elif "time_ms" in data_group:
                data["time"] = data_group["time_ms"][:] / 1000.0  # Convert to seconds

            # Get position (looking for position_Spacecraft or similar)
            pos_cols = [col for col in data_group.keys() if "position" in col.lower()]
            if pos_cols:
                data["position"] = data_group[pos_cols[0]][:]

            # Get velocity
            vel_cols = [col for col in data_group.keys() if "velocity" in col.lower()]
            if vel_cols:
                data["velocity"] = data_group[vel_cols[0]][:]

            # Get acceleration
            acc_cols = [col for col in data_group.keys() if "acceleration" in col.lower()]
            if acc_cols:
                data["acceleration"] = data_group[acc_cols[0]][:]

        # Check if using old steps structure
        elif "steps" in f.keys():
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
        else:
            # New group-based structure (Mosaik)
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


def create_comparison_plot(mosaik_data, pure_data, mosaik_label, pure_label, output_path: Path):
    """Create comparison plot with position and velocity trajectories and their differences"""

    # Create figure with 4 rows x 1 column layout
    fig, axes = plt.subplots(4, 1, figsize=(14, 16))

    # Calculate error metrics
    pos_metrics = calculate_error_metrics(mosaik_data["position"], pure_data["position"])
    vel_metrics = calculate_error_metrics(mosaik_data["velocity"], pure_data["velocity"])

    # Ensure same time length for plotting
    min_len = min(len(mosaik_data["time"]), len(pure_data["time"]))
    time_mosaik = mosaik_data["time"][:min_len]
    time_pure = pure_data["time"][:min_len]

    # Use mosaik time as reference (they should be identical)
    time_ref = time_mosaik

    # --- Row 1: Position Trajectories ---
    ax = axes[0]
    ax.plot(time_mosaik, mosaik_data["position"][:min_len], "b-", linewidth=2, label="Mosaik", alpha=0.8)
    ax.plot(time_pure, pure_data["position"][:min_len], "r--", linewidth=2, label="Pure Python", alpha=0.8)
    ax.set_xlabel("Time [s]", fontsize=12, fontweight="bold")
    ax.set_ylabel("Position [m]", fontsize=12, fontweight="bold")
    ax.set_title("(a) Position Trajectories", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=11)

    # --- Row 2: Position Difference ---
    ax = axes[1]
    ax.plot(time_ref, pos_metrics["error_signal"], "g-", linewidth=1.5, alpha=0.8)
    ax.axhline(y=0, color="k", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlabel("Time [s]", fontsize=12, fontweight="bold")
    ax.set_ylabel("Position Error [m]", fontsize=12, fontweight="bold")
    ax.set_title("(b) Position Difference (Pure - Mosaik)", fontsize=14, fontweight="bold")
    ax.set_ylim(-1e-15, 1e-15)  # Fix y-axis to 10^-16 order
    ax.grid(True, alpha=0.3)

    # --- Row 3: Velocity Trajectories ---
    ax = axes[2]
    ax.plot(time_mosaik, mosaik_data["velocity"][:min_len], "b-", linewidth=2, label="Mosaik", alpha=0.8)
    ax.plot(time_pure, pure_data["velocity"][:min_len], "r--", linewidth=2, label="Pure Python", alpha=0.8)
    ax.set_xlabel("Time [s]", fontsize=12, fontweight="bold")
    ax.set_ylabel("Velocity [m/s]", fontsize=12, fontweight="bold")
    ax.set_title("(c) Velocity Trajectories", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=11)

    # --- Row 4: Velocity Difference ---
    ax = axes[3]
    ax.plot(time_ref, vel_metrics["error_signal"], "purple", linewidth=1.5, alpha=0.8)
    ax.axhline(y=0, color="k", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlabel("Time [s]", fontsize=12, fontweight="bold")
    ax.set_ylabel("Velocity Error [m/s]", fontsize=12, fontweight="bold")
    ax.set_title("(d) Velocity Difference (Pure - Mosaik)", fontsize=14, fontweight="bold")
    # Fix y-axis to 10^-16 order
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Comparison plot saved: {output_path}")

    return pos_metrics, vel_metrics


def main():
    # Get script directory
    script_dir = Path(__file__).parent

    # Find simulation directories
    sim_dirs = sorted([d for d in script_dir.iterdir() if d.is_dir() and d.name.startswith("202")])

    if len(sim_dirs) < 2:
        print(f"Error: Expected 2 simulation directories, found {len(sim_dirs)}")
        print(f"Available directories: {[d.name for d in sim_dirs]}")
        return

    # Assume first is Mosaik, second is Pure Python (based on timestamp order)
    mosaik_dir = sim_dirs[0]
    pure_dir = sim_dirs[1]

    print("=" * 70)
    print("Mosaik vs Pure Python Comparison")
    print("=" * 70)
    print(f"Mosaik directory: {mosaik_dir.name}")
    print(f"Pure directory:   {pure_dir.name}")
    print()

    # Load data
    print("Loading data...")
    mosaik_h5 = mosaik_dir / "hils_data.h5"
    pure_h5 = pure_dir / "hils_data.h5"

    if not mosaik_h5.exists():
        print(f"Error: Mosaik HDF5 file not found: {mosaik_h5}")
        return

    if not pure_h5.exists():
        print(f"Error: Pure Python HDF5 file not found: {pure_h5}")
        return

    mosaik_data = load_simulation_data(mosaik_h5)
    pure_data = load_simulation_data(pure_h5)

    mosaik_config = load_config(mosaik_dir)
    pure_config = load_config(pure_dir)

    print(f"Mosaik data points: {len(mosaik_data.get('time', []))}")
    print(f"Pure data points:   {len(pure_data.get('time', []))}")
    print()

    # Verify data availability
    if "position" not in mosaik_data or "velocity" not in mosaik_data:
        print("Error: Mosaik data missing position or velocity")
        return

    if "position" not in pure_data or "velocity" not in pure_data:
        print("Error: Pure Python data missing position or velocity")
        return

    # Create comparison plot
    print("Creating comparison plot...")
    output_path = script_dir / "mosaik_vs_pure_comparison.png"

    pos_metrics, vel_metrics = create_comparison_plot(
        mosaik_data, pure_data, f"Mosaik ({mosaik_dir.name})", f"Pure Python ({pure_dir.name})", output_path
    )

    # Print summary
    print()
    print("=" * 70)
    print("Error Metrics Summary")
    print("=" * 70)
    print("\nPosition Errors:")
    print(f"  RMSE:      {pos_metrics['rmse']:.8f} m ({pos_metrics['rmse_percent']:.4f}%)")
    print(f"  MAE:       {pos_metrics['mae']:.8f} m ({pos_metrics['mae_percent']:.4f}%)")
    print(f"  Max Error: {pos_metrics['max_error']:.8f} m ({pos_metrics['max_error_percent']:.4f}%)")

    print("\nVelocity Errors:")
    print(f"  RMSE:      {vel_metrics['rmse']:.8f} m/s ({vel_metrics['rmse_percent']:.4f}%)")
    print(f"  MAE:       {vel_metrics['mae']:.8f} m/s ({vel_metrics['mae_percent']:.4f}%)")
    print(f"  Max Error: {vel_metrics['max_error']:.8f} m/s ({vel_metrics['max_error_percent']:.4f}%)")

    print("\n" + "=" * 70)
    print("Comparison Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
