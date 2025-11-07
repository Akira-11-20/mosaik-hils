"""
Analyze plant delay in actual HILS simulation data

Compares:
1. command_thrust: Controller output (before comm delay)
2. measured_thrust: Plant ideal response (after comm delay, no lag)
3. actual_thrust: Plant response with 1st-order lag (after comm delay + lag)
"""

import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


def load_latest_hils_data():
    """Find and load the latest HILS data file"""
    results_dir = Path("results")

    if not results_dir.exists():
        print("❌ No results directory found")
        sys.exit(1)

    # Find all subdirectories
    subdirs = [d for d in results_dir.iterdir() if d.is_dir()]

    if not subdirs:
        print("❌ No result directories found")
        sys.exit(1)

    # Sort by modification time (latest first)
    latest_dir = sorted(subdirs, key=lambda x: x.stat().st_mtime, reverse=True)[0]

    h5_file = latest_dir / "hils_data.h5"

    if not h5_file.exists():
        print(f"❌ No HDF5 file found in {latest_dir}")
        sys.exit(1)

    print(f"📂 Loading: {h5_file}")
    return h5_file


def load_hdf5_hierarchical(h5_path):
    """Load HDF5 data (hierarchical structure)"""
    data = {}

    with h5py.File(h5_path, "r") as f:

        def read_group(group, prefix=""):
            for key in group.keys():
                item = group[key]
                if isinstance(item, h5py.Group):
                    read_group(item, f"{key}_")
                else:
                    # Flatten key: attr_name_group_name
                    parts = item.name.split("/")
                    if len(parts) >= 2:
                        group_name = parts[1]
                        attr_name = parts[-1]
                        flat_key = f"{attr_name}_{group_name}" if group_name != "time" else attr_name
                    else:
                        flat_key = key
                    data[flat_key] = item[:]

        read_group(f)

    return data


def analyze_plant_delay(h5_file):
    """Analyze plant delay characteristics"""

    data = load_hdf5_hierarchical(h5_file)

    print(f"\n✓ Loaded {len(data)} datasets")

    # Extract time
    time_s = data.get("time_s", data.get("time_ms", np.array([])) / 1000.0)

    # Find thrust keys
    command_key = None
    measured_key = None
    actual_key = None

    for key in data.keys():
        if "command_thrust" in key and "Controller" in key:
            command_key = key
        elif "measured_thrust" in key and "Plant" in key:
            measured_key = key
        elif "actual_thrust" in key and "Plant" in key:
            actual_key = key

    if not all([command_key, measured_key, actual_key]):
        print("❌ Missing thrust data")
        print(f"  command: {command_key}")
        print(f"  measured: {measured_key}")
        print(f"  actual: {actual_key}")
        return

    command_thrust = data[command_key]
    measured_thrust = data[measured_key]
    actual_thrust = data[actual_key]

    print("\n📊 Data keys:")
    print(f"  command_thrust: {command_key}")
    print(f"  measured_thrust: {measured_key}")
    print(f"  actual_thrust: {actual_key}")

    print("\n📈 Data statistics:")
    print(f"  Time range: {time_s[0]:.3f}s to {time_s[-1]:.3f}s ({len(time_s)} samples)")
    print(f"  command_thrust: [{command_thrust.min():.2f}, {command_thrust.max():.2f}] N")
    print(f"  measured_thrust: [{measured_thrust.min():.2f}, {measured_thrust.max():.2f}] N")
    print(f"  actual_thrust: [{actual_thrust.min():.2f}, {actual_thrust.max():.2f}] N")

    # Calculate delays
    print("\n🔍 Delay Analysis:")

    # 1. Communication delay: command → measured
    # Find first non-zero command
    cmd_nonzero = np.where(command_thrust > 0.1)[0]
    meas_nonzero = np.where(measured_thrust > 0.1)[0]

    if len(cmd_nonzero) > 0 and len(meas_nonzero) > 0:
        comm_delay_samples = meas_nonzero[0] - cmd_nonzero[0]
        comm_delay_ms = comm_delay_samples * (time_s[1] - time_s[0]) * 1000
        print("  Communication delay (command → measured):")
        print(f"    {comm_delay_samples} samples = {comm_delay_ms:.1f} ms")

    # 2. Plant lag: measured → actual
    # Find step change in measured_thrust
    meas_diff = np.diff(measured_thrust)
    step_indices = np.where(np.abs(meas_diff) > 10.0)[0]  # Large changes

    if len(step_indices) > 0:
        # Analyze first major step
        step_idx = step_indices[0]
        time_s[step_idx]

        # Extract window around step
        window = 1000  # samples (0.1s at 0.0001s resolution)
        start_idx = max(0, step_idx - 100)
        end_idx = min(len(time_s), step_idx + window)

        time_s[start_idx:end_idx]
        measured_thrust[start_idx:end_idx]
        actual_thrust[start_idx:end_idx]

        # Find when actual reaches 63.2% of step
        if step_idx < len(measured_thrust) - 1:
            before_val = measured_thrust[step_idx]
            after_val = measured_thrust[step_idx + 1]
            step_size = after_val - before_val

            if abs(step_size) > 1.0:  # Significant step
                target_val = before_val + 0.632 * step_size

                # Find crossing point
                actual_after = actual_thrust[step_idx:]
                crossing = np.where(np.diff(np.sign(actual_after - target_val)))[0]

                if len(crossing) > 0:
                    tau_samples = crossing[0]
                    tau_ms = tau_samples * (time_s[1] - time_s[0]) * 1000
                    print("  Plant time constant (estimated from step response):")
                    print(f"    τ ≈ {tau_ms:.1f} ms")

    # 3. RMS difference
    rms_meas_actual = np.sqrt(np.mean((measured_thrust - actual_thrust) ** 2))
    print(f"  RMS difference (measured vs actual): {rms_meas_actual:.2f} N")

    # Plotting
    fig = plt.figure(figsize=(14, 10))

    # Plot 1: Full time series
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(time_s, command_thrust, "b-", label="command_thrust (Controller output)", lw=1.5, alpha=0.7)
    ax1.plot(time_s, measured_thrust, "g-", label="measured_thrust (ideal, after comm delay)", lw=1.5, alpha=0.7)
    ax1.plot(time_s, actual_thrust, "r-", label="actual_thrust (with plant lag)", lw=1.5)
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Thrust [N]")
    ax1.set_title("Plant Delay Analysis: Full Simulation")
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)

    # Plot 2: Zoom on initial response (first 0.5s)
    ax2 = plt.subplot(3, 1, 2)
    zoom_end = int(0.5 / (time_s[1] - time_s[0]))
    ax2.plot(time_s[:zoom_end], command_thrust[:zoom_end], "b-", label="command_thrust", lw=2, alpha=0.7)
    ax2.plot(time_s[:zoom_end], measured_thrust[:zoom_end], "g-", label="measured_thrust", lw=2, alpha=0.7)
    ax2.plot(time_s[:zoom_end], actual_thrust[:zoom_end], "r-", label="actual_thrust", lw=2)
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Thrust [N]")
    ax2.set_title("Plant Delay Analysis: Initial Response (0-0.5s)")
    ax2.legend(loc="best")
    ax2.grid(True, alpha=0.3)

    # Plot 3: Lag analysis
    ax3 = plt.subplot(3, 1, 3)
    comm_lag = measured_thrust - command_thrust
    plant_lag = actual_thrust - measured_thrust
    total_lag = actual_thrust - command_thrust

    ax3.plot(time_s, comm_lag, "g-", label="Communication lag (measured - command)", lw=1.5, alpha=0.7)
    ax3.plot(time_s, plant_lag, "orange", label="Plant lag (actual - measured)", lw=1.5, alpha=0.7)
    ax3.plot(time_s, total_lag, "r-", label="Total lag (actual - command)", lw=1.5)
    ax3.axhline(0, color="k", linestyle=":", lw=1)
    ax3.set_xlabel("Time [s]")
    ax3.set_ylabel("Lag [N]")
    ax3.set_title("Lag Decomposition")
    ax3.legend(loc="best")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    output_dir = h5_file.parent
    output_file = output_dir / "plant_delay_analysis.png"
    plt.savefig(output_file, dpi=300)
    print(f"\n📊 Saved: {output_file}")

    return fig


if __name__ == "__main__":
    print("=" * 70)
    print("PLANT DELAY ANALYSIS - HILS Simulation Data")
    print("=" * 70 + "\n")

    h5_file = load_latest_hils_data()
    analyze_plant_delay(h5_file)

    print("\n" + "=" * 70)
    print("✅ Analysis completed")
    print("=" * 70 + "\n")

    plt.show()
