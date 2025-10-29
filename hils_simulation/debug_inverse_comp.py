"""
Debug script to analyze inverse compensation simulation results
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import json
import sys


def analyze_inverse_comp_data(filepath):
    """Analyze and visualize inverse compensation simulation data"""

    with h5py.File(filepath, "r") as f:
        # Load time series
        time_s = f["data"]["time_s"][:]

        # Load position and control signals
        position = f["data"]["position_EnvSim-0.Spacecraft1DOF_0"][:]
        velocity = f["data"]["velocity_EnvSim-0.Spacecraft1DOF_0"][:]
        acceleration = f["data"]["acceleration_EnvSim-0.Spacecraft1DOF_0"][:]

        # Load control signals
        error = f["data"]["error_ControllerSim-0.PIDController_0"][:]
        controller_thrust = f["data"]["command_ControllerSim-0.PIDController_0_thrust"][:]

        # Load compensated output
        compensated_thrust = f["data"]["compensated_output_InverseCompSim-0.cmd_compensator_thrust"][:]

        # Load plant measurement
        measured_thrust = f["data"]["measured_thrust_PlantSim-0.ThrustStand_0"][:]

        # Load force applied to spacecraft
        force = f["data"]["force_EnvSim-0.Spacecraft1DOF_0"][:]

        # Load metadata if available
        metadata = {}
        for key in f.attrs.keys():
            try:
                metadata[key] = json.loads(f.attrs[key]) if isinstance(f.attrs[key], str) else f.attrs[key]
            except:
                metadata[key] = str(f.attrs[key])

    # Print summary statistics
    print("=" * 80)
    print("INVERSE COMPENSATION SIMULATION ANALYSIS")
    print("=" * 80)
    print(f"\nSimulation time: {time_s[0]:.6f}s to {time_s[-1]:.6f}s")
    print(f"Number of data points: {len(time_s)}")
    print(f"Time resolution: {(time_s[-1] - time_s[0]) / (len(time_s) - 1):.6f}s")

    print("\n" + "-" * 80)
    print("POSITION TRACKING")
    print("-" * 80)
    print(f"Initial position: {position[0]:.6f} m")
    print(f"Final position: {position[-1]:.6f} m")
    print(f"Position range: [{np.min(position):.6f}, {np.max(position):.6f}] m")
    print(f"Position std dev: {np.std(position):.6f} m")

    print("\n" + "-" * 80)
    print("VELOCITY & ACCELERATION")
    print("-" * 80)
    print(f"Velocity range: [{np.min(velocity):.6f}, {np.max(velocity):.6f}] m/s")
    print(f"Velocity std dev: {np.std(velocity):.6f} m/s")
    print(f"Acceleration range: [{np.min(acceleration):.6f}, {np.max(acceleration):.6f}] m/s²")

    print("\n" + "-" * 80)
    print("CONTROL SIGNALS")
    print("-" * 80)
    print(f"Control error range: [{np.min(error):.6f}, {np.max(error):.6f}] m")
    print(f"Controller thrust range: [{np.min(controller_thrust):.6f}, {np.max(controller_thrust):.6f}] N")
    print(f"Compensated thrust range: [{np.min(compensated_thrust):.6f}, {np.max(compensated_thrust):.6f}] N")
    print(f"Measured thrust range: [{np.min(measured_thrust):.6f}, {np.max(measured_thrust):.6f}] N")
    print(f"Applied force range: [{np.min(force):.6f}, {np.max(force):.6f}] N")

    # Check for instability
    print("\n" + "-" * 80)
    print("STABILITY ANALYSIS")
    print("-" * 80)

    # Check if position is diverging
    pos_diff = np.diff(np.abs(position))
    if np.mean(pos_diff[-1000:]) > 0:
        print("⚠ WARNING: Position appears to be diverging!")

    # Check if velocity is increasing
    vel_abs = np.abs(velocity)
    if np.mean(vel_abs[-1000:]) > np.mean(vel_abs[:1000]) * 2:
        print("⚠ WARNING: Velocity is increasing significantly!")

    # Check for NaN or Inf
    signals = {
        "position": position,
        "velocity": velocity,
        "controller_thrust": controller_thrust,
        "compensated_thrust": compensated_thrust,
        "force": force,
    }

    for name, signal in signals.items():
        if np.any(np.isnan(signal)):
            print(f"⚠ WARNING: NaN detected in {name}!")
        if np.any(np.isinf(signal)):
            print(f"⚠ WARNING: Inf detected in {name}!")

    # Metadata
    if metadata:
        print("\n" + "-" * 80)
        print("SIMULATION METADATA")
        print("-" * 80)
        for key, value in metadata.items():
            if isinstance(value, dict):
                print(f"{key}:")
                for k, v in value.items():
                    print(f"  {k}: {v}")
            else:
                print(f"{key}: {value}")

    # Create detailed visualization
    fig, axes = plt.subplots(5, 1, figsize=(12, 14), sharex=True)

    # Plot 1: Position and error
    ax1 = axes[0]
    ax1.plot(time_s, position, "b-", label="Position", linewidth=1.5)
    ax1.axhline(y=0, color="k", linestyle="--", alpha=0.3)
    ax1.set_ylabel("Position [m]", fontsize=11)
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Inverse Compensation Simulation - Detailed Analysis", fontsize=13, fontweight="bold")

    ax1_twin = ax1.twinx()
    ax1_twin.plot(time_s, error, "r-", label="Error", alpha=0.7, linewidth=1)
    ax1_twin.set_ylabel("Error [m]", color="r", fontsize=11)
    ax1_twin.tick_params(axis="y", labelcolor="r")
    ax1_twin.legend(loc="upper left")

    # Plot 2: Velocity
    ax2 = axes[1]
    ax2.plot(time_s, velocity, "g-", label="Velocity", linewidth=1.5)
    ax2.axhline(y=0, color="k", linestyle="--", alpha=0.3)
    ax2.set_ylabel("Velocity [m/s]", fontsize=11)
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    # Plot 3: Thrust signals comparison
    ax3 = axes[2]
    ax3.plot(time_s, controller_thrust, "b-", label="Controller Output", alpha=0.7, linewidth=1)
    ax3.plot(
        time_s,
        compensated_thrust,
        "r-",
        label="Compensated (After Inv. Comp)",
        alpha=0.7,
        linewidth=1,
    )
    ax3.plot(time_s, measured_thrust, "g--", label="Measured (Plant)", alpha=0.5, linewidth=1)
    ax3.set_ylabel("Thrust [N]", fontsize=11)
    ax3.legend(loc="upper right")
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color="k", linestyle="--", alpha=0.3)

    # Plot 4: Applied force to spacecraft
    ax4 = axes[3]
    ax4.plot(time_s, force, "purple", label="Applied Force", linewidth=1.5)
    ax4.axhline(y=0, color="k", linestyle="--", alpha=0.3)
    # Add gravity reference
    ax4.axhline(y=-9.81, color="orange", linestyle=":", alpha=0.5, label="Gravity (-9.81 N)")
    ax4.set_ylabel("Force [N]", fontsize=11)
    ax4.legend(loc="upper right")
    ax4.grid(True, alpha=0.3)

    # Plot 5: Acceleration
    ax5 = axes[4]
    ax5.plot(time_s, acceleration, "orange", label="Acceleration", linewidth=1.5)
    ax5.axhline(y=0, color="k", linestyle="--", alpha=0.3)
    ax5.axhline(y=-9.81, color="r", linestyle=":", alpha=0.5, label="Gravity (-9.81 m/s²)")
    ax5.set_ylabel("Acceleration [m/s²]", fontsize=11)
    ax5.set_xlabel("Time [s]", fontsize=11)
    ax5.legend(loc="upper right")
    ax5.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_path = filepath.replace(".h5", "_detailed_analysis.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n✓ Detailed plot saved to: {output_path}")

    # Create zoomed-in view for early time behavior
    fig2, axes2 = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    # Find first 0.5 seconds or 20% of data
    zoom_idx = int(min(0.5 / (time_s[1] - time_s[0]), len(time_s) * 0.2))
    time_zoom = time_s[:zoom_idx]

    axes2[0].plot(time_zoom, position[:zoom_idx], "b-", linewidth=1.5)
    axes2[0].set_ylabel("Position [m]", fontsize=11)
    axes2[0].set_title("Early Time Behavior (First 0.5s or 20%)", fontsize=13, fontweight="bold")
    axes2[0].grid(True, alpha=0.3)

    axes2[1].plot(time_zoom, controller_thrust[:zoom_idx], "b-", label="Controller", alpha=0.7)
    axes2[1].plot(time_zoom, compensated_thrust[:zoom_idx], "r-", label="Compensated", alpha=0.7)
    axes2[1].set_ylabel("Thrust [N]", fontsize=11)
    axes2[1].legend()
    axes2[1].grid(True, alpha=0.3)

    axes2[2].plot(time_zoom, force[:zoom_idx], "purple", linewidth=1.5)
    axes2[2].axhline(y=-9.81, color="orange", linestyle=":", alpha=0.5, label="Gravity")
    axes2[2].set_ylabel("Applied Force [N]", fontsize=11)
    axes2[2].set_xlabel("Time [s]", fontsize=11)
    axes2[2].legend()
    axes2[2].grid(True, alpha=0.3)

    plt.tight_layout()

    output_path_zoom = filepath.replace(".h5", "_early_time_zoom.png")
    plt.savefig(output_path_zoom, dpi=150, bbox_inches="tight")
    print(f"✓ Zoomed plot saved to: {output_path_zoom}")

    plt.show()

    return {
        "time": time_s,
        "position": position,
        "velocity": velocity,
        "error": error,
        "controller_thrust": controller_thrust,
        "compensated_thrust": compensated_thrust,
        "measured_thrust": measured_thrust,
        "force": force,
    }


if __name__ == "__main__":
    filepath = "results/20251022-013456_inverse_comp/hils_data.h5"
    if len(sys.argv) > 1:
        filepath = sys.argv[1]

    print(f"Analyzing: {filepath}\n")
    data = analyze_inverse_comp_data(filepath)
