"""
Analyze integral windup and inverse compensation interaction
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent / "archive" / "comparisons"))

import json

import h5py
import matplotlib.pyplot as plt
import numpy as np
from hdf5_helper import get_dataset, load_hdf5_data


def analyze_integral_behavior(filepath):
    """Analyze how inverse compensation affects integral windup"""

    # Load data with new helper (supports both old and new formats)
    data = load_hdf5_data(filepath)

    # Load time series
    time_s = data.get("time_s")

    # Load control signals
    position = get_dataset(data, "position_EnvSim-0.Spacecraft1DOF_0") or get_dataset(
        data, "position_EnvSim-0_Spacecraft1DOF_0"
    )
    velocity = get_dataset(data, "velocity_EnvSim-0.Spacecraft1DOF_0") or get_dataset(
        data, "velocity_EnvSim-0_Spacecraft1DOF_0"
    )
    error = get_dataset(data, "error_ControllerSim-0.PIDController_0") or get_dataset(
        data, "error_ControllerSim-0_PIDController_0"
    )

    # Load thrust signals
    controller_thrust = get_dataset(data, "command_ControllerSim-0.PIDController_0_thrust") or get_dataset(
        data, "command_thrust_ControllerSim-0_PIDController_0"
    )
    compensated_thrust = get_dataset(data, "compensated_output_InverseCompSim-0.cmd_compensator_thrust") or get_dataset(
        data, "compensated_output_thrust_InverseCompSim-0_cmd_compensator"
    )
    get_dataset(data, "measured_thrust_PlantSim-0.ThrustStand_0") or get_dataset(
        data, "measured_thrust_PlantSim-0_ThrustStand_0"
    )
    applied_force = get_dataset(data, "force_EnvSim-0.Spacecraft1DOF_0") or get_dataset(
        data, "force_EnvSim-0_Spacecraft1DOF_0"
    )

    # Load metadata
    with h5py.File(filepath, "r") as f:
        metadata = {}
        for key in f.attrs.keys():
            try:
                metadata[key] = json.loads(f.attrs[key]) if isinstance(f.attrs[key], str) else f.attrs[key]
            except:
                metadata[key] = str(f.attrs[key])

    # Extract control parameters
    if "simulation_parameters" in metadata:
        params = metadata["simulation_parameters"]
        kp = params["control"]["kp"]
        ki = params["control"]["ki"]
        kd = params["control"]["kd"]
        integral_limit = params["control"]["integral_limit"]
        target = params["control"]["target_position_m"]
        comp_gain = params["inverse_compensation"]["gain"]
    else:
        print("Warning: Could not find simulation parameters in metadata")
        kp, ki, kd = 15.0, 5.0, 8.0
        integral_limit = 100.0
        target = 5.0
        comp_gain = 15.0

    print("=" * 80)
    print("INTEGRAL WINDUP & INVERSE COMPENSATION ANALYSIS")
    print("=" * 80)
    print("\nControl Parameters:")
    print(f"  Kp = {kp}, Ki = {ki}, Kd = {kd}")
    print(f"  Target position = {target} m")
    print(f"  Integral limit = {integral_limit} m·s")
    print(f"  Compensation gain = {comp_gain}")

    # Reconstruct integral term from controller behavior
    # From controller: thrust = Kp*error + Ki*integral - Kd*velocity
    # Therefore: integral ≈ (thrust - Kp*error + Kd*velocity) / Ki

    dt = time_s[1] - time_s[0]

    # Estimate integral from controller output
    estimated_integral = (controller_thrust - kp * error + kd * velocity) / ki

    # Compute cumulative integral (what it would be without limits)
    cumulative_integral = np.cumsum(error * dt)

    # Apply saturation to see limited integral
    limited_integral = np.clip(cumulative_integral, -integral_limit, integral_limit)

    print(f"\n{'=' * 80}")
    print("INTEGRAL TERM ANALYSIS")
    print(f"{'=' * 80}")
    print(f"Estimated integral range: [{np.min(estimated_integral):.3f}, {np.max(estimated_integral):.3f}] m·s")
    print(f"Cumulative integral range: [{np.min(cumulative_integral):.3f}, {np.max(cumulative_integral):.3f}] m·s")
    print(f"Limited integral range: [{np.min(limited_integral):.3f}, {np.max(limited_integral):.3f}] m·s")

    # Check if integral is saturating
    saturation_points = np.sum((cumulative_integral > integral_limit) | (cumulative_integral < -integral_limit))
    saturation_percent = 100 * saturation_points / len(cumulative_integral)
    print(f"\nIntegral saturation: {saturation_points}/{len(cumulative_integral)} points ({saturation_percent:.1f}%)")

    if saturation_percent > 10:
        print("⚠ WARNING: Significant integral saturation detected!")

    # Analyze amplification from inverse compensation
    print(f"\n{'=' * 80}")
    print("INVERSE COMPENSATION AMPLIFICATION")
    print(f"{'=' * 80}")

    # Compute thrust changes (first derivative)
    controller_delta = np.diff(controller_thrust)
    compensated_delta = np.diff(compensated_thrust)

    # Compensation formula: y_comp = gain * y[k] - (gain-1) * y[k-1]
    # Change in output: Δy_comp ≈ gain * Δy (for slowly varying signals)

    amplification_ratio = np.zeros_like(controller_delta)
    nonzero_mask = np.abs(controller_delta) > 1e-6
    amplification_ratio[nonzero_mask] = compensated_delta[nonzero_mask] / controller_delta[nonzero_mask]

    print(f"Controller thrust change range: [{np.min(controller_delta):.3f}, {np.max(controller_delta):.3f}] N")
    print(f"Compensated thrust change range: [{np.min(compensated_delta):.3f}, {np.max(compensated_delta):.3f}] N")
    print(
        f"Amplification ratio range: [{np.min(amplification_ratio[nonzero_mask]):.3f}, {np.max(amplification_ratio[nonzero_mask]):.3f}]"
    )
    print(f"Mean amplification ratio: {np.mean(np.abs(amplification_ratio[nonzero_mask])):.3f}")
    print(f"Expected amplification (gain): {comp_gain:.3f}")

    # Check for instability onset
    print(f"\n{'=' * 80}")
    print("INSTABILITY ANALYSIS")
    print(f"{'=' * 80}")

    # Find when position starts diverging significantly
    pos_abs = np.abs(position)
    threshold = 10 * target  # 10x target position
    divergence_idx = np.where(pos_abs > threshold)[0]

    if len(divergence_idx) > 0:
        divergence_time = time_s[divergence_idx[0]]
        print(f"⚠ Divergence detected at t = {divergence_time:.3f}s (position > {threshold:.1f}m)")

        # Look at behavior just before divergence
        window_start = max(0, divergence_idx[0] - 1000)
        window_end = min(len(time_s), divergence_idx[0] + 100)

        print("\nBehavior around divergence onset:")
        print(f"  Time window: [{time_s[window_start]:.3f}, {time_s[window_end]:.3f}]s")
        print(
            f"  Position range: [{np.min(position[window_start:window_end]):.3f}, {np.max(position[window_start:window_end]):.3f}]m"
        )
        print(
            f"  Controller thrust range: [{np.min(controller_thrust[window_start:window_end]):.3f}, {np.max(controller_thrust[window_start:window_end]):.3f}]N"
        )
        print(
            f"  Compensated thrust range: [{np.min(compensated_thrust[window_start:window_end]):.3f}, {np.max(compensated_thrust[window_start:window_end]):.3f}]N"
        )
        print(
            f"  Estimated integral range: [{np.min(estimated_integral[window_start:window_end]):.3f}, {np.max(estimated_integral[window_start:window_end]):.3f}]m·s"
        )
    else:
        print("✓ No significant divergence detected (threshold = 10x target)")

    # Create comprehensive visualization
    fig, axes = plt.subplots(6, 1, figsize=(14, 16), sharex=True)

    # Plot 1: Position and error
    ax1 = axes[0]
    ax1.plot(time_s, position, "b-", label="Position", linewidth=1.5)
    ax1.axhline(y=target, color="g", linestyle="--", alpha=0.5, label=f"Target ({target}m)")
    ax1.axhline(y=0, color="k", linestyle="--", alpha=0.3)
    ax1.set_ylabel("Position [m]", fontsize=11)
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Integral Windup & Inverse Compensation Analysis", fontsize=13, fontweight="bold")

    ax1_twin = ax1.twinx()
    ax1_twin.plot(time_s, error, "r-", label="Error", alpha=0.7, linewidth=1)
    ax1_twin.set_ylabel("Error [m]", color="r", fontsize=11)
    ax1_twin.tick_params(axis="y", labelcolor="r")
    ax1_twin.legend(loc="upper right")

    # Plot 2: Integral terms comparison
    ax2 = axes[1]
    ax2.plot(
        time_s,
        estimated_integral,
        "b-",
        label="Estimated Integral (from controller)",
        linewidth=1.5,
        alpha=0.8,
    )
    ax2.plot(
        time_s,
        cumulative_integral,
        "g--",
        label="Cumulative Integral (unlimited)",
        linewidth=1,
        alpha=0.6,
    )
    ax2.plot(time_s, limited_integral, "r:", label="Limited Integral (saturated)", linewidth=2, alpha=0.7)
    ax2.axhline(
        y=integral_limit,
        color="orange",
        linestyle="--",
        alpha=0.5,
        label=f"Limit (±{integral_limit})",
    )
    ax2.axhline(y=-integral_limit, color="orange", linestyle="--", alpha=0.5)
    ax2.set_ylabel("Integral [m·s]", fontsize=11)
    ax2.legend(loc="best", fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Thrust signals
    ax3 = axes[2]
    ax3.plot(time_s, controller_thrust, "b-", label="Controller Output", alpha=0.7, linewidth=1.5)
    ax3.plot(time_s, compensated_thrust, "r-", label="Compensated (×gain)", alpha=0.7, linewidth=1.5)
    ax3.set_ylabel("Thrust [N]", fontsize=11)
    ax3.legend(loc="best")
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color="k", linestyle="--", alpha=0.3)

    # Plot 4: Thrust rate of change (derivative)
    ax4 = axes[3]
    time_diff = time_s[1:]
    ax4.plot(time_diff, controller_delta, "b-", label="Controller Δthrust", alpha=0.7, linewidth=1)
    ax4.plot(time_diff, compensated_delta, "r-", label="Compensated Δthrust", alpha=0.7, linewidth=1)
    ax4.set_ylabel("Thrust Change [N/step]", fontsize=11)
    ax4.legend(loc="best")
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color="k", linestyle="--", alpha=0.3)

    # Plot 5: Amplification ratio
    ax5 = axes[4]
    # Only plot where controller delta is significant
    plot_mask = np.abs(controller_delta) > 1e-3
    if np.any(plot_mask):
        ax5.scatter(
            time_diff[plot_mask],
            amplification_ratio[plot_mask],
            c="purple",
            alpha=0.3,
            s=1,
            label="Amplification Ratio",
        )
        ax5.axhline(
            y=comp_gain,
            color="r",
            linestyle="--",
            alpha=0.7,
            label=f"Expected Gain ({comp_gain:.1f})",
            linewidth=2,
        )
        ax5.axhline(y=1, color="g", linestyle="--", alpha=0.5, label="No amplification (1.0)")
        ax5.set_ylabel("Amplification Ratio", fontsize=11)
        ax5.set_ylim([-50, 50])  # Limit y-axis for readability
        ax5.legend(loc="best")
        ax5.grid(True, alpha=0.3)
    else:
        ax5.text(
            0.5,
            0.5,
            "No significant controller changes",
            ha="center",
            va="center",
            transform=ax5.transAxes,
        )

    # Plot 6: Applied force and velocity
    ax6 = axes[5]
    ax6.plot(time_s, applied_force, "purple", label="Applied Force", linewidth=1.5, alpha=0.7)
    ax6.axhline(y=0, color="k", linestyle="--", alpha=0.3)
    ax6.axhline(y=-9.81, color="orange", linestyle=":", alpha=0.5, label="Gravity (-9.81 N)")
    ax6.set_ylabel("Force [N]", fontsize=11)
    ax6.set_xlabel("Time [s]", fontsize=11)
    ax6.legend(loc="upper left")
    ax6.grid(True, alpha=0.3)

    ax6_twin = ax6.twinx()
    ax6_twin.plot(time_s, velocity, "g-", label="Velocity", alpha=0.6, linewidth=1)
    ax6_twin.set_ylabel("Velocity [m/s]", color="g", fontsize=11)
    ax6_twin.tick_params(axis="y", labelcolor="g")
    ax6_twin.legend(loc="upper right")

    plt.tight_layout()

    # Save figure
    output_path = filepath.replace(".h5", "_integral_analysis.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n✓ Analysis plot saved to: {output_path}")

    # Create zoomed view around divergence if detected
    if len(divergence_idx) > 0:
        fig2, axes2 = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

        # Zoom window: ±500 samples around divergence
        zoom_start = max(0, divergence_idx[0] - 5000)
        zoom_end = min(len(time_s), divergence_idx[0] + 1000)
        time_zoom = time_s[zoom_start:zoom_end]

        axes2[0].plot(time_zoom, position[zoom_start:zoom_end], "b-", linewidth=1.5)
        axes2[0].axvline(x=divergence_time, color="r", linestyle="--", alpha=0.5, label="Divergence onset")
        axes2[0].axhline(y=target, color="g", linestyle="--", alpha=0.5)
        axes2[0].set_ylabel("Position [m]", fontsize=11)
        axes2[0].set_title(f"Behavior Around Divergence (t={divergence_time:.3f}s)", fontsize=13, fontweight="bold")
        axes2[0].legend()
        axes2[0].grid(True, alpha=0.3)

        axes2[1].plot(time_zoom, estimated_integral[zoom_start:zoom_end], "b-", linewidth=1.5)
        axes2[1].axvline(x=divergence_time, color="r", linestyle="--", alpha=0.5)
        axes2[1].axhline(y=integral_limit, color="orange", linestyle="--", alpha=0.5)
        axes2[1].axhline(y=-integral_limit, color="orange", linestyle="--", alpha=0.5)
        axes2[1].set_ylabel("Integral [m·s]", fontsize=11)
        axes2[1].grid(True, alpha=0.3)

        axes2[2].plot(time_zoom, controller_thrust[zoom_start:zoom_end], "b-", label="Controller", alpha=0.7)
        axes2[2].plot(time_zoom, compensated_thrust[zoom_start:zoom_end], "r-", label="Compensated", alpha=0.7)
        axes2[2].axvline(x=divergence_time, color="r", linestyle="--", alpha=0.5)
        axes2[2].set_ylabel("Thrust [N]", fontsize=11)
        axes2[2].legend()
        axes2[2].grid(True, alpha=0.3)

        axes2[3].plot(time_zoom, velocity[zoom_start:zoom_end], "g-", linewidth=1.5)
        axes2[3].axvline(x=divergence_time, color="r", linestyle="--", alpha=0.5)
        axes2[3].set_ylabel("Velocity [m/s]", fontsize=11)
        axes2[3].set_xlabel("Time [s]", fontsize=11)
        axes2[3].grid(True, alpha=0.3)

        plt.tight_layout()

        output_path_zoom = filepath.replace(".h5", "_divergence_zoom.png")
        plt.savefig(output_path_zoom, dpi=150, bbox_inches="tight")
        print(f"✓ Divergence zoom plot saved to: {output_path_zoom}")

    plt.show()

    return {
        "time": time_s,
        "estimated_integral": estimated_integral,
        "cumulative_integral": cumulative_integral,
        "limited_integral": limited_integral,
        "controller_thrust": controller_thrust,
        "compensated_thrust": compensated_thrust,
        "amplification_ratio": amplification_ratio,
    }


if __name__ == "__main__":
    filepath = "results/20251022-013456_inverse_comp/hils_data.h5"
    if len(sys.argv) > 1:
        filepath = sys.argv[1]

    print(f"Analyzing: {filepath}\n")
    data = analyze_integral_behavior(filepath)
