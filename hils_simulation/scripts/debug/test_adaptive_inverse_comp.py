"""
Test Adaptive Inverse Compensation with Dynamic Time Constants

This script demonstrates how the inverse compensator adapts its gain
based on dynamic time constants calculated using the same model as the plant.

Tests:
1. Fixed time constant (baseline)
2. Linear thrust-dependent time constant
3. Hybrid model (thrust + thermal effects)
4. Comparison of compensation effectiveness

Updated: Now uses tau_model to calculate time constants (same as plant)
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add hils_simulation to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models import create_time_constant_model


def simulate_first_order_lag(u_profile, tau_values, dt):
    """
    Simulate first-order lag with time-varying time constant

    Args:
        u_profile: Input signal array
        tau_values: Time constant array (same length as u_profile)
        dt: Time step [ms]

    Returns:
        Output signal array
    """
    y = np.zeros(len(u_profile))

    for k in range(len(u_profile) - 1):
        tau = tau_values[k]
        y[k + 1] = y[k] + (dt / tau) * (u_profile[k] - y[k])

    return y


def apply_inverse_compensation(signal, gain_values):
    """
    Apply inverse compensation with time-varying gain

    Formula: y_comp[k] = gain * y[k] - (gain-1) * y[k-1]

    Args:
        signal: Input signal array
        gain_values: Gain array (same length as signal)

    Returns:
        Compensated signal array
    """
    compensated = np.zeros(len(signal))
    compensated[0] = signal[0]

    for k in range(1, len(signal)):
        gain = gain_values[k]
        compensated[k] = gain * signal[k] - (gain - 1.0) * signal[k - 1]

    return compensated


def test_adaptive_compensation():
    """Main test: Adaptive inverse compensation with dynamic tau"""

    print("=" * 80)
    print("Adaptive Inverse Compensation Test")
    print("=" * 80)

    # Simulation parameters
    dt = 1.0  # [ms] time step
    t_sim = 2000.0  # [ms] simulation time
    n_steps = int(t_sim / dt)
    time = np.arange(n_steps) * dt

    # Create thrust profile (step changes to trigger different tau values)
    thrust_profile = np.zeros(n_steps)
    thrust_profile[100:600] = 50.0  # 50N from 100-600ms
    thrust_profile[800:1200] = 75.0  # 75N from 800-1200ms
    thrust_profile[1400:1800] = 30.0  # 30N from 1400-1800ms

    # Base time constant
    base_tau = 100.0  # [ms]

    # Create time constant models to test
    models_to_test = {
        "Constant (τ=100ms)": {
            "model": create_time_constant_model("constant"),
            "color": "blue",
            "linestyle": "-",
        },
        "Linear (k=0.3)": {
            "model": create_time_constant_model("linear", sensitivity=0.3),
            "color": "green",
            "linestyle": "--",
        },
        "Hybrid (k=0.2, thermal)": {
            "model": create_time_constant_model(
                "hybrid",
                thrust_sensitivity=0.2,
                heating_rate=0.0005,
                cooling_rate=0.01,
                thermal_sensitivity=0.03,
            ),
            "color": "purple",
            "linestyle": "-.",
        },
    }

    # Create figure with multiple subplots
    fig, axes = plt.subplots(5, 1, figsize=(14, 18))

    # Subplot 0: Thrust profile (input)
    ax = axes[0]
    ax.plot(time, thrust_profile, "k-", linewidth=2, label="Thrust Command")
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Thrust [N]")
    ax.set_title("Input Thrust Profile")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Store results for each model
    results = {}

    for model_name, model_info in models_to_test.items():
        print(f"\n{'=' * 80}")
        print(f"Testing: {model_name}")
        print(f"{'=' * 80}")

        tau_model = model_info["model"]
        color = model_info["color"]
        linestyle = model_info["linestyle"]

        # Calculate time constants based on thrust
        tau_values = np.zeros(n_steps)
        for i, thrust in enumerate(thrust_profile):
            tau_values[i] = tau_model.get_time_constant(
                thrust=thrust,
                base_tau=base_tau,
                dt=dt,
            )

        # Simulate plant with first-order lag
        plant_output = simulate_first_order_lag(thrust_profile, tau_values, dt)

        # Test both fixed gain and adaptive gain

        # 1. Fixed gain (traditional approach - uses average tau)
        avg_tau = np.mean(tau_values)
        fixed_gain = avg_tau * 0.1  # Using 0.1 as tau_to_gain_ratio
        fixed_gain_values = np.full(n_steps, fixed_gain)
        compensated_fixed = apply_inverse_compensation(plant_output, fixed_gain_values)

        # 2. Adaptive gain (proposed approach - tracks actual tau)
        tau_to_gain_ratio = 0.1
        adaptive_gain_values = tau_values * tau_to_gain_ratio
        compensated_adaptive = apply_inverse_compensation(plant_output, adaptive_gain_values)

        # Calculate errors (compared to ideal thrust command)
        error_no_comp = thrust_profile - plant_output
        error_fixed_comp = thrust_profile - compensated_fixed
        error_adaptive_comp = thrust_profile - compensated_adaptive

        # Metrics
        rmse_no_comp = np.sqrt(np.mean(error_no_comp**2))
        rmse_fixed = np.sqrt(np.mean(error_fixed_comp**2))
        rmse_adaptive = np.sqrt(np.mean(error_adaptive_comp**2))

        print("\nRMSE Results:")
        print(f"  No compensation:    {rmse_no_comp:.4f} N")
        print(f"  Fixed gain (α={fixed_gain:.2f}):   {rmse_fixed:.4f} N ({(rmse_fixed / rmse_no_comp) * 100:.1f}%)")
        print(f"  Adaptive gain:      {rmse_adaptive:.4f} N ({(rmse_adaptive / rmse_no_comp) * 100:.1f}%)")
        print(f"  Improvement: {((rmse_fixed - rmse_adaptive) / rmse_fixed * 100):.1f}% reduction")

        # Store results
        results[model_name] = {
            "tau_values": tau_values,
            "plant_output": plant_output,
            "compensated_fixed": compensated_fixed,
            "compensated_adaptive": compensated_adaptive,
            "adaptive_gain_values": adaptive_gain_values,
            "fixed_gain": fixed_gain,
            "color": color,
            "linestyle": linestyle,
            "rmse_no_comp": rmse_no_comp,
            "rmse_fixed": rmse_fixed,
            "rmse_adaptive": rmse_adaptive,
        }

    # Subplot 1: Time constant evolution
    ax = axes[1]
    for model_name, result in results.items():
        ax.plot(
            time,
            result["tau_values"],
            color=result["color"],
            linestyle=result["linestyle"],
            linewidth=2,
            label=model_name,
        )
    ax.axhline(base_tau, color="gray", linestyle=":", alpha=0.5, label=f"Base τ = {base_tau}ms")
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Time Constant τ [ms]")
    ax.set_title("Dynamic Time Constant")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Subplot 2: Adaptive gain evolution
    ax = axes[2]
    for model_name, result in results.items():
        ax.plot(
            time,
            result["adaptive_gain_values"],
            color=result["color"],
            linestyle=result["linestyle"],
            linewidth=2,
            label=f"{model_name} (adaptive)",
        )
        # Also plot fixed gain as horizontal line
        ax.axhline(
            result["fixed_gain"],
            color=result["color"],
            linestyle=":",
            alpha=0.5,
            linewidth=1.5,
            label=f"{model_name} (fixed α={result['fixed_gain']:.2f})",
        )
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Compensation Gain α")
    ax.set_title("Compensation Gain (Adaptive vs Fixed)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Subplot 3: Plant output comparison
    ax = axes[3]
    ax.plot(time, thrust_profile, "k--", linewidth=2, alpha=0.5, label="Target (Ideal)")
    for model_name, result in results.items():
        ax.plot(
            time,
            result["plant_output"],
            color=result["color"],
            linestyle=result["linestyle"],
            linewidth=1.5,
            label=f"{model_name} (no comp)",
        )
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Thrust [N]")
    ax.set_title("Plant Output (No Compensation)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Subplot 4: Compensated output comparison
    ax = axes[4]
    ax.plot(time, thrust_profile, "k--", linewidth=2, alpha=0.5, label="Target (Ideal)")

    # Plot only one model for clarity (or plot all with transparency)
    # Let's plot all with transparency
    for model_name, result in results.items():
        # Fixed gain compensation
        ax.plot(
            time,
            result["compensated_fixed"],
            color=result["color"],
            linestyle=":",
            linewidth=1.5,
            alpha=0.6,
            label=f"{model_name} (fixed)",
        )
        # Adaptive gain compensation
        ax.plot(
            time,
            result["compensated_adaptive"],
            color=result["color"],
            linestyle=result["linestyle"],
            linewidth=2,
            label=f"{model_name} (adaptive)",
        )

    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Thrust [N]")
    ax.set_title("Compensated Output (Fixed vs Adaptive Gain)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    output_dir = Path(__file__).parent.parent.parent / "results"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "adaptive_inverse_compensation_test.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"\n{'=' * 80}")
    print(f"Plot saved to: {output_file}")
    print(f"{'=' * 80}")

    # Summary table
    print(f"\n{'=' * 80}")
    print("RMSE Comparison Summary")
    print(f"{'=' * 80}")
    print(f"{'Model':<30} {'No Comp':>12} {'Fixed Gain':>12} {'Adaptive':>12} {'Improvement':>12}")
    print("-" * 80)
    for model_name, result in results.items():
        improvement = (result["rmse_fixed"] - result["rmse_adaptive"]) / result["rmse_fixed"] * 100
        print(
            f"{model_name:<30} {result['rmse_no_comp']:>12.4f} {result['rmse_fixed']:>12.4f} "
            f"{result['rmse_adaptive']:>12.4f} {improvement:>11.1f}%"
        )
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    test_adaptive_compensation()
    print("\n✅ Test complete! Check the generated plot for visualization.")
