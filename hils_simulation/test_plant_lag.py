"""
Test script to verify Plant time lag implementation

Tests:
1. Plant 1st-order lag dynamics
2. Comparison of measured_thrust (ideal) vs actual_thrust (with lag)
3. Timing analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def test_first_order_lag():
    """Test discrete first-order lag implementation"""

    print("="*70)
    print("Test 1: First-Order Lag Dynamics (Unit Step Response)")
    print("="*70)

    # Parameters
    dt = 0.1  # [ms] - time step
    tau = 50.0  # [ms] - time constant
    t_sim = 500.0  # [ms] - simulation time

    # Time array
    n_steps = int(t_sim / dt)
    time = np.arange(0, n_steps) * dt

    # Input: unit step at t=0
    u = np.ones(n_steps)

    # Output: first-order lag response
    y = np.zeros(n_steps)

    for k in range(n_steps - 1):
        # Discrete first-order lag: y[k+1] = y[k] + (dt/tau) * (u[k] - y[k])
        y[k+1] = y[k] + (dt / tau) * (u[k] - y[k])

    # Analytical solution: y(t) = 1 - exp(-t/tau)
    y_analytical = 1 - np.exp(-time / tau)

    # Time constants
    t_63 = tau  # 63.2% of final value
    t_95 = 3 * tau  # 95% of final value

    print(f"\nParameters:")
    print(f"  dt = {dt} ms")
    print(f"  tau = {tau} ms")
    print(f"  t_sim = {t_sim} ms")

    print(f"\nExpected behavior:")
    print(f"  At t = tau ({tau} ms): y = {1 - np.exp(-1):.4f} (63.2%)")
    print(f"  At t = 3*tau ({3*tau} ms): y = {1 - np.exp(-3):.4f} (95.0%)")

    # Find actual values at these times
    idx_63 = int(t_63 / dt)
    idx_95 = int(t_95 / dt)

    print(f"\nSimulated values:")
    print(f"  At t = {time[idx_63]:.1f} ms: y = {y[idx_63]:.4f}")
    print(f"  At t = {time[idx_95]:.1f} ms: y = {y[idx_95]:.4f}")

    # Error analysis
    error = np.abs(y - y_analytical)
    max_error = np.max(error)
    mean_error = np.mean(error)

    print(f"\nError analysis (discrete vs analytical):")
    print(f"  Max error: {max_error:.6f}")
    print(f"  Mean error: {mean_error:.6f}")

    if max_error < 0.01:
        print(f"  ✅ PASS: Discrete implementation matches analytical solution")
    else:
        print(f"  ❌ FAIL: Error too large")

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.plot(time, u, 'k--', label='Input (step)', lw=2)
    ax1.plot(time, y, 'b-', label='Output (discrete)', lw=2)
    ax1.plot(time, y_analytical, 'r:', label='Output (analytical)', lw=2)
    ax1.axvline(tau, color='gray', linestyle=':', alpha=0.5, label=f'τ = {tau} ms')
    ax1.axhline(1 - np.exp(-1), color='gray', linestyle=':', alpha=0.5)
    ax1.set_xlabel('Time [ms]')
    ax1.set_ylabel('Amplitude')
    ax1.set_title('First-Order Lag: Step Response')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(time, error, 'r-', lw=1.5)
    ax2.set_xlabel('Time [ms]')
    ax2.set_ylabel('Error')
    ax2.set_title('Discrete vs Analytical Error')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    return fig, (y, y_analytical, error)


def test_plant_with_varying_input():
    """Test plant response with varying thrust command"""

    print("\n" + "="*70)
    print("Test 2: Plant Response with Varying Thrust Command")
    print("="*70)

    # Parameters matching HILS simulation
    dt = 0.1  # [ms] - time step (0.0001s * 1000)
    tau = 50.0  # [ms] - time constant
    t_sim = 2000.0  # [ms] - 2 seconds

    n_steps = int(t_sim / dt)
    time = np.arange(0, n_steps) * dt

    # Create varying thrust command (simulating PID controller output)
    u = np.zeros(n_steps)

    # Step 1: Ramp up (0-500ms)
    u[0:5000] = np.linspace(0, 100, 5000)

    # Step 2: Hold (500-1000ms)
    u[5000:10000] = 100

    # Step 3: Step down (1000-1500ms)
    u[10000:15000] = 50

    # Step 4: Ramp to zero (1500-2000ms)
    u[15000:20000] = np.linspace(50, 0, 5000)

    # Simulate measured_thrust (ideal, no lag)
    measured_thrust = u.copy()

    # Simulate actual_thrust (with 1st-order lag)
    actual_thrust = np.zeros(n_steps)

    for k in range(n_steps - 1):
        actual_thrust[k+1] = actual_thrust[k] + (dt / tau) * (measured_thrust[k] - actual_thrust[k])

    # Analysis
    print(f"\nParameters:")
    print(f"  dt = {dt} ms")
    print(f"  tau = {tau} ms")
    print(f"  t_sim = {t_sim} ms")

    # Calculate lag at different points
    print(f"\nDelay analysis:")

    # At step change (t=1000ms)
    idx_step = 10000
    print(f"  At step down (t={time[idx_step]:.1f}ms):")
    print(f"    Command: {measured_thrust[idx_step]:.2f} N")
    print(f"    Actual: {actual_thrust[idx_step]:.2f} N")
    print(f"    Lag: {measured_thrust[idx_step] - actual_thrust[idx_step]:.2f} N")

    # Peak lag
    lag = measured_thrust - actual_thrust
    max_lag_idx = np.argmax(np.abs(lag))
    print(f"\n  Maximum lag:")
    print(f"    Time: {time[max_lag_idx]:.1f} ms")
    print(f"    Command: {measured_thrust[max_lag_idx]:.2f} N")
    print(f"    Actual: {actual_thrust[max_lag_idx]:.2f} N")
    print(f"    Lag: {lag[max_lag_idx]:.2f} N")

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    ax1.plot(time, measured_thrust, 'b-', label='measured_thrust (ideal)', lw=2, alpha=0.7)
    ax1.plot(time, actual_thrust, 'r-', label='actual_thrust (with lag)', lw=2)
    ax1.set_xlabel('Time [ms]')
    ax1.set_ylabel('Thrust [N]')
    ax1.set_title('Plant Response: Measured vs Actual Thrust')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(time, lag, 'g-', lw=2)
    ax2.axhline(0, color='k', linestyle=':', lw=1)
    ax2.set_xlabel('Time [ms]')
    ax2.set_ylabel('Lag [N]')
    ax2.set_title('Thrust Lag (measured - actual)')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    return fig, (measured_thrust, actual_thrust, lag)


if __name__ == "__main__":
    print("\n" + "="*70)
    print("PLANT TIME LAG VERIFICATION")
    print("="*70 + "\n")

    # Test 1: Basic first-order lag
    fig1, _ = test_first_order_lag()

    # Test 2: Varying input
    fig2, _ = test_plant_with_varying_input()

    # Save figures
    output_dir = Path("test_results")
    output_dir.mkdir(exist_ok=True)

    fig1.savefig(output_dir / "plant_lag_test1_step_response.png", dpi=300)
    print(f"\n📊 Saved: {output_dir / 'plant_lag_test1_step_response.png'}")

    fig2.savefig(output_dir / "plant_lag_test2_varying_input.png", dpi=300)
    print(f"📊 Saved: {output_dir / 'plant_lag_test2_varying_input.png'}")

    print("\n" + "="*70)
    print("✅ All tests completed")
    print("="*70 + "\n")

    plt.show()
