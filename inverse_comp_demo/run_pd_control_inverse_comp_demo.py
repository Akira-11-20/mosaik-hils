"""PD Control with Inverse Compensation Demo

This demo uses the same PD control structure as the HILS simulation
to demonstrate the effect of command inverse compensation on control performance.

System:
- Plant: 1-DOF spacecraft (F = ma with gravity)
- Controller: PD controller tracking target position
- Communication: Command delay on controller output
- Compensation: Inverse compensation to counteract delay

制御系は HILS シミュレーションと同じPD制御を使用
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class PDControlConfig:
    """PD control with inverse compensation configuration"""

    # Time parameters
    Ts: float = 0.01  # Sampling period [s] (10ms like HILS)
    Tend: float = 5.0  # Simulation end time [s]

    # Communication delay (command path: Controller → Actuator)
    tau_cmd: float = 0.15  # Command delay [s] (150ms like HILS)

    # Plant dynamics (1-DOF spacecraft)
    mass: float = 1.0  # Spacecraft mass [kg]
    gravity: float = 9.81  # Gravity acceleration [m/s²]

    # PD Controller parameters (same as HILS)
    kp: float = 15.0  # Proportional gain
    kd: float = 5.0  # Derivative gain
    target_position: float = 5.0  # Target position [m]
    max_thrust: float = 100.0  # Maximum thrust [N]

    # Initial conditions
    x0: float = 0.0  # Initial position [m]
    v0: float = 10.0  # Initial velocity [m/s]

    @property
    def sample_count(self) -> int:
        """Total number of samples"""
        return int(self.Tend / self.Ts)

    @property
    def delay_samples(self) -> int:
        """Command delay in samples"""
        return int(round(self.tau_cmd / self.Ts))

    def inverse_gain_for_test(self, gain_value: float) -> float:
        """Return specified inverse gain for testing"""
        return gain_value


def pd_controller(position: float, velocity: float, cfg: PDControlConfig) -> float:
    """PD controller (same as HILS)

    Args:
        position: Current position [m]
        velocity: Current velocity [m/s]
        cfg: Configuration

    Returns:
        thrust: Commanded thrust [N]
    """
    error = cfg.target_position - position
    thrust = cfg.kp * error - cfg.kd * velocity

    # Clamp to max thrust
    thrust = np.clip(thrust, -cfg.max_thrust, cfg.max_thrust)

    return thrust


def apply_command_delay(cmd: np.ndarray, delay_samples: int) -> np.ndarray:
    """Apply delay to command signal using FIFO buffer

    Args:
        cmd: Original command signal
        delay_samples: Delay in samples

    Returns:
        cmd_delayed: Delayed command signal
    """
    if delay_samples < 0:
        raise ValueError("delay_samples must be non-negative")

    buffer = deque([0.0] * delay_samples, maxlen=delay_samples or 1)
    cmd_delayed = np.zeros_like(cmd)

    for k, value in enumerate(cmd):
        buffer.append(value)
        cmd_delayed[k] = buffer[0]

    return cmd_delayed


def apply_command_inverse_compensation(u_ref: np.ndarray, gain: float) -> np.ndarray:
    """Apply command inverse compensation

    Formula: u_comp[k] = a * u_ref[k] - (a - 1) * u_ref[k-1]

    Args:
        u_ref: Reference command from controller
        gain: Inverse compensation gain

    Returns:
        u_comp: Pre-compensated command
    """
    if len(u_ref) == 0:
        return np.array([])

    u_comp = np.zeros_like(u_ref)
    prev = u_ref[0]
    u_comp[0] = prev

    for k in range(1, len(u_ref)):
        curr = u_ref[k]
        u_comp[k] = gain * curr - (gain - 1.0) * prev
        prev = curr

    return u_comp


def simulate_spacecraft_pd_control(cfg: PDControlConfig, thrust_cmd: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate 1-DOF spacecraft with PD control

    Dynamics: F = m*a
              a = (thrust - m*g) / m
              v[k+1] = v[k] + a * Ts
              x[k+1] = x[k] + v[k] * Ts

    Args:
        cfg: Configuration
        thrust_cmd: Thrust command time series [N]

    Returns:
        position: Position time series [m]
        velocity: Velocity time series [m/s]
    """
    N = len(thrust_cmd)
    position = np.zeros(N)
    velocity = np.zeros(N)

    position[0] = cfg.x0
    velocity[0] = cfg.v0

    for k in range(N - 1):
        # Apply thrust (actuator)
        force = thrust_cmd[k]

        # Dynamics: a = F/m - g
        accel = force / cfg.mass - cfg.gravity

        # Integrate
        velocity[k + 1] = velocity[k] + accel * cfg.Ts
        position[k + 1] = position[k] + velocity[k] * cfg.Ts

    return position, velocity


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate Root Mean Square Error"""
    return float(np.sqrt(np.mean((a - b) ** 2)))


def settling_time(t: np.ndarray, pos: np.ndarray, target: float, tolerance: float = 0.02) -> float:
    """Calculate settling time (time to reach within ±2% of target)

    Args:
        t: Time array
        pos: Position array
        target: Target position
        tolerance: Settling tolerance (default 2%)

    Returns:
        Settling time [s], or nan if never settles
    """
    threshold = target * tolerance

    # Find first time when error becomes small
    error = np.abs(pos - target)

    for i in range(len(error) - 100):  # Check last 100 samples stay within bound
        if np.all(error[i:] < threshold):
            return t[i]

    return float("nan")


def plot_results(
    t: np.ndarray,
    pos_no_delay: np.ndarray,
    pos_delayed: np.ndarray,
    pos_compensated_dict: dict,
    cfg: PDControlConfig,
) -> None:
    """Plot comparison results

    Args:
        t: Time array
        pos_no_delay: Position with no delay
        pos_delayed: Position with delay (no compensation)
        pos_compensated_dict: Dict of {gain: position} for compensated cases
        cfg: Configuration
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Top plot: Position tracking
    ax1.axhline(cfg.target_position, color="black", linestyle="--", linewidth=1, alpha=0.5, label="Target")
    ax1.plot(t, pos_no_delay, label="No delay (ideal)", linewidth=2, color="green")
    ax1.plot(
        t,
        pos_delayed,
        label=f"With delay ({cfg.tau_cmd * 1000:.0f}ms, no comp)",
        linewidth=2,
        color="red",
        alpha=0.7,
    )

    colors = ["blue", "purple", "orange", "cyan"]
    for idx, (gain, pos) in enumerate(pos_compensated_dict.items()):
        ax1.plot(
            t,
            pos,
            label=f"Inverse comp (gain={gain:.0f})",
            linewidth=2,
            linestyle="--",
            color=colors[idx % len(colors)],
        )

    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Position [m]")
    ax1.set_title("PD Control with Command Inverse Compensation")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_xlim([0, cfg.Tend])

    # Bottom plot: Tracking error
    ax2.plot(
        t,
        np.abs(cfg.target_position - pos_no_delay),
        label="No delay (ideal)",
        linewidth=2,
        color="green",
    )
    ax2.plot(
        t,
        np.abs(cfg.target_position - pos_delayed),
        label=f"With delay (no comp)",
        linewidth=2,
        color="red",
        alpha=0.7,
    )

    for idx, (gain, pos) in enumerate(pos_compensated_dict.items()):
        ax2.plot(
            t,
            np.abs(cfg.target_position - pos),
            label=f"Inverse comp (gain={gain:.0f})",
            linewidth=2,
            linestyle="--",
            color=colors[idx % len(colors)],
        )

    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Tracking Error [m]")
    ax2.set_title("Absolute Tracking Error")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_xlim([0, cfg.Tend])
    ax2.set_yscale("log")

    plt.tight_layout()

    # Save
    output_path = Path(__file__).resolve().parent / "pd_control_inverse_comp_demo.png"
    plt.savefig(output_path, dpi=200)
    print(f"\nPlot saved to: {output_path}")
    plt.show()


def main() -> None:
    """Main execution for PD control inverse compensation demo"""
    np.random.seed(42)
    cfg = PDControlConfig()

    print("=== PD Control with Command Inverse Compensation Demo ===")
    print(f"Sampling period: {cfg.Ts * 1000:.1f} ms")
    print(f"Command delay: {cfg.tau_cmd * 1000:.1f} ms ({cfg.delay_samples} samples)")
    print(f"PD gains: Kp={cfg.kp}, Kd={cfg.kd}")
    print(f"Target position: {cfg.target_position} m")
    print()

    # Time array
    t = np.arange(cfg.sample_count) * cfg.Ts

    # ===== Case A: Ideal (no delay) =====
    print("Running Case A: No delay (ideal)...")
    pos_ideal = np.zeros(cfg.sample_count)
    vel_ideal = np.zeros(cfg.sample_count)
    cmd_ideal = np.zeros(cfg.sample_count)

    pos_ideal[0] = cfg.x0
    vel_ideal[0] = cfg.v0

    for k in range(cfg.sample_count - 1):
        # Controller
        cmd_ideal[k] = pd_controller(pos_ideal[k], vel_ideal[k], cfg)

        # Plant dynamics (no delay)
        force = cmd_ideal[k]
        accel = force / cfg.mass - cfg.gravity
        vel_ideal[k + 1] = vel_ideal[k] + accel * cfg.Ts
        pos_ideal[k + 1] = pos_ideal[k] + vel_ideal[k] * cfg.Ts

    # ===== Case B: With delay, no compensation =====
    print("Running Case B: With delay, no compensation...")
    pos_delayed = np.zeros(cfg.sample_count)
    vel_delayed = np.zeros(cfg.sample_count)
    cmd_delayed_input = np.zeros(cfg.sample_count)

    pos_delayed[0] = cfg.x0
    vel_delayed[0] = cfg.v0

    # Generate controller commands
    for k in range(cfg.sample_count):
        cmd_delayed_input[k] = pd_controller(pos_delayed[k], vel_delayed[k], cfg)

    # Apply delay
    cmd_delayed_output = apply_command_delay(cmd_delayed_input, cfg.delay_samples)

    # Simulate with delayed commands
    for k in range(cfg.sample_count - 1):
        force = cmd_delayed_output[k]
        accel = force / cfg.mass - cfg.gravity
        vel_delayed[k + 1] = vel_delayed[k] + accel * cfg.Ts
        pos_delayed[k + 1] = pos_delayed[k] + vel_delayed[k] * cfg.Ts

    # ===== Case C: With delay AND inverse compensation =====
    # Test multiple gains
    test_gains = [5.0, 15.0, 30.0, 50.0]
    pos_compensated_dict = {}

    for gain in test_gains:
        print(f"Running Case C: With delay and inverse comp (gain={gain:.0f})...")

        pos_comp = np.zeros(cfg.sample_count)
        vel_comp = np.zeros(cfg.sample_count)
        cmd_raw = np.zeros(cfg.sample_count)
        cmd_compensated = np.zeros(cfg.sample_count)

        pos_comp[0] = cfg.x0
        vel_comp[0] = cfg.v0

        # Delay buffer for compensated commands
        cmd_buffer = deque([0.0] * cfg.delay_samples, maxlen=cfg.delay_samples or 1)

        # Previous command for inverse compensation
        prev_cmd = 0.0

        # Real-time simulation with feedback
        for k in range(cfg.sample_count - 1):
            # 1. Controller computes command based on current state
            cmd_raw[k] = pd_controller(pos_comp[k], vel_comp[k], cfg)

            # 2. Apply inverse compensation
            if k == 0:
                cmd_compensated[k] = cmd_raw[k]
            else:
                cmd_compensated[k] = gain * cmd_raw[k] - (gain - 1.0) * prev_cmd
            prev_cmd = cmd_raw[k]

            # 3. Pass through delay buffer
            cmd_buffer.append(cmd_compensated[k])
            delayed_cmd = cmd_buffer[0]

            # 4. Apply to plant
            force = delayed_cmd
            accel = force / cfg.mass - cfg.gravity

            # 5. Update state
            vel_comp[k + 1] = vel_comp[k] + accel * cfg.Ts
            pos_comp[k + 1] = pos_comp[k] + vel_comp[k] * cfg.Ts

        pos_compensated_dict[gain] = pos_comp

    # ===== Metrics =====
    print("\n=== Performance Metrics ===")

    # RMSE
    rmse_ideal = rmse(pos_ideal, np.full_like(pos_ideal, cfg.target_position))
    rmse_delayed = rmse(pos_delayed, np.full_like(pos_delayed, cfg.target_position))

    print(f"RMSE from target:")
    print(f"  No delay:        {rmse_ideal:.4f} m")
    print(f"  With delay (no comp): {rmse_delayed:.4f} m")

    for gain, pos_comp in pos_compensated_dict.items():
        rmse_comp = rmse(pos_comp, np.full_like(pos_comp, cfg.target_position))
        improvement = (1 - rmse_comp / rmse_delayed) * 100
        print(f"  Inverse comp (gain={gain:.0f}): {rmse_comp:.4f} m (improvement: {improvement:.1f}%)")

    # Settling time
    print(f"\nSettling time (±2% of target):")
    settle_ideal = settling_time(t, pos_ideal, cfg.target_position)
    settle_delayed = settling_time(t, pos_delayed, cfg.target_position)

    print(f"  No delay:        {settle_ideal:.3f} s")
    print(f"  With delay (no comp): {settle_delayed:.3f} s")

    for gain, pos_comp in pos_compensated_dict.items():
        settle_comp = settling_time(t, pos_comp, cfg.target_position)
        print(f"  Inverse comp (gain={gain:.0f}): {settle_comp:.3f} s")

    # ===== Plot =====
    plot_results(t, pos_ideal, pos_delayed, pos_compensated_dict, cfg)


if __name__ == "__main__":
    main()
