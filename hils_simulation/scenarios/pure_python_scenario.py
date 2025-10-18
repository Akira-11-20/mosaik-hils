"""
Pure Python scenario without Mosaik framework.

This scenario implements a simple numerical simulation without any
framework overhead, serving as a baseline for performance comparison.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List

import h5py
import numpy as np

from config.parameters import SimulationParameters
from .base_scenario import BaseScenario


class Spacecraft1DOF:
    """1-DOF spacecraft dynamics simulator."""

    def __init__(self, mass: float, initial_position: float, initial_velocity: float, gravity: float):
        self.mass = mass
        self.position = initial_position
        self.velocity = initial_velocity
        self.gravity = gravity
        self.force = 0.0
        self.acceleration = 0.0

    def step(self, dt: float, force: float):
        """
        Integrate equations of motion using Explicit Euler method.

        Args:
            dt: Time step [s]
            force: Applied thrust [N]
        """
        self.force = force

        # F = ma → a = F/m - g
        self.acceleration = (self.force / self.mass) - self.gravity

        # Explicit Euler integration (same as Mosaik version)
        self.position += self.velocity * dt
        self.velocity += self.acceleration * dt


class PDController:
    """PD controller for position tracking."""

    def __init__(self, kp: float, kd: float, target_position: float, max_thrust: float):
        self.kp = kp
        self.kd = kd
        self.target_position = target_position
        self.max_thrust = max_thrust
        self.error = 0.0

    def compute_control(self, position: float, velocity: float) -> float:
        """
        Compute control input.

        Args:
            position: Current position [m]
            velocity: Current velocity [m/s]

        Returns:
            Thrust command [N]
        """
        self.error = self.target_position - position
        thrust = self.kp * self.error - self.kd * velocity
        return thrust  # No saturation (allows negative thrust)


class ThrustStand:
    """Ideal thrust measurement sensor."""

    def __init__(self):
        self.measured_thrust = 0.0

    def measure(self, thrust: float) -> float:
        """Measure thrust (ideal sensor, no noise)."""
        self.measured_thrust = thrust
        return self.measured_thrust


class PurePythonScenario(BaseScenario):
    """
    Pure Python simulation scenario without Mosaik framework.

    Features:
    - No framework overhead
    - Direct numerical integration
    - Ideal baseline for performance comparison
    - No communication delays
    """

    def __init__(self, params: SimulationParameters = None):
        """Initialize Pure Python scenario."""
        # Note: PurePythonScenario doesn't use Mosaik, so world is always None
        super().__init__(params)
        self.spacecraft = None
        self.controller = None
        self.plant = None
        self.data: Dict[str, List] = {}

    @property
    def scenario_name(self) -> str:
        return "PurePython"

    @property
    def scenario_description(self) -> str:
        return "Pure Python Simulation (No Mosaik Framework)"

    @property
    def results_base_dir(self) -> str:
        return "results_pure"

    def create_world(self):
        """Pure Python doesn't use Mosaik world."""
        return None

    def setup_entities(self):
        """Create pure Python simulation components."""
        self.spacecraft = Spacecraft1DOF(
            mass=self.params.spacecraft.mass,
            initial_position=self.params.spacecraft.initial_position,
            initial_velocity=self.params.spacecraft.initial_velocity,
            gravity=self.params.spacecraft.gravity,
        )

        self.controller = PDController(
            kp=self.params.control.kp,
            kd=self.params.control.kd,
            target_position=self.params.control.target_position,
            max_thrust=self.params.control.max_thrust,
        )

        self.plant = ThrustStand()

        print(f"   Spacecraft: mass={self.params.spacecraft.mass}kg")
        print(f"   Controller: Kp={self.params.control.kp}, Kd={self.params.control.kd}")

    def connect_entities(self):
        """Pure Python doesn't need explicit connections."""
        print("   Direct function calls (no Mosaik connections)")

    def setup_data_collection(self):
        """Initialize data collection arrays."""
        self.data = {
            "time_s": [],
            "time_ms": [],
            "position_Spacecraft": [],
            "velocity_Spacecraft": [],
            "acceleration_Spacecraft": [],
            "force_Spacecraft": [],
            "command_Controller_thrust": [],
            "error_Controller": [],
            "measured_thrust_Plant": [],
        }

    def save_data_to_hdf5(self):
        """Save simulation data to HDF5 file."""
        h5_path = self.run_dir / "hils_data.h5"

        with h5py.File(h5_path, "w") as f:
            data_group = f.create_group("data")

            for key, values in self.data.items():
                data_group.create_dataset(key, data=np.array(values), compression="gzip")

        print(f"📁 HDF5 data saved: {h5_path}")
        return h5_path

    def run_simulation(self):
        """Execute pure Python simulation loop."""
        control_period_steps = int(
            self.params.control.control_period / 1000.0 / self.params.time_resolution
        )
        thrust = 0.0
        log_interval_steps = int(1.0 / self.params.time_resolution)  # Log every 1 second

        for step in range(self.params.simulation_steps):
            time_s = step * self.params.time_resolution
            time_ms = time_s * 1000

            # Compute control at specified period
            if step % control_period_steps == 0:
                thrust = self.controller.compute_control(
                    self.spacecraft.position, self.spacecraft.velocity
                )
                measured_thrust = self.plant.measure(thrust)

                # Periodic logging
                if step % log_interval_steps == 0:
                    print(
                        f"[t={time_ms:.0f}ms] pos={self.spacecraft.position:.3f}m, "
                        f"vel={self.spacecraft.velocity:.3f}m/s, error={self.controller.error:.3f}m, "
                        f"thrust={thrust:.3f}N"
                    )

            # Record data (before physics update, like Mosaik)
            self.data["time_s"].append(time_s)
            self.data["time_ms"].append(time_ms)
            self.data["position_Spacecraft"].append(self.spacecraft.position)
            self.data["velocity_Spacecraft"].append(self.spacecraft.velocity)
            self.data["acceleration_Spacecraft"].append(self.spacecraft.acceleration)
            self.data["force_Spacecraft"].append(self.spacecraft.force)
            self.data["command_Controller_thrust"].append(thrust)
            self.data["error_Controller"].append(self.controller.error)
            self.data["measured_thrust_Plant"].append(self.plant.measured_thrust)

            # Update spacecraft dynamics
            self.spacecraft.step(self.params.time_resolution, thrust)

    def generate_graphs(self):
        """Pure Python scenario doesn't generate Mosaik graphs."""
        print(f"\n📊 Skipping graph generation (Pure Python scenario)")

    def run(self):
        """Execute complete Pure Python simulation."""
        # Print header
        self.print_header()

        # Setup output directory
        self.setup_output_directory()
        print(f"📁 Log directory: {self.run_dir}")

        # Save configuration
        self.save_configuration()

        # Print simulation info
        self.print_simulation_info()

        # Create components
        print(f"\n📦 Creating components...")
        self.setup_entities()

        # Connect (dummy for Pure Python)
        print(f"\n🔗 Connecting data flows...")
        self.connect_entities()

        # Setup data collection
        print(f"\n📊 Setting up data collection...")
        self.setup_data_collection()

        # Run simulation
        print(f"\n▶️  Running simulation until {self.params.simulation_time}s ({self.params.simulation_steps} steps)...")
        print("=" * 70)

        self.run_simulation()

        print("=" * 70)
        print("✅ Simulation completed successfully!")

        # Save data
        print("\n💾 Saving data...")
        self.save_data_to_hdf5()

        # Print final state
        print(f"\n📊 Final state:")
        print(f"   Position: {self.spacecraft.position:.3f} m (target: {self.params.control.target_position} m)")
        print(f"   Velocity: {self.spacecraft.velocity:.3f} m/s")
        print(f"   Error: {self.controller.error:.3f} m")

        # Skip graph generation
        self.generate_graphs()

        # Print footer
        print("\n" + "=" * 70)
        print(f"{self.scenario_name} Simulation Finished")
        print("=" * 70)
