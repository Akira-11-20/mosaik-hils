"""
HILS scenario with Inverse Compensation.

This scenario extends the basic HILS setup by adding an inverse compensator
in the command path to pre-compensate for communication delays.
"""

from pathlib import Path
from typing import Optional

import mosaik
import mosaik.util
from config.parameters import SimulationParameters
from config.sim_config import get_simulator_config

from .base_scenario import BaseScenario


class InverseCompScenario(BaseScenario):
    """
    HILS scenario with inverse compensation in command path.

    Data flow:
        Env → Controller → [Inverse Comp] → Bridge(cmd) → Plant → Bridge(sense) → Env

    Features:
    - Command inverse compensation to pre-shape control signals
    - Full communication delays and bridges
    - Configurable compensation gain
    - Can be disabled via parameters for comparison
    """

    def __init__(self, params: SimulationParameters = None, minimal_data_mode: bool = False):
        """Initialize Inverse Compensation scenario."""
        super().__init__(params, minimal_data_mode)
        self.controller = None
        self.inverse_comp = None
        self.plant = None
        self.spacecraft = None
        self.bridge_cmd = None
        self.bridge_sense = None
        self.collector = None

    @property
    def scenario_name(self) -> str:
        return "HILS+InverseComp"

    @property
    def scenario_description(self) -> str:
        comp_status = "Enabled" if self.params.inverse_comp.enabled else "Disabled"
        return f"HILS with Command Inverse Compensation ({comp_status})"

    @property
    def results_base_dir(self) -> str:
        return "results"

    def setup_output_directory(self, suffix: str = "", parent_dir: Optional[Path] = None) -> Path:
        """Setup output directory with _inverse_comp suffix."""
        if not suffix:
            suffix = "_inverse_comp"
        return super().setup_output_directory(suffix, parent_dir=parent_dir)

    def create_world(self) -> mosaik.World:
        """Create Mosaik world with inverse compensator."""
        sim_config = get_simulator_config(include_bridge=True, include_inverse_comp=True)

        world = mosaik.World(
            sim_config,
            time_resolution=self.params.time_resolution,
            debug=False,
        )

        return world

    def setup_entities(self):
        """Create all simulation entities including inverse compensator."""
        print("\n🚀 Starting simulators...")

        controller_sim = self.world.start(
            "ControllerSim",
            step_size=self.params.control_period_steps,
        )

        # Inverse compensator (runs at same rate as controller)
        if self.params.inverse_comp.enabled:
            inverse_comp_sim = self.world.start(
                "InverseCompSim",
                step_size=self.params.control_period_steps,
            )
            print(f" ✨ Inverse Compensator enabled (gain={self.params.inverse_comp.gain})")

        plant_sim = self.world.start("PlantSim", step_size=self.params.plant_sim_period_steps)
        env_sim = self.world.start("EnvSim", step_size=self.params.env_sim_period_steps)

        bridge_cmd_sim = self.world.start("BridgeSim", step_size=1, log_dir=str(self.run_dir))
        bridge_sense_sim = self.world.start("BridgeSim", step_size=1, log_dir=str(self.run_dir))

        # Create entities
        self.controller = controller_sim.PIDController(
            kp=self.params.control.kp,
            ki=self.params.control.ki,
            kd=self.params.control.kd,
            target_position=self.params.control.target_position,
            max_thrust=self.params.control.max_thrust,
            thrust_duration=self.params.control.control_period,
            integral_limit=self.params.control.integral_limit,
        )

        if self.params.inverse_comp.enabled:
            self.inverse_comp = inverse_comp_sim.InverseCompensator(
                comp_id="cmd_compensator",
                gain=self.params.inverse_comp.gain,
                comp_type="command",
            )

        self.plant = plant_sim.ThrustStand(
            stand_id="stand_01",
            time_constant=self.params.plant.time_constant,
            time_constant_std=self.params.plant.time_constant_std,
            time_constant_noise=self.params.plant.time_constant_noise,
            enable_lag=self.params.plant.enable_lag,
        )

        self.spacecraft = env_sim.Spacecraft1DOF(
            mass=self.params.spacecraft.mass,
            initial_position=self.params.spacecraft.initial_position,
            initial_velocity=self.params.spacecraft.initial_velocity,
            gravity=self.params.spacecraft.gravity,
        )

        # Command path bridge with time_shifted compensation
        # Since InverseComp->Bridge uses time_shifted=True, it adds 1 step delay
        # We compensate for this in the Bridge to achieve the desired total delay
        self.bridge_cmd = bridge_cmd_sim.CommBridge(
            bridge_type="cmd",
            base_delay=self.params.communication.cmd_delay,
            jitter_std=self.params.communication.cmd_jitter,
            packet_loss_rate=self.params.communication.cmd_loss_rate,
            preserve_order=True,
            compensate_time_shifted=True,
            time_shifted_delay_ms=self.params.time_resolution * 1000,  # 1 step in ms
        )

        self.bridge_sense = bridge_sense_sim.CommBridge(
            bridge_type="sense",
            base_delay=self.params.communication.sense_delay,
            jitter_std=self.params.communication.sense_jitter,
            packet_loss_rate=self.params.communication.sense_loss_rate,
            preserve_order=True,
        )

    def connect_entities(self):
        """Connect entities with inverse compensator in command path."""
        if self.params.inverse_comp.enabled:
            # With inverse compensation
            print("   Controller → Inverse Compensator: same-step connection")
            self.world.connect(
                self.controller,
                self.inverse_comp,
                ("command", "input"),
            )

            print("   ⏱️  Inverse Compensator → Bridge(cmd): time-shifted connection")
            self.world.connect(
                self.inverse_comp,
                self.bridge_cmd,
                ("compensated_output", "input"),
                time_shifted=True,
                initial_data={
                    "compensated_output": {
                        "thrust": 0.0,
                        "duration": self.params.control.control_period,
                    }
                },
            )
        else:
            # Without inverse compensation (fallback to standard HILS)
            print("   ⏱️  Controller → Bridge(cmd): time-shifted connection (no compensation)")
            self.world.connect(
                self.controller,
                self.bridge_cmd,
                ("command", "input"),
                time_shifted=True,
                initial_data={
                    "command": {
                        "thrust": 0.0,
                        "duration": self.params.control.control_period,
                    }
                },
            )

        # Rest of the connections are same as HILS
        self.world.connect(
            self.bridge_cmd,
            self.plant,
            ("delayed_output", "command"),
        )

        # Plant → Bridge(sense) - measurement path
        # Uses actual_thrust (with first-order lag) instead of measured_thrust (ideal)
        self.world.connect(
            self.plant,
            self.bridge_sense,
            ("actual_thrust", "input"),
        )

        self.world.connect(
            self.bridge_sense,
            self.spacecraft,
            ("delayed_output", "force"),
        )

        print("   📡 Env → Controller: same-step connection (state feedback)")
        self.world.connect(
            self.spacecraft,
            self.controller,
            "position",
            "velocity",
        )

        print("\n✅ Data flow configured:")
        print("   Env → Controller (same step)")
        if self.params.inverse_comp.enabled:
            print("   Controller → [Inverse Comp] → Bridge(cmd) → Plant (time-shifted)")
        else:
            print("   Controller → Bridge(cmd) → Plant (time-shifted)")
        print("   Plant → Bridge(sense) → Env")

    def setup_data_collection(self):
        """Setup data collection including inverse compensator data."""
        data_collector_sim = self.world.start("DataCollector", step_size=1)
        self.collector = data_collector_sim.Collector(
            output_dir=str(self.run_dir),
            minimal_mode=self.minimal_data_mode
        )

        if self.minimal_data_mode:
            # Minimal mode: only collect position and velocity from spacecraft
            mosaik.util.connect_many_to_one(
                self.world,
                [self.spacecraft],
                self.collector,
                "position",
                "velocity",
            )
            print("   ⚡ Minimal mode: Data collection (position, velocity only)")
            return

        # Collect data from all entities
        mosaik.util.connect_many_to_one(
            self.world,
            [self.controller],
            self.collector,
            "command",
            "error",
        )

        # Collect inverse compensator data if enabled
        if self.params.inverse_comp.enabled:
            mosaik.util.connect_many_to_one(
                self.world,
                [self.inverse_comp],
                self.collector,
                # "compensated_output",
                # "stats",
                # # Debug attributes
                # "raw_input",
                # "prev_input",
                # "delta",
                # "compensation_amount",
                "input_thrust",
                "output_thrust",
            )

        # mosaik.util.connect_many_to_one(
        #     self.world,
        #     [self.bridge_cmd],
        #     self.collector,
        #     "stats",
        #     "packet_receive_time",
        #     "packet_send_time",
        #     "packet_actual_delay",
        #     # Debug attributes
        #     "buffer_size",
        #     # "buffer_content",  # Commented out to reduce log size
        #     "oldest_packet_time",
        #     "newest_packet_time",
        # )

        mosaik.util.connect_many_to_one(
            self.world,
            [self.plant],
            self.collector,
            "measured_thrust",
            "actual_thrust",
            # "status",
            "time_constant",  # Record actual sampled time constant
        )

        # mosaik.util.connect_many_to_one(
        #     self.world,
        #     [self.bridge_sense],
        #     self.collector,
        #     "stats",
        #     "packet_receive_time",
        #     "packet_send_time",
        #     "packet_actual_delay",
        #     # Debug attributes
        #     "buffer_size",
        #     # "buffer_content",  # Commented out to reduce log size
        #     "oldest_packet_time",
        #     "newest_packet_time",
        # )

        mosaik.util.connect_many_to_one(
            self.world,
            [self.spacecraft],
            self.collector,
            "position",
            "velocity",
            "acceleration",
            "force",
        )

        print("   All data → DataCollector → HDF5")
