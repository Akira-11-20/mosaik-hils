"""
HILS (Hardware-in-the-Loop Simulation) scenario.

This scenario implements the full HILS setup with communication delays
and bridges in both command and sensing paths.
"""

import mosaik
import mosaik.util

from config.parameters import SimulationParameters
from config.sim_config import get_simulator_config
from .base_scenario import BaseScenario


class HILSScenario(BaseScenario):
    """
    HILS scenario with communication delays.

    Data flow:
        Env → Controller (same step) → Bridge(cmd) → Plant (time-shifted) → Bridge(sense) → Env

    Features:
    - Command path delay and jitter
    - Sensing path delay and jitter
    - Time-shifted connections to break circular dependencies
    - Full packet loss simulation
    """

    def __init__(self, params: SimulationParameters = None):
        """Initialize HILS scenario."""
        super().__init__(params)
        self.controller = None
        self.plant = None
        self.spacecraft = None
        self.bridge_cmd = None
        self.bridge_sense = None
        self.collector = None

    @property
    def scenario_name(self) -> str:
        return "HILS"

    @property
    def scenario_description(self) -> str:
        return "Hardware-in-the-Loop Simulation with Communication Delays"

    @property
    def results_base_dir(self) -> str:
        return "results"

    def create_world(self) -> mosaik.World:
        """Create Mosaik world with HILS configuration."""
        sim_config = get_simulator_config(include_bridge=True, include_inverse_comp=False)

        world = mosaik.World(
            sim_config,
            time_resolution=self.params.time_resolution,
            debug=False,  # Disable debug mode for performance
        )

        return world

    def setup_entities(self):
        """Create all simulation entities for HILS scenario."""
        # Start simulators
        print("\n🚀 Starting simulators...")

        controller_sim = self.world.start(
            "ControllerSim",
            step_size=self.params.control_period_steps,
        )

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

        self.plant = plant_sim.ThrustStand(
            stand_id="stand_01",
            time_constant=self.params.plant.time_constant,
            enable_lag=self.params.plant.enable_lag,
        )

        self.spacecraft = env_sim.Spacecraft1DOF(
            mass=self.params.spacecraft.mass,
            initial_position=self.params.spacecraft.initial_position,
            initial_velocity=self.params.spacecraft.initial_velocity,
            gravity=self.params.spacecraft.gravity,
        )

        # Command path bridge with time_shifted compensation
        # Since Controller->Bridge uses time_shifted=True, it adds 1 step delay
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
        """Connect entities to form HILS dataflow."""
        # 1. Controller → Bridge(cmd) - time-shifted to execute on next step
        print("   ⏱️  Controller → Bridge(cmd): time-shifted connection")
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

        # 2. Bridge(cmd) → Plant - delayed command
        self.world.connect(
            self.bridge_cmd,
            self.plant,
            ("delayed_output", "command"),
        )

        # 3. Plant → Bridge(sense) - measurement path
        # Uses actual_thrust (with first-order lag) instead of measured_thrust (ideal)
        self.world.connect(
            self.plant,
            self.bridge_sense,
            ("actual_thrust", "input"),
        )

        # 4. Bridge(sense) → Env - delayed measurement
        self.world.connect(
            self.bridge_sense,
            self.spacecraft,
            ("delayed_output", "force"),
        )

        # 5. Env → Controller - state feedback (same step)
        print("   📡 Env → Controller: same-step connection (state feedback)")
        self.world.connect(
            self.spacecraft,
            self.controller,
            "position",
            "velocity",
        )

        print("\n✅ Data flow configured:")
        print("   Env → Controller (same step)")
        print("   Controller → Bridge(cmd) → Plant (time-shifted)")
        print("   Plant → Bridge(sense) → Env")
        print("   ℹ️  Command format: JSON/dict {thrust, duration}")

    def setup_data_collection(self):
        """Setup data collection for all entities."""
        data_collector_sim = self.world.start("DataCollector", step_size=1)
        self.collector = data_collector_sim.Collector(output_dir=str(self.run_dir))

        # Collect data from all entities
        mosaik.util.connect_many_to_one(
            self.world,
            [self.controller],
            self.collector,
            "command",
            "error",
        )

        mosaik.util.connect_many_to_one(
            self.world,
            [self.bridge_cmd],
            self.collector,
            "stats",
            "packet_receive_time",
            "packet_send_time",
            "packet_actual_delay",
            # Debug attributes
            "buffer_size",
            # "buffer_content",  # Commented out to reduce log size
            "oldest_packet_time",
            "newest_packet_time",
        )

        mosaik.util.connect_many_to_one(
            self.world,
            [self.plant],
            self.collector,
            "measured_thrust",
            "actual_thrust",
            "status",
        )

        mosaik.util.connect_many_to_one(
            self.world,
            [self.bridge_sense],
            self.collector,
            "stats",
            "packet_receive_time",
            "packet_send_time",
            "packet_actual_delay",
            # Debug attributes
            "buffer_size",
            # "buffer_content",  # Commented out to reduce log size
            "oldest_packet_time",
            "newest_packet_time",
        )

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
