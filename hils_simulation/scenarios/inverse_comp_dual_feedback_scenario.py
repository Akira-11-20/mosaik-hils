"""
HILS scenario with Inverse Compensation and Dual Feedback.

This scenario extends the post-compensation setup by adding a second bridge
from Plant to Inverse Compensator for delayed feedback.
"""

from pathlib import Path
from typing import Optional

import mosaik
import mosaik.util
from config.parameters import SimulationParameters
from config.sim_config import get_simulator_config

from .base_scenario import BaseScenario


class InverseCompDualFeedbackScenario(BaseScenario):
    """
    HILS scenario with inverse compensation and dual feedback paths.

    Data flow:
        Main Loop:
            Controller → Bridge-0 → Plant → Inverse → Bridge-1 → Env → Controller
        Dual Feedback:
            Bridge-0 → Inverse (delayed feedback)

    Features:
    - Full communication delays in both command and sensing paths
    - Inverse compensator positioned between Plant and Env (Bridge-1)
    - Dual feedback: Bridge-0 output also fed back to Inverse Compensator
    - Allows Inverse Compensator to access delayed command information
    """

    def __init__(self, params: SimulationParameters = None, minimal_data_mode: bool = False):
        """Initialize Inverse Compensation Dual Feedback scenario."""
        super().__init__(params, minimal_data_mode)
        self.controller = None
        self.inverse_comp = None
        self.plant = None
        self.spacecraft = None
        self.bridge_cmd = None
        self.bridge_feedback = None  # New bridge for Plant → InverseComp
        self.collector = None

    @property
    def scenario_name(self) -> str:
        return "HILS+InverseComp+DualFeedback"

    @property
    def scenario_description(self) -> str:
        comp_status = "Enabled" if self.params.inverse_comp.enabled else "Disabled"
        return f"HILS with Dual Feedback Inverse Compensation ({comp_status})"

    @property
    def results_base_dir(self) -> str:
        return "results"

    def setup_output_directory(self, suffix: str = "", parent_dir: Optional[Path] = None) -> Path:
        """Setup output directory with _inverse_comp_dual_feedback suffix."""
        if not suffix:
            suffix = "_inverse_comp_dual_feedback"
        return super().setup_output_directory(suffix, parent_dir=parent_dir)

    def create_world(self) -> mosaik.World:
        """Create Mosaik world with dual feedback inverse compensator."""
        sim_config = get_simulator_config(
            include_bridge=True, include_dual_feedback_inverse_comp=True
        )

        world = mosaik.World(
            sim_config,
            time_resolution=self.params.time_resolution,
            debug=False,
        )

        return world

    def setup_entities(self):
        """Create all simulation entities including inverse compensator and dual bridges."""
        print("\n🚀 Starting simulators...")

        controller_sim = self.world.start(
            "ControllerSim",
            step_size=self.params.control_period_steps,
        )

        # Dual Feedback Inverse compensator (runs at same rate as controller)
        if self.params.inverse_comp.enabled:
            inverse_comp_sim = self.world.start(
                "DualFeedbackInverseCompSim",
                step_size=self.params.control_period_steps,
            )
            mode_info = f"mode={self.params.inverse_comp.tau_model_type}"
            if self.params.inverse_comp.tau_model_type == "constant":
                mode_info += f", gain={self.params.inverse_comp.gain}"
            else:
                mode_info += f", base_tau={self.params.inverse_comp.base_tau}ms, ratio={self.params.inverse_comp.tau_to_gain_ratio}"
            print(f" ✨ Dual Feedback Inverse Compensator enabled ({mode_info})")
            print(f" 📡 Dual Feedback mode: Plant → InverseComp, Bridge-0 → InverseComp")

        plant_sim = self.world.start("PlantSim", step_size=self.params.plant_sim_period_steps)
        env_sim = self.world.start("EnvSim", step_size=self.params.env_sim_period_steps)

        bridge_cmd_sim = self.world.start("BridgeSim", step_size=1, log_dir=str(self.run_dir))
        bridge_feedback_sim = self.world.start("BridgeSim", step_size=1, log_dir=str(self.run_dir))

        # Create entities
        self.controller = controller_sim.PIDController(
            kp=self.params.control.kp,
            ki=self.params.control.ki,
            kd=self.params.control.kd,
            target_position=self.params.control.target_position,
            min_thrust=self.params.control.min_thrust,
            max_thrust=self.params.control.max_thrust,
            thrust_duration=self.params.control.control_period,
            integral_limit=self.params.control.integral_limit,
        )

        if self.params.inverse_comp.enabled:
            self.inverse_comp = inverse_comp_sim.DualFeedbackInverseCompensator(
                comp_id="dual_fb_compensator",
                gain=self.params.inverse_comp.gain,
                comp_type="command",
                tau_to_gain_ratio=self.params.inverse_comp.tau_to_gain_ratio,
                base_tau=self.params.inverse_comp.base_tau,
                tau_model_type=self.params.inverse_comp.tau_model_type,
                tau_model_params=self.params.inverse_comp.tau_model_params,
                feedback_weight=0.5,  # Customize this parameter
                enable_dual_compensation=True,  # Enable dual feedback compensation
            )

        self.plant = plant_sim.ThrustStand(
            stand_id="stand_01",
            time_constant=self.params.plant.time_constant,
            time_constant_std=self.params.plant.time_constant_std,
            time_constant_noise=self.params.plant.time_constant_noise,
            enable_lag=self.params.plant.enable_lag,
            tau_model_type=self.params.plant.tau_model_type,
            tau_model_params=self.params.plant.tau_model_params,
            min_thrust=self.params.plant.min_thrust,
            max_thrust=self.params.plant.max_thrust,
        )

        self.spacecraft = env_sim.Spacecraft1DOF(
            mass=self.params.spacecraft.mass,
            initial_position=self.params.spacecraft.initial_position,
            initial_velocity=self.params.spacecraft.initial_velocity,
            gravity=self.params.spacecraft.gravity,
        )

        # Command path bridge
        self.bridge_cmd = bridge_cmd_sim.CommBridge(
            bridge_type="cmd",
            base_delay=self.params.communication.cmd_delay,
            jitter_std=self.params.communication.cmd_jitter,
            packet_loss_rate=self.params.communication.cmd_loss_rate,
            preserve_order=True,
            compensate_time_shifted=True,
            time_shifted_delay_ms=self.params.time_resolution * 1000,  # 1 step in ms
        )

        # Feedback path bridge (Plant → InverseComp)
        self.bridge_feedback = bridge_feedback_sim.CommBridge(
            bridge_type="feedback",
            base_delay=self.params.communication.sense_delay,
            jitter_std=self.params.communication.sense_jitter,
            packet_loss_rate=self.params.communication.sense_loss_rate,
            preserve_order=True,
        )

    def connect_entities(self):
        """Connect entities with dual feedback paths."""
        comp_enabled = self.params.inverse_comp.enabled

        if comp_enabled:
            # Main loop: Controller → Bridge-0 → Plant → Inverse → Bridge-1 → Env → Controller
            print("   ⏱️  Controller → Bridge-0 (cmd): time-shifted connection")
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

            print("   Bridge-0 (cmd) → Plant: delayed command")
            self.world.connect(
                self.bridge_cmd,
                self.plant,
                ("delayed_output", "command"),
            )

            print("   Plant → Inverse Compensator: actual_thrust connection")
            self.world.connect(
                self.plant,
                self.inverse_comp,
                ("actual_thrust", "input"),
            )

            print("   Inverse Compensator → Bridge-1 (sense): compensated output")
            self.world.connect(
                self.inverse_comp,
                self.bridge_feedback,
                ("compensated_output", "input"),
            )

            print("   Bridge-1 (sense) → Env: delayed force")
            self.world.connect(
                self.bridge_feedback,
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

            # Additional feedback: Bridge-0 → Inverse (dual feedback)
            print("   🔄 Bridge-0 (cmd) → Inverse Compensator: delayed feedback (dual path)")
            self.world.connect(
                self.bridge_cmd,
                self.inverse_comp,
                ("delayed_output", "delayed_feedback"),
            )

        else:
            # NO COMPENSATION: Controller → Bridge(cmd) → Plant → Env
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

            print("   Bridge(cmd) → Plant: delayed command")
            self.world.connect(
                self.bridge_cmd,
                self.plant,
                ("delayed_output", "command"),
            )

            print("   Plant → Env: direct force connection")
            self.world.connect(
                self.plant,
                self.spacecraft,
                ("actual_thrust", "force"),
            )

            print("   📡 Env → Controller: same-step connection (state feedback)")
            self.world.connect(
                self.spacecraft,
                self.controller,
                "position",
                "velocity",
            )

        print("\n✅ Data flow configured:")
        if comp_enabled:
            print("   Main Loop:")
            print("     Controller → Bridge-0 (cmd) → Plant → Inverse → Bridge-1 (sense) → Env → Controller")
            print("   Dual Feedback:")
            print("     Bridge-0 (cmd) → Inverse (delayed feedback)")
        else:
            print("   Controller → Bridge(cmd) → Plant → Env → Controller")

    def setup_data_collection(self):
        """Setup data collection including inverse compensator and feedback bridge data."""
        data_collector_sim = self.world.start("DataCollector", step_size=1)
        self.collector = data_collector_sim.Collector(output_dir=str(self.run_dir), minimal_mode=self.minimal_data_mode)

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

        # Collect dual feedback inverse compensator data if enabled
        if self.params.inverse_comp.enabled:
            mosaik.util.connect_many_to_one(
                self.world,
                [self.inverse_comp],
                self.collector,
                "compensated_output",
                "stats",
                # Debug attributes
                "raw_input",
                "input_thrust",
                "output_thrust",
                "current_gain",
                "current_tau",
                "delta",
                "compensation_amount",
                "delayed_feedback_value",  # Delayed feedback from Bridge-0
                "delayed_command_thrust",  # Extracted thrust from delayed feedback
                "feedback_contribution",  # Contribution from dual feedback
            )

        mosaik.util.connect_many_to_one(
            self.world,
            [self.bridge_cmd],
            self.collector,
            "stats",
            "packet_receive_time",
            "packet_send_time",
            "packet_actual_delay",
            "buffer_size",
            "oldest_packet_time",
            "newest_packet_time",
        )

        # Collect feedback bridge data
        mosaik.util.connect_many_to_one(
            self.world,
            [self.bridge_feedback],
            self.collector,
            "stats",
            "packet_receive_time",
            "packet_send_time",
            "packet_actual_delay",
            "buffer_size",
            "oldest_packet_time",
            "newest_packet_time",
        )

        mosaik.util.connect_many_to_one(
            self.world,
            [self.plant],
            self.collector,
            "measured_thrust",
            "actual_thrust",
            "time_constant",
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
