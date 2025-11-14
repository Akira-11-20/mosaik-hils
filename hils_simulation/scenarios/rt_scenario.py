"""
RT (Real-Time) scenario without communication delays.

This scenario implements a direct connection setup without bridges,
representing an ideal control loop for comparison with HILS.
"""

import mosaik
import mosaik.util
from config.parameters import SimulationParameters
from config.sim_config import get_simulator_config

from .base_scenario import BaseScenario


class RTScenario(BaseScenario):
    """
    RT scenario without communication delays.

    Data flow:
        Env → Controller → Plant → Env (all direct connections, no delays)

    Features:
    - No communication bridges
    - No delays or packet loss
    - Ideal control loop for baseline comparison
    - Uses same time resolution as HILS for fair comparison
    """

    def __init__(self, params: SimulationParameters = None, minimal_data_mode: bool = False):
        """Initialize RT scenario."""
        super().__init__(params, minimal_data_mode)
        self.controller = None
        self.plant = None
        self.spacecraft = None
        self.collector = None

    @property
    def scenario_name(self) -> str:
        return "RT"

    @property
    def scenario_description(self) -> str:
        return "Real-Time Simulation (No Communication Delays)"

    @property
    def results_base_dir(self) -> str:
        return "results_rt"

    def create_world(self) -> mosaik.World:
        """Create Mosaik world with RT configuration (no bridges)."""
        sim_config = get_simulator_config(include_bridge=False, include_inverse_comp=False)

        world = mosaik.World(
            sim_config,
            time_resolution=self.params.time_resolution,
            debug=False,
        )

        return world

    def setup_entities(self):
        """Create all simulation entities for RT scenario."""
        print("\n🚀 Starting simulators...")

        controller_sim = self.world.start(
            "ControllerSim",
            step_size=self.params.control_period_steps,
        )

        plant_sim = self.world.start("PlantSim", step_size=self.params.plant_sim_period_steps)
        env_sim = self.world.start("EnvSim", step_size=self.params.env_sim_period_steps)

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

        self.plant = plant_sim.ThrustStand(
            stand_id="stand_01",
            time_constant=self.params.plant.time_constant,
            time_constant_std=self.params.plant.time_constant_std,
            time_constant_noise=self.params.plant.time_constant_noise,
            enable_lag=False,
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

    def connect_entities(self):
        """Connect entities to form RT dataflow (direct connections)."""
        # 1. Controller → Plant - direct connection (time-shifted to break cycle)
        # Note: RT scenario doesn't use Bridge, so time_shifted adds only 1 step delay
        # This is acceptable as RT is intended to have minimal delay
        print("   ⚡ Controller → Plant: 1-step shifted (to break cycle)")
        self.world.connect(
            self.controller,
            self.plant,
            ("command", "command"),
            time_shifted=True,
            initial_data={
                "command": {
                    "thrust": 0.0,
                    "duration": self.params.control.control_period,
                }
            },
        )

        # 2. Plant → Env - direct connection (no delay)
        # Uses actual_thrust (with first-order lag) instead of measured_thrust (ideal)
        print("   ⚡ Plant → Env: direct connection (no delay)")
        self.world.connect(
            self.plant,
            self.spacecraft,
            ("actual_thrust", "force"),
        )

        # 3. Env → Controller - state feedback (same step)
        print("   📡 Env → Controller: same-step connection (state feedback)")
        self.world.connect(
            self.spacecraft,
            self.controller,
            "position",
            "velocity",
        )

        print("\n✅ Data flow configured:")
        print("   Env → Controller (same step)")
        print("   Controller → Plant (1-step shifted, to break cycle)")
        print("   Plant → Env (direct, no delay)")
        print("   ⚠️  Note: 1-step shift = minimal overhead for cycle breaking")

    def setup_data_collection(self):
        """Setup data collection for all entities."""
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

        mosaik.util.connect_many_to_one(
            self.world,
            [self.plant],
            self.collector,
            "measured_thrust",
            "actual_thrust",
            "status",
            "time_constant",  # Record actual sampled time constant
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

    def generate_graphs(self):
        """Generate execution graphs with RT-specific styling."""
        if self.run_dir is None or self.world is None:
            return

        print("\n📊 Generating execution graphs...")
        try:
            from utils.plot_utils import plot_dataflow_graph_custom

            plot_kwargs = {
                "folder": str(self.run_dir),
                "show_plot": False,
            }

            # Custom dataflow graph with green color for RT
            plot_dataflow_graph_custom(
                self.world,
                folder=str(self.run_dir),
                show_plot=False,
                dpi=600,
                format="png",
                node_size=150,
                node_label_size=12,
                edge_label_size=8,
                node_color="tab:green",  # Green for RT scenario
                node_alpha=0.8,
                label_alpha=0.8,
                edge_alpha=0.6,
                arrow_size=25,
                figsize=(6, 5),
                exclude_nodes=["DataCollector"],
            )

            # Execution time graph
            mosaik.util.plot_execution_time(self.world, **plot_kwargs)

            print(f"   Graphs saved to {self.run_dir}/")
        except Exception as e:
            print(f"   ⚠️  Graph generation failed: {e}")
