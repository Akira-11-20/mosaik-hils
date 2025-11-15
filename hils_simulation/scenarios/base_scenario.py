"""
Base scenario class for all HILS simulation scenarios.

This module provides the abstract base class that defines the common interface
and shared functionality for all simulation scenarios.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Optional

import mosaik
import mosaik.util
from config.parameters import SimulationParameters


class BaseScenario(ABC):
    """
    Abstract base class for all simulation scenarios.

    This class defines the common interface and provides shared functionality
    for scenario setup, execution, and post-processing.
    """

    def __init__(self, params: Optional[SimulationParameters] = None, minimal_data_mode: bool = False):
        """
        Initialize scenario with parameters.

        Args:
            params: Simulation parameters. If None, loads from environment.
            minimal_data_mode: If True, only collect minimal data (time, position, velocity) for faster simulation
        """
        self.params = params if params is not None else SimulationParameters.from_env()
        self.world: Optional[mosaik.World] = None
        self.run_dir: Optional[Path] = None
        self.minimal_data_mode = minimal_data_mode

    @property
    @abstractmethod
    def scenario_name(self) -> str:
        """Return the name of this scenario (e.g., 'HILS', 'RT')."""
        pass

    @property
    @abstractmethod
    def scenario_description(self) -> str:
        """Return a brief description of this scenario."""
        pass

    @property
    def results_base_dir(self) -> str:
        """Return the base directory name for results (e.g., 'results', 'results_rt')."""
        return "results"

    def setup_output_directory(self, suffix: str = "", parent_dir: Optional[Path] = None) -> Path:
        """
        Create output directory for this simulation run.

        Args:
            suffix: Optional suffix to append to timestamp
            parent_dir: Optional parent directory path. If provided, creates output directory under this parent.
                       Useful for organizing multiple runs (e.g., parameter sweeps) under a common directory.

        Returns:
            Path to created output directory
        """
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        dir_name = f"{timestamp}{suffix}" if suffix else timestamp

        if parent_dir is not None:
            # Create under specified parent directory
            self.run_dir = parent_dir / dir_name
        else:
            # Create under default results base directory
            self.run_dir = Path(self.results_base_dir) / dir_name

        self.run_dir.mkdir(parents=True, exist_ok=True)
        return self.run_dir

    def save_configuration(self):
        """Save simulation configuration to JSON file."""
        if self.run_dir is None:
            raise RuntimeError("Output directory not set. Call setup_output_directory() first.")

        config_path = self.params.save_to_json(self.run_dir, self.scenario_name)
        print(f"💾 Configuration saved: {config_path}")

    def print_header(self):
        """Print scenario header information."""
        print("=" * 70)
        print(f"{self.scenario_name} Simulation - 1DOF Configuration")
        print(f"{self.scenario_description}")
        print("=" * 70)

    def print_simulation_info(self):
        """Print simulation configuration information."""
        print("\n🌍 Simulation Info:")
        print(f"   Time: {self.params.simulation_time}s")
        print(f"   Resolution: {self.params.time_resolution}s ({self.params.time_resolution * 1000}ms)")
        print(f"   Steps: {self.params.simulation_steps}")
        print(f"   RT Factor: {self.params.rt_factor}")

    @abstractmethod
    def create_world(self) -> mosaik.World:
        """
        Create and configure Mosaik world.

        Returns:
            Configured Mosaik world instance
        """
        pass

    @abstractmethod
    def setup_entities(self):
        """Create and configure simulation entities."""
        pass

    @abstractmethod
    def connect_entities(self):
        """Connect entities to form the simulation dataflow."""
        pass

    def setup_data_collection(self):
        """
        Setup data collection (common implementation).

        Subclasses can override to customize data collection.
        """
        pass

    def generate_graphs(self):
        """
        Generate execution graphs and plots.

        Default implementation generates standard Mosaik graphs.
        Subclasses can override to customize graph generation.
        """
        if self.run_dir is None or self.world is None:
            return

        print("\n📊 Generating execution graphs...")
        try:
            import sys
            from pathlib import Path

            # プロジェクトルートをパスに追加
            project_root = Path(__file__).parent.parent.parent
            sys.path.insert(0, str(project_root))

            from common_utils.plot_utils import plot_dataflow_graph_custom

            plot_kwargs = {
                "folder": str(self.run_dir),
                "show_plot": False,
            }

            # Custom dataflow graph
            plot_dataflow_graph_custom(
                self.world,
                folder=str(self.run_dir),
                show_plot=False,
                dpi=600,
                format="png",
                node_size=150,
                node_label_size=12,
                edge_label_size=8,
                node_color="tab:blue",
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

    def run(self):
        """
        Execute the complete simulation scenario.

        This is the main entry point that orchestrates all steps.
        """
        # Print header
        self.print_header()

        # Setup output directory
        self.setup_output_directory()
        print(f"📁 Log directory: {self.run_dir}")

        # Save configuration
        self.save_configuration()

        # Print simulation info
        self.print_simulation_info()

        # Create world
        print("\n🌍 Creating Mosaik World...")
        self.world = self.create_world()

        # Setup entities
        print("\n📦 Creating entities...")
        self.setup_entities()

        # Connect entities
        print("\n🔗 Connecting data flows...")
        self.connect_entities()

        # Setup data collection
        print("\n📊 Setting up data collection...")
        self.setup_data_collection()

        # Run simulation
        print(f"\n▶️  Running simulation until {self.params.simulation_time}s ({self.params.simulation_steps} steps)...")
        print("=" * 70)

        self.world.run(until=self.params.simulation_steps, rt_factor=self.params.rt_factor)

        print("=" * 70)
        print("✅ Simulation completed successfully!")

        # Generate graphs
        self.generate_graphs()

        # Print footer
        print("\n" + "=" * 70)
        print(f"{self.scenario_name} Simulation Finished")
        print("=" * 70)

    def run_with_custom_suffix(self, suffix: str, parent_dir: Optional[Path] = None):
        """
        Run simulation with custom output directory suffix.

        Args:
            suffix: Suffix to append to timestamp in directory name
            parent_dir: Optional parent directory path for organizing multiple runs
        """
        # Print header
        self.print_header()

        # Setup output directory with suffix
        self.setup_output_directory(suffix=suffix, parent_dir=parent_dir)
        print(f"📁 Log directory: {self.run_dir}")

        # Continue with normal run
        self.save_configuration()
        self.print_simulation_info()

        print("\n🌍 Creating Mosaik World...")
        self.world = self.create_world()

        print("\n📦 Creating entities...")
        self.setup_entities()

        print("\n🔗 Connecting data flows...")
        self.connect_entities()

        print("\n📊 Setting up data collection...")
        self.setup_data_collection()

        print(f"\n▶️  Running simulation until {self.params.simulation_time}s ({self.params.simulation_steps} steps)...")
        print("=" * 70)

        self.world.run(until=self.params.simulation_steps, rt_factor=self.params.rt_factor)

        print("=" * 70)
        print("✅ Simulation completed successfully!")

        self.generate_graphs()

        print("\n" + "=" * 70)
        print(f"{self.scenario_name} Simulation Finished")
        print("=" * 70)
