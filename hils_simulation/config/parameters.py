"""
Unified parameter management for HILS simulation scenarios.

This module provides a centralized way to load and manage simulation parameters
from environment variables and default values.
"""

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


def get_env_float(key: str, default: float) -> float:
    """
    Get float value from environment variable.

    Args:
        key: Environment variable name
        default: Default value if not found

    Returns:
        Float value from environment or default
    """
    value = os.getenv(key)
    if value is None or value == "None":
        return default
    return float(value)


def get_env_bool(key: str, default: bool) -> bool:
    """
    Get boolean value from environment variable.

    Args:
        key: Environment variable name
        default: Default value if not found

    Returns:
        Boolean value from environment or default
    """
    value = os.getenv(key)
    if value is None:
        return default
    return value.lower() == "true"


@dataclass
class CommunicationParams:
    """Communication delay parameters."""

    cmd_delay: float = 20.0  # Command path delay [ms]
    cmd_jitter: float = 0.0  # Command path jitter std [ms]
    cmd_loss_rate: float = 0.0  # Command path packet loss rate

    sense_delay: float = 30.0  # Sensing path delay [ms]
    sense_jitter: float = 0.0  # Sensing path jitter std [ms]
    sense_loss_rate: float = 0.0  # Sensing path packet loss rate

    @classmethod
    def from_env(cls) -> "CommunicationParams":
        """Load communication parameters from environment variables."""
        return cls(
            cmd_delay=get_env_float("CMD_DELAY", 20.0),
            cmd_jitter=get_env_float("CMD_JITTER", 0.0),
            cmd_loss_rate=get_env_float("CMD_LOSS_RATE", 0.0),
            sense_delay=get_env_float("SENSE_DELAY", 30.0),
            sense_jitter=get_env_float("SENSE_JITTER", 0.0),
            sense_loss_rate=get_env_float("SENSE_LOSS_RATE", 0.0),
        )


@dataclass
class ControlParams:
    """Control system parameters."""

    control_period: float = 10.0  # Control period [ms]
    kp: float = 15.0  # Proportional gain
    ki: float = 0.5  # Integral gain
    kd: float = 5.0  # Derivative gain
    target_position: float = 5.0  # Target position [m]
    max_thrust: float = 100.0  # Maximum thrust [N]
    integral_limit: float = 100.0  # Integral term limit

    @classmethod
    def from_env(cls) -> "ControlParams":
        """Load control parameters from environment variables."""
        return cls(
            control_period=get_env_float("CONTROL_PERIOD", 10.0),
            kp=get_env_float("KP", 15.0),
            ki=get_env_float("KI", 0.5),
            kd=get_env_float("KD", 5.0),
            target_position=get_env_float("TARGET_POSITION", 5.0),
            max_thrust=get_env_float("MAX_THRUST", 100.0),
            integral_limit=get_env_float("INTEGRAL_LIMIT", 100.0),
        )


@dataclass
class SimulatorParams:
    """Simulator timing parameters."""

    env_sim_period: float = 10.0  # Environment simulator period [ms]
    plant_sim_period: float = 10.0  # Plant simulator period [ms]

    @classmethod
    def from_env(cls) -> "SimulatorParams":
        """Load simulator parameters from environment variables."""
        return cls(
            env_sim_period=get_env_float("ENV_SIM_PERIOD", 10.0),
            plant_sim_period=get_env_float("PLANT_SIM_PERIOD", 10.0),
        )


@dataclass
class SpacecraftParams:
    """Spacecraft physical parameters."""

    mass: float = 1.0  # Mass [kg]
    initial_position: float = 0.0  # Initial position [m]
    initial_velocity: float = 10.0  # Initial velocity [m/s]
    gravity: float = 9.81  # Gravity acceleration [m/s^2]

    @classmethod
    def from_env(cls) -> "SpacecraftParams":
        """Load spacecraft parameters from environment variables."""
        return cls(
            mass=get_env_float("SPACECRAFT_MASS", 1.0),
            initial_position=get_env_float("INITIAL_POSITION", 0.0),
            initial_velocity=get_env_float("INITIAL_VELOCITY", 10.0),
            gravity=get_env_float("GRAVITY", 9.81),
        )


@dataclass
class PlantParams:
    """Plant (actuator) dynamics parameters."""

    time_constant: float = 50.0  # First-order lag time constant [ms]
    enable_lag: bool = True  # Enable first-order lag dynamics

    @classmethod
    def from_env(cls) -> "PlantParams":
        """Load plant parameters from environment variables."""
        return cls(
            time_constant=get_env_float("PLANT_TIME_CONSTANT", 50.0),
            enable_lag=get_env_bool("PLANT_ENABLE_LAG", True),
        )


@dataclass
class InverseCompParams:
    """Inverse compensation parameters."""

    enabled: bool = True  # Enable inverse compensation
    gain: float = 15.0  # Compensation gain

    @classmethod
    def from_env(cls) -> "InverseCompParams":
        """Load inverse compensation parameters from environment variables."""
        return cls(
            enabled=get_env_bool("ENABLE_INVERSE_COMP", True),
            gain=get_env_float("INVERSE_COMP_GAIN", 15.0),
        )


@dataclass
class SimulationParameters:
    """
    Complete set of simulation parameters.

    This class aggregates all parameter groups and provides methods for
    loading from environment variables and saving to JSON.
    """

    # Simulation timing
    simulation_time: float = 2.0  # Total simulation time [s]
    time_resolution: float = 0.0001  # Time step [s/step]
    rt_factor: Optional[float] = None  # Real-time factor (None = as fast as possible)

    # Parameter groups
    communication: CommunicationParams = None
    control: ControlParams = None
    simulators: SimulatorParams = None
    spacecraft: SpacecraftParams = None
    plant: PlantParams = None
    inverse_comp: InverseCompParams = None

    def __post_init__(self):
        """Initialize parameter groups if not provided."""
        if self.communication is None:
            self.communication = CommunicationParams()
        if self.control is None:
            self.control = ControlParams()
        if self.simulators is None:
            self.simulators = SimulatorParams()
        if self.spacecraft is None:
            self.spacecraft = SpacecraftParams()
        if self.plant is None:
            self.plant = PlantParams()
        if self.inverse_comp is None:
            self.inverse_comp = InverseCompParams()

    @classmethod
    def from_env(cls) -> "SimulationParameters":
        """
        Load all parameters from environment variables.

        Returns:
            SimulationParameters instance with values from environment
        """
        load_dotenv()  # Load .env file

        rt_factor_str = os.getenv("RT_FACTOR", "None")
        rt_factor = None if rt_factor_str == "None" else float(rt_factor_str)

        return cls(
            simulation_time=get_env_float("SIMULATION_TIME", 2.0),
            time_resolution=get_env_float("TIME_RESOLUTION", 0.0001),
            rt_factor=rt_factor,
            communication=CommunicationParams.from_env(),
            control=ControlParams.from_env(),
            simulators=SimulatorParams.from_env(),
            spacecraft=SpacecraftParams.from_env(),
            plant=PlantParams.from_env(),
            inverse_comp=InverseCompParams.from_env(),
        )

    @property
    def simulation_steps(self) -> int:
        """Calculate total number of simulation steps."""
        return int(self.simulation_time / self.time_resolution)

    @property
    def control_period_steps(self) -> int:
        """Calculate control period in steps."""
        return int(self.control.control_period / 1000 / self.time_resolution)

    @property
    def env_sim_period_steps(self) -> int:
        """Calculate environment simulator period in steps."""
        return int(self.simulators.env_sim_period / 1000 / self.time_resolution)

    @property
    def plant_sim_period_steps(self) -> int:
        """Calculate plant simulator period in steps."""
        return int(self.simulators.plant_sim_period / 1000 / self.time_resolution)

    def to_dict(self, scenario_type: str = "HILS") -> dict:
        """
        Convert parameters to dictionary format for JSON export.

        Args:
            scenario_type: Type of scenario (for metadata)

        Returns:
            Dictionary representation of all parameters
        """
        return {
            "simulation": {
                "simulation_time_s": self.simulation_time,
                "time_resolution_s": self.time_resolution,
                "simulation_steps": self.simulation_steps,
                "rt_factor": self.rt_factor,
                "type": scenario_type,
            },
            "communication": {
                "cmd_delay_s": self.communication.cmd_delay / 1000.0,
                "cmd_jitter_s": self.communication.cmd_jitter / 1000.0,
                "cmd_loss_rate": self.communication.cmd_loss_rate,
                "sense_delay_s": self.communication.sense_delay / 1000.0,
                "sense_jitter_s": self.communication.sense_jitter / 1000.0,
                "sense_loss_rate": self.communication.sense_loss_rate,
            },
            "control": {
                "control_period_s": self.control.control_period / 1000.0,
                "kp": self.control.kp,
                "ki": self.control.ki,
                "kd": self.control.kd,
                "target_position_m": self.control.target_position,
                "max_thrust_N": self.control.max_thrust,
                "integral_limit": self.control.integral_limit,
            },
            "simulators": {
                "env_sim_period_s": self.env_sim_period_steps * self.time_resolution,
                "plant_sim_period_s": self.plant_sim_period_steps * self.time_resolution,
            },
            "spacecraft": {
                "mass_kg": self.spacecraft.mass,
                "initial_position_m": self.spacecraft.initial_position,
                "initial_velocity_m_s": self.spacecraft.initial_velocity,
                "gravity_m_s2": self.spacecraft.gravity,
            },
            "plant": {
                "time_constant_s": self.plant.time_constant / 1000.0,
                "enable_lag": self.plant.enable_lag,
            },
            "inverse_compensation": {
                "enabled": self.inverse_comp.enabled,
                "gain": self.inverse_comp.gain,
            },
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "description": f"{scenario_type} 1-DOF Spacecraft Control Simulation",
                "note": "All time units are in seconds (s)",
            },
        }

    def save_to_json(self, output_dir: Path, scenario_type: str = "HILS") -> Path:
        """
        Save parameters to JSON file.

        Args:
            output_dir: Directory to save configuration
            scenario_type: Type of scenario (for metadata)

        Returns:
            Path to saved configuration file
        """
        config_path = output_dir / "simulation_config.json"
        with open(config_path, "w") as f:
            json.dump(self.to_dict(scenario_type), f, indent=2)
        return config_path
