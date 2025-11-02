"""Parameter configuration for delay estimation experiments"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class SystemParameters:
    """System dynamics parameters"""
    # State space dimensions
    state_dim: int = 2  # [position, velocity]
    measurement_dim: int = 1  # [position]
    control_dim: int = 1  # [force]

    # System matrices (1-DOF system: x'' = F/m)
    mass: float = 10.0  # kg
    dt: float = 0.01  # sampling time [s]

    # Initial conditions
    initial_position: float = 0.0  # m
    initial_velocity: float = 0.0  # m/s


@dataclass
class NetworkParameters:
    """Network delay and noise parameters"""
    # Delay characteristics
    mean_delay: float = 0.05  # mean delay [s]
    delay_std: float = 0.01  # delay standard deviation [s]
    max_delay: int = 10  # maximum delay in time steps

    # Noise characteristics
    process_noise_std: float = 0.1  # process noise std
    measurement_noise_std: float = 0.5  # measurement noise std


@dataclass
class KalmanParameters:
    """Kalman filter parameters"""
    # Process noise covariance Q
    q_position: float = 0.01
    q_velocity: float = 0.01

    # Measurement noise covariance R
    r_measurement: float = 0.25

    # Initial covariance P0
    p0_position: float = 1.0
    p0_velocity: float = 1.0


@dataclass
class EstimatorParameters:
    """Delay estimator parameters"""
    # Estimation method
    method: str = "innovation"  # "innovation", "ml", "bayesian"

    # Window size for estimation
    window_size: int = 10

    # Threshold for delay change detection
    detection_threshold: float = 0.5


@dataclass
class SimulationParameters:
    """Overall simulation parameters"""
    total_time: float = 10.0  # total simulation time [s]
    dt: float = 0.01  # time step [s]

    # Sub-parameter groups
    system: SystemParameters = None
    network: NetworkParameters = None
    kalman: KalmanParameters = None
    estimator: EstimatorParameters = None

    # Results
    save_results: bool = True
    results_dir: str = "results"

    def __post_init__(self):
        """Initialize sub-parameters if not provided"""
        if self.system is None:
            self.system = SystemParameters(dt=self.dt)
        if self.network is None:
            self.network = NetworkParameters()
        if self.kalman is None:
            self.kalman = KalmanParameters()
        if self.estimator is None:
            self.estimator = EstimatorParameters()

    @property
    def num_steps(self) -> int:
        """Calculate total number of simulation steps"""
        return int(self.total_time / self.dt)

    @classmethod
    def default(cls) -> "SimulationParameters":
        """Create default simulation parameters"""
        return cls()
