"""System dynamics simulator"""

import numpy as np
from typing import Tuple


class LinearSystem:
    """
    Linear time-invariant system

    Dynamics:
        x(k+1) = A*x(k) + B*u(k) + w(k)
        y(k) = C*x(k) + v(k)
    """

    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        C: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        x0: np.ndarray,
        dt: float = 0.01,
    ):
        """
        Initialize linear system

        Args:
            A: State transition matrix (n x n)
            B: Control input matrix (n x m)
            C: Measurement matrix (p x n)
            Q: Process noise covariance (n x n)
            R: Measurement noise covariance (p x p)
            x0: Initial state (n,)
            dt: Time step [s]
        """
        self.A = A
        self.B = B
        self.C = C
        self.Q = Q
        self.R = R
        self.x = x0.copy()
        self.dt = dt

        self.n = A.shape[0]  # State dimension
        self.m = B.shape[1]  # Control dimension
        self.p = C.shape[0]  # Measurement dimension

    def step(self, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate one time step

        Args:
            u: Control input (m,)

        Returns:
            x: Current state (n,)
            y: Measurement (p,)
        """
        # Process noise
        w = np.random.multivariate_normal(np.zeros(self.n), self.Q)

        # State update
        self.x = self.A @ self.x + self.B @ u + w

        # Measurement noise
        v = np.random.multivariate_normal(np.zeros(self.p), self.R)

        # Measurement
        y = self.C @ self.x + v

        return self.x.copy(), y

    def get_state(self) -> np.ndarray:
        """Get current state"""
        return self.x.copy()

    def reset(self, x0: np.ndarray):
        """Reset state"""
        self.x = x0.copy()


class OneDOFSystem(LinearSystem):
    """
    1-DOF mechanical system (position-velocity)

    State: [position, velocity]
    Input: [force]
    Output: [position]

    Dynamics:
        x'' = F/m
    """

    @staticmethod
    def create_matrices(mass: float, dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create system matrices for 1-DOF system

        Args:
            mass: Mass [kg]
            dt: Time step [s]

        Returns:
            A: State transition matrix (2 x 2)
            B: Control input matrix (2 x 1)
            C: Measurement matrix (1 x 2)
        """
        # Continuous-time system: x_dot = [0 1; 0 0] * x + [0; 1/m] * u
        # Discretization using Euler method
        A = np.array([[1.0, dt], [0.0, 1.0]])

        B = np.array([[0.0], [dt / mass]])

        C = np.array([[1.0, 0.0]])  # Measure position only

        return A, B, C

    def __init__(
        self,
        mass: float,
        dt: float,
        process_noise_std: float = 0.1,
        measurement_noise_std: float = 0.5,
        x0: np.ndarray = None,
    ):
        """
        Initialize 1-DOF system

        Args:
            mass: Mass [kg]
            dt: Time step [s]
            process_noise_std: Process noise standard deviation
            measurement_noise_std: Measurement noise standard deviation
            x0: Initial state [position, velocity]
        """
        A, B, C = self.create_matrices(mass, dt)

        # Noise covariance matrices
        Q = np.diag([process_noise_std**2, process_noise_std**2])
        R = np.array([[measurement_noise_std**2]])

        if x0 is None:
            x0 = np.zeros(2)

        super().__init__(A, B, C, Q, R, x0, dt)
        self.mass = mass
