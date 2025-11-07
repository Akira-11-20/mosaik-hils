"""Standard Kalman Filter implementation"""

from typing import Optional, Tuple

import numpy as np


class KalmanFilter:
    """
    Standard Kalman Filter for linear systems

    State space model:
        x(k+1) = A*x(k) + B*u(k) + w(k)  # w ~ N(0, Q)
        y(k) = C*x(k) + v(k)              # v ~ N(0, R)
    """

    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        C: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        x0: np.ndarray,
        P0: np.ndarray,
    ):
        """
        Initialize Kalman Filter

        Args:
            A: State transition matrix (n x n)
            B: Control input matrix (n x m)
            C: Measurement matrix (p x n)
            Q: Process noise covariance (n x n)
            R: Measurement noise covariance (p x p)
            x0: Initial state estimate (n,)
            P0: Initial state covariance (n x n)
        """
        self.A = A
        self.B = B
        self.C = C
        self.Q = Q
        self.R = R

        self.x = x0.copy()  # State estimate
        self.P = P0.copy()  # State covariance

        self.n = A.shape[0]  # State dimension
        self.m = B.shape[1]  # Control dimension
        self.p = C.shape[0]  # Measurement dimension

    def predict(self, u: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prediction step

        Args:
            u: Control input (m,), optional

        Returns:
            x_pred: Predicted state (n,)
            P_pred: Predicted covariance (n x n)
        """
        if u is None:
            u = np.zeros(self.m)

        # Predicted state
        self.x = self.A @ self.x + self.B @ u

        # Predicted covariance
        self.P = self.A @ self.P @ self.A.T + self.Q

        return self.x.copy(), self.P.copy()

    def update(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Update step (correction)

        Args:
            y: Measurement (p,)

        Returns:
            x_updated: Updated state estimate (n,)
            P_updated: Updated covariance (n x n)
            innovation: Measurement innovation (p,)
        """
        # Innovation (measurement residual)
        innovation = y - self.C @ self.x

        # Innovation covariance
        S = self.C @ self.P @ self.C.T + self.R

        # Kalman gain
        K = self.P @ self.C.T @ np.linalg.inv(S)

        # Updated state
        self.x = self.x + K @ innovation

        # Updated covariance (Joseph form for numerical stability)
        I_KC = np.eye(self.n) - K @ self.C
        self.P = I_KC @ self.P @ I_KC.T + K @ self.R @ K.T

        return self.x.copy(), self.P.copy(), innovation

    def step(self, y: np.ndarray, u: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Complete filter step (predict + update)

        Args:
            y: Measurement (p,)
            u: Control input (m,), optional

        Returns:
            x: State estimate (n,)
            P: State covariance (n x n)
            innovation: Measurement innovation (p,)
        """
        self.predict(u)
        return self.update(y)

    def get_state(self) -> np.ndarray:
        """Get current state estimate"""
        return self.x.copy()

    def get_covariance(self) -> np.ndarray:
        """Get current state covariance"""
        return self.P.copy()
