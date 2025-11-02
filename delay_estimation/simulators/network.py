"""Network delay simulator"""

import numpy as np
from collections import deque
from typing import Optional, Tuple


class NetworkDelay:
    """
    Network delay model

    Simulates time-varying delays in measurement transmission
    """

    def __init__(
        self,
        mean_delay: float,
        delay_std: float,
        dt: float,
        max_delay: int = 10,
        delay_type: str = "constant",
    ):
        """
        Initialize network delay model

        Args:
            mean_delay: Mean delay [s]
            delay_std: Delay standard deviation [s]
            dt: Time step [s]
            max_delay: Maximum delay in time steps
            delay_type: Type of delay ("constant", "random", "varying")
        """
        self.mean_delay = mean_delay
        self.delay_std = delay_std
        self.dt = dt
        self.max_delay = max_delay
        self.delay_type = delay_type

        # Convert delay from seconds to time steps
        self.mean_delay_steps = int(mean_delay / dt)
        self.delay_std_steps = int(delay_std / dt)

        # Buffer to store delayed measurements
        self.buffer = deque(maxlen=max_delay + 1)

        # Current delay
        self.current_delay = self.mean_delay_steps

    def add_measurement(self, measurement: np.ndarray, time_step: int) -> None:
        """
        Add measurement to delay buffer

        Args:
            measurement: Measurement to be delayed
            time_step: Current time step
        """
        self.buffer.append((time_step, measurement.copy()))

    def get_delayed_measurement(
        self, current_time_step: int
    ) -> Tuple[Optional[np.ndarray], int]:
        """
        Get delayed measurement

        Args:
            current_time_step: Current time step

        Returns:
            measurement: Delayed measurement (or None if buffer empty)
            actual_delay: Actual delay in time steps
        """
        if len(self.buffer) == 0:
            return None, 0

        # Update delay based on type
        if self.delay_type == "constant":
            delay = self.mean_delay_steps
        elif self.delay_type == "random":
            delay = max(
                0,
                int(
                    np.random.normal(self.mean_delay_steps, self.delay_std_steps)
                ),
            )
            delay = min(delay, self.max_delay)
        elif self.delay_type == "varying":
            # Sinusoidal variation
            delay = int(
                self.mean_delay_steps
                + self.delay_std_steps * np.sin(current_time_step * 0.1)
            )
            delay = max(0, min(delay, self.max_delay))
        else:
            delay = self.mean_delay_steps

        self.current_delay = delay

        # Find measurement with the appropriate delay
        target_time = current_time_step - delay

        # Search for closest measurement
        best_measurement = None
        best_time_diff = float("inf")
        actual_delay = 0

        for time_step, measurement in self.buffer:
            time_diff = abs(time_step - target_time)
            if time_diff < best_time_diff:
                best_time_diff = time_diff
                best_measurement = measurement
                actual_delay = current_time_step - time_step

        return best_measurement, actual_delay

    def get_current_delay(self) -> int:
        """Get current delay in time steps"""
        return self.current_delay

    def reset(self):
        """Reset buffer"""
        self.buffer.clear()
        self.current_delay = self.mean_delay_steps
