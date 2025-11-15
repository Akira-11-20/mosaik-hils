"""
Inverse Compensator Simulator for HILS

This simulator applies inverse transfer compensation to counteract communication delays.
Two types of compensation are supported:

1. Command Inverse Compensation (Command Path):
   - Applied before command delay: Controller → [Inverse Comp] → [Delay] → Plant
   - Pre-shapes the command signal to cancel upcoming delay

2. Sensor Inverse Compensation (Sensing Path):
   - Applied after sensor delay: Sensor → [Delay] → [Inverse Comp] → Controller
   - Post-processes delayed measurement to recover timing

Compensation formula:
    y_comp[k] = y[k] + gain * (y[k] - y[k-1])

where 'gain' is the compensation gain that controls prediction strength.
This is a lead compensator that predicts future values based on current trends.

逆伝達補償シミュレータ - HILS用
通信遅延を補償するための2つのモード：
1. 指令補償（Command Path）: 遅延前に適用
2. センサー補償（Sensing Path）: 遅延後に適用
"""

import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import mosaik_api

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

meta = {
    "type": "time-based",
    "models": {
        "InverseCompensator": {
            "public": True,
            "params": [
                "gain",
            ],
            "attrs": [
                # 入力
                "input_force_x",  # 推力 X成分 [N]
                "input_force_y",  # 推力 Y成分 [N]
                "input_force_z",  # 推力 Z成分 [N]
                "input_norm_force",  # 推力ノルム [N]
                # 出力
                "compensated_force_x",  # 補償後推力 X成分 [N]
                "compensated_force_y",  # 補償後推力 Y成分 [N]
                "compensated_force_z",  # 補償後推力 Z成分 [N]
                "compensated_norm_force",  # 補償後推力ノルム [N]
                # デバッグ情報
                "gain",
            ],
        }
    },
}


class InverseCompensatorSimulator(mosaik_api.Simulator):
    """
    Mosaik Inverse Compensator Simulator

    Applies inverse transfer compensation to signals to counteract delays.
    """

    def __init__(self):
        """Initialize the simulator"""
        super().__init__(meta)
        self.entities = {}
        self.step_size: int = 1
        self.time_resolution: float = 0.001

    def init(self, sid: str, time_resolution: float = 1.0, step_size: int = 1, **sim_params):
        """
        Initialize the simulator

        Args:
            sid: Simulator ID
            time_resolution: Time resolution in seconds per step
            **sim_params: Additional simulator parameters
        """
        self.sid = sid
        self.time_resolution = self.time_resolution
        self.step_size = self.step_size
        return self.meta

    def create(self, num: int, model: str, **model_params) -> list[dict]:
        """
        Create InverseCompensator entities

        Args:
            num: Number of entities to create
            model: Model name (must be "InverseCompensator")
            **model_params: Model parameters
                - comp_id: Compensator ID (str)
                - gain: Direct compensation gain (float, default: 15.0)
                - comp_type: "command" or "sensor" (str, default: "command")
                - tau_to_gain_ratio: Ratio for tau->gain conversion (float, default: 0.1)
                - base_tau: Base time constant (float, default: 100.0)
                - tau_model_type: "constant" (use direct gain) or "linear", "hybrid", etc.
                - tau_model_params: Time constant model parameters (dict)

        Returns:
            List of entity info dictionaries
        """
        entities = []
        for i in range(num):
            eid = f"{model}_{i}"
            gain = model_params.get("gain", 15.0)
            time_resolution = model_params.get("time_resolution", self.time_resolution)
            step_size = model_params.get("step_size", self.step_size)
            comp = InverseCompensator(
                gain=gain,
                time_resolution=time_resolution,
                step_size=step_size,
            )
            self.entities[eid] = {
                "compensator": comp,
                "force": np.zeros(3),
                "norm_force": 0.0,
                "compensated_force": np.zeros(3),
                "compensated_norm_force": 0.0,
            }

            entities.append({"eid": eid, "type": model})

        return entities

    def step(self, time: int, inputs, max_advance: int) -> int:
        """
        Execute one simulation step

        Args:
            time: Current simulation time (in steps)
            inputs: Input data from connected entities
            max_advance: Maximum time advance allowed

        Returns:
            Next step time
        """
        for eid, entity in self.entities.items():
            force = np.zeros(3)
            if eid in inputs:
                for axis, attr in enumerate(["force_x", "force_y", "force_z"]):
                    if attr in inputs[eid]:
                        force_value = list(inputs[eid][attr].values())[0]
                        force[axis] = force_value if force_value is not None else 0.0
            
            
            entity["force"] = force
            
            compensator = entity["compensator"]
            
            compensated_force = compensator._process_input(force)
            
            entity["compensated_force"] = compensated_force
            entity["compensated_norm_force"] = np.linalg.norm(compensated_force)

        return time + self.step_size

    def get_data(self, outputs: Dict[str, list[str]]) -> Dict[str, Dict[str, Any]]:
        """
        Return output data for requested entities and attributes

        Args:
            outputs: Dict mapping entity IDs to lists of requested attribute names

        Returns:
            Dict mapping entity IDs to dicts of attribute values
        """
        data = {}
        for eid, attrs in outputs.items():
            if eid not in self.entities:
                continue
            
            entity = self.entities[eid]
            data[eid] = {}
            
            attr_map = {
                "input_force_x": entity["force"][0],
                "input_force_y": entity["force"][1],
                "input_force_z": entity["force"][2],
                "input_norm_force": entity["norm_force"],
                "compensated_force_x": entity["compensated_force"][0],
                "compensated_force_y": entity["compensated_force"][1],
                "compensated_force_z": entity["compensated_force"][2],
                "compensated_norm_force": entity["compensated_norm_force"],
                "gain": entity["compensator"].gain,
            }
            
            for attr in attrs:
                data[eid][attr] = attr_map.get(attr, None)

        return data


class InverseCompensator:
    """
    Inverse Compensator Entity for 3D Force Vectors

    Applies inverse transfer compensation to cancel delays for orbital HILS.

    Compensation approach:
        1. Calculate compensation ratio based on force magnitude (norm)
        2. Apply ratio to each component of the force vector

    Compensation formula for norm:
        norm_comp[k] = norm[k] + gain * (norm[k] - norm[k-1])
        ratio = norm_comp[k] / norm[k]  (if norm[k] != 0)

    Then apply ratio to each component:
        force_comp[k] = force[k] * ratio

    where:
        - gain: compensation gain that controls prediction strength
        - norm[k]: current force magnitude
        - norm[k-1]: previous force magnitude
    """

    def __init__(
        self,
        gain: float = 15.0,
        time_resolution: float = 0.001,
        step_size: int = 1,
    ):
        """
        Initialize the compensator

        Args:
            gain: Compensation gain (controls prediction strength)
            time_resolution: Time resolution [s]
            step_size: Step size in simulation steps
        """
        self.gain = gain
        self.time_resolution = time_resolution
        self.step_size = step_size

        # State - store previous force vector and norm
        self.prev_force: np.ndarray = np.zeros(3)
        self.prev_norm: float = 0.0
        self.input_count: int = 0

    def _process_input(self, force: np.ndarray) -> np.ndarray:
        """
        Process 3D force vector input and apply inverse compensation

        Args:
            force: Input force vector [Fx, Fy, Fz] in Newtons

        Returns:
            Compensated force vector [Fx_comp, Fy_comp, Fz_comp]
        """
        self.input_count += 1

        # Calculate current force norm
        current_norm = np.linalg.norm(force)

        # Apply compensation to the norm (scalar)
        compensated_norm = self._apply_compensation_to_norm(current_norm)

        # Apply ratio to each component
        compensated_force = compensated_norm * force / current_norm if current_norm > 0 else np.zeros(3)

        # Update state for next iteration
        self.prev_force = force.copy()
        self.prev_norm = current_norm

        # Debug logging (every 5000 steps to reduce output)
        if self.input_count % 5000 == 0:
            print(
                f"[InverseComp] Step {self.input_count}: "
                f"norm={current_norm:.3f}N → comp_norm={compensated_norm:.3f}N "
            )

        return compensated_force

    def _apply_compensation_to_norm(self, current_norm: float) -> float:
        """
        Apply inverse compensation formula to force magnitude (norm)

        Formula: norm_comp[k] = gain * norm[k] - (gain-1) * norm[k-1]

        This can be rewritten as: norm_comp[k] = norm[k] + (gain-1) * (norm[k] - norm[k-1])

        This is a lead compensator that predicts future force magnitude based on
        the current trend.

        Args:
            current_norm: Current force magnitude

        Returns:
            Compensated force magnitude
        """
        # Apply compensation formula
        # For first input, prev_norm is 0.0 (initialized in __init__)
        compensated_norm = self.gain * current_norm - (self.gain - 1.0) * self.prev_norm

        # Ensure non-negative (force magnitude cannot be negative)
        compensated_norm = max(0.0, compensated_norm)

        # Debug: show first few compensations
        if self.input_count <= 10:
            delta = current_norm - self.prev_norm
            print(
                f"[InverseComp] Step {self.input_count}: "
                f"curr_norm={current_norm:.3f}, prev_norm={self.prev_norm:.3f}, "
                f"delta={delta:.3f}, gain={self.gain:.1f} → comp_norm={compensated_norm:.3f}"
            )

        return compensated_norm


# Mosaik entry point
if __name__ == "__main__":
    mosaik_api.start_simulation(InverseCompensatorSimulator())
