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

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import mosaik_api

# Import time constant models (same as plant_simulator_with_model.py)
sys.path.insert(0, str(Path(__file__).parent.parent))
from models import create_time_constant_model

meta = {
    "type": "time-based",
    "models": {
        "InverseCompensator": {
            "public": True,
            "params": [
                "comp_id",
                "gain",  # Direct gain (used when tau_model_type="constant")
                "comp_type",
                "tau_to_gain_ratio",  # Ratio for tau->gain conversion (for non-constant models)
                "base_tau",  # Base time constant for tau_model
                "tau_model_type",  # "constant" (direct gain) or "linear", "hybrid", etc.
                "tau_model_params",  # Time constant model parameters
            ],
            "attrs": [
                "input",
                "compensated_output",
                "stats",
                # Debug attributes
                "raw_input",
                "prev_input",
                "delta",
                "compensation_amount",
                "input_thrust",
                "output_thrust",
                "current_gain",  # Current gain being used
                "current_tau",  # Current tau (for non-constant models)
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
        self.sid: Optional[str] = None
        self.step_size: int = 1
        self.time_resolution: float = 0.001
        self.compensators: Dict[str, InverseCompensator] = {}

    def init(self, sid: str, time_resolution: float = 1.0, **sim_params):
        """
        Initialize the simulator

        Args:
            sid: Simulator ID
            time_resolution: Time resolution in seconds per step
            **sim_params: Additional simulator parameters
        """
        self.sid = sid
        self.time_resolution = time_resolution
        self.step_size = sim_params.get("step_size", 1)
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
            comp_id = model_params.get("comp_id", f"comp_{len(self.compensators)}")
            gain = model_params.get("gain", 15.0)
            comp_type = model_params.get("comp_type", "command")
            tau_to_gain_ratio = model_params.get("tau_to_gain_ratio", 0.1)
            base_tau = model_params.get("base_tau", 100.0)
            tau_model_type = model_params.get("tau_model_type", "constant")
            tau_model_params = model_params.get("tau_model_params", {})

            comp = InverseCompensator(
                comp_id=comp_id,
                gain=gain,
                comp_type=comp_type,
                tau_to_gain_ratio=tau_to_gain_ratio,
                base_tau=base_tau,
                tau_model_type=tau_model_type,
                tau_model_params=tau_model_params,
                time_resolution=self.time_resolution,
                step_size=self.step_size,
            )
            self.compensators[comp_id] = comp

            entities.append({"eid": comp_id, "type": model})

        return entities

    def step(self, time: int, inputs: Dict[str, Dict[str, Any]], max_advance: int) -> int:
        """
        Execute one simulation step

        Args:
            time: Current simulation time (in steps)
            inputs: Input data from connected entities
            max_advance: Maximum time advance allowed

        Returns:
            Next step time
        """
        for comp_id, comp in self.compensators.items():
            comp_inputs = inputs.get(comp_id, {})

            # Process input signal
            for attr, sources in comp_inputs.items():
                if attr == "input":
                    # Get the latest value from all sources
                    for src_id, value_dict in sources.items():
                        if isinstance(value_dict, dict) and "input" in value_dict:
                            value = value_dict["input"]
                        else:
                            value = value_dict

                        # Handle JSON strings
                        if isinstance(value, (str, bytes)):
                            try:
                                if isinstance(value, bytes):
                                    value = value.decode("utf-8")
                                value = json.loads(value)
                            except (json.JSONDecodeError, ValueError):
                                pass

                        comp.process_input(value)

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
        for comp_id, attrs in outputs.items():
            comp = self.compensators.get(comp_id)
            if comp is None:
                continue

            data[comp_id] = {}
            for attr in attrs:
                if attr == "compensated_output":
                    data[comp_id][attr] = comp.get_output()
                elif attr == "stats":
                    data[comp_id][attr] = comp.get_stats()
                elif attr == "raw_input":
                    data[comp_id][attr] = comp.raw_input_value
                elif attr == "prev_input":
                    data[comp_id][attr] = comp.prev_input_value
                elif attr == "delta":
                    data[comp_id][attr] = comp.delta_value
                elif attr == "compensation_amount":
                    data[comp_id][attr] = comp.compensation_amount_value
                elif attr == "input_thrust":
                    data[comp_id][attr] = comp.input_thrust_value
                elif attr == "output_thrust":
                    data[comp_id][attr] = comp.output_thrust_value
                elif attr == "current_gain":
                    data[comp_id][attr] = comp.current_gain
                elif attr == "current_tau":
                    data[comp_id][attr] = comp.current_tau

        return data


class InverseCompensator:
    """
    Inverse Compensator Entity

    Applies inverse transfer compensation to cancel delays.

    Compensation formula:
        y_comp[k] = y[k] + gain * (y[k] - y[k-1])

    where:
        - gain: compensation gain that controls prediction strength
        - y[k]: current input signal
        - y[k-1]: previous input signal

    The gain parameter should be tuned based on the delay characteristics:
        - Higher gain = stronger prediction (more aggressive compensation)
        - Lower gain = weaker prediction (more conservative)
        - Typical range: 1.0 to 50.0 (depending on signal characteristics)
    """

    def __init__(
        self,
        comp_id: str,
        gain: float = 15.0,
        comp_type: str = "command",
        tau_to_gain_ratio: float = 0.1,
        base_tau: float = 100.0,
        tau_model_type: str = "constant",
        tau_model_params: dict = None,
        time_resolution: float = 0.001,
        step_size: int = 1,
    ):
        """
        Initialize the compensator

        Args:
            comp_id: Compensator ID
            gain: Direct compensation gain (used when tau_model_type="constant")
            comp_type: "command" (pre-delay) or "sensor" (post-delay)
            tau_to_gain_ratio: Ratio to convert tau to gain (for non-constant models)
            base_tau: Base time constant for tau_model [ms]
            tau_model_type: "constant" (use direct gain) or "linear", "hybrid", etc.
            tau_model_params: Time constant model parameters (dict)
            time_resolution: Time resolution [s]
            step_size: Step size in simulation steps
        """
        self.comp_id = comp_id
        self.base_gain = gain  # Store original direct gain
        self.comp_type = comp_type
        self.tau_to_gain_ratio = tau_to_gain_ratio
        self.time_resolution = time_resolution
        self.step_size = step_size

        # Time constant model setup
        self.base_tau = base_tau
        self.tau_model_type = tau_model_type
        if tau_model_params is None:
            tau_model_params = {}

        # Determine if we use adaptive (tau-based) gain or constant (direct) gain
        self.use_adaptive_gain = tau_model_type != "constant"

        if self.use_adaptive_gain:
            # Adaptive mode: create tau_model for calculating gain from thrust
            self.tau_model = create_time_constant_model(tau_model_type, **tau_model_params)
            self.gain = base_tau * tau_to_gain_ratio  # Initial gain from base_tau
        else:
            # Constant mode: use direct gain parameter
            self.tau_model = None
            self.gain = gain

        # State
        self.prev_value: float = 0.0
        self.current_output: Any = 0.0
        self.input_count: int = 0
        self.current_gain: float = self.gain  # Track current gain for debugging
        self.current_tau: float = base_tau  # Track current tau for debugging

        # Debug information
        self.raw_input_value: Any = 0.0
        self.prev_input_value: float = 0.0
        self.delta_value: float = 0.0
        self.compensation_amount_value: float = 0.0
        self.input_thrust_value: float = 0.0
        self.output_thrust_value: float = 0.0

    def process_input(self, value: Any) -> None:
        """
        Process input and apply inverse compensation

        Args:
            value: Input signal (numeric or dict/JSON)
        """
        self.input_count += 1
        self.raw_input_value = value  # Store raw input for debugging

        # Extract numeric value if input is dict/JSON
        if isinstance(value, dict):
            # For command signals: extract "thrust" field
            if "thrust" in value:
                numeric_value = value["thrust"]
                self.input_thrust_value = numeric_value

                # Adaptive mode: calculate gain from thrust-based time constant
                if self.use_adaptive_gain:
                    dt = self.step_size * self.time_resolution * 1000  # [ms]
                    self.current_tau = self.tau_model.get_time_constant(
                        thrust=numeric_value,
                        base_tau=self.base_tau,
                        dt=dt,
                    )
                    # Update gain based on calculated tau
                    self.gain = max(1.0, self.current_tau * self.tau_to_gain_ratio)
                    self.current_gain = self.gain

                # Apply compensation
                compensated_thrust = self._apply_compensation(numeric_value)
                self.output_thrust_value = compensated_thrust

                # Debug logging (every 1000 steps)
                if self.input_count % 1000 == 0:
                    mode_str = f"tau={self.current_tau:.1f}ms" if self.use_adaptive_gain else "constant"
                    print(
                        f"[InverseComp-{self.comp_id}] Step {self.input_count}: input={numeric_value:.3f}N → output={compensated_thrust:.3f}N (gain={self.gain:.1f}, {mode_str})"
                    )

                # Reconstruct command dict with compensated value
                self.current_output = {
                    "thrust": compensated_thrust,
                    "duration": value.get("duration", 10),
                }
            else:
                # Generic dict - try to compensate first numeric field
                for key, val in value.items():
                    if isinstance(val, (int, float)):
                        compensated_val = self._apply_compensation(val)
                        self.current_output = {**value, key: compensated_val}
                        break
        # Numeric input - direct compensation (for actual_thrust from plant)
        elif isinstance(value, (int, float)):
            numeric_value = value
            self.input_thrust_value = numeric_value

            # Adaptive mode: calculate gain from thrust-based time constant
            if self.use_adaptive_gain:
                dt = self.step_size * self.time_resolution * 1000  # [ms]
                self.current_tau = self.tau_model.get_time_constant(
                    thrust=numeric_value,
                    base_tau=self.base_tau,
                    dt=dt,
                )
                # Update gain based on calculated tau
                self.gain = max(1.0, self.current_tau * self.tau_to_gain_ratio)
                self.current_gain = self.gain

            # Apply compensation
            compensated_value = self._apply_compensation(value)
            self.output_thrust_value = compensated_value
            self.current_output = compensated_value

            # Debug logging (every 1000 steps)
            if self.input_count % 1000 == 0:
                mode_str = (
                    f"tau={self.current_tau:.1f}ms, gain={self.gain:.1f}"
                    if self.use_adaptive_gain
                    else f"gain={self.gain:.1f} (constant)"
                )
                print(
                    f"[InverseComp-{self.comp_id}] Step {self.input_count}: "
                    f"input={numeric_value:.3f}N → output={compensated_value:.3f}N ({mode_str})"
                )
        else:
            # Pass through if unable to process
            self.current_output = value

    def _apply_compensation(self, value: float) -> float:
        """
        Apply inverse compensation formula

        Formula: y_comp[k] = a * y[k] - (a-1) * y[k-1]

        This can be rewritten as: y_comp[k] = y[k] + (a-1) * (y[k] - y[k-1])

        This is a lead compensator that pre-shapes the signal to counteract
        upcoming delays. The formula creates a "lead" effect by emphasizing
        changes in the signal.

        Args:
            value: Current input value

        Returns:
            Compensated value
        """
        # Store previous input for debugging (use 0 if first input)
        self.prev_input_value = self.prev_value

        # Calculate delta and compensation amount
        # For first input, prev_value is 0.0 (initialized in __init__)
        self.delta_value = value - self.prev_value
        self.compensation_amount_value = (self.gain - 1.0) * self.delta_value

        # Apply compensation formula (same as demo code)
        # This emphasizes current value and subtracts previous value
        compensated = self.gain * value - (self.gain - 1.0) * self.prev_value

        # Alternative formula (equivalent): y[k] + (gain-1) * (y[k] - y[k-1])
        # compensated_alt = value + self.compensation_amount_value

        # Debug: show first few compensations
        if self.input_count <= 10:
            print(
                f"[InverseComp] Step {self.input_count}: "
                f"curr={value:.3f}, prev={self.prev_value:.3f}, "
                f"delta={self.delta_value:.3f}, comp_amt={self.compensation_amount_value:.3f}, "
                f"gain={self.gain:.1f} → comp={compensated:.3f}"
            )

        # Update state
        self.prev_value = value

        return compensated

    def get_output(self) -> Any:
        """
        Get compensated output signal

        Returns:
            Compensated output value
        """
        return self.current_output

    def get_stats(self) -> str:
        """
        Get compensator statistics as JSON string

        Returns:
            JSON string with statistics
        """
        stats = {
            "comp_id": self.comp_id,
            "gain": self.gain,
            "base_gain": self.base_gain,
            "comp_type": self.comp_type,
            "use_adaptive_gain": self.use_adaptive_gain,
            "tau_model_type": self.tau_model_type,
            "base_tau": self.base_tau,
            "current_tau": self.current_tau,
            "tau_to_gain_ratio": self.tau_to_gain_ratio,
            "input_count": self.input_count,
            "prev_value": self.prev_value,
        }
        return json.dumps(stats)


# Mosaik entry point
if __name__ == "__main__":
    mosaik_api.start_simulation(InverseCompensatorSimulator())
