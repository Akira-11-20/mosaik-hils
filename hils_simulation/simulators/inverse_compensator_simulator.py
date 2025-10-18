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
from typing import Any, Dict, Optional

import mosaik_api


meta = {
    "type": "time-based",
    "models": {
        "InverseCompensator": {
            "public": True,
            "params": ["comp_id", "gain", "comp_type"],
            "attrs": ["input", "compensated_output", "stats"],
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
        self.compensators: Dict[str, "InverseCompensator"] = {}

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
                - gain: Compensation gain (float, default: delay_samples)
                - comp_type: "command" or "sensor" (str, default: "command")

        Returns:
            List of entity info dictionaries
        """
        entities = []
        for i in range(num):
            comp_id = model_params.get("comp_id", f"comp_{len(self.compensators)}")
            gain = model_params.get("gain", 15.0)
            comp_type = model_params.get("comp_type", "command")

            comp = InverseCompensator(
                comp_id=comp_id,
                gain=gain,
                comp_type=comp_type,
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
    ):
        """
        Initialize the compensator

        Args:
            comp_id: Compensator ID
            gain: Compensation gain (typically equals delay samples)
            comp_type: "command" (pre-delay) or "sensor" (post-delay)
        """
        self.comp_id = comp_id
        self.gain = gain
        self.comp_type = comp_type

        # State
        self.prev_value: float = 0.0
        self.current_output: Any = 0.0
        self.input_count: int = 0

    def process_input(self, value: Any) -> None:
        """
        Process input and apply inverse compensation

        Args:
            value: Input signal (numeric or dict/JSON)
        """
        self.input_count += 1

        # Extract numeric value if input is dict/JSON
        if isinstance(value, dict):
            # For command signals: extract "thrust" field
            if "thrust" in value:
                numeric_value = value["thrust"]
                # Apply compensation
                compensated_thrust = self._apply_compensation(numeric_value)

                # Debug logging (every 1000 steps)
                if self.input_count % 1000 == 0:
                    print(
                        f"[InverseComp-{self.comp_id}] Step {self.input_count}: "
                        f"input={numeric_value:.3f}N → output={compensated_thrust:.3f}N "
                        f"(gain={self.gain:.1f})"
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
        else:
            # Numeric input - direct compensation
            if isinstance(value, (int, float)):
                self.current_output = self._apply_compensation(value)
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
        # First input - no previous value to use
        if self.input_count == 1:
            self.prev_value = value
            if self.input_count <= 10:
                print(f"[InverseComp] First input: {value:.3f} (no compensation)")
            return value

        # Apply compensation formula (same as demo code)
        # This emphasizes current value and subtracts previous value
        compensated = self.gain * value - (self.gain - 1.0) * self.prev_value

        # Debug: show first few compensations
        if self.input_count <= 10:
            delta = value - self.prev_value
            print(
                f"[InverseComp] Step {self.input_count}: "
                f"curr={value:.3f}, prev={self.prev_value:.3f}, "
                f"delta={delta:.3f}, gain={self.gain:.1f} → "
                f"comp={compensated:.3f}"
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
            "comp_type": self.comp_type,
            "input_count": self.input_count,
            "prev_value": self.prev_value,
        }
        return json.dumps(stats)


# Mosaik entry point
if __name__ == "__main__":
    mosaik_api.start_simulation(InverseCompensatorSimulator())
