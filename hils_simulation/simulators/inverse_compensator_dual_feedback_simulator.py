"""
Dual Feedback Inverse Compensator Simulator for HILS

This simulator receives inputs from two sources:
1. Plant actual thrust (direct input)
2. Bridge-0 delayed command (delayed feedback)

The compensator can use both inputs to perform more sophisticated compensation
algorithms that take into account both the current plant state and the delayed
command information.

Data Flow:
    Plant → [input] → InverseComp
    Bridge-0 → [delayed_feedback] → InverseComp
    InverseComp → [compensated_output] → Bridge-1 → Env

デュアルフィードバック逆補償シミュレータ - HILS用
PlantとBridge-0の両方からデータを受け取り、より高度な補償を実現
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import mosaik_api

# Import time constant models
sys.path.insert(0, str(Path(__file__).parent.parent))
from models import create_time_constant_model

meta = {
    "type": "time-based",
    "models": {
        "DualFeedbackInverseCompensator": {
            "public": True,
            "params": [
                "comp_id",
                "gain",  # Direct compensation gain
                "comp_type",
                "tau_to_gain_ratio",  # Ratio for tau->gain conversion
                "base_tau",  # Base time constant
                "tau_model_type",  # "constant" or "linear", "hybrid", etc.
                "tau_model_params",  # Time constant model parameters
                # NEW: Dual feedback specific parameters
                "feedback_weight",  # Weight for delayed feedback (0.0-1.0)
                "enable_dual_compensation",  # Enable/disable dual feedback compensation
            ],
            "attrs": [
                "input",  # From Plant (actual_thrust)
                "ideal_input",  # From Bridge-0 (delayed command - ideal reference)
                "compensated_output",
                "stats",
                # Debug attributes
                "raw_input",
                "prev_input",
                "delta",
                "compensation_amount",
                "input_thrust",
                "output_thrust",
                "current_gain",
                "current_tau",
                "ideal_input_value",
                "ideal_command_thrust",  # Extracted thrust from ideal input
                "ideal_input_contribution",  # Contribution from ideal input
            ],
        }
    },
}


class DualFeedbackInverseCompensatorSimulator(mosaik_api.Simulator):
    """
    Mosaik Dual Feedback Inverse Compensator Simulator

    Applies inverse compensation using both plant output and delayed command feedback.
    """

    def __init__(self):
        """Initialize the simulator"""
        super().__init__(meta)
        self.sid: Optional[str] = None
        self.step_size: int = 1
        self.time_resolution: float = 0.001
        self.compensators: Dict[str, DualFeedbackInverseCompensator] = {}

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
        Create DualFeedbackInverseCompensator entities

        Args:
            num: Number of entities to create
            model: Model name (must be "DualFeedbackInverseCompensator")
            **model_params: Model parameters

        Returns:
            List of entity info dictionaries
        """
        entities = []
        for i in range(num):
            comp_id = model_params.get("comp_id", f"dual_comp_{len(self.compensators)}")
            gain = model_params.get("gain", 15.0)
            comp_type = model_params.get("comp_type", "command")
            tau_to_gain_ratio = model_params.get("tau_to_gain_ratio", 0.1)
            base_tau = model_params.get("base_tau", 100.0)
            tau_model_type = model_params.get("tau_model_type", "constant")
            tau_model_params = model_params.get("tau_model_params", {})
            feedback_weight = model_params.get("feedback_weight", 0.5)
            enable_dual_compensation = model_params.get("enable_dual_compensation", True)

            comp = DualFeedbackInverseCompensator(
                comp_id=comp_id,
                gain=gain,
                comp_type=comp_type,
                tau_to_gain_ratio=tau_to_gain_ratio,
                base_tau=base_tau,
                tau_model_type=tau_model_type,
                tau_model_params=tau_model_params,
                feedback_weight=feedback_weight,
                enable_dual_compensation=enable_dual_compensation,
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

            # Process input signal (from Plant)
            for attr, sources in comp_inputs.items():
                if attr == "input":
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

                elif attr == "ideal_input":
                    # Process ideal input from Bridge-0
                    for src_id, value_dict in sources.items():
                        if isinstance(value_dict, dict) and "ideal_input" in value_dict:
                            value = value_dict["ideal_input"]
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

                        comp.process_ideal_input(value)

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
                elif attr == "ideal_input_value":
                    data[comp_id][attr] = comp.ideal_input_value
                elif attr == "ideal_command_thrust":
                    data[comp_id][attr] = comp.ideal_command_thrust
                elif attr == "ideal_input_contribution":
                    data[comp_id][attr] = comp.ideal_input_contribution

        return data


class DualFeedbackInverseCompensator:
    """
    Dual Feedback Inverse Compensator Entity

    Receives inputs from:
    1. Plant (actual thrust) - primary input for compensation
    2. Bridge-0 (ideal command) - ideal reference for enhanced compensation

    Compensation strategy:
    - Uses plant output as primary signal
    - Optionally incorporates ideal command information from Bridge-0
    - Can implement custom compensation algorithms using both signals
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
        feedback_weight: float = 0.5,
        enable_dual_compensation: bool = True,
        time_resolution: float = 0.001,
        step_size: int = 1,
    ):
        """
        Initialize the dual feedback compensator

        Args:
            comp_id: Compensator ID
            gain: Direct compensation gain
            comp_type: "command" or "sensor"
            tau_to_gain_ratio: Ratio to convert tau to gain
            base_tau: Base time constant [ms]
            tau_model_type: "constant" or model type
            tau_model_params: Time constant model parameters
            feedback_weight: Weight for delayed feedback contribution (0.0-1.0)
            enable_dual_compensation: Enable dual feedback compensation
            time_resolution: Time resolution [s]
            step_size: Step size in simulation steps
        """
        self.comp_id = comp_id
        self.base_gain = gain
        self.comp_type = comp_type
        self.tau_to_gain_ratio = tau_to_gain_ratio
        self.time_resolution = time_resolution
        self.step_size = step_size
        self.feedback_weight = feedback_weight
        self.enable_dual_compensation = enable_dual_compensation

        # Time constant model setup
        self.base_tau = base_tau
        self.tau_model_type = tau_model_type
        if tau_model_params is None:
            tau_model_params = {}

        # Determine if we use adaptive gain
        self.use_adaptive_gain = tau_model_type != "constant"

        if self.use_adaptive_gain:
            self.tau_model = create_time_constant_model(tau_model_type, **tau_model_params)
            self.gain = base_tau * tau_to_gain_ratio
        else:
            self.tau_model = None
            self.gain = gain

        # State
        self.prev_value: float = 0.0
        self.current_output: Any = 0.0
        self.input_count: int = 0
        self.current_gain: float = self.gain
        self.current_tau: float = base_tau

        # Debug information
        self.raw_input_value: Any = 0.0
        self.prev_input_value: float = 0.0
        self.delta_value: float = 0.0
        self.compensation_amount_value: float = 0.0
        self.input_thrust_value: float = 0.0
        self.output_thrust_value: float = 0.0
        self.ideal_input_value: Any = 0.0
        self.ideal_command_thrust: float = 0.0
        self.ideal_input_contribution: float = 0.0

    def process_input(self, value: Any) -> None:
        """
        Process primary input from Plant and apply compensation

        Args:
            value: Input signal (numeric thrust value from Plant)
        """
        self.input_count += 1
        self.raw_input_value = value

        # Extract numeric value (Plant sends actual_thrust as float)
        if isinstance(value, (int, float)):
            numeric_value = value
            self.input_thrust_value = numeric_value

            # Adaptive mode: calculate gain from thrust
            if self.use_adaptive_gain:
                dt = self.step_size * self.time_resolution * 1000  # [ms]
                self.current_tau = self.tau_model.get_time_constant(
                    thrust=numeric_value,
                    base_tau=self.base_tau,
                    dt=dt,
                )
                self.gain = max(1.0, self.current_tau * self.tau_to_gain_ratio)
                self.current_gain = self.gain

            # Apply compensation
            compensated_value = self._apply_compensation(numeric_value)
            self.output_thrust_value = compensated_value
            self.current_output = compensated_value

            # Debug logging (every 1000 steps)
            if self.input_count % 1000 == 0:
                mode_str = (
                    f"tau={self.current_tau:.1f}ms, gain={self.gain:.1f}"
                    if self.use_adaptive_gain
                    else f"gain={self.gain:.1f} (constant)"
                )
                fb_str = f", ideal_contrib={self.ideal_input_contribution:.3f}N" if self.enable_dual_compensation else ""
                print(
                    f"[DualFbInverseComp-{self.comp_id}] Step {self.input_count}: "
                    f"input={numeric_value:.3f}N → output={compensated_value:.3f}N ({mode_str}{fb_str})"
                )
        else:
            # Pass through if unable to process
            self.current_output = value

    def process_ideal_input(self, value: Any) -> None:
        """
        Process ideal input from Bridge-0

        This receives the ideal command (delayed through Bridge-0) and can be used
        to enhance the compensation algorithm by providing a reference signal.

        Args:
            value: Ideal input signal (command dict from Bridge-0)
        """
        self.ideal_input_value = value

        # Extract thrust from ideal command
        if isinstance(value, dict) and "thrust" in value:
            self.ideal_command_thrust = value["thrust"]

            # Calculate ideal input contribution if dual compensation is enabled
            if self.enable_dual_compensation:
                # Example: Use the difference between ideal command and current output
                # This is a placeholder - you can customize this algorithm
                self.ideal_input_contribution = self.feedback_weight * (
                    self.ideal_command_thrust - self.input_thrust_value
                )
            else:
                self.ideal_input_contribution = 0.0

            # Debug logging (every 1000 steps)
            if self.input_count % 1000 == 0 and self.enable_dual_compensation:
                print(
                    f"[DualFbInverseComp-{self.comp_id}] Ideal cmd: {self.ideal_command_thrust:.3f}N, "
                    f"current input: {self.input_thrust_value:.3f}N, "
                    f"contribution: {self.ideal_input_contribution:.3f}N"
                )
        elif isinstance(value, (int, float)):
            self.ideal_command_thrust = value
            if self.enable_dual_compensation:
                self.ideal_input_contribution = self.feedback_weight * (
                    self.ideal_command_thrust - self.input_thrust_value
                )
            else:
                self.ideal_input_contribution = 0.0

    def _apply_compensation(self, value: float) -> float:
        """
        Apply dual feedback compensation formula

        Base formula: y_comp[k] = gain * y[k] - (gain-1) * y[k-1]

        With ideal input:
        y_comp[k] = gain * y[k] - (gain-1) * y[k-1] + ideal_input_contribution

        Args:
            value: Current input value

        Returns:
            Compensated value
        """
        # Store previous input for debugging
        self.prev_input_value = self.prev_value

        # Calculate delta and base compensation amount
        self.delta_value = value - self.prev_value
        self.compensation_amount_value = (self.gain - 1.0) * self.delta_value

        # Apply base compensation formula
        compensated = self.gain * value - (self.gain - 1.0) * self.prev_value

        # Add ideal input contribution if enabled
        if self.enable_dual_compensation:
            compensated += self.ideal_input_contribution

        # Debug: show first few compensations
        if self.input_count <= 10:
            fb_str = f", ideal_contrib={self.ideal_input_contribution:.3f}" if self.enable_dual_compensation else ""
            print(
                f"[DualFbInverseComp] Step {self.input_count}: "
                f"curr={value:.3f}, prev={self.prev_value:.3f}, "
                f"delta={self.delta_value:.3f}, comp_amt={self.compensation_amount_value:.3f}, "
                f"gain={self.gain:.1f}{fb_str} → comp={compensated:.3f}"
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
            "feedback_weight": self.feedback_weight,
            "enable_dual_compensation": self.enable_dual_compensation,
            "ideal_command_thrust": self.ideal_command_thrust,
            "ideal_input_contribution": self.ideal_input_contribution,
        }
        return json.dumps(stats)


# Mosaik entry point
if __name__ == "__main__":
    mosaik_api.start_simulation(DualFeedbackInverseCompensatorSimulator())
