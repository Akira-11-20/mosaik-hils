"""
Dual Feedback Inverse Compensator Simulator for HILS (Alpha-based Compensation)

This simulator receives inputs from two sources:
1. Plant actual thrust (direct input) - used for compensation calculation
2. Bridge-0 delayed command (ideal reference) - used to calculate time constant τ

Compensation Strategy:
- Uses ideal command thrust to calculate time constant τ via linear model
- Computes alpha: α = 1 - exp(-dt/τ)
- Applies inverse compensation: y_comp[k] = (y[k] - (1-α) * y[k-1]) / α
- This compensates for the first-order hold dynamics of the plant

Data Flow:
    Plant → [input] → InverseComp → [compensated_output] → Bridge-1 → Env
    Bridge-0 → [ideal_input] → InverseComp (for τ calculation)

デュアルフィードバック逆補償シミュレータ (Alpha方式) - HILS用
理想推力から時定数τを計算し、αベースの補償を実現
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
                "current_alpha",  # α = 1 - exp(-dt/τ)
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

        IMPORTANT: Process ideal_input FIRST to update alpha, then process input.
        This ensures compensation uses the correct alpha value.

        Args:
            time: Current simulation time (in steps)
            inputs: Input data from connected entities
            max_advance: Maximum time advance allowed

        Returns:
            Next step time
        """
        for comp_id, comp in self.compensators.items():
            comp_inputs = inputs.get(comp_id, {})

            # STEP 1: Process ideal_input FIRST to calculate tau and alpha
            if "ideal_input" in comp_inputs:
                for src_id, value_dict in comp_inputs["ideal_input"].items():
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

            # STEP 2: Process input AFTER alpha is updated
            if "input" in comp_inputs:
                for src_id, value_dict in comp_inputs["input"].items():
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
                elif attr == "current_alpha":
                    data[comp_id][attr] = comp.current_alpha
                elif attr == "ideal_input_value":
                    data[comp_id][attr] = comp.ideal_input_value
                elif attr == "ideal_command_thrust":
                    data[comp_id][attr] = comp.ideal_command_thrust
                elif attr == "ideal_input_contribution":
                    data[comp_id][attr] = comp.ideal_input_contribution

        return data


class DualFeedbackInverseCompensator:
    """
    Dual Feedback Inverse Compensator Entity (Alpha-based)

    Receives inputs from:
    1. Plant (actual thrust) - primary input for compensation
    2. Bridge-0 (ideal command) - used to calculate time constant τ

    Compensation strategy (adaptive alpha mode):
    - Extract ideal thrust from Bridge-0 delayed command
    - Calculate time constant τ using linear model: τ(F) = τ_base + k * |dF/dt|
    - Compute alpha: α = 1 - exp(-dt/τ)
    - Apply inverse first-order hold compensation:
        y_comp[k] = (y[k] - (1-α) * y[k-1]) / α

    Compensation strategy (constant mode):
    - Traditional gain-based compensation: y_comp[k] = gain * y[k] - (gain-1) * y[k-1]
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
            gain: Direct compensation gain (used only when tau_model_type="constant")
            comp_type: "command" or "sensor"
            tau_to_gain_ratio: Ratio to convert tau to gain (deprecated for linear model)
            base_tau: Base time constant [ms]
            tau_model_type: "constant", "linear", etc.
            tau_model_params: Time constant model parameters
            feedback_weight: Weight for delayed feedback contribution (0.0-1.0) (deprecated)
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

        # Determine if we use adaptive alpha from tau model
        self.use_adaptive_alpha = tau_model_type != "constant"

        if self.use_adaptive_alpha:
            # Create tau model (e.g., LinearThrustDependentModel)
            self.tau_model = create_time_constant_model(tau_model_type, **tau_model_params)
        else:
            self.tau_model = None

        # For constant mode, use fixed gain
        self.gain = gain

        # State
        self.prev_value: float = 0.0
        self.current_output: Any = 0.0
        self.input_count: int = 0
        self.current_gain: float = self.gain
        self.current_tau: float = base_tau
        # Initialize alpha from base_tau to ensure valid compensation from step 0
        if self.use_adaptive_alpha:
            self.current_alpha: float = base_tau * tau_to_gain_ratio
        else:
            self.current_alpha: float = gain  # Use gain for constant mode

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

            # Apply compensation using alpha (if adaptive) or fixed gain (if constant)
            compensated_value = self._apply_compensation(numeric_value)
            self.output_thrust_value = compensated_value
            self.current_output = compensated_value

            # Debug logging (every 1000 steps)
            if self.input_count % 1000 == 0:
                if self.use_adaptive_alpha:
                    mode_str = f"τ={self.current_tau:.2f}ms, α={self.current_alpha:.6f}"
                else:
                    mode_str = f"gain={self.gain:.2f} (constant)"
                print(
                    f"[DualFbInverseComp-{self.comp_id}] Step {self.input_count}: "
                    f"input={numeric_value:.3f}N → output={compensated_value:.3f}N ({mode_str})"
                )
        else:
            # Pass through if unable to process
            self.current_output = value

    def process_ideal_input(self, value: Any) -> None:
        """
        Process ideal input from Bridge-0

        This receives the ideal command (delayed through Bridge-0) and uses it
        to calculate the time constant (tau) from the linear model, then computes
        alpha = 1 - exp(-dt/tau) for compensation.

        Args:
            value: Ideal input signal (command dict from Bridge-0)
        """
        self.ideal_input_value = value

        # Extract thrust from ideal command
        if isinstance(value, dict) and "thrust" in value:
            self.ideal_command_thrust = value["thrust"]
        elif isinstance(value, (int, float)):
            self.ideal_command_thrust = value
        else:
            # Cannot extract thrust, skip tau/alpha calculation
            return

        # Calculate tau from ideal thrust using the model
        if self.use_adaptive_alpha and self.tau_model is not None:
            dt_ms = self.step_size * self.time_resolution * 1000  # [ms]

            # Use ideal_command_thrust to calculate tau from linear model
            self.current_tau = self.tau_model.get_time_constant(
                thrust=self.ideal_command_thrust,
                base_tau=self.base_tau,
                dt=dt_ms,
            )

            # Calculate alpha from tau using tau_to_gain_ratio
            if self.current_tau > 0:
                self.current_alpha = self.current_tau * self.tau_to_gain_ratio
                self.current_gain = self.current_alpha
            else:
                self.current_alpha = 1.0  # Prevent division by zero
                self.current_gain = self.current_alpha

            # Debug logging (every 1000 steps)
            if self.input_count % 1000 == 0:
                print(
                    f"[DualFbInverseComp-{self.comp_id}] Ideal thrust: {self.ideal_command_thrust:.3f}N → "
                    f"τ={self.current_tau:.2f}ms, α={self.current_alpha:.6f}"
                )

    def _apply_compensation(self, value: float) -> float:
        """
        Apply inverse compensation using alpha calculated from ideal thrust

        For adaptive mode (use_adaptive_alpha=True):
            Alpha-based formula: y_comp[k] = (y[k] - (1-α) * y[k-1]) / α
            where α = 1 - exp(-dt/τ)
            τ is calculated from ideal thrust using linear model

        For constant mode (use_adaptive_alpha=False):
            Traditional gain-based formula: y_comp[k] = gain * y[k] - (gain-1) * y[k-1]

        Args:
            value: Current input value (actual thrust from plant)

        Returns:
            Compensated value
        """
        # Store previous input for debugging
        self.prev_input_value = self.prev_value

        if self.use_adaptive_alpha:
            # Alpha-based compensation (inverse of first-order hold)
            # y[k] = α * y_comp[k] + (1-α) * y[k-1]

            if self.current_alpha > 0:
                compensated = self.current_alpha*value - (self.current_alpha-1.0) * self.prev_value
                self.compensation_amount_value = compensated - value
            else:
                # If alpha is zero, pass through
                compensated = value
                self.compensation_amount_value = 0.0

            # Calculate delta for debugging
            self.delta_value = value - self.prev_value

            # Debug: show first few compensations
            if self.input_count <= 10:
                print(
                    f"[DualFbInverseComp] Step {self.input_count}: "
                    f"curr={value:.3f}, prev={self.prev_value:.3f}, "
                    f"delta={self.delta_value:.3f}, α={self.current_alpha:.6f}, "
                    f"τ={self.current_tau:.2f}ms → comp={compensated:.3f}"
                )
        else:
            # Traditional gain-based compensation (constant mode)
            self.delta_value = value - self.prev_value
            self.compensation_amount_value = (self.gain - 1.0) * self.delta_value
            compensated = self.gain * value - (self.gain - 1.0) * self.prev_value

            # Debug: show first few compensations
            if self.input_count <= 10:
                print(
                    f"[DualFbInverseComp] Step {self.input_count}: "
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
            "use_adaptive_alpha": self.use_adaptive_alpha,
            "tau_model_type": self.tau_model_type,
            "base_tau": self.base_tau,
            "current_tau": self.current_tau,
            "current_alpha": self.current_alpha,
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
