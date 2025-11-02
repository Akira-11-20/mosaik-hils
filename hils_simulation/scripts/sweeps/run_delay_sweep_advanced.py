"""
Advanced delay sweep script with fine-grained control
- Separate command/sense delay configurations
- Custom inverse compensation gain per delay
- Plant time constant (actuator lag) configuration
- Multiple test modes for comprehensive analysis
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add parent directory to path to enable imports from hils_simulation root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.parameters import SimulationParameters
from scenarios import HILSScenario, InverseCompScenario


class DelayConfig:
    """Configuration for a single delay test case"""

    def __init__(
        self,
        cmd_delay: float,
        sense_delay: float,
        use_inverse_comp: bool = False,
        comp_gain: Optional[float] = None,
        plant_time_constant: Optional[float] = None,
        plant_time_constant_std: Optional[float] = None,
        plant_time_constant_noise: Optional[float] = None,
        plant_enable_lag: Optional[bool] = None,
        label: Optional[str] = None
    ):
        """
        Args:
            cmd_delay: Command path delay in milliseconds
            sense_delay: Sensing path delay in milliseconds
            use_inverse_comp: Enable inverse compensation
            comp_gain: Custom compensation gain (if None, uses default from .env)
            plant_time_constant: Plant 1st-order lag time constant in milliseconds (if None, uses default from .env)
            plant_time_constant_std: Standard deviation for time constant variability in milliseconds (if None, uses default from .env)
            plant_time_constant_noise: Time-varying noise for time constant in milliseconds (if None, uses default from .env)
            plant_enable_lag: Enable plant lag (if None, uses default from .env)
            label: Custom label for this configuration
        """
        self.cmd_delay = cmd_delay
        self.sense_delay = sense_delay
        self.use_inverse_comp = use_inverse_comp
        self.comp_gain = comp_gain
        self.plant_time_constant = plant_time_constant
        self.plant_time_constant_std = plant_time_constant_std
        self.plant_time_constant_noise = plant_time_constant_noise
        self.plant_enable_lag = plant_enable_lag
        self.label = label or self._generate_label()

    def _generate_label(self) -> str:
        """Generate a descriptive label"""
        comp_str = "comp" if self.use_inverse_comp else "nocomp"

        # Base label with delays
        if self.cmd_delay == self.sense_delay:
            label = f"delay{self.cmd_delay:.0f}ms_{comp_str}"
        else:
            label = f"cmd{self.cmd_delay:.0f}ms_sense{self.sense_delay:.0f}ms_{comp_str}"

        # Add plant time constant if specified
        if self.plant_time_constant is not None:
            label += f"_tau{self.plant_time_constant:.0f}ms"

        # Add plant time constant std if specified
        if self.plant_time_constant_std is not None and self.plant_time_constant_std > 0:
            label += f"_std{self.plant_time_constant_std:.0f}ms"

        # Add plant time constant noise if specified
        if self.plant_time_constant_noise is not None and self.plant_time_constant_noise > 0:
            label += f"_noise{self.plant_time_constant_noise:.0f}ms"

        # Add plant lag disabled flag if specified
        if self.plant_enable_lag is not None and not self.plant_enable_lag:
            label += "_nolag"

        return label

    def __repr__(self) -> str:
        parts = [f"cmd={self.cmd_delay}ms", f"sense={self.sense_delay}ms"]

        if self.use_inverse_comp:
            gain_str = f"{self.comp_gain}" if self.comp_gain else "default"
            parts.append(f"comp_gain={gain_str}")
        else:
            parts.append("no_comp")

        if self.plant_time_constant is not None:
            parts.append(f"plant_tau={self.plant_time_constant}ms")

        if self.plant_time_constant_std is not None:
            parts.append(f"plant_tau_std={self.plant_time_constant_std}ms")

        if self.plant_time_constant_noise is not None:
            parts.append(f"plant_tau_noise={self.plant_time_constant_noise}ms")

        if self.plant_enable_lag is not None:
            parts.append(f"plant_lag={self.plant_enable_lag}")

        return f"DelayConfig({', '.join(parts)})"


def create_symmetric_sweep(
    delays: List[float],
    use_inverse_comp: bool = False
) -> List[DelayConfig]:
    """
    Create configurations with symmetric delays (cmd_delay = sense_delay)

    Args:
        delays: List of delay values in milliseconds
        use_inverse_comp: Enable inverse compensation

    Returns:
        List of DelayConfig objects
    """
    return [
        DelayConfig(d, d, use_inverse_comp=use_inverse_comp)
        for d in delays
    ]


def create_asymmetric_sweep(
    cmd_delays: List[float],
    sense_delays: List[float],
    use_inverse_comp: bool = False
) -> List[DelayConfig]:
    """
    Create configurations with asymmetric delays (different cmd/sense)

    Args:
        cmd_delays: List of command delay values in milliseconds
        sense_delays: List of sensing delay values in milliseconds
        use_inverse_comp: Enable inverse compensation

    Returns:
        List of DelayConfig objects
    """
    configs = []
    for cmd_delay in cmd_delays:
        for sense_delay in sense_delays:
            configs.append(
                DelayConfig(cmd_delay, sense_delay, use_inverse_comp=use_inverse_comp)
            )
    return configs


def create_comparison_sweep(
    delays: List[float]
) -> List[DelayConfig]:
    """
    Create configurations comparing HILS vs Inverse Compensation for each delay

    Args:
        delays: List of delay values in milliseconds

    Returns:
        List of DelayConfig objects (each delay tested with/without compensation)
    """
    configs = []
    for delay in delays:
        # Without compensation
        configs.append(DelayConfig(delay, delay, use_inverse_comp=False))
        # With compensation
        configs.append(DelayConfig(delay, delay, use_inverse_comp=True))
    return configs


def run_simulation(config: DelayConfig) -> Dict[str, Any]:
    """
    Run a single simulation with given delay configuration

    Args:
        config: DelayConfig object

    Returns:
        Dictionary with simulation results
    """
    print(f"\n{'='*60}")
    print(f"Running: {config}")
    print(f"{'='*60}")

    try:
        # Load parameters from .env
        params = SimulationParameters.from_env()

        # Override delay parameters
        params.communication.cmd_delay = config.cmd_delay
        params.communication.sense_delay = config.sense_delay

        # Override plant parameters
        if config.plant_time_constant is not None:
            params.plant.time_constant = config.plant_time_constant
        if config.plant_time_constant_std is not None:
            params.plant.time_constant_std = config.plant_time_constant_std
        if config.plant_time_constant_noise is not None:
            params.plant.time_constant_noise = config.plant_time_constant_noise
        if config.plant_enable_lag is not None:
            params.plant.enable_lag = config.plant_enable_lag

        # Select scenario based on inverse compensation flag
        if config.use_inverse_comp:
            params.inverse_comp.enabled = True
            if config.comp_gain is not None:
                params.inverse_comp.gain = config.comp_gain
            scenario = InverseCompScenario(params)
            scenario_type = "InverseComp"
        else:
            scenario = HILSScenario(params)
            scenario_type = "HILS"

        # Run scenario
        scenario.run()

        result = {
            "config": config,
            "scenario_type": scenario_type,
            "status": "success",
            "output_dir": scenario.run_dir
        }

        print(f"\n✅ Completed: {config.label}")
        print(f"   Results saved to: {scenario.run_dir}")

        return result

    except Exception as e:
        print(f"\n❌ Failed: {config.label}")
        print(f"   Error: {e}")

        return {
            "config": config,
            "scenario_type": scenario_type if 'scenario_type' in locals() else "unknown",
            "status": "failed",
            "error": str(e)
        }


def print_summary(results: List[Dict[str, Any]]):
    """Print summary of all simulation results"""
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for result in results:
        config = result["config"]
        status_icon = "✅" if result["status"] == "success" else "❌"

        print(f"\n{status_icon} {config.label}")
        print(f"   cmd_delay={config.cmd_delay} ms, sense_delay={config.sense_delay} ms")

        if config.use_inverse_comp:
            gain_str = f"{config.comp_gain}" if config.comp_gain else "default"
            print(f"   inverse_comp=True (gain={gain_str})")
        else:
            print(f"   inverse_comp=False")

        if config.plant_time_constant is not None:
            print(f"   plant_time_constant={config.plant_time_constant} ms")

        if config.plant_enable_lag is not None:
            print(f"   plant_enable_lag={config.plant_enable_lag}")

        print(f"   status={result['status']}")

        if result["status"] == "success":
            print(f"   → {result['output_dir']}")

    print("\n" + "=" * 60)
    successful = sum(1 for r in results if r["status"] == "success")
    total = len(results)
    print(f"Completed: {successful}/{total} simulations successful")
    print("=" * 60)


def main():
    """Main entry point with example configurations"""

    # ========================================
    # Configuration Examples
    # ========================================

    # Choose one of the following configurations:

    # Example 1: Symmetric delays with comparison (recommended)
    # Tests each delay with and without inverse compensation
    # delays = [10.0, 20.0, 30.0, 50.0, 100.0]
    # configs = create_comparison_sweep(delays)

    # Example 2: Symmetric delays without compensation only
    # configs = create_symmetric_sweep([10.0, 20.0, 30.0, 50.0, 100.0], use_inverse_comp=False)

    # Example 3: Symmetric delays with compensation only
    # configs = create_symmetric_sweep([10.0, 20.0, 30.0, 50.0, 100.0], use_inverse_comp=True)

    # Example 4: Asymmetric delays (different cmd/sense delays)
    # configs = create_asymmetric_sweep(
    #     cmd_delays=[20.0, 30.0],
    #     sense_delays=[10.0, 20.0],
    #     use_inverse_comp=False
    # )

    # Example 5: Plant time constant sweep
    # Test different actuator dynamics (plant lag)
    # configs = [
    #     DelayConfig(0.0, 0.0, plant_time_constant=100.0),   # Fast actuator
    #     DelayConfig(0.0, 0.0, plant_time_constant=500.0),   # Medium (default)
    #     DelayConfig(0.0, 0.0, plant_time_constant=1000.0),  # Slow actuator
    #     DelayConfig(0.0, 0.0, plant_enable_lag=False),      # Ideal actuator (no lag)
    # ]

    # Example 6: Combined delay and plant lag sweep
    # configs = [
    #     DelayConfig(20.0, 20.0, plant_time_constant=100.0),
    #     DelayConfig(20.0, 20.0, plant_time_constant=500.0),
    #     DelayConfig(20.0, 20.0, plant_time_constant=1000.0),
    # ]

    # Example 7: Custom configurations with specific gains
    configs = [
        DelayConfig(120.0, 0.0, use_inverse_comp=True, comp_gain=8.0),
        DelayConfig(120.0, 0.0, use_inverse_comp=True, comp_gain=6.0),
        DelayConfig(120.0, 0.0, use_inverse_comp=True, comp_gain=4.0),
        DelayConfig(120.0, 0.0, use_inverse_comp=True, comp_gain=2.0),
        DelayConfig(0.0, 0.0, use_inverse_comp=False),
    ]

    # ========================================
    # Run Simulations
    # ========================================

    print("=" * 60)
    print("Advanced Communication Delay Sweep")
    print("=" * 60)
    print(f"Total configurations: {len(configs)}")
    print()

    for i, config in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}] {config}")

    # Confirm before running
    print("\n" + "=" * 60)
    response = input("Proceed with simulations? [y/N]: ")
    if response.lower() != 'y':
        print("Cancelled.")
        return

    # Run all simulations
    results = []
    for config in configs:
        result = run_simulation(config)
        results.append(result)

    # Print summary
    print_summary(results)


if __name__ == "__main__":
    main()
