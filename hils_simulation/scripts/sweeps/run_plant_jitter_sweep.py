"""
Plant time constant jitter sweep - Evaluate inverse compensation accuracy with plant dynamics variability

This script evaluates how plant time constant variability (jitter) affects inverse compensation performance:
- Tests different levels of plant time constant standard deviation
- Compares HILS (no compensation) vs Inverse Compensation scenarios
- Fixed communication delays with varying plant dynamics uncertainty
"""

import sys
from pathlib import Path

# Add parent directory to path to enable imports from hils_simulation root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.sweeps.run_delay_sweep_advanced import DelayConfig, run_simulation, print_summary


def create_jitter_sweep_configs(
    base_time_constant: float = 100.0,
    jitter_noise_ratios: list = None,
    cmd_delay: float = 0.0,
    sense_delay: float = 0.0,
    comp_gain: float = None,
) -> list:
    """
    Create configurations for plant jitter sweep

    Args:
        base_time_constant: Base plant time constant in milliseconds
        jitter_noise_ratios: List of time-varying noise ratios (e.g., [0.0, 0.05, 0.10, 0.20] for 0%, 5%, 10%, 20%)
        cmd_delay: Command path delay in milliseconds
        sense_delay: Sensing path delay in milliseconds
        comp_gain: Inverse compensation gain (None = use default from .env)

    Returns:
        List of DelayConfig objects
    """
    if jitter_noise_ratios is None:
        jitter_noise_ratios = [0.0, 0.05, 0.10, 0.15, 0.20]

    configs = []

    for i, ratio in enumerate(jitter_noise_ratios):
        time_constant_noise = base_time_constant * ratio

        # Calculate compensation gain for this jitter level
        # If comp_gain is None, calculate based on base_time_constant
        # Formula: gain = base_time_constant / 10 + 1
        if comp_gain is None:
            current_comp_gain = base_time_constant / 10 + 1
        elif isinstance(comp_gain, list):
            # If comp_gain is a list, use corresponding value
            current_comp_gain = comp_gain[i] if i < len(comp_gain) else comp_gain[-1]
        else:
            # If comp_gain is a single value, use it for all
            current_comp_gain = comp_gain

        # Without inverse compensation
        configs.append(
            DelayConfig(
                cmd_delay=cmd_delay,
                sense_delay=sense_delay,
                use_inverse_comp=False,
                plant_time_constant=base_time_constant,
                plant_time_constant_noise=time_constant_noise,
                plant_enable_lag=True,
                label=f"HILS_tau{base_time_constant:.0f}ms_noise{ratio*100:.0f}pct"
            )
        )

        # With inverse compensation
        configs.append(
            DelayConfig(
                cmd_delay=cmd_delay,
                sense_delay=sense_delay,
                use_inverse_comp=True,
                comp_gain=current_comp_gain,
                plant_time_constant=base_time_constant,
                plant_time_constant_noise=time_constant_noise,
                plant_enable_lag=True,
                label=f"InvComp_tau{base_time_constant:.0f}ms_noise{ratio*100:.0f}pct_gain{current_comp_gain:.1f}"
            )
        )

    return configs


def main():
    """
    Main entry point for plant jitter sweep
    """

    # ========================================
    # Configuration
    # ========================================

    # Plant parameters
    base_time_constant = 100.0  # Base plant time constant [ms]
    jitter_noise_ratios = [0.0, 0.05, 0.10, 0.15, 0.20]  # Time-varying noise as % of base

    # Communication delays (set to 0 for pure plant dynamics evaluation)
    cmd_delay = 0.0  # [ms]
    sense_delay = 0.0  # [ms]

    # Inverse compensation gain (None = use default calculation)
    comp_gain = 11

    # Generate configurations
    configs = create_jitter_sweep_configs(
        base_time_constant=base_time_constant,
        jitter_noise_ratios=jitter_noise_ratios,
        cmd_delay=cmd_delay,
        sense_delay=sense_delay,
        comp_gain=comp_gain,
    )

    # ========================================
    # Display Configuration
    # ========================================

    print("=" * 70)
    print("Plant Time Constant Jitter Sweep (Time-Varying Noise)")
    print("=" * 70)
    print(f"Base plant time constant: {base_time_constant} ms")
    print(f"Jitter levels (noise/tau): {[f'{r*100:.0f}%' for r in jitter_noise_ratios]}")
    print(f"Communication delays: cmd={cmd_delay}ms, sense={sense_delay}ms")
    print(f"Total configurations: {len(configs)}")
    print()

    print("Configuration details:")
    print("-" * 70)
    for i, config in enumerate(configs, 1):
        comp_type = "InvComp" if config.use_inverse_comp else "HILS   "
        noise_ratio = config.plant_time_constant_noise / base_time_constant if config.plant_time_constant_noise and config.plant_time_constant_noise > 0 else 0.0
        print(f"{i:2d}. [{comp_type}] tau={config.plant_time_constant:.0f}ms, "
              f"noise={config.plant_time_constant_noise:.1f}ms ({noise_ratio*100:.0f}%), "
              f"delays=({config.cmd_delay:.0f}, {config.sense_delay:.0f})ms")
        if config.use_inverse_comp and config.comp_gain is not None:
            print(f"    Compensation gain: {config.comp_gain}")

    # ========================================
    # Confirmation
    # ========================================

    print("\n" + "=" * 70)
    print("This will run the following experiment:")
    print(f"- {len(jitter_noise_ratios)} jitter levels (time-varying noise)")
    print(f"- 2 scenarios per jitter level (HILS + InverseComp)")
    print(f"- Total: {len(configs)} simulations")
    print()
    print("Expected evaluation metrics:")
    print("- Position error RMS")
    print("- Settling time")
    print("- Overshoot")
    print("- Steady-state error")
    print("=" * 70)

    response = input("Proceed with simulations? [y/N]: ")

    if response.lower() != 'y':
        print("Cancelled.")
        return

    # ========================================
    # Run Simulations
    # ========================================

    print("\n" + "=" * 70)
    print("Running simulations...")
    print("=" * 70)

    results = []
    for i, config in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}]")
        result = run_simulation(config)
        results.append(result)

    # ========================================
    # Print Summary
    # ========================================

    print_summary(results)

    # Additional analysis hints
    print("\n" + "=" * 70)
    print("Next steps:")
    print("=" * 70)
    print("1. Analyze results using:")
    print("   cd hils_simulation")
    print("   uv run python scripts/analysis/visualize_results.py results/<timestamp>/hils_data.h5")
    print()
    print("2. Compare performance metrics across jitter levels")
    print("3. Evaluate inverse compensator robustness to plant uncertainty")
    print("=" * 70)


if __name__ == "__main__":
    main()
