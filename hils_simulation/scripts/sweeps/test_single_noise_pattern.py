"""
Single pattern test for plant noise compensation evaluation
Tests one noise level with Ideal, HILS, and InverseComp scenarios
"""

import sys
from pathlib import Path

# Add parent directory to path to enable imports from hils_simulation root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.sweeps.run_delay_sweep_advanced import DelayConfig, run_simulation, print_summary


def main():
    """
    Test single noise pattern with comprehensive comparison
    """

    # ========================================
    # Configuration
    # ========================================

    base_time_constant = 300.0  # [ms]
    noise_level = 50.0  # [ms] - 33.3% of tau
    comp_gain = 31.0  # Compensation gain

    # Create configurations for comparison
    configs = [
        # 1. Ideal: No lag (best case baseline)
        DelayConfig(
            cmd_delay=0.0,
            sense_delay=0.0,
            use_inverse_comp=False,
            plant_enable_lag=False,
            label="Ideal_NoLag"
        ),

        # 2. HILS: With lag and noise, no compensation
        DelayConfig(
            cmd_delay=0.0,
            sense_delay=0.0,
            use_inverse_comp=False,
            plant_time_constant=base_time_constant,
            plant_time_constant_noise=noise_level,
            plant_enable_lag=True,
            label=f"HILS_tau{base_time_constant:.0f}ms_noise{noise_level:.0f}ms"
        ),

        # 3. InverseComp: With lag and noise, with compensation
        DelayConfig(
            cmd_delay=0.0,
            sense_delay=0.0,
            use_inverse_comp=True,
            comp_gain=comp_gain,
            plant_time_constant=base_time_constant,
            plant_time_constant_noise=noise_level,
            plant_enable_lag=True,
            label=f"InvComp_tau{base_time_constant:.0f}ms_noise{noise_level:.0f}ms_gain{comp_gain:.0f}"
        ),
    ]

    # ========================================
    # Display Configuration
    # ========================================

    print("=" * 70)
    print("Plant Noise Compensation Evaluation - Single Pattern Test")
    print("=" * 70)
    print(f"Base plant time constant: {base_time_constant} ms")
    print(f"Noise level: {noise_level} ms ({noise_level/base_time_constant*100:.1f}% of tau)")
    print(f"Compensation gain: {comp_gain}")
    print(f"Total configurations: {len(configs)}")
    print()

    print("Test scenarios:")
    print("-" * 70)
    for i, config in enumerate(configs, 1):
        print(f"{i}. {config.label}")
        if hasattr(config, 'plant_time_constant') and config.plant_time_constant:
            print(f"   Plant tau: {config.plant_time_constant} ms")
        if hasattr(config, 'plant_time_constant_noise') and config.plant_time_constant_noise:
            print(f"   Plant noise: {config.plant_time_constant_noise} ms")
        if config.use_inverse_comp:
            print(f"   Compensation: ON (gain={config.comp_gain})")
        else:
            print(f"   Compensation: OFF")
        print()

    # ========================================
    # Confirmation
    # ========================================

    print("=" * 70)
    print("This test will:")
    print("1. Run Ideal scenario (no lag) as baseline")
    print("2. Run HILS scenario (lag + noise, no compensation)")
    print("3. Run InverseComp scenario (lag + noise, with compensation)")
    print()
    print("After completion, run the analysis script to compute:")
    print("- Position error RMS")
    print("- Compensation improvement ratio")
    print("- Deviation from ideal performance")
    print("=" * 70)

    response = input("\nProceed with test? [y/N]: ")

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

    # ========================================
    # Next Steps
    # ========================================

    print("\n" + "=" * 70)
    print("Next steps:")
    print("=" * 70)
    print("Run the analysis script to compute evaluation metrics:")
    print("  cd hils_simulation")
    print("  uv run python scripts/analysis/analyze_compensation_metrics.py \\")
    print("    --ideal results/<ideal_timestamp>/hils_data.h5 \\")
    print("    --hils results/<hils_timestamp>/hils_data.h5 \\")
    print("    --invcomp results/<invcomp_timestamp>/hils_data.h5")
    print()
    print("Or manually specify the output directories from above.")
    print("=" * 70)


if __name__ == "__main__":
    main()
