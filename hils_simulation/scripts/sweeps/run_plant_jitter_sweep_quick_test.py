"""
Quick test version of plant jitter sweep with fewer configurations
"""

import sys
from pathlib import Path

# Add parent directory to path to enable imports from hils_simulation root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.sweeps.run_delay_sweep_advanced import DelayConfig, run_simulation, print_summary


def main():
    """
    Quick test with only 2 jitter levels
    """

    # Quick test configuration - only 2 jitter levels
    configs = [
        # 0% jitter (baseline)
        DelayConfig(
            cmd_delay=0.0,
            sense_delay=0.0,
            use_inverse_comp=False,
            plant_time_constant=50.0,
            plant_time_constant_std=0.0,
            plant_enable_lag=True,
            label="HILS_tau50ms_std0pct"
        ),
        DelayConfig(
            cmd_delay=0.0,
            sense_delay=0.0,
            use_inverse_comp=True,
            plant_time_constant=50.0,
            plant_time_constant_std=0.0,
            plant_enable_lag=True,
            label="InvComp_tau50ms_std0pct"
        ),
        # 10% jitter
        DelayConfig(
            cmd_delay=0.0,
            sense_delay=0.0,
            use_inverse_comp=False,
            plant_time_constant=50.0,
            plant_time_constant_std=5.0,
            plant_enable_lag=True,
            label="HILS_tau50ms_std10pct"
        ),
        DelayConfig(
            cmd_delay=0.0,
            sense_delay=0.0,
            use_inverse_comp=True,
            plant_time_constant=50.0,
            plant_time_constant_std=5.0,
            plant_enable_lag=True,
            label="InvComp_tau50ms_std10pct"
        ),
    ]

    print("=" * 70)
    print("Plant Time Constant Jitter Sweep - QUICK TEST")
    print("=" * 70)
    print(f"Testing 2 jitter levels: 0%, 10%")
    print(f"Total configurations: {len(configs)}")
    print()

    for i, config in enumerate(configs, 1):
        comp_type = "InvComp" if config.use_inverse_comp else "HILS   "
        std_ratio = config.plant_time_constant_std / 50.0 if config.plant_time_constant_std > 0 else 0.0
        print(f"{i}. [{comp_type}] tau={config.plant_time_constant:.0f}ms, "
              f"std={config.plant_time_constant_std:.1f}ms ({std_ratio*100:.0f}%)")

    print("\n" + "=" * 70)
    response = input("Proceed with quick test? [y/N]: ")

    if response.lower() != 'y':
        print("Cancelled.")
        return

    print("\n" + "=" * 70)
    print("Running simulations...")
    print("=" * 70)

    results = []
    for i, config in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}]")
        result = run_simulation(config)
        results.append(result)

    print_summary(results)


if __name__ == "__main__":
    main()
