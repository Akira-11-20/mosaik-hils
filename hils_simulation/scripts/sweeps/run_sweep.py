"""
Plant time constant sweep - Run simulations with different actuator dynamics

This script demonstrates various plant parameter sweep scenarios:
- Time constant variation (τ)
- Individual variability (std)
- Time-varying noise
- Combined effects with communication delays and inverse compensation
"""

import sys
from pathlib import Path

# Add parent directory to path to enable imports from hils_simulation root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.sweeps.run_delay_sweep_advanced import DelayConfig, run_simulation, print_summary

# ============================================================================
# Configure your sweep scenario here
# ============================================================================

# Scenario: Time constant sweep with time-varying noise
configs = [
    DelayConfig(0.0, 0.0, plant_time_constant=300.0, plant_time_constant_noise=50.0, comp_gain=31, plant_enable_lag=True, use_inverse_comp=True),
    DelayConfig(0.0, 0.0, plant_time_constant=300.0, plant_time_constant_noise=100.0, comp_gain=31, plant_enable_lag=True, use_inverse_comp=True),
    DelayConfig(0.0, 0.0, plant_time_constant=300.0, plant_time_constant_noise=150.0, comp_gain=31, plant_enable_lag=True, use_inverse_comp=True),
    DelayConfig(0.0, 0.0, plant_time_constant=300.0, plant_time_constant_noise=220.0, comp_gain=31, plant_enable_lag=True, use_inverse_comp=True),
    DelayConfig(0.0, 0.0, plant_time_constant_std=0.0, plant_enable_lag=False),  # No lag (ideal)
]

print("=" * 70)
print("Plant Time Constant Sweep")
print("=" * 70)
print(f"Total configurations: {len(configs)}\n")

print("Testing DelayConfig with plant parameters:\n")

for i, config in enumerate(configs, 1):
    print(f"{i}. {config}")
    print(f"   Label: {config.label}")
    print(f"   Delays: cmd={config.cmd_delay}ms, sense={config.sense_delay}ms")
    if config.plant_time_constant is not None:
        print(f"   Plant τ: {config.plant_time_constant}ms")
    if config.plant_time_constant_std is not None and config.plant_time_constant_std > 0:
        print(f"   Plant τ std: {config.plant_time_constant_std}ms (±{3*config.plant_time_constant_std:.1f}ms @ 3σ)")
    if config.plant_time_constant_noise is not None and config.plant_time_constant_noise > 0:
        print(f"   Plant τ noise: {config.plant_time_constant_noise}ms (time-varying)")
    if config.plant_enable_lag is not None:
        print(f"   Plant lag enabled: {config.plant_enable_lag}")
    print(f"   Inverse compensation: {config.use_inverse_comp}")
    if config.use_inverse_comp and config.comp_gain is not None:
        print(f"   Compensation gain: {config.comp_gain}")
    print()

print("=" * 70)
response = input("Proceed with simulations? [y/N]: ")

if response.lower() != 'y':
    print("Cancelled.")
    exit()

# Run all simulations
print("\n" + "=" * 70)
print("Running simulations...")
print("=" * 70)

results = []
for config in configs:
    result = run_simulation(config)
    results.append(result)

# Print summary
print_summary(results)
