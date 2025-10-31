"""
Plant time constant sweep - Run simulations with different actuator dynamics
"""

import sys
from pathlib import Path

# Add parent directory to path to enable imports from hils_simulation root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.sweeps.run_delay_sweep_advanced import DelayConfig, run_simulation, print_summary

# Test plant time constant sweep
configs = [
    DelayConfig(0.0, 0.0, plant_time_constant=30.0, plant_enable_lag=True, use_inverse_comp=True, comp_gain=4),
    DelayConfig(0.0, 0.0, plant_time_constant=90.0, plant_enable_lag=True, use_inverse_comp=True, comp_gain=10),
    DelayConfig(0.0, 0.0, plant_time_constant=150.0, plant_enable_lag=True, use_inverse_comp=True, comp_gain=16),
    DelayConfig(0.0, 0.0, plant_time_constant=300.0, plant_enable_lag=True, use_inverse_comp=True, comp_gain=31),
    DelayConfig(0.0, 0.0, plant_enable_lag=False),
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
