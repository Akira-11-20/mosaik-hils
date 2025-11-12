"""
Adaptive Inverse Compensation Sweep

Test adaptive inverse compensation with dynamic time constant models.
Compares traditional fixed-gain compensation vs adaptive compensation
that tracks plant dynamics in real-time.

Scenarios tested:
1. Baseline (no compensation, no lag)
2. Fixed-gain inverse compensation with constant tau
3. Fixed-gain inverse compensation with dynamic tau (linear model)
4. Adaptive inverse compensation with dynamic tau (linear model)
5. Adaptive inverse compensation with hybrid tau model
"""

import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path to enable imports from hils_simulation root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.sweeps.run_delay_sweep_advanced import DelayConfig, print_summary, run_simulation

# ============================================================================
# Sweep Configuration
# ============================================================================

# Base parameters
BASE_TAU = 100.0  # ms
TAU_TO_GAIN_RATIO = 0.1  # Ratio for converting tau to gain (gain = tau * ratio)

# Linear model parameters (thrust-rate dependent)
LINEAR_MODEL_PARAMS = {
    "sensitivity": 50,  # [s/N] - how much tau changes with thrust rate
}

# Hybrid model parameters (thrust-rate + thermal)
HYBRID_MODEL_PARAMS = {
    "thrust_sensitivity": 0.2,  # Thrust rate sensitivity
    "heating_rate": 0.0005,  # Heating rate [K/(N^2·ms)]
    "cooling_rate": 0.01,  # Cooling rate [1/ms]
    "thermal_sensitivity": 0.03,  # Thermal sensitivity [1/K]
}

# ============================================================================
# Test Cases
# ============================================================================

configs = []

configs.append(
    DelayConfig(
        cmd_delay=0.0,
        sense_delay=0.0,
        plant_time_constant=None,  # Don't set time constant (use default but won't appear in heatmap)
        plant_time_constant_std=0.0,
        plant_time_constant_noise=0.0,  # Explicitly set to 0 to avoid default noise
        plant_enable_lag=False,
        use_inverse_comp=False,
        label="baseline_rt",
    )
)


# 4. HILS with linear tau model, no compensation
configs.append(
    DelayConfig(
        cmd_delay=0.0,
        sense_delay=0.0,
        use_inverse_comp=False,
        plant_time_constant=BASE_TAU,
        plant_enable_lag=True,
        use_plant_model=True,
        plant_tau_model_type="linear",
        plant_tau_model_params=LINEAR_MODEL_PARAMS,
        label="hils_linear_tau_nocomp",
    )
)

# 5. HILS with linear tau model + fixed-gain compensation (mismatch!)
configs.append(
    DelayConfig(
        cmd_delay=0.0,
        sense_delay=0.0,
        use_inverse_comp=True,
        use_adaptive_comp=False,  # Fixed gain - will be suboptimal
        comp_gain=BASE_TAU * TAU_TO_GAIN_RATIO,  # Only matches base tau
        plant_time_constant=BASE_TAU,
        plant_enable_lag=True,
        use_plant_model=True,
        plant_tau_model_type="linear",
        plant_tau_model_params=LINEAR_MODEL_PARAMS,
        label="hils_linear_tau_fixed_comp",
    )
)

# 6. HILS with linear tau model + adaptive compensation (perfect match!)
configs.append(
    DelayConfig(
        cmd_delay=0.0,
        sense_delay=0.0,
        use_inverse_comp=True,
        use_adaptive_comp=True,  # Adaptive - tracks plant tau
        comp_tau_to_gain_ratio=TAU_TO_GAIN_RATIO,
        plant_time_constant=BASE_TAU,
        plant_enable_lag=True,
        use_plant_model=True,
        plant_tau_model_type="linear",
        plant_tau_model_params=LINEAR_MODEL_PARAMS,
        # Compensator uses same model as plant
        comp_tau_model_type="linear",
        comp_tau_model_params=LINEAR_MODEL_PARAMS,
        label="hils_linear_tau_adaptive_comp",
    )
)

# ============================================================================
# Run the sweep
# ============================================================================
print("=" * 80)
print("Adaptive Inverse Compensation Sweep")
print("=" * 80)
print(f"Total configurations: {len(configs)}\n")

print("Test cases:\n")
for i, config in enumerate(configs, 1):
    print(f"{i}. {config.label}")
    if config.use_plant_model:
        print(f"   Plant model: {config.plant_tau_model_type}")
    if config.use_inverse_comp:
        if config.use_adaptive_comp:
            print(f"   Compensation: Adaptive (ratio={config.comp_tau_to_gain_ratio})")
            print(f"   Comp model: {config.comp_tau_model_type}")
        else:
            print(f"   Compensation: Fixed gain ({config.comp_gain})")
    else:
        print("   Compensation: None")
    print()

print("=" * 80)
response = input("Proceed with simulations? [y/N]: ")

if response.lower() != "y":
    print("Cancelled.")
    exit()

# Create sweep directory
base_dir = Path(__file__).parent.parent.parent
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
sweep_dir = base_dir / "results" / f"{timestamp}_adaptive_comp_sweep"
sweep_dir.mkdir(parents=True, exist_ok=True)

print(f"\n📁 Sweep results directory: {sweep_dir}")

# Run all simulations
print("\n" + "=" * 80)
print("Running simulations...")
print("=" * 80)

results = []
for config in configs:
    result = run_simulation(config, sweep_dir=sweep_dir)
    results.append(result)

# Print summary
print_summary(results)

# ============================================================================
# Analysis suggestions
# ============================================================================
print("\n" + "=" * 80)
print("Analysis Suggestions")
print("=" * 80)
print("\nCompare the following cases to demonstrate adaptive compensation benefits:")
print("\n1. Constant tau:")
print(f"   - Fixed comp:    {configs[2].label}")
print(f"   - Adaptive comp: {configs[2].label} (should be identical)")
print("\n2. Linear tau model:")
print(f"   - No comp:       {configs[3].label}")
print(f"   - Fixed comp:    {configs[4].label}")
print(f"   - Adaptive comp: {configs[5].label} (should outperform fixed)")
print("\n3. Hybrid tau model:")
print(f"   - No comp:       {configs[6].label}")
print(f"   - Fixed comp:    {configs[7].label}")
print(f"   - Adaptive comp: {configs[8].label} (should significantly outperform fixed)")

print("\n" + "=" * 80)
print(f"Sweep complete! Results in: {sweep_dir}")
print("=" * 80)
