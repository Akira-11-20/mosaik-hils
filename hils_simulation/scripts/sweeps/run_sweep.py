"""
Plant time constant sweep - Run simulations with different actuator dynamics

This script demonstrates various plant parameter sweep scenarios:
- Time constant variation (τ)
- Individual variability (std)
- Time-varying noise
- Dynamic time constant models (linear, thermal, hybrid, etc.)
- Adaptive inverse compensation with dynamic tau models
- Combined effects with communication delays and inverse compensation

Example configurations:

1. Constant tau sweep (default):
   USE_PLANT_MODEL = False
   PLANT_TAU_MODEL_TYPE = None
   USE_ADAPTIVE_COMP = False

2. Linear tau model with fixed gain compensation:
   USE_PLANT_MODEL = True
   PLANT_TAU_MODEL_TYPE = "linear"
   PLANT_TAU_MODEL_PARAMS = {"sensitivity": 0.1}
   USE_ADAPTIVE_COMP = False

3. Linear tau model with adaptive compensation:
   USE_PLANT_MODEL = True
   PLANT_TAU_MODEL_TYPE = "linear"
   PLANT_TAU_MODEL_PARAMS = {"sensitivity": 0.1}
   USE_ADAPTIVE_COMP = True
   INVERSE_COMP_TAU_TO_GAIN_RATIO = 0.1
   INVERSE_COMP_TAU_MODEL_TYPE = "linear"
   INVERSE_COMP_TAU_MODEL_PARAMS = {"sensitivity": 0.1}

4. Hybrid thermal model with adaptive compensation:
   USE_PLANT_MODEL = True
   PLANT_TAU_MODEL_TYPE = "hybrid"
   PLANT_TAU_MODEL_PARAMS = {
       "thrust_sensitivity": 0.25,
       "heating_rate": 0.001,
       "cooling_rate": 0.01,
       "thermal_sensitivity": 0.04
   }
   USE_ADAPTIVE_COMP = True
   INVERSE_COMP_TAU_MODEL_TYPE = "hybrid"
   INVERSE_COMP_TAU_MODEL_PARAMS = {
       "thrust_sensitivity": 0.25,
       "heating_rate": 0.001,
       "cooling_rate": 0.01,
       "thermal_sensitivity": 0.04
   }
"""

import sys
from datetime import datetime
from itertools import product
from pathlib import Path

# Add parent directory to path to enable imports from hils_simulation root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.sweeps.run_delay_sweep_advanced import DelayConfig, print_summary, run_simulation

# ============================================================================
# Configure your sweep scenario here
# ============================================================================

# Define time constant and compensation gain pairs
# command_delay = [
#     (10.0, 1),
#     (20.0, 2),
#     (30.0, 3),
#     (40.0, 4),
#     (0.0, 5),
# ]

# Format: (time_constant_ms, corresponding_compensation_gain)
time_constants = [
    (100, 8),
    (100, 9),
    (100, 10),
    (100, 11),
    (100, 12),
]

# Define noise levels to test
time_constant_noises = [
    0,
    5,
    10,
    15,
]

# Define whether to test with/without inverse compensation
test_inverse_comp = [True] * 10

comp_positions = ["post"]  # Compensation position: "pre" or "post"

# ============================================================================
# Plant Time Constant Model Configuration
# ============================================================================
# Enable dynamic plant model (plant_simulator_with_model.py)
# Set to True to use advanced tau models, False to use constant tau (sweep each value)
# IMPORTANT: For tau sweep, set this to False so each tau value in the sweep is used
USE_PLANT_MODEL = False

# Time constant model type
# Options: "constant", "linear", "saturation", "thermal", "hybrid", "stochastic"
PLANT_TAU_MODEL_TYPE = "linear"  # Set to None to use default from .env

# Time constant model parameters (JSON dict)
# Examples:
#   - constant: {} or None (no parameters needed)
#   - linear: {"sensitivity": 0.1}
#   - hybrid: {"thrust_sensitivity": 0.25, "heating_rate": 0.001, "cooling_rate": 0.01, "thermal_sensitivity": 0.04}
#   - thermal: {"heating_rate": 0.001, "cooling_rate": 0.01, "thermal_sensitivity": 0.05}
PLANT_TAU_MODEL_PARAMS = {"sensitivity": 1}  # Set to None for constant model (no parameters needed)

# Example: Enable linear model with sensitivity
# USE_PLANT_MODEL = True
# PLANT_TAU_MODEL_TYPE = "linear"
# PLANT_TAU_MODEL_PARAMS = {"sensitivity": 0.1}

# Example: Enable hybrid thermal model
# USE_PLANT_MODEL = True
# PLANT_TAU_MODEL_TYPE = "hybrid"
# PLANT_TAU_MODEL_PARAMS = {
#     "thrust_sensitivity": 0.25,
#     "heating_rate": 0.001,
#     "cooling_rate": 0.01,
#     "thermal_sensitivity": 0.04
# }

# ============================================================================
# Inverse Compensator Adaptive Configuration
# ============================================================================
# Enable adaptive compensation (compensator gain adapts to plant tau model)
# When True, the compensator will use the same tau model as the plant
USE_ADAPTIVE_COMP = False

# Tau to gain conversion ratio (gain = tau * ratio)
# Used when USE_ADAPTIVE_COMP = True
INVERSE_COMP_TAU_TO_GAIN_RATIO = None  # Set to None to use default from .env (e.g., 0.1)

# Base time constant for compensator [ms]
# Used when USE_ADAPTIVE_COMP = True
INVERSE_COMP_BASE_TAU = None  # Set to None to use plant time constant

# Compensator tau model type
# Options: "constant" (fixed gain), "linear", "hybrid", etc.
# If None, will match PLANT_TAU_MODEL_TYPE when USE_ADAPTIVE_COMP = True
INVERSE_COMP_TAU_MODEL_TYPE = None

# Compensator tau model parameters (JSON dict)
# If None, will match PLANT_TAU_MODEL_PARAMS when USE_ADAPTIVE_COMP = True
INVERSE_COMP_TAU_MODEL_PARAMS = None

# Example: Enable adaptive compensation with linear model
# USE_ADAPTIVE_COMP = True
# INVERSE_COMP_TAU_TO_GAIN_RATIO = 0.1
# INVERSE_COMP_BASE_TAU = 100.0
# INVERSE_COMP_TAU_MODEL_TYPE = "linear"  # Match plant model
# INVERSE_COMP_TAU_MODEL_PARAMS = {"sensitivity": 0.1}  # Match plant params

# Generate all combinations using itertools.product
configs = []
for use_inverse, comp_position, time_constant, time_constant_noise in product(
    test_inverse_comp, comp_positions, time_constants, time_constant_noises
):  # product( #, (tau, gain), noise, use_inv
    # command_delay,
    # time_constants,
    # time_constant_noises,
    # test_inverse_comp,
    configs.append(
        DelayConfig(
            cmd_delay=0.0,
            sense_delay=0.0,
            plant_time_constant=time_constant[0],
            plant_time_constant_noise=time_constant_noise,
            comp_gain=time_constant[1],
            plant_enable_lag=True,  # Enable first-order lag
            use_inverse_comp=use_inverse,
            comp_position=comp_position,  # Set compensator position to "pre" (before plant)
            # Plant model configuration
            use_plant_model=USE_PLANT_MODEL,
            plant_tau_model_type=PLANT_TAU_MODEL_TYPE,
            plant_tau_model_params=PLANT_TAU_MODEL_PARAMS,
            # Inverse compensator adaptive configuration
            use_adaptive_comp=USE_ADAPTIVE_COMP,
            comp_tau_to_gain_ratio=INVERSE_COMP_TAU_TO_GAIN_RATIO,
            comp_base_tau=INVERSE_COMP_BASE_TAU if INVERSE_COMP_BASE_TAU is not None else time_constant[0],
            comp_tau_model_type=INVERSE_COMP_TAU_MODEL_TYPE,
            comp_tau_model_params=INVERSE_COMP_TAU_MODEL_PARAMS,
        )
    )

# Create baseline (ideal) configuration - will be included in sweep
baseline_config = DelayConfig(
    cmd_delay=0.0,
    sense_delay=0.0,
    plant_time_constant=0.0,  # Don't set time constant (use default but won't appear in heatmap)
    plant_time_constant_std=0.0,
    plant_time_constant_noise=0.0,  # Explicitly set to 0 to avoid default noise
    plant_enable_lag=False,
    use_inverse_comp=False,
    label="baseline_rt",
    # Baseline doesn't use plant model (keeps it simple)
    use_plant_model=False,
    plant_tau_model_type=None,
    plant_tau_model_params=None,
)

# Add baseline to configs (will be run first)
configs.insert(0, baseline_config)

# ============================================================================
# Run the sweep
# ============================================================================
print("=" * 70)
print("Plant Time Constant Sweep")
print("=" * 70)
print(f"Total configurations: {len(configs)} (including baseline)\n")

print("Configurations:\n")

for i, config in enumerate(configs, 1):
    print(f"{i}. {config}")
    print(f"   Label: {config.label}")
    print(f"   Delays: cmd={config.cmd_delay}ms, sense={config.sense_delay}ms")
    if config.plant_time_constant is not None:
        print(f"   Plant τ: {config.plant_time_constant}ms")
    if config.plant_time_constant_std is not None and config.plant_time_constant_std > 0:
        print(f"   Plant τ std: {config.plant_time_constant_std}ms (±{3 * config.plant_time_constant_std:.1f}ms @ 3σ)")
    if config.plant_time_constant_noise is not None and config.plant_time_constant_noise > 0:
        print(f"   Plant τ noise: {config.plant_time_constant_noise}ms (time-varying)")
    if config.plant_enable_lag is not None:
        print(f"   Plant lag enabled: {config.plant_enable_lag}")
    # Display plant model information
    if config.use_plant_model:
        print("   Plant model: ENABLED")
        if config.plant_tau_model_type:
            print(f"   Plant τ model type: {config.plant_tau_model_type}")
        if config.plant_tau_model_params:
            import json

            params_str = json.dumps(config.plant_tau_model_params)
            print(f"   Plant τ model params: {params_str}")
    print(f"   Inverse compensation: {config.use_inverse_comp}")
    if config.use_inverse_comp:
        if config.comp_gain is not None:
            print(f"   Compensation gain: {config.comp_gain}")
        if config.comp_position is not None:
            print(f"   Compensation position: {config.comp_position}")
        # Display adaptive compensation information
        if config.use_adaptive_comp:
            print("   Adaptive compensation: ENABLED")
            if config.comp_tau_to_gain_ratio is not None:
                print(f"   Tau-to-gain ratio: {config.comp_tau_to_gain_ratio}")
            if config.comp_tau_model_type:
                print(f"   Comp τ model type: {config.comp_tau_model_type}")
            if config.comp_tau_model_params:
                import json

                params_str = json.dumps(config.comp_tau_model_params)
                print(f"   Comp τ model params: {params_str}")
    print()

print("=" * 70)
response = input("Proceed with simulations? [y/N]: ")

if response.lower() != "y":
    print("Cancelled.")
    exit()

# Create sweep directory to organize all simulation results
base_dir = Path(__file__).parent.parent.parent
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
sweep_dir = base_dir / "results" / f"{timestamp}_sweep"
sweep_dir.mkdir(parents=True, exist_ok=True)

print(f"\n📁 Sweep results directory: {sweep_dir}")

# Run all simulations
print("\n" + "=" * 70)
print("Running simulations...")
print("=" * 70)

results = []
for config in configs:
    result = run_simulation(config, sweep_dir=sweep_dir)
    results.append(result)

# Print summary
print_summary(results)

# ============================================================================
# Run comparison with RT baseline (optional)
# ============================================================================
print("\n" + "=" * 70)
print("Post-Sweep Analysis")
print("=" * 70)

# Extract result directories
result_dirs = [r["output_dir"] for r in results if r["status"] == "success" and "output_dir" in r]

if result_dirs:
    print("\nSimulation results saved to:")
    for i, result_dir in enumerate(result_dirs, 1):
        print(f"  {i}. {result_dir}")

    # Ask if user wants to run comparison analysis
    print("\n" + "=" * 70)

    # Find baseline in result_dirs (it should be the first one with "baseline_rt" label)
    baseline_dir = None
    for result_dir in result_dirs:
        if "baseline_rt" in str(result_dir):
            baseline_dir = result_dir
            break

    if baseline_dir:
        print(f"Baseline found in sweep: {baseline_dir.name}")
        response = input("Run comparison analysis with this baseline? [Y/n]: ").strip().lower()
        use_baseline = response in ["", "y", "yes"]
    else:
        response = input("Run comparison analysis with RT baseline? [y/N]: ").strip().lower()
        use_baseline = response in ["y", "yes"]

    if use_baseline:
        rt_dir = None

        if baseline_dir:
            # Use the baseline from this sweep
            rt_dir = baseline_dir
            print(f"Using baseline from sweep: {rt_dir.name}")
        else:
            # Ask for RT baseline directory
            print("\nPlease specify RT baseline directory.")
            print("Options:")
            print("  1. Use latest baseline simulation (default)")
            print("  2. Specify custom directory")
            rt_choice = input("Enter choice [1]: ").strip() or "1"

            if rt_choice == "1":
                # Find latest baseline (no inverse_comp suffix, no sweep directories)
                base_dir = Path(__file__).parent.parent.parent
                results_path = base_dir / "results"
                baseline_dirs = sorted(
                    [
                        d
                        for d in results_path.iterdir()
                        if d.is_dir()
                        and "_inverse_comp" not in d.name
                        and "_sweep" not in d.name
                        and "_delay_sweep" not in d.name
                        and "_gain_sweep" not in d.name
                        and "visualizations" not in d.name
                        and "comparison" not in d.name
                    ],
                    key=lambda x: x.name,
                    reverse=True,
                )
                if baseline_dirs:
                    rt_dir = baseline_dirs[0]
                    print(f"Using baseline: {rt_dir.name}")
                else:
                    print("No baseline found in results directory.")
            elif rt_choice == "2":
                custom_dir = input("Enter RT baseline directory path: ").strip()
                rt_dir = Path(custom_dir)
                if not rt_dir.exists():
                    print(f"Error: Directory not found: {rt_dir}")
                    rt_dir = None

        if rt_dir:
            # Import comparison module
            import subprocess

            # Filter out RT baseline from result_dirs if it's in the list
            # Also filter out any directory with "baseline" in the name
            filtered_result_dirs = [
                d for d in result_dirs if str(rt_dir) not in str(d) and "baseline" not in str(d).lower()
            ]

            if not filtered_result_dirs:
                print("\n⚠️  No simulation results to compare (all results were filtered out).")
                print("    This usually means the RT baseline is the same as the sweep directory.")
                print("    Please run a proper RT baseline simulation first using:")
                print("    uv run python main.py r")
            else:
                print(f"\nComparing {len(filtered_result_dirs)} simulation(s) against RT baseline.")

                # Build command - pass specific result directories
                cmd = [
                    "uv",
                    "run",
                    "python",
                    "scripts/analysis/compare_with_rt.py",
                    "--rt-dir",
                    str(rt_dir),
                    "--result-dirs",
                ] + [str(d) for d in filtered_result_dirs]

                print(f"Command: {' '.join(cmd)}")
                print()

                # Run comparison
                try:
                    subprocess.run(cmd, check=True, cwd=str(Path(__file__).parent.parent.parent))
                    print("\n" + "=" * 70)
                    print("Comparison analysis complete!")
                    print("Results saved to: results/comparison_with_rt/")
                    print("=" * 70)
                except subprocess.CalledProcessError as e:
                    print(f"\nError running comparison analysis: {e}")
        else:
            print("Skipping comparison analysis.")
    else:
        print("Skipping comparison analysis.")
else:
    print("\nNo successful simulation results to analyze.")
