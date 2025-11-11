"""
Plant time constant sweep - Run simulations with different actuator dynamics

This script demonstrates various plant parameter sweep scenarios:
- Time constant variation (τ)
- Individual variability (std)
- Time-varying noise
- Combined effects with communication delays and inverse compensation
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
#     50.0,
#     100.0,
#     150.0,
#     200.0,
# ]

# Format: (time_constant_ms, corresponding_compensation_gain)
time_constants = [
    (100,7),
    (100,8),
    (100,9),
    (100, 10),
    (100, 11),
    (100,12),
    (100,13),
]

# Define noise levels to test
# time_constant_noises = [
#     5,
#     10,
#     15,
#     20,
# ]

# Define whether to test with/without inverse compensation
test_inverse_comp = [True]

# Generate all combinations using itertools.product
configs = []
for time_constant, use_inverse in product(time_constants, test_inverse_comp):  # product( #, (tau, gain), noise, use_inv
    # command_delay,
    # time_constants,
    # time_constant_noises,
    # test_inverse_comp,
    configs.append(
        DelayConfig(
            cmd_delay=0.0,
            sense_delay=0.0,
            plant_time_constant=time_constant[0],
            plant_time_constant_noise=0,
            comp_gain=time_constant[1],
            plant_enable_lag=True,
            use_inverse_comp=use_inverse,
        )
    )

# Create baseline (ideal) configuration - will be included in sweep
baseline_config = DelayConfig(
    cmd_delay=0.0,
    sense_delay=0.0,
    plant_time_constant=None,  # Don't set time constant (use default but won't appear in heatmap)
    plant_time_constant_std=0.0,
    plant_time_constant_noise=0.0,  # Explicitly set to 0 to avoid default noise
    plant_enable_lag=False,
    use_inverse_comp=False,
    label="baseline_rt",
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
    print(f"   Inverse compensation: {config.use_inverse_comp}")
    if config.use_inverse_comp and config.comp_gain is not None:
        print(f"   Compensation gain: {config.comp_gain}")
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
