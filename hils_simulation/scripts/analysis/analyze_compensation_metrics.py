"""
Analyze and compare compensation metrics for plant noise robustness evaluation

Computes:
- Position error RMS
- Compensation improvement ratio
- Deviation from ideal performance
"""

import argparse
import h5py
import numpy as np
from pathlib import Path
import json


def load_hdf5_data(filepath: str) -> dict:
    """Load position and time data from HDF5 file"""
    data = {}
    with h5py.File(filepath, 'r') as f:
        # Get time data
        data['time_s'] = f['time/time_s'][:]

        # Find position dataset (look for EnvSim)
        for key in f.keys():
            if 'EnvSim' in key:
                data['position'] = f[f'{key}/position'][:]
                data['velocity'] = f[f'{key}/velocity'][:]
                break

        # Find controller error if available
        for key in f.keys():
            if 'ControllerSim' in key:
                data['error'] = f[f'{key}/error'][:]
                break

    return data


def compute_rms(values: np.ndarray) -> float:
    """Compute Root Mean Square"""
    return np.sqrt(np.mean(values ** 2))


def compute_metrics(data: dict, target_position: float = 5.0) -> dict:
    """
    Compute performance metrics

    Args:
        data: Dictionary with 'time_s', 'position', 'velocity', 'error'
        target_position: Target position in meters

    Returns:
        Dictionary of computed metrics
    """
    metrics = {}

    # Position error
    position_error = data['position'] - target_position
    metrics['position_rms'] = compute_rms(position_error)
    metrics['position_max_error'] = np.max(np.abs(position_error))

    # Settling time (within 2% of target)
    settling_threshold = 0.02 * target_position
    settled = np.abs(position_error) < settling_threshold
    if np.any(settled):
        settling_idx = np.where(settled)[0][0]
        metrics['settling_time'] = data['time_s'][settling_idx]
    else:
        metrics['settling_time'] = None

    # Overshoot
    overshoot = np.max(data['position']) - target_position
    metrics['overshoot'] = max(0, overshoot)
    metrics['overshoot_percent'] = max(0, overshoot / target_position * 100)

    # Steady-state error (last 20% of simulation)
    steady_state_idx = int(len(position_error) * 0.8)
    metrics['steady_state_error'] = np.mean(np.abs(position_error[steady_state_idx:]))

    return metrics


def compare_scenarios(ideal_metrics: dict, hils_metrics: dict, invcomp_metrics: dict) -> dict:
    """
    Compare scenarios and compute improvement metrics

    Args:
        ideal_metrics: Metrics from ideal scenario (no lag)
        hils_metrics: Metrics from HILS scenario (no compensation)
        invcomp_metrics: Metrics from inverse compensation scenario

    Returns:
        Dictionary of comparison metrics
    """
    comparison = {}

    # RMS improvement
    comparison['hils_rms'] = hils_metrics['position_rms']
    comparison['invcomp_rms'] = invcomp_metrics['position_rms']
    comparison['ideal_rms'] = ideal_metrics['position_rms']

    # Absolute improvement
    comparison['rms_improvement_absolute'] = hils_metrics['position_rms'] - invcomp_metrics['position_rms']

    # Relative improvement (%)
    if hils_metrics['position_rms'] > 0:
        comparison['rms_improvement_percent'] = (
            (hils_metrics['position_rms'] - invcomp_metrics['position_rms'])
            / hils_metrics['position_rms'] * 100
        )
    else:
        comparison['rms_improvement_percent'] = 0.0

    # Deviation from ideal
    if ideal_metrics['position_rms'] > 0:
        comparison['hils_deviation_from_ideal'] = (
            hils_metrics['position_rms'] / ideal_metrics['position_rms']
        )
        comparison['invcomp_deviation_from_ideal'] = (
            invcomp_metrics['position_rms'] / ideal_metrics['position_rms']
        )
    else:
        comparison['hils_deviation_from_ideal'] = float('inf')
        comparison['invcomp_deviation_from_ideal'] = float('inf')

    # Settling time comparison
    comparison['hils_settling_time'] = hils_metrics['settling_time']
    comparison['invcomp_settling_time'] = invcomp_metrics['settling_time']
    comparison['ideal_settling_time'] = ideal_metrics['settling_time']

    # Overshoot comparison
    comparison['hils_overshoot'] = hils_metrics['overshoot_percent']
    comparison['invcomp_overshoot'] = invcomp_metrics['overshoot_percent']
    comparison['ideal_overshoot'] = ideal_metrics['overshoot_percent']

    return comparison


def print_results(ideal_metrics: dict, hils_metrics: dict, invcomp_metrics: dict, comparison: dict):
    """Print formatted results"""

    print("\n" + "=" * 70)
    print("PERFORMANCE METRICS ANALYSIS")
    print("=" * 70)

    print("\n1. Absolute Metrics (compared to Ideal baseline)")
    print("-" * 70)
    print(f"{'Metric':<25} {'Ideal':<12} {'HILS':<12} {'InverseComp':<12}")
    print("-" * 70)

    print(f"{'Position RMS Error [m]':<25} {ideal_metrics['position_rms']:<12.6f} "
          f"{hils_metrics['position_rms']:<12.6f} {invcomp_metrics['position_rms']:<12.6f}")
    print(f"{'Max Error [m]':<25} {ideal_metrics['position_max_error']:<12.6f} "
          f"{hils_metrics['position_max_error']:<12.6f} {invcomp_metrics['position_max_error']:<12.6f}")

    if all([ideal_metrics['settling_time'], hils_metrics['settling_time'], invcomp_metrics['settling_time']]):
        print(f"{'Settling Time [s]':<25} {ideal_metrics['settling_time']:<12.3f} "
              f"{hils_metrics['settling_time']:<12.3f} {invcomp_metrics['settling_time']:<12.3f}")

    print(f"{'Overshoot [%]':<25} {ideal_metrics['overshoot_percent']:<12.2f} "
          f"{hils_metrics['overshoot_percent']:<12.2f} {invcomp_metrics['overshoot_percent']:<12.2f}")
    print(f"{'Steady-State Error [m]':<25} {ideal_metrics['steady_state_error']:<12.6f} "
          f"{hils_metrics['steady_state_error']:<12.6f} {invcomp_metrics['steady_state_error']:<12.6f}")

    print("\n2. Relative Performance (normalized to Ideal = 1.00x)")
    print("-" * 70)
    print(f"{'Metric':<25} {'Ideal':<12} {'HILS':<12} {'InverseComp':<12}")
    print("-" * 70)

    # Calculate relative metrics
    rms_hils_rel = hils_metrics['position_rms'] / ideal_metrics['position_rms'] if ideal_metrics['position_rms'] > 0 else float('inf')
    rms_invcomp_rel = invcomp_metrics['position_rms'] / ideal_metrics['position_rms'] if ideal_metrics['position_rms'] > 0 else float('inf')

    max_hils_rel = hils_metrics['position_max_error'] / ideal_metrics['position_max_error'] if ideal_metrics['position_max_error'] > 0 else float('inf')
    max_invcomp_rel = invcomp_metrics['position_max_error'] / ideal_metrics['position_max_error'] if ideal_metrics['position_max_error'] > 0 else float('inf')

    overshoot_hils_rel = hils_metrics['overshoot'] / ideal_metrics['overshoot'] if ideal_metrics['overshoot'] > 0 else float('inf')
    overshoot_invcomp_rel = invcomp_metrics['overshoot'] / ideal_metrics['overshoot'] if ideal_metrics['overshoot'] > 0 else float('inf')

    ss_hils_rel = hils_metrics['steady_state_error'] / ideal_metrics['steady_state_error'] if ideal_metrics['steady_state_error'] > 0 else float('inf')
    ss_invcomp_rel = invcomp_metrics['steady_state_error'] / ideal_metrics['steady_state_error'] if ideal_metrics['steady_state_error'] > 0 else float('inf')

    print(f"{'RMS Error':<25} {'1.00x':<12} {rms_hils_rel:<12.2f}x {rms_invcomp_rel:<12.2f}x")
    print(f"{'Max Error':<25} {'1.00x':<12} {max_hils_rel:<12.2f}x {max_invcomp_rel:<12.2f}x")
    print(f"{'Overshoot':<25} {'1.00x':<12} {overshoot_hils_rel:<12.2f}x {overshoot_invcomp_rel:<12.2f}x")
    print(f"{'Steady-State Error':<25} {'1.00x':<12} {ss_hils_rel:<12.2f}x {ss_invcomp_rel:<12.2f}x")

    # Average relative performance
    avg_hils = np.mean([rms_hils_rel, max_hils_rel, overshoot_hils_rel, ss_hils_rel])
    avg_invcomp = np.mean([rms_invcomp_rel, max_invcomp_rel, overshoot_invcomp_rel, ss_invcomp_rel])

    print("-" * 70)
    print(f"{'Average (all metrics)':<25} {'1.00x':<12} {avg_hils:<12.2f}x {avg_invcomp:<12.2f}x")
    print("-" * 70)

    print("\n3. Compensation Effectiveness")
    print("-" * 70)

    def format_improvement(pct):
        if pct > 0:
            return f"✓ {pct:+.2f}% (improved)"
        elif pct < 0:
            return f"✗ {pct:+.2f}% (degraded)"
        else:
            return f"  {pct:+.2f}% (no change)"

    # A. HILS vs InverseComp
    print(f"\nA. HILS → InverseComp (% improvement from HILS):")
    print("-" * 70)

    # Calculate improvement percentages from HILS
    rms_from_hils_pct = (hils_metrics['position_rms'] - invcomp_metrics['position_rms']) / hils_metrics['position_rms'] * 100
    max_from_hils_pct = (hils_metrics['position_max_error'] - invcomp_metrics['position_max_error']) / hils_metrics['position_max_error'] * 100
    overshoot_from_hils_pct = (hils_metrics['overshoot'] - invcomp_metrics['overshoot']) / hils_metrics['overshoot'] * 100 if hils_metrics['overshoot'] > 0 else 0
    ss_from_hils_pct = (hils_metrics['steady_state_error'] - invcomp_metrics['steady_state_error']) / hils_metrics['steady_state_error'] * 100

    print(f"  RMS Error:             {format_improvement(rms_from_hils_pct)}")
    print(f"  Max Error:             {format_improvement(max_from_hils_pct)}")
    print(f"  Overshoot:             {format_improvement(overshoot_from_hils_pct)}")
    print(f"  Steady-State Error:    {format_improvement(ss_from_hils_pct)}")

    avg_from_hils = np.mean([rms_from_hils_pct, max_from_hils_pct, overshoot_from_hils_pct, ss_from_hils_pct])
    print("-" * 70)
    print(f"  Average:               {format_improvement(avg_from_hils)}")

    # B. Ideal vs InverseComp
    print(f"\nB. Ideal → InverseComp (% deviation from Ideal):")
    print("-" * 70)

    # Calculate deviation percentages from Ideal (negative means worse than ideal)
    rms_from_ideal_pct = (ideal_metrics['position_rms'] - invcomp_metrics['position_rms']) / ideal_metrics['position_rms'] * 100
    max_from_ideal_pct = (ideal_metrics['position_max_error'] - invcomp_metrics['position_max_error']) / ideal_metrics['position_max_error'] * 100
    overshoot_from_ideal_pct = (ideal_metrics['overshoot'] - invcomp_metrics['overshoot']) / ideal_metrics['overshoot'] * 100 if ideal_metrics['overshoot'] > 0 else 0
    ss_from_ideal_pct = (ideal_metrics['steady_state_error'] - invcomp_metrics['steady_state_error']) / ideal_metrics['steady_state_error'] * 100

    def format_deviation(pct):
        if abs(pct) < 5:
            return f"✓✓ {pct:+.2f}% (near-ideal)"
        elif pct > 0:
            return f"✓ {pct:+.2f}% (better than ideal)"
        elif pct > -20:
            return f"~ {pct:+.2f}% (acceptable)"
        else:
            return f"✗ {pct:+.2f}% (significant degradation)"

    print(f"  RMS Error:             {format_deviation(rms_from_ideal_pct)}")
    print(f"  Max Error:             {format_deviation(max_from_ideal_pct)}")
    print(f"  Overshoot:             {format_deviation(overshoot_from_ideal_pct)}")
    print(f"  Steady-State Error:    {format_deviation(ss_from_ideal_pct)}")

    avg_from_ideal = np.mean([rms_from_ideal_pct, max_from_ideal_pct, overshoot_from_ideal_pct, ss_from_ideal_pct])
    print("-" * 70)
    print(f"  Average:               {format_deviation(avg_from_ideal)}")

    print("\n4. Summary")
    print("-" * 70)

    # Overall assessment
    print(f"\nCompensation vs HILS:")
    if avg_from_hils > 10:
        print(f"  ✓✓ Strong improvement: {avg_from_hils:.1f}% better than HILS")
    elif avg_from_hils > 0:
        print(f"  ✓ Moderate improvement: {avg_from_hils:.1f}% better than HILS")
    elif avg_from_hils > -10:
        print(f"  ~ Marginal change: {avg_from_hils:.1f}% from HILS")
    else:
        print(f"  ✗ Degradation: {avg_from_hils:.1f}% worse than HILS")

    print(f"\nCompensation vs Ideal:")
    if abs(avg_from_ideal) < 5:
        print(f"  ✓✓ Near-ideal performance: {avg_from_ideal:.1f}% from Ideal")
    elif avg_from_ideal > 0:
        print(f"  ✓ Better than ideal: {avg_from_ideal:.1f}% (unexpected)")
    elif avg_from_ideal > -20:
        print(f"  ~ Acceptable degradation: {avg_from_ideal:.1f}% from Ideal")
    else:
        print(f"  ✗ Significant degradation: {avg_from_ideal:.1f}% from Ideal")

    # Gap recovery
    if avg_invcomp < avg_hils:
        gap_recovery = (avg_hils - avg_invcomp) / (avg_hils - 1.0) * 100 if avg_hils > 1.0 else 0
        print(f"\nGap Recovery:")
        print(f"  ✓ Recovered {gap_recovery:.1f}% of the gap between HILS and Ideal")
        print(f"  InverseComp: {avg_invcomp:.2f}x vs HILS: {avg_hils:.2f}x (relative to Ideal: 1.00x)")
    else:
        print(f"\n✗ InverseComp ({avg_invcomp:.2f}x) did not improve over HILS ({avg_hils:.2f}x)")

    print("=" * 70)


def save_results(output_file: str, ideal_metrics: dict, hils_metrics: dict,
                invcomp_metrics: dict, comparison: dict):
    """Save results to JSON file"""
    results = {
        'ideal': ideal_metrics,
        'hils': hils_metrics,
        'inverse_comp': invcomp_metrics,
        'comparison': comparison
    }

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze compensation performance metrics"
    )
    parser.add_argument('--ideal', required=True, help='Path to ideal scenario HDF5 file')
    parser.add_argument('--hils', required=True, help='Path to HILS scenario HDF5 file')
    parser.add_argument('--invcomp', required=True, help='Path to InverseComp scenario HDF5 file')
    parser.add_argument('--target', type=float, default=5.0, help='Target position in meters (default: 5.0)')
    parser.add_argument('--output', help='Output JSON file for results (optional)')

    args = parser.parse_args()

    print("=" * 70)
    print("Loading simulation data...")
    print("=" * 70)

    # Load data
    print(f"Ideal:      {args.ideal}")
    ideal_data = load_hdf5_data(args.ideal)

    print(f"HILS:       {args.hils}")
    hils_data = load_hdf5_data(args.hils)

    print(f"InverseComp: {args.invcomp}")
    invcomp_data = load_hdf5_data(args.invcomp)

    # Compute metrics
    print("\nComputing metrics...")
    ideal_metrics = compute_metrics(ideal_data, args.target)
    hils_metrics = compute_metrics(hils_data, args.target)
    invcomp_metrics = compute_metrics(invcomp_data, args.target)

    # Compare scenarios
    comparison = compare_scenarios(ideal_metrics, hils_metrics, invcomp_metrics)

    # Print results
    print_results(ideal_metrics, hils_metrics, invcomp_metrics, comparison)

    # Save results if output file specified
    if args.output:
        save_results(args.output, ideal_metrics, hils_metrics, invcomp_metrics, comparison)


if __name__ == "__main__":
    main()
