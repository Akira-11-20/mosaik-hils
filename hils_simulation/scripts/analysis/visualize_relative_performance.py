"""
Visualize relative performance analysis for compensation evaluation

Creates comprehensive visualizations comparing Ideal, HILS, and InverseComp scenarios
"""

import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json


def load_hdf5_data(filepath: str) -> dict:
    """Load simulation data from HDF5 file"""
    data = {}
    with h5py.File(filepath, 'r') as f:
        # Get time data
        data['time_s'] = f['time/time_s'][:]

        # Find datasets
        for key in f.keys():
            if 'EnvSim' in key:
                data['position'] = f[f'{key}/position'][:]
                data['velocity'] = f[f'{key}/velocity'][:]
                data['acceleration'] = f[f'{key}/acceleration'][:]
                break

        for key in f.keys():
            if 'ControllerSim' in key:
                data['error'] = f[f'{key}/error'][:]
                data['command_thrust'] = f[f'{key}/command_thrust'][:]
                break

        for key in f.keys():
            if 'PlantSim' in key:
                data['measured_thrust'] = f[f'{key}/measured_thrust'][:]
                if 'time_constant' in f[key]:
                    data['time_constant'] = f[f'{key}/time_constant'][:]
                break

    return data


def create_comparison_plots(ideal_data: dict, hils_data: dict, invcomp_data: dict,
                           target_position: float, output_dir: Path):
    """
    Create comprehensive comparison plots

    Args:
        ideal_data: Data from ideal scenario
        hils_data: Data from HILS scenario
        invcomp_data: Data from inverse compensation scenario
        target_position: Target position in meters
        output_dir: Directory to save plots
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    colors = {
        'ideal': '#2ecc71',      # Green
        'hils': '#e74c3c',       # Red
        'invcomp': '#3498db'     # Blue
    }

    # ========================================
    # Figure 1: Position Tracking
    # ========================================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Position vs Time
    ax1.plot(ideal_data['time_s'], ideal_data['position'],
             label='Ideal (No Lag)', color=colors['ideal'], linewidth=2)
    ax1.plot(hils_data['time_s'], hils_data['position'],
             label='HILS (No Comp)', color=colors['hils'], linewidth=2)
    ax1.plot(invcomp_data['time_s'], invcomp_data['position'],
             label='InverseComp', color=colors['invcomp'], linewidth=2)
    ax1.axhline(y=target_position, color='black', linestyle='--',
                linewidth=1.5, label='Target', alpha=0.7)
    ax1.set_xlabel('Time [s]', fontsize=12)
    ax1.set_ylabel('Position [m]', fontsize=12)
    ax1.set_title('Position Tracking Comparison', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Position Error vs Time
    ideal_error = ideal_data['position'] - target_position
    hils_error = hils_data['position'] - target_position
    invcomp_error = invcomp_data['position'] - target_position

    ax2.plot(ideal_data['time_s'], ideal_error,
             label='Ideal', color=colors['ideal'], linewidth=2)
    ax2.plot(hils_data['time_s'], hils_error,
             label='HILS', color=colors['hils'], linewidth=2)
    ax2.plot(invcomp_data['time_s'], invcomp_error,
             label='InverseComp', color=colors['invcomp'], linewidth=2)
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_xlabel('Time [s]', fontsize=12)
    ax2.set_ylabel('Position Error [m]', fontsize=12)
    ax2.set_title('Position Error Comparison', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'position_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    # ========================================
    # Figure 2: Control Input and Velocity
    # ========================================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Control Input (Thrust Command)
    ax1.plot(ideal_data['time_s'], ideal_data['command_thrust'],
             label='Ideal', color=colors['ideal'], linewidth=2)
    ax1.plot(hils_data['time_s'], hils_data['command_thrust'],
             label='HILS', color=colors['hils'], linewidth=2)
    ax1.plot(invcomp_data['time_s'], invcomp_data['command_thrust'],
             label='InverseComp', color=colors['invcomp'], linewidth=2)
    ax1.set_xlabel('Time [s]', fontsize=12)
    ax1.set_ylabel('Thrust Command [N]', fontsize=12)
    ax1.set_title('Control Input Comparison', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Velocity
    ax2.plot(ideal_data['time_s'], ideal_data['velocity'],
             label='Ideal', color=colors['ideal'], linewidth=2)
    ax2.plot(hils_data['time_s'], hils_data['velocity'],
             label='HILS', color=colors['hils'], linewidth=2)
    ax2.plot(invcomp_data['time_s'], invcomp_data['velocity'],
             label='InverseComp', color=colors['invcomp'], linewidth=2)
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_xlabel('Time [s]', fontsize=12)
    ax2.set_ylabel('Velocity [m/s]', fontsize=12)
    ax2.set_title('Velocity Comparison', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'control_velocity_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    # ========================================
    # Figure 3: Performance Metrics Bar Chart
    # ========================================
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    # Compute metrics
    metrics = {}
    for name, data, error in [('Ideal', ideal_data, ideal_error),
                               ('HILS', hils_data, hils_error),
                               ('InverseComp', invcomp_data, invcomp_error)]:
        metrics[name] = {
            'rms': np.sqrt(np.mean(error ** 2)),
            'max_error': np.max(np.abs(error)),
            'overshoot': max(0, np.max(data['position']) - target_position),
            'steady_state': np.mean(np.abs(error[int(len(error) * 0.8):]))
        }

    scenarios = ['Ideal', 'HILS', 'InverseComp']
    x_pos = np.arange(len(scenarios))
    bar_colors = [colors['ideal'], colors['hils'], colors['invcomp']]

    # RMS Error
    rms_values = [metrics[s]['rms'] for s in scenarios]
    bars1 = ax1.bar(x_pos, rms_values, color=bar_colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('RMS Error [m]', fontsize=11)
    ax1.set_title('Position RMS Error', fontsize=12, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(scenarios, fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    for i, (bar, val) in enumerate(zip(bars1, rms_values)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    # Max Error
    max_values = [metrics[s]['max_error'] for s in scenarios]
    bars2 = ax2.bar(x_pos, max_values, color=bar_colors, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Max Error [m]', fontsize=11)
    ax2.set_title('Maximum Position Error', fontsize=12, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(scenarios, fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    for i, (bar, val) in enumerate(zip(bars2, max_values)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    # Overshoot
    overshoot_values = [metrics[s]['overshoot'] / target_position * 100 for s in scenarios]
    bars3 = ax3.bar(x_pos, overshoot_values, color=bar_colors, alpha=0.7, edgecolor='black')
    ax3.set_ylabel('Overshoot [%]', fontsize=11)
    ax3.set_title('Overshoot Percentage', fontsize=12, fontweight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(scenarios, fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')
    for i, (bar, val) in enumerate(zip(bars3, overshoot_values)):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f'{val:.2f}%', ha='center', va='bottom', fontsize=9)

    # Steady-State Error
    ss_values = [metrics[s]['steady_state'] for s in scenarios]
    bars4 = ax4.bar(x_pos, ss_values, color=bar_colors, alpha=0.7, edgecolor='black')
    ax4.set_ylabel('Steady-State Error [m]', fontsize=11)
    ax4.set_title('Steady-State Error', fontsize=12, fontweight='bold')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(scenarios, fontsize=10)
    ax4.grid(True, alpha=0.3, axis='y')
    for i, (bar, val) in enumerate(zip(bars4, ss_values)):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / 'performance_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()

    # ========================================
    # Figure 4: Relative Performance (Normalized to Ideal)
    # ========================================
    fig, ax = plt.subplots(figsize=(10, 6))

    metric_names = ['RMS Error', 'Max Error', 'Overshoot', 'Steady-State Error']
    ideal_values = [metrics['Ideal']['rms'], metrics['Ideal']['max_error'],
                   metrics['Ideal']['overshoot'], metrics['Ideal']['steady_state']]
    hils_values = [metrics['HILS']['rms'], metrics['HILS']['max_error'],
                  metrics['HILS']['overshoot'], metrics['HILS']['steady_state']]
    invcomp_values = [metrics['InverseComp']['rms'], metrics['InverseComp']['max_error'],
                     metrics['InverseComp']['overshoot'], metrics['InverseComp']['steady_state']]

    # Normalize to ideal (ideal = 1.0)
    hils_normalized = [h / i if i > 0 else 0 for h, i in zip(hils_values, ideal_values)]
    invcomp_normalized = [ic / i if i > 0 else 0 for ic, i in zip(invcomp_values, ideal_values)]

    x = np.arange(len(metric_names))
    width = 0.25

    bars1 = ax.bar(x - width, [1.0] * len(metric_names), width,
                   label='Ideal', color=colors['ideal'], alpha=0.7, edgecolor='black')
    bars2 = ax.bar(x, hils_normalized, width,
                   label='HILS', color=colors['hils'], alpha=0.7, edgecolor='black')
    bars3 = ax.bar(x + width, invcomp_normalized, width,
                   label='InverseComp', color=colors['invcomp'], alpha=0.7, edgecolor='black')

    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
    ax.set_ylabel('Relative Performance (normalized to Ideal)', fontsize=12)
    ax.set_title('Relative Performance Comparison\n(Lower is Better, Ideal = 1.0)',
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=10, rotation=15, ha='right')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bars in [bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.05,
                   f'{height:.2f}x', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / 'relative_performance.png', dpi=300, bbox_inches='tight')
    plt.close()

    # ========================================
    # Figure 5: Time Constant Variation (if available)
    # ========================================
    if 'time_constant' in hils_data or 'time_constant' in invcomp_data:
        fig, ax = plt.subplots(figsize=(12, 5))

        if 'time_constant' in hils_data:
            ax.plot(hils_data['time_s'], hils_data['time_constant'],
                   label='HILS Plant τ', color=colors['hils'], linewidth=1.5, alpha=0.7)
        if 'time_constant' in invcomp_data:
            ax.plot(invcomp_data['time_s'], invcomp_data['time_constant'],
                   label='InverseComp Plant τ', color=colors['invcomp'], linewidth=1.5, alpha=0.7)

        ax.set_xlabel('Time [s]', fontsize=12)
        ax.set_ylabel('Plant Time Constant [ms]', fontsize=12)
        ax.set_title('Plant Time Constant Variation (Time-Varying Noise)',
                    fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'time_constant_variation.png', dpi=300, bbox_inches='tight')
        plt.close()

    print(f"\n✅ Visualizations saved to: {output_dir}")
    print(f"   - position_comparison.png")
    print(f"   - control_velocity_comparison.png")
    print(f"   - performance_metrics.png")
    print(f"   - relative_performance.png")
    if 'time_constant' in hils_data or 'time_constant' in invcomp_data:
        print(f"   - time_constant_variation.png")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize relative performance of compensation strategies"
    )
    parser.add_argument('--ideal', required=True, help='Path to ideal scenario HDF5 file')
    parser.add_argument('--hils', required=True, help='Path to HILS scenario HDF5 file')
    parser.add_argument('--invcomp', required=True, help='Path to InverseComp scenario HDF5 file')
    parser.add_argument('--target', type=float, default=5.0, help='Target position (default: 5.0m)')
    parser.add_argument('--output', default='results/visualizations', help='Output directory for plots')

    args = parser.parse_args()

    print("=" * 70)
    print("RELATIVE PERFORMANCE VISUALIZATION")
    print("=" * 70)
    print(f"Loading data...")
    print(f"  Ideal:      {args.ideal}")
    print(f"  HILS:       {args.hils}")
    print(f"  InverseComp: {args.invcomp}")

    # Load data
    ideal_data = load_hdf5_data(args.ideal)
    hils_data = load_hdf5_data(args.hils)
    invcomp_data = load_hdf5_data(args.invcomp)

    print(f"\nGenerating comparison plots...")

    # Create visualizations
    create_comparison_plots(ideal_data, hils_data, invcomp_data, args.target, args.output)

    print("\n" + "=" * 70)
    print("VISUALIZATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
