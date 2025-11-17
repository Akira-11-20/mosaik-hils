"""
Main simulation script for PD Inverse Compensation Verification

Architecture:
  Plant (outputs target) → PD Controller (tracks target) → [optional: Inverse Comp] → feedback

Scenarios:
1. No Compensation: Plant → PD Controller → (feedback to PD)
2. With Compensation: Plant → PD Controller → Inverse Comp → (feedback to PD)
"""
import mosaik
import sys
from pathlib import Path
import json
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np


# Simulation parameters
SIM_CONFIG = {
    'PDController': {
        'python': 'simulators.pd_controller:PDControllerSimulator',
    },
    'Plant': {
        'python': 'simulators.plant_simulator:PlantSimulator',
    },
    'InverseComp': {
        'python': 'simulators.inverse_compensator:InverseCompensatorSimulator',
    },
    'DataCollector': {
        'python': 'simulators.data_collector:DataCollectorSimulator',
    }
}


class PDInverseSimulation:
    """Main simulation class"""

    def __init__(self, params):
        """
        Args:
            params: Dictionary with simulation parameters
        """
        self.params = params
        self.results_dir = None

    def create_results_dir(self, scenario_name):
        """Create timestamped results directory"""
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        results_dir = Path('results') / f'{timestamp}_{scenario_name}'
        results_dir.mkdir(parents=True, exist_ok=True)

        # Save parameters
        with open(results_dir / 'params.json', 'w') as f:
            json.dump(self.params, f, indent=2)

        self.results_dir = results_dir
        return results_dir

    def run_no_compensation(self):
        """Run scenario without inverse compensation"""
        print("\n" + "="*60)
        print("Running: Plant → PD Controller (NO COMPENSATION)")
        print("="*60)

        results_dir = self.create_results_dir('no_comp')

        # Create Mosaik world
        world = mosaik.World(SIM_CONFIG, time_resolution=self.params['time_resolution'])

        # Create simulators
        plant_sim = world.start('Plant', step_size=self.params['step_size'])
        pd_sim = world.start('PDController', step_size=self.params['step_size'])
        data_collector_sim = world.start('DataCollector', step_size=self.params['step_size'])

        # Create entities
        plant = plant_sim.SimplePlant(target=self.params['target'])

        pd = pd_sim.PDController(
            kp=self.params['kp'],
            kd=self.params['kd'],
            dt=self.params['dt'],
            initial_position=self.params['initial_position']
        )

        collector = data_collector_sim.Collector(output_dir=str(results_dir), minimal_mode=False)

        # Connect: Plant → PD Controller
        world.connect(plant, pd, ('target_output', 'target_position'))

        # Connect to data collector
        world.connect(plant, collector, 'target_output')
        world.connect(pd, collector, 'position', 'control_output', 'velocity', 'error')

        # Run simulation
        world.run(until=self.params['sim_time'])

        print(f"✓ Results saved to: {results_dir}")

        return results_dir

    def run_with_compensation(self):
        """Run scenario with inverse compensation"""
        print("\n" + "="*60)
        print("Running: Plant → PD Controller → Inverse Compensator → PD")
        print("="*60)

        results_dir = self.create_results_dir('with_comp')

        # Create Mosaik world
        world = mosaik.World(SIM_CONFIG, time_resolution=self.params['time_resolution'])

        # Create simulators
        plant_sim = world.start('Plant', step_size=self.params['step_size'])
        pd_sim = world.start('PDController', step_size=self.params['step_size'])
        inv_sim = world.start('InverseComp', step_size=self.params['step_size'])
        data_collector_sim = world.start('DataCollector', step_size=self.params['step_size'])

        # Create entities
        plant = plant_sim.SimplePlant(target=self.params['target'])

        pd = pd_sim.PDController(
            kp=self.params['kp'],
            kd=self.params['kd'],
            dt=self.params['dt'],
            initial_position=self.params['initial_position']
        )

        inv_comp = inv_sim.InverseCompensator(
            comp_gain=self.params['comp_gain'],
            dt=self.params['dt']
        )

        collector = data_collector_sim.Collector(output_dir=str(results_dir), minimal_mode=False)

        # Connect: Plant → PD Controller
        world.connect(plant, pd, ('target_output', 'target_position'))

        # Connect: PD → Inverse Comp (for data collection only, no feedback)
        world.connect(pd, inv_comp, ('error', 'position'))

        # Connect to data collector
        world.connect(plant, collector, 'target_output')
        world.connect(pd, collector, 'position', 'control_output', 'velocity', 'error')
        world.connect(inv_comp, collector, 'position', 'output')

        # Run simulation
        world.run(until=self.params['sim_time'])

        print(f"✓ Results saved to: {results_dir}")

        return results_dir


def load_results(results_dir):
    """Load HDF5 results"""
    import h5py

    h5_file = list(Path(results_dir).glob('hils_data.h5'))[0]

    with h5py.File(h5_file, 'r') as f:
        # Get time
        time_s = f['time']['time_s'][:]

        # Get all data keys
        keys = [k for k in f.keys() if k != 'time']

        # Find target output from Plant
        target = None
        for key in keys:
            if 'target_output' in f[key]:
                target = f[key]['target_output'][:]
                break

        # Find PD controller data
        position = None
        control_output = None
        velocity = None
        error = None
        for key in keys:
            if 'position' in f[key]:
                position = f[key]['position'][:]
            if 'control_output' in f[key]:
                control_output = f[key]['control_output'][:]
            if 'velocity' in f[key]:
                velocity = f[key]['velocity'][:]
            if 'error' in f[key]:
                error = f[key]['error'][:]

        # Find inverse comp output (if exists)
        inv_output = None
        for key in keys:
            if 'output_InverseComp' in key or ('output' in f[key] and 'InverseComp' in key):
                inv_output = f[key]['output'][:]
                break

    return {
        'time': time_s,
        'target': target,
        'position': position,
        'control_output': control_output,
        'velocity': velocity,
        'error': error,
        'inv_output': inv_output
    }


def plot_comparison(no_comp_dir, with_comp_dir, params):
    """Plot comparison of both scenarios"""
    print("\n" + "="*60)
    print("Generating comparison plots...")
    print("="*60)

    # Load results
    no_comp = load_results(no_comp_dir)
    with_comp = load_results(with_comp_dir)

    # Create figure
    fig, axes = plt.subplots(4, 1, figsize=(12, 12))

    # Plot 1: Position tracking
    ax = axes[0]
    if no_comp['target'] is not None:
        ax.plot(no_comp['time'], no_comp['target'], 'r:', label='Target', linewidth=2)
    ax.plot(no_comp['time'], no_comp['position'], label='Position (No Comp)', linewidth=2)
    ax.plot(with_comp['time'], with_comp['position'], label='Position (With Comp)', linewidth=2, linestyle='--')
    ax.set_ylabel('Position [m]')
    ax.set_xlabel('Time [s]')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title(f'Position Tracking (Target={params["target"]}m, Comp Gain={params["comp_gain"]})')

    # Plot 2: Tracking error
    ax = axes[1]
    ax.plot(no_comp['time'], no_comp['error'], label='Error (No Comp)', linewidth=2)
    ax.plot(with_comp['time'], with_comp['error'], label='Error (With Comp)', linewidth=2, linestyle='--')
    ax.axhline(y=0, color='k', linestyle=':', linewidth=1, alpha=0.5)
    ax.set_ylabel('Error [m]')
    ax.set_xlabel('Time [s]')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title('Tracking Error')

    # Plot 3: Control output (velocity command)
    ax = axes[2]
    ax.plot(no_comp['time'], no_comp['control_output'], label='Control Output (No Comp)', linewidth=2, alpha=0.7)
    ax.plot(with_comp['time'], with_comp['control_output'], label='Control Output (With Comp)', linewidth=2, alpha=0.7, linestyle='--')
    ax.set_ylabel('Control Output [m/s]')
    ax.set_xlabel('Time [s]')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title('Control Output (Velocity Command)')

    # Plot 4: Velocity
    ax = axes[3]
    ax.plot(no_comp['time'], no_comp['velocity'], label='Velocity (No Comp)', linewidth=2, alpha=0.7)
    ax.plot(with_comp['time'], with_comp['velocity'], label='Velocity (With Comp)', linewidth=2, alpha=0.7, linestyle='--')
    if with_comp['inv_output'] is not None:
        ax.plot(with_comp['time'], with_comp['inv_output'], label='Inv Comp Output', linewidth=2, alpha=0.7, linestyle=':')
    ax.set_ylabel('Velocity [m/s]')
    ax.set_xlabel('Time [s]')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title('Velocity and Compensator Output')

    plt.tight_layout()

    # Save figure
    save_path = Path('results') / 'comparison.png'
    save_path.parent.mkdir(exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Comparison plot saved to: {save_path}")

    plt.show()

    # Print performance metrics
    print("\n" + "="*60)
    print("Performance Metrics")
    print("="*60)

    no_comp_rmse = np.sqrt(np.mean(no_comp['error']**2))
    with_comp_rmse = np.sqrt(np.mean(with_comp['error']**2))

    no_comp_settling = calculate_settling_time(no_comp['time'], no_comp['error'], params['target'])
    with_comp_settling = calculate_settling_time(with_comp['time'], with_comp['error'], params['target'])

    print(f"No Compensation RMSE:       {no_comp_rmse:.4f} m")
    print(f"With Compensation RMSE:     {with_comp_rmse:.4f} m")
    if no_comp_rmse > 0:
        print(f"RMSE Improvement:           {(1 - with_comp_rmse/no_comp_rmse)*100:.2f}%")
    print()
    print(f"No Compensation Settling:   {no_comp_settling:.4f} s")
    print(f"With Compensation Settling: {with_comp_settling:.4f} s")


def calculate_settling_time(time, error, target, threshold_pct=0.02):
    """Calculate 2% settling time"""
    threshold = target * threshold_pct
    abs_error = np.abs(error)

    # Find last time error exceeded threshold
    above_threshold = abs_error > threshold
    if not any(above_threshold):
        return 0.0

    last_idx = np.where(above_threshold)[0][-1]
    return time[last_idx]


def main():
    """Main entry point"""
    # Default parameters
    params = {
        'sim_time': 5000,           # Total simulation time [ms]
        'time_resolution': 1.0,     # Mosaik time resolution (1ms per step)
        'step_size': 10,            # Simulator step size [steps] = 10ms
        'dt': 0.01,                 # Physical time step [s] = 10ms
        'kp': 2.0,                  # Proportional gain
        'kd': 0.5,                  # Derivative gain
        'target': 10.0,             # Target position [m]
        'initial_position': 0.0,    # Initial position [m]
        'comp_gain': 90,           # Inverse compensation gain (1.0=no effect, <1.0=damping, >1.0=amplification)
    }

    # Parse command line arguments (simple override)
    if len(sys.argv) > 1:
        if sys.argv[1] in ['--help', '-h']:
            print("Usage: python main.py [comp_gain]")
            print(f"Default: comp_gain={params['comp_gain']}")
            print("\nCompensation gain effects:")
            print("  1.0 = No compensation effect")
            print("  <1.0 = Damping effect on feedback")
            print("  >1.0 = Amplification effect on feedback")
            return

        if len(sys.argv) >= 2:
            params['comp_gain'] = float(sys.argv[1])

    print("\n" + "="*60)
    print("PD Inverse Compensation Verification")
    print("="*60)
    print(f"Target Position:       {params['target']} m")
    print(f"Initial Position:      {params['initial_position']} m")
    print(f"PD Gains:              Kp={params['kp']}, Kd={params['kd']}")
    print(f"Compensation Gain:     {params['comp_gain']}")
    print(f"Simulation Time:       {params['sim_time']/1000:.1f} s")
    print(f"Time Step:             {params['dt']*1000:.0f} ms")

    # Create simulation instance
    sim = PDInverseSimulation(params)

    # Run both scenarios
    no_comp_dir = sim.run_no_compensation()
    with_comp_dir = sim.run_with_compensation()

    # Plot comparison
    plot_comparison(no_comp_dir, with_comp_dir, params)

    print("\n" + "="*60)
    print("Simulation Complete!")
    print("="*60)


if __name__ == '__main__':
    main()
