"""Main entry point for delay estimation experiments"""

import numpy as np
from pathlib import Path
from datetime import datetime
import argparse

from config.parameters import SimulationParameters
from simulators.system import OneDOFSystem
from simulators.network import NetworkDelay
from estimators.kalman_filter import KalmanFilter
from estimators.delay_estimator import create_estimator
from utils.visualization import (
    plot_estimation_results,
    plot_delay_comparison,
    compute_metrics,
)


def run_simulation(params: SimulationParameters, use_delay_estimation: bool = False):
    """
    Run simulation with or without delay estimation

    Args:
        params: Simulation parameters
        use_delay_estimation: Whether to use delay estimation

    Returns:
        results: Dictionary containing simulation results
    """
    # Create system
    system = OneDOFSystem(
        mass=params.system.mass,
        dt=params.system.dt,
        process_noise_std=params.network.process_noise_std,
        measurement_noise_std=params.network.measurement_noise_std,
        x0=np.array([params.system.initial_position, params.system.initial_velocity]),
    )

    # Create network delay model
    network = NetworkDelay(
        mean_delay=params.network.mean_delay,
        delay_std=params.network.delay_std,
        dt=params.system.dt,
        max_delay=params.network.max_delay,
        delay_type="random",  # or "constant", "varying"
    )

    # Create Kalman filter
    A, B, C = OneDOFSystem.create_matrices(params.system.mass, params.system.dt)
    Q = np.diag([params.kalman.q_position, params.kalman.q_velocity])
    R = np.array([[params.kalman.r_measurement]])
    P0 = np.diag([params.kalman.p0_position, params.kalman.p0_velocity])
    x0 = np.array([params.system.initial_position, params.system.initial_velocity])

    kf = KalmanFilter(A, B, C, Q, R, x0, P0)

    # Create delay estimator if needed
    if use_delay_estimation:
        delay_estimator = create_estimator(
            method=params.estimator.method,
            max_delay=params.network.max_delay,
            window_size=params.estimator.window_size,
        )
    else:
        delay_estimator = None

    # Storage for results
    num_steps = params.num_steps
    results = {
        "time": np.zeros(num_steps),
        "true_position": np.zeros(num_steps),
        "true_velocity": np.zeros(num_steps),
        "measured_position": np.zeros(num_steps),
        "estimated_position": np.zeros(num_steps),
        "estimated_velocity": np.zeros(num_steps),
        "true_delay": np.zeros(num_steps),
        "estimated_delay": np.zeros(num_steps) if use_delay_estimation else None,
    }

    # Simulation loop
    for k in range(num_steps):
        current_time = k * params.system.dt
        results["time"][k] = current_time

        # Control input (for now, zero control)
        u = np.zeros(1)

        # System step
        true_state, measurement = system.step(u)

        # Store true state
        results["true_position"][k] = true_state[0]
        results["true_velocity"][k] = true_state[1]

        # Add measurement to network delay buffer
        network.add_measurement(measurement, k)

        # Get delayed measurement
        delayed_measurement, actual_delay = network.get_delayed_measurement(k)
        results["true_delay"][k] = actual_delay

        if delayed_measurement is not None:
            results["measured_position"][k] = delayed_measurement[0]

            # Kalman filter step
            kf.predict(u)
            x_est, P_est, innovation = kf.update(delayed_measurement)

            # Delay estimation
            if use_delay_estimation:
                S = C @ P_est @ C.T + R  # Innovation covariance
                estimated_delay = delay_estimator.estimate_delay(innovation, S)
                results["estimated_delay"][k] = estimated_delay

            # Store estimates
            results["estimated_position"][k] = x_est[0]
            results["estimated_velocity"][k] = x_est[1]
        else:
            # No measurement available yet (initial steps)
            x_est = kf.get_state()
            results["estimated_position"][k] = x_est[0]
            results["estimated_velocity"][k] = x_est[1]

    return results


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Delay Estimation Experiments")
    parser.add_argument(
        "--method",
        type=str,
        default="innovation",
        choices=["innovation", "ml", "bayesian"],
        help="Delay estimation method",
    )
    parser.add_argument(
        "--compare", action="store_true", help="Compare with and without estimation"
    )
    parser.add_argument("--show", action="store_true", help="Show plots")
    args = parser.parse_args()

    # Load parameters
    params = SimulationParameters.default()
    params.estimator.method = args.method

    print("=" * 60)
    print("Delay Estimation Experiment")
    print("=" * 60)
    print(f"Method: {params.estimator.method}")
    print(f"Simulation time: {params.total_time} s")
    print(f"Time step: {params.system.dt} s")
    print(f"Mean delay: {params.network.mean_delay} s")
    print(f"Max delay: {params.network.max_delay} steps")
    print("=" * 60)

    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(params.results_dir) / timestamp
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nResults will be saved to: {results_dir}")

    if args.compare:
        print("\nRunning comparison...")

        # Run without delay estimation
        print("  - Standard Kalman Filter...")
        results_no_est = run_simulation(params, use_delay_estimation=False)

        # Run with delay estimation
        print(f"  - Kalman Filter with Delay Estimation ({args.method})...")
        results_with_est = run_simulation(params, use_delay_estimation=True)

        # Compute metrics
        metrics_no_est = compute_metrics(results_no_est)
        metrics_with_est = compute_metrics(results_with_est)

        print("\n" + "=" * 60)
        print("Results:")
        print("=" * 60)
        print("\nStandard KF:")
        print(f"  Position RMSE: {metrics_no_est['position_rmse']:.4f} m")
        print(f"  Position MAE:  {metrics_no_est['position_mae']:.4f} m")
        print(f"  Max Error:     {metrics_no_est['position_max_error']:.4f} m")

        print(f"\nKF with Delay Estimation ({args.method}):")
        print(f"  Position RMSE: {metrics_with_est['position_rmse']:.4f} m")
        print(f"  Position MAE:  {metrics_with_est['position_mae']:.4f} m")
        print(f"  Max Error:     {metrics_with_est['position_max_error']:.4f} m")
        if "delay_rmse" in metrics_with_est:
            print(f"  Delay RMSE:    {metrics_with_est['delay_rmse']:.4f} steps")

        # Plot comparison
        plot_delay_comparison(
            results_no_est,
            results_with_est,
            save_path=results_dir / "comparison.png",
            show=args.show,
        )

    else:
        print("\nRunning single simulation...")
        results = run_simulation(params, use_delay_estimation=True)

        # Compute metrics
        metrics = compute_metrics(results)

        print("\n" + "=" * 60)
        print("Results:")
        print("=" * 60)
        print(f"Position RMSE: {metrics['position_rmse']:.4f} m")
        print(f"Position MAE:  {metrics['position_mae']:.4f} m")
        print(f"Max Error:     {metrics['position_max_error']:.4f} m")

        # Plot results
        plot_estimation_results(
            results, save_path=results_dir / "results.png", show=args.show
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
