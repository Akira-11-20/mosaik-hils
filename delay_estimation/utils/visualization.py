"""Visualization utilities for delay estimation results"""

from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_estimation_results(
    results: Dict[str, np.ndarray],
    save_path: Optional[Path] = None,
    show: bool = True,
):
    """
    Plot estimation results

    Args:
        results: Dictionary containing simulation results
        save_path: Path to save figure (optional)
        show: Whether to show the plot
    """
    time = results["time"]

    fig, axes = plt.subplots(4, 1, figsize=(12, 10))

    # Position
    axes[0].plot(time, results["true_position"], "k-", label="True", linewidth=1.5)
    axes[0].plot(time, results["measured_position"], "r.", label="Measured", markersize=3, alpha=0.5)
    axes[0].plot(
        time,
        results["estimated_position"],
        "b-",
        label="Estimated (KF)",
        linewidth=1,
    )
    axes[0].set_ylabel("Position [m]")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title("State Estimation with Delay")

    # Velocity
    axes[1].plot(time, results["true_velocity"], "k-", label="True", linewidth=1.5)
    axes[1].plot(time, results["estimated_velocity"], "b-", label="Estimated (KF)", linewidth=1)
    axes[1].set_ylabel("Velocity [m/s]")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Delay
    axes[2].plot(time, results["true_delay"], "k-", label="True Delay", linewidth=1.5)
    if "estimated_delay" in results:
        axes[2].plot(
            time,
            results["estimated_delay"],
            "r--",
            label="Estimated Delay",
            linewidth=1,
        )
    axes[2].set_ylabel("Delay [steps]")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    # Estimation error
    position_error = results["true_position"] - results["estimated_position"]
    axes[3].plot(time, position_error, "b-", linewidth=1)
    axes[3].set_ylabel("Position Error [m]")
    axes[3].set_xlabel("Time [s]")
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_delay_comparison(
    results_no_est: Dict[str, np.ndarray],
    results_with_est: Dict[str, np.ndarray],
    save_path: Optional[Path] = None,
    show: bool = True,
):
    """
    Compare results with and without delay estimation

    Args:
        results_no_est: Results without delay estimation
        results_with_est: Results with delay estimation
        save_path: Path to save figure (optional)
        show: Whether to show the plot
    """
    time = results_no_est["time"]

    fig, axes = plt.subplots(3, 1, figsize=(12, 8))

    # Position comparison
    axes[0].plot(time, results_no_est["true_position"], "k-", label="True", linewidth=1.5)
    axes[0].plot(
        time,
        results_no_est["estimated_position"],
        "r--",
        label="Standard KF",
        linewidth=1,
    )
    axes[0].plot(
        time,
        results_with_est["estimated_position"],
        "b-",
        label="KF with Delay Est.",
        linewidth=1,
    )
    axes[0].set_ylabel("Position [m]")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title("Comparison: Standard KF vs KF with Delay Estimation")

    # Estimation errors
    error_no_est = results_no_est["true_position"] - results_no_est["estimated_position"]
    error_with_est = results_with_est["true_position"] - results_with_est["estimated_position"]

    axes[1].plot(time, error_no_est, "r--", label="Standard KF", linewidth=1)
    axes[1].plot(time, error_with_est, "b-", label="KF with Delay Est.", linewidth=1)
    axes[1].set_ylabel("Position Error [m]")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # RMSE over time (sliding window)
    window = 100
    rmse_no_est = np.sqrt(np.convolve(error_no_est**2, np.ones(window) / window, mode="same"))
    rmse_with_est = np.sqrt(np.convolve(error_with_est**2, np.ones(window) / window, mode="same"))

    axes[2].plot(time, rmse_no_est, "r--", label="Standard KF", linewidth=1.5)
    axes[2].plot(time, rmse_with_est, "b-", label="KF with Delay Est.", linewidth=1.5)
    axes[2].set_ylabel("RMSE [m]")
    axes[2].set_xlabel("Time [s]")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Comparison figure saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def compute_metrics(results: Dict[str, np.ndarray]) -> Dict[str, float]:
    """
    Compute performance metrics

    Args:
        results: Simulation results

    Returns:
        metrics: Dictionary of performance metrics
    """
    position_error = results["true_position"] - results["estimated_position"]
    velocity_error = results["true_velocity"] - results["estimated_velocity"]

    metrics = {
        "position_rmse": np.sqrt(np.mean(position_error**2)),
        "position_mae": np.mean(np.abs(position_error)),
        "position_max_error": np.max(np.abs(position_error)),
        "velocity_rmse": np.sqrt(np.mean(velocity_error**2)),
        "velocity_mae": np.mean(np.abs(velocity_error)),
    }

    if "estimated_delay" in results:
        delay_error = results["true_delay"] - results["estimated_delay"]
        metrics.update(
            {
                "delay_rmse": np.sqrt(np.mean(delay_error**2)),
                "delay_mae": np.mean(np.abs(delay_error)),
            }
        )

    return metrics
