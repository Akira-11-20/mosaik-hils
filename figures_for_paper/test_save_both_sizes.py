#!/usr/bin/env python3
"""
Test script for save_figure_both_sizes function.
Creates a simple plot and saves it in both normal and large text sizes.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Import save_figure_both_sizes from plot_config
sys.path.insert(0, str(Path(__file__).parent))
from plot_config import save_figure_both_sizes

def main():
    # Create test data
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(x, y1, label="sin(x)", linewidth=2)
    ax.plot(x, y2, label="cos(x)", linewidth=2)

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Amplitude")
    ax.set_title("Test Plot for save_figure_both_sizes")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Save using save_figure_both_sizes
    output_dir = Path(__file__).parent
    normal_path, large_path = save_figure_both_sizes(
        fig, output_dir, base_name="test_plot"
    )

    print("\n" + "=" * 70)
    print("Test completed successfully!")
    print(f"Normal version: {normal_path}")
    print(f"Large text version: {large_path}")
    print("=" * 70)

    # Show the plot
    plt.show()


if __name__ == "__main__":
    main()
