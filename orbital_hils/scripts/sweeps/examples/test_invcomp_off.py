"""
Test: INVERSE_COMPENSATION=False is correctly applied
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from scripts.sweeps.run_parameter_sweep import ParameterSweepConfig, run_sweep

# Sweep with InvComp OFF explicitly
sweep_params = {
    "CONTROLLER_TYPE": ["hohmann"],
    "INVERSE_COMPENSATION": [False],  # Explicitly OFF
    "PLANT_TIME_CONSTANT": [10.0],
    "SIMULATION_TIME": [500.0],
    "TIME_RESOLUTION": [1.0],
    "MINIMAL_DATA_MODE": [False],
    "AUTO_VISUALIZE": [False],
}

config = ParameterSweepConfig(
    sweep_params=sweep_params,
    base_env_file=".env",
    output_base_dir="results_sweep",
    description="Test InvComp OFF",
)

if __name__ == "__main__":
    print("🧪 Testing INVERSE_COMPENSATION=False\n")
    run_sweep(config, dry_run=False)
