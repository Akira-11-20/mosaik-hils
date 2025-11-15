"""
Test: Verify MAX_THRUST from .env is correctly used
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from scripts.sweeps.run_parameter_sweep import ParameterSweepConfig, run_sweep

# スイープパラメータ (MAX_THRUSTは含めない → .envから読むはず)
sweep_params = {
    "CONTROLLER_TYPE": ["hohmann"],
    "INVERSE_COMPENSATION": [True],
    "SIMULATION_TIME": [500.0],  # Short simulation
    "TIME_RESOLUTION": [1.0],
    "MINIMAL_DATA_MODE": [False],
    "AUTO_VISUALIZE": [False],  # Disable visualization for speed
}

# スイープ設定の作成
config = ParameterSweepConfig(
    sweep_params=sweep_params,
    base_env_file=".env",
    output_base_dir="results_sweep",
    description="Max Thrust Test (.env should give 100.0 N)",
)

if __name__ == "__main__":
    import sys
    dry_run = "--dry-run" in sys.argv

    if dry_run:
        print("🔍 Dry run mode - showing configuration without execution\n")

    print("🧪 Testing MAX_THRUST loading from .env (should be 100.0 N)\n")
    run_sweep(config, dry_run=dry_run)
