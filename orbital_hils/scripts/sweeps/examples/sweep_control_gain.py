"""
Control Gain Sweep Example

制御ゲインのスイープ。
異なる制御ゲインでの軌道制御性能を比較。
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from scripts.sweeps.run_parameter_sweep import ParameterSweepConfig, run_sweep

# スイープパラメータの定義
sweep_params = {
    "CONTROL_GAIN": [0.0001, 0.001, 0.01, 0.1, 1.0],
    "PLANT_TIME_CONSTANT": [10.0],
    "SIMULATION_TIME": [500.0],
    "MINIMAL_DATA_MODE": [True],
    "AUTO_VISUALIZE": [False],
}

# スイープ設定の作成
config = ParameterSweepConfig(
    sweep_params=sweep_params,
    base_env_file=".env",
    output_base_dir="results_sweep",
    description="Control Gain Sweep",
)

if __name__ == "__main__":
    import sys
    dry_run = "--dry-run" in sys.argv

    if dry_run:
        print("🔍 Dry run mode - showing configuration without execution\n")

    run_sweep(config, dry_run=dry_run)
