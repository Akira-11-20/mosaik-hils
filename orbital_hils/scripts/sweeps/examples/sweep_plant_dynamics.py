"""
Plant Dynamics Sweep Example

Plant（推力計測デバイス）の動特性パラメータのスイープ。
時定数とノイズレベルの影響を調査。
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from scripts.sweeps.run_parameter_sweep import ParameterSweepConfig, run_sweep

# スイープパラメータの定義
sweep_params = {
    "PLANT_TIME_CONSTANT": [1.0, 5.0, 10.0, 20.0, 50.0, 100.0],
    "PLANT_NOISE_STD": [0.0, 0.01, 0.05, 0.1],
    "SIMULATION_TIME": [300.0],
    "MINIMAL_DATA_MODE": [True],
    "AUTO_VISUALIZE": [False],  # スイープ中は可視化無効
}

# スイープ設定の作成
config = ParameterSweepConfig(
    sweep_params=sweep_params,
    base_env_file=".env",
    output_base_dir="results_sweep",
    description="Plant Dynamics Parameter Sweep",
)

if __name__ == "__main__":
    import sys
    dry_run = "--dry-run" in sys.argv

    if dry_run:
        print("🔍 Dry run mode - showing configuration without execution\n")

    run_sweep(config, dry_run=dry_run)
