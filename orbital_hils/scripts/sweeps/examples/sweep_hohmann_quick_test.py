"""
Hohmann Transfer Quick Test

ホーマン遷移の動作確認用クイックテスト。
短時間（500秒）でInverse CompensationのON/OFFを比較。
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from scripts.sweeps.run_parameter_sweep import ParameterSweepConfig, run_sweep

# スイープパラメータの定義（短時間テスト用）
sweep_params = {
    "CONTROLLER_TYPE": ["hohmann"],  # ★ Hohmann transfer controller
    "PLANT_TIME_CONSTANT": [10.0],  # Single value for quick test
    "INVERSE_COMPENSATION": [True, False],  # ON/OFF comparison
    "INVERSE_COMPENSATION_GAIN": [10.0],  # Fixed gain
    "SIMULATION_TIME": [500.0],  # Short 500s test
    "TIME_RESOLUTION": [1.0],  # 1s steps
    "MINIMAL_DATA_MODE": [False],  # Full data for analysis
    "AUTO_VISUALIZE": [True],  # Generate plots
    "HOHMANN_INITIAL_ALTITUDE_KM": [408.0],  # ISS altitude
    "HOHMANN_TARGET_ALTITUDE_KM": [500.0],  # Target altitude
    "HOHMANN_START_TIME": [100.0],  # Start transfer at 100s
}

# スイープ設定の作成
config = ParameterSweepConfig(
    sweep_params=sweep_params,
    base_env_file=".env",
    output_base_dir="results_sweep",
    description="Hohmann Transfer Quick Test (500s)",
)

if __name__ == "__main__":
    import sys

    dry_run = "--dry-run" in sys.argv

    if dry_run:
        print("🔍 Dry run mode - showing configuration without execution\n")

    print("🚀 Hohmann transfer quick test - 500s simulation\n")
    run_sweep(config, dry_run=dry_run)
