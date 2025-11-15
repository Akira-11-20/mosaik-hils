"""
Hohmann Transfer with Inverse Compensation Sweep

ホーマン遷移におけるInverse Compensationの効果を検証するスイープ。
Plant time constantとcompensation gain、およびON/OFFの組み合わせをテスト。
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from scripts.sweeps.run_parameter_sweep import ParameterSweepConfig, run_sweep

# スイープパラメータの定義
sweep_params = {
    "CONTROLLER_TYPE": ["hohmann"],  # Hohmann transfer controller
    "PLANT_TIME_CONSTANT": [10.0, 50.0],  # Plant lag
    "INVERSE_COMPENSATION": [True, False],  # ON/OFF
    "INVERSE_COMPENSATION_GAIN": [10.0],  # Fixed gain
    "SIMULATION_TIME": [3000.0],  # ~50 min for Hohmann
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
    description="Hohmann Transfer with Inverse Compensation Sweep",
)

if __name__ == "__main__":
    import sys
    dry_run = "--dry-run" in sys.argv

    if dry_run:
        print("🔍 Dry run mode - showing configuration without execution\n")

    print("🚀 Hohmann transfer sweep - testing inverse compensation effect\n")
    run_sweep(config, dry_run=dry_run)
