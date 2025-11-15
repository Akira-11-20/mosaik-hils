"""
Inverse Compensation Sweep Example

Inverse compensationの効果を検証するスイープ。
異なるplant time constantとcompensation gainの組み合わせをテスト。
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from scripts.sweeps.run_parameter_sweep import ParameterSweepConfig, run_sweep

# スイープパラメータの定義
sweep_params = {
    "PLANT_TIME_CONSTANT": [10.0, 20.0, 50.0, 100.0],
    "INVERSE_COMPENSATION": [True, False],
    "INVERSE_COMPENSATION_GAIN": [1.0, 2.0, 5.0, 10.0],
    "SIMULATION_TIME": [200.0],  # 短めのシミュレーション時間
    "MINIMAL_DATA_MODE": [True],  # データ量削減
}

# スイープ設定の作成
config = ParameterSweepConfig(
    sweep_params=sweep_params,
    base_env_file=".env",
    output_base_dir="results_sweep",
    description="Inverse Compensation Effect Sweep",
)

if __name__ == "__main__":
    # dry_run=True で設定確認のみ
    import sys
    dry_run = "--dry-run" in sys.argv

    if dry_run:
        print("🔍 Dry run mode - showing configuration without execution\n")

    run_sweep(config, dry_run=dry_run)
