"""
Quick Test Sweep Example

動作確認用の小規模スイープ。
短いシミュレーション時間で2-3パラメータのみをテスト。
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from scripts.sweeps.run_parameter_sweep import ParameterSweepConfig, run_sweep

# スイープパラメータの定義（小規模テスト用）
sweep_params = {
    "PLANT_TIME_CONSTANT": [10.0, 20.0],  # 2値のみ
    "SIMULATION_TIME": [50.0],            # 50秒の短いシミュレーション
    "TIME_RESOLUTION": [1.0],             # 1秒刻み
    "MINIMAL_DATA_MODE": [True],          # データ量削減
    "AUTO_VISUALIZE": [False],            # 可視化無効
}

# スイープ設定の作成
config = ParameterSweepConfig(
    sweep_params=sweep_params,
    base_env_file=".env",
    output_base_dir="results_sweep",
    description="Quick Test Sweep (for verification)",
)

if __name__ == "__main__":
    import sys
    dry_run = "--dry-run" in sys.argv

    if dry_run:
        print("🔍 Dry run mode - showing configuration without execution\n")

    print("⚡ Quick test sweep - 2 configurations, 50s each\n")
    run_sweep(config, dry_run=dry_run)
