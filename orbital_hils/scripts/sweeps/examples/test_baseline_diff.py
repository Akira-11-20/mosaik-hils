"""
Test Baseline Difference Plots

ベースライン差分プロットのテスト用スイープ。
小規模なformation flyingスイープでベースラインとの差分を確認。
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from scripts.sweeps.run_parameter_sweep import ParameterSweepConfig, run_sweep

# 小規模なスイープパラメータ（テスト用）
sweep_params = {
    "CONTROLLER_TYPE": ["formation"],
    "FORMATION_CONTROLLER_TYPE": ["hcw"],
    "PLANT_TIME_CONSTANT": [20.0],  # Single value for quick test
    "INVERSE_COMPENSATION": [True, False],
    "INVERSE_COMPENSATION_GAIN": [100.0],
    "FORMATION_OFFSET_X": [100.0],
    "SIMULATION_TIME": [500.0],  # Short simulation
    "TIME_RESOLUTION": [1.0],
}

# スイープ設定の作成
config = ParameterSweepConfig(
    sweep_params=sweep_params,
    base_env_file=".env",
    output_base_dir="results_sweep",
    description="Test: Baseline Difference Plots",
    include_baseline=True,  # ベースラインを追加
)

if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv

    if dry_run:
        print("🔍 Dry run mode - configuration preview only\n")

    print("🧪 Testing baseline difference plots with small sweep\n")
    run_sweep(config, dry_run=dry_run)

    if not dry_run:
        print("\n" + "=" * 70)
        print("📊 Test Complete!")
        print("=" * 70)
        print("\nCheck the comparison directory for baseline difference plots:")
        print("  - formation_baseline_difference.png")
        print("  - formation_baseline_position_difference.png")
        print("=" * 70)
