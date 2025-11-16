"""
Formation Flying Parameter Sweep - 編隊飛行制御のパラメータスイープ

Chaser-Target編隊飛行における以下のパラメータの影響を評価:
- 制御方式（HCW vs PD）
- 初期オフセット
- 制御ゲイン
- Plant遅れ（τ）
- 逆補償の効果

使用方法:
    cd /home/akira/mosaik-hils/orbital_hils
    uv run python scripts/sweeps/examples/sweep_formation_flying.py
"""

import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.sweeps.run_parameter_sweep import ParameterSweepConfig, run_sweep

# ============================================================================
# Formation Flying用のスイープ設定
# ============================================================================

# Example 1: 制御方式とオフセットの比較
sweep_params_control_type = {
    "CONTROLLER_TYPE": ["formation"],
    "FORMATION_CONTROLLER_TYPE": ["hcw", "pd"],  # HCW vs PD
    "FORMATION_OFFSET_X": [50.0, 100.0, 200.0],  # 初期オフセット [m]
    "SIMULATION_TIME": [1000.0],  # 短時間テスト
    "TIME_RESOLUTION": [1.0],
}

# Example 2: Plant遅れの影響評価
sweep_params_plant_lag = {
    "CONTROLLER_TYPE": ["formation"],
    "FORMATION_CONTROLLER_TYPE": ["hcw"],
    "PLANT_TIME_CONSTANT": [0.0, 5.0, 10.0, 20.0, 50.0],  # Plant τ [s]
    "PLANT_NOISE_STD": [0.0, 0.01],  # ノイズレベル
    "FORMATION_OFFSET_X": [100.0],
    "SIMULATION_TIME": [1000.0],
}

# Example 3: 逆補償の効果（HCW制御）
sweep_params_inverse_comp = {
    "CONTROLLER_TYPE": ["formation"],
    "FORMATION_CONTROLLER_TYPE": ["hcw"],
    "PLANT_TIME_CONSTANT": [20.0],  # 大きめの遅れ
    "INVERSE_COMPENSATION": [True, False],  # 逆補償ON/OFF
    "INVERSE_COMPENSATION_GAIN": [50.0, 100.0, 200.0],  # 補償ゲイン
    "FORMATION_OFFSET_X": [100.0],
    "SIMULATION_TIME": [2000.0],
}

# Example 4: 制御ゲインのチューニング
sweep_params_gain_tuning = {
    "CONTROLLER_TYPE": ["formation"],
    "FORMATION_CONTROLLER_TYPE": ["hcw"],
    "CONTROL_GAIN": [0.001, 0.01, 0.1, 1.0, 10.0],  # 制御ゲイン
    "FORMATION_OFFSET_X": [100.0],
    "PLANT_TIME_CONSTANT": [10.0],
    "SIMULATION_TIME": [2000.0],
}

# Example 5: 初期オフセットと制御ゲインの組み合わせ
sweep_params_offset_gain = {
    "CONTROLLER_TYPE": ["formation"],
    "FORMATION_CONTROLLER_TYPE": ["hcw"],
    "FORMATION_OFFSET_X": [50.0, 100.0, 200.0, 500.0],  # 初期オフセット
    "FORMATION_OFFSET_Y": [0.0, 50.0],  # Y方向オフセット
    "CONTROL_GAIN": [0.01, 0.1, 1.0],
    "SIMULATION_TIME": [2000.0],
}

# Example 6: 完全なパラメータスタディ（時間注意）
sweep_params_full_study = {
    "CONTROLLER_TYPE": ["formation"],
    "FORMATION_CONTROLLER_TYPE": ["hcw", "pd"],
    "FORMATION_OFFSET_X": [100.0, 200.0],
    "CONTROL_GAIN": [0.1, 1.0],
    "PLANT_TIME_CONSTANT": [0.0, 10.0, 20.0],
    "INVERSE_COMPENSATION": [True, False],
    "INVERSE_COMPENSATION_GAIN": [100.0],
    "SIMULATION_TIME": [1000.0],
}

# ============================================================================
# スイープの選択と実行
# ============================================================================

if __name__ == "__main__":
    # 実行するスイープを選択（ここを変更）
    sweep_choice = (
        "control_type"  # オプション: control_type, plant_lag, inverse_comp, gain_tuning, offset_gain, full_study
    )

    sweep_params_map = {
        "control_type": (sweep_params_control_type, "Formation: Control Type & Offset Comparison"),
        "plant_lag": (sweep_params_plant_lag, "Formation: Plant Lag Effect"),
        "inverse_comp": (sweep_params_inverse_comp, "Formation: Inverse Compensation Effect"),
        "gain_tuning": (sweep_params_gain_tuning, "Formation: Control Gain Tuning"),
        "offset_gain": (sweep_params_offset_gain, "Formation: Offset & Gain Study"),
        "full_study": (sweep_params_full_study, "Formation: Full Parameter Study"),
    }

    if sweep_choice not in sweep_params_map:
        print(f"❌ Invalid sweep choice: {sweep_choice}")
        print(f"   Available options: {list(sweep_params_map.keys())}")
        sys.exit(1)

    sweep_params, description = sweep_params_map[sweep_choice]

    # スイープ設定の作成
    config = ParameterSweepConfig(
        sweep_params=sweep_params,
        base_env_file=".env",
        output_base_dir="results_sweep",
        description=description,
    )

    # コマンドライン引数チェック
    dry_run = "--dry-run" in sys.argv

    if dry_run:
        print("🔍 Dry run mode - configuration preview only\n")

    # 実行
    run_sweep(config, dry_run=dry_run)

    print("\n" + "=" * 70)
    print("📊 Formation Flying Sweep Complete!")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Check sweep results in results_sweep/YYYYMMDD-HHMMSS_sweep/")
    print("  2. View comparison plots (auto-generated)")
    print("  3. Analyze formation metrics:")
    print("     - Relative position convergence")
    print("     - Control effort (thrust magnitude)")
    print("     - Tracking error vs time")
    print("=" * 70)
