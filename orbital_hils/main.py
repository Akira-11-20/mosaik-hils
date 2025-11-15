"""
Orbital HILS Simulation - Main Entry Point

6DOF軌道力学シミュレーション with 制御フィードバックループ

.envファイルのCONTROLLER_TYPEに応じて、適切なシナリオを自動選択:
- zero: 自由軌道運動（デフォルト）
- pd: PD制御
- hohmann: ホーマン遷移制御

使用方法:
    cd /home/akira/mosaik-hils/orbital_hils
    uv run python main.py
"""

import sys
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent))

from config.orbital_parameters import get_env_param, load_config_from_env
from scenarios.hohmann_scenario import HohmannScenario
from scenarios.orbital_scenario import OrbitalScenario


def main():
    """メインエントリーポイント"""
    print("=" * 70)
    print("Orbital HILS Simulation")
    print("6DOF Orbital Dynamics with Control Feedback Loop")
    print("=" * 70)
    print()

    # .envからコントローラータイプを読み込み
    controller_type = get_env_param("CONTROLLER_TYPE", "zero", str)
    print(f"Controller Type: {controller_type}")
    print()
    print("Architecture:")
    print("  OrbitalEnv (RK4) → OrbitalController → OrbitalPlant → OrbitalEnv")
    print("  All components → DataCollector → HDF5")
    print("=" * 70)
    print()

    # .envファイルから設定を読み込み
    print("Loading configuration from .env...")
    config = load_config_from_env()
    print()

    # コントローラータイプに応じてシナリオを選択
    if controller_type == "hohmann":
        print("🚀 Hohmann Transfer Scenario Selected")
        print()
        scenario = HohmannScenario(config=config)
    elif controller_type == "pd":
        print("🎯 PD Control Scenario Selected")
        print("   (Using base OrbitalScenario with PD controller)")
        print()
        scenario = OrbitalScenario(config=config)
    else:
        print("🌌 Free Orbit Scenario Selected (zero thrust)")
        print()
        scenario = OrbitalScenario(config=config)

    output_dir = scenario.run()

    print("\n" + "=" * 70)
    print("✅ Simulation Complete!")
    print("=" * 70)
    print(f"📁 Results directory: {output_dir}")
    print(f"📊 HDF5 data file: {output_dir / 'hils_data.h5'}")
    print()

    if controller_type == "hohmann":
        print("💡 Hohmann transfer plots (PNG & HTML) were auto-generated!")
        print()

    print("Next steps:")
    print("  1. Visualize results:")
    print(f"     uv run python scripts/analysis/visualize_orbital_results.py {output_dir / 'hils_data.h5'}")
    print("  2. Interactive 3D plot:")
    print(f"     uv run python scripts/analysis/visualize_orbital_interactive.py {output_dir / 'hils_data.h5'}")
    if controller_type == "hohmann":
        print("  3. Phase-colored plots (already generated):")
        print(f"     {output_dir}/orbital_3d_trajectory_phases.png")
        print(f"     {output_dir}/orbital_3d_trajectory_phases_interactive.html")
    print("=" * 70)


if __name__ == "__main__":
    main()
