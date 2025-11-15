"""
Orbital HILS Simulation - Main Entry Point

6DOF軌道力学シミュレーション with 制御フィードバックループ

使用方法:
    cd /home/akira/mosaik-hils/orbital_hils
    uv run python main.py
"""

import sys
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent))

from config.orbital_parameters import load_config_from_env
from scenarios.orbital_scenario import OrbitalScenario


def main():
    """メインエントリーポイント"""
    print("=" * 70)
    print("Orbital HILS Simulation")
    print("6DOF Orbital Dynamics with Control Feedback Loop")
    print("=" * 70)
    print()
    print("Architecture:")
    print("  OrbitalEnv (RK4) → OrbitalController → OrbitalPlant → OrbitalEnv")
    print("  All components → DataCollector → HDF5")
    print("=" * 70)
    print()

    # .envファイルから設定を読み込み（存在しない場合はデフォルトISS設定）
    print("Loading configuration...")
    config = load_config_from_env()
    print()

    scenario = OrbitalScenario(config=config)
    output_dir = scenario.run()

    print("\n" + "=" * 70)
    print("✅ Simulation Complete!")
    print("=" * 70)
    print(f"📁 Results directory: {output_dir}")
    print(f"📊 HDF5 data file: {output_dir / 'hils_data.h5'}")
    print()
    print("Next steps:")
    print("  1. Visualize results:")
    print(f"     uv run python scripts/analysis/visualize_orbital_results.py {output_dir / 'hils_data.h5'}")
    print("  2. Interactive 3D plot:")
    print(f"     uv run python scripts/analysis/visualize_orbital_interactive.py {output_dir / 'hils_data.h5'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
