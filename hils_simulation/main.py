"""
HILS Simulation - Unified Entry Point

このスクリプトは、複数のシミュレーションシナリオを統一されたインターフェースで実行するための
メインエントリーポイントです。

Available Scenarios:
    hils          - HILS (Hardware-in-the-Loop Simulation)
                    通信遅延を含む完全なHILS構成

    rt            - RT (Real-Time)
                    通信遅延なしのベースライン比較用

    inverse_comp  - HILS + Inverse Compensation
                    逆補償を用いた遅延補償の評価

    pure_python   - Pure Python
                    Mosaikフレームワークなしの純粋なPython実装

Usage:
    python main.py [scenario]

Examples:
    python main.py              # デフォルト（HILS）
    python main.py hils         # HILS明示的指定
    python main.py rt           # RT
    python main.py inverse_comp # 逆補償
    python main.py pure_python  # Pure Python
    python main.py --help       # ヘルプ表示

Configuration:
    シミュレーションパラメータは .env ファイルから読み込まれます。
    詳細は README_scenarios.md を参照してください。

Architecture:
    このスクリプトは scenarios/ モジュールの各シナリオクラスを呼び出します。
    詳細なアーキテクチャについては V2_ARCHITECTURE.md を参照してください。
"""

import argparse
import sys
from typing import Optional

from config.parameters import SimulationParameters
from scenarios import (
    HILSScenario,
    RTScenario,
    InverseCompScenario,
    PurePythonScenario,
)


def create_parser():
    """Create argument parser with scenario choices."""
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="HILS Simulation - Unified Entry Point for multiple simulation scenarios",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py              # Default (HILS)
  python main.py h            # HILS (shortcut)
  python main.py r            # RT (shortcut)
  python main.py i            # Inverse Compensation (shortcut)
  python main.py p            # Pure Python (shortcut)

  # Full names also work:
  python main.py hils
  python main.py rt
  python main.py inverse_comp
  python main.py pure_python

Documentation:
  README_scenarios.md   - Quick start guide
  V2_ARCHITECTURE.md    - Architecture details

Scenario Descriptions:
  hils          - HILS with communication delays
                  通信遅延を含む完全なHILS構成

  rt            - Real-time simulation without delays
                  通信遅延なしのベースライン比較用

  inverse_comp  - HILS with inverse compensation
                  逆補償を用いた遅延補償の評価

  pure_python   - Pure Python simulation (no Mosaik)
                  Mosaikフレームワークなしの純粋なPython実装
        """,
    )

    parser.add_argument(
        "scenario",
        nargs="?",
        default="h",
        choices=["h", "hils", "r", "rt", "i", "inverse_comp", "p", "pure_python"],
        help="Simulation scenario to run (default: h=hils). Shortcuts: h=hils, r=rt, i=inverse_comp, p=pure_python",
    )

    return parser


def get_scenario(scenario_name: str, params: Optional[SimulationParameters] = None):
    """
    Get scenario instance by name.

    Args:
        scenario_name: Name of the scenario (accepts shortcuts: h, r, i, p)
        params: Simulation parameters (if None, loads from environment)

    Returns:
        Scenario instance

    Raises:
        ValueError: If scenario name is invalid
    """
    # Shortcut mapping (1 character aliases)
    shortcuts = {
        "h": "hils",
        "r": "rt",
        "i": "inverse_comp",
        "p": "pure_python",
    }

    # Scenario class mapping
    scenario_map = {
        "hils": HILSScenario,
        "rt": RTScenario,
        "inverse_comp": InverseCompScenario,
        "pure_python": PurePythonScenario,
    }

    # Resolve shortcut to full name
    resolved_name = shortcuts.get(scenario_name.lower(), scenario_name.lower())

    scenario_class = scenario_map.get(resolved_name)
    if scenario_class is None:
        raise ValueError(f"Invalid scenario: {scenario_name}. Available scenarios: {', '.join(scenario_map.keys())} (shortcuts: h, r, i, p)")

    return scenario_class(params)


def main():
    """Main entry point."""
    # Create argument parser
    parser = create_parser()

    # Parse command line arguments
    args = parser.parse_args()

    # Load parameters from environment
    params = SimulationParameters.from_env()

    # Create and run scenario
    try:
        scenario = get_scenario(args.scenario, params)
        scenario.run()
    except Exception as e:
        print(f"\n❌ Simulation failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
