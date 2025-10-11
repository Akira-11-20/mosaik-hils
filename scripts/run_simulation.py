#!/usr/bin/env python3
"""
Simulation Runner Script
シミュレーション実行用のコンビニエンススクリプト

使用方法:
    python scripts/run_simulation.py                    # 通常実行
    python scripts/run_simulation.py --no-webvis       # WebVisなし
    python scripts/run_simulation.py --steps 100       # ステップ数指定
    python scripts/run_simulation.py --rt-factor 0.1   # リアルタイムファクター指定
"""

import sys
import os
import argparse
from pathlib import Path

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from main import main as simulation_main


def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser(description="Mosaik HILS Simulation Runner")
    parser.add_argument(
        "--no-webvis",
        action="store_true",
        help="Skip WebVis (set SKIP_MOSAIK_WEBVIS=1)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=300,
        help="Number of simulation steps (default: 300)",
    )
    parser.add_argument(
        "--rt-factor", type=float, default=0.5, help="Real-time factor (default: 0.5)"
    )
    parser.add_argument(
        "--delay", type=float, default=3.0, help="Base delay in steps (default: 3.0)"
    )
    parser.add_argument(
        "--jitter", type=float, default=1.0, help="Jitter std deviation (default: 1.0)"
    )
    parser.add_argument(
        "--packet-loss",
        type=float,
        default=0.001,
        help="Packet loss rate (default: 0.001)",
    )

    args = parser.parse_args()

    print("🚀 Starting Mosaik HILS Simulation")
    print(f"📊 Steps: {args.steps}")
    print(f"⏱️  RT Factor: {args.rt_factor}")
    print(f"🔄 Delay: {args.delay} steps")
    print(f"📈 Jitter: {args.jitter} std")
    print(f"📦 Packet Loss: {args.packet_loss * 100:.1f}%")
    print("-" * 50)

    # 環境変数設定
    if args.no_webvis:
        os.environ["SKIP_MOSAIK_WEBVIS"] = "1"
        print("🚫 WebVis disabled")
    else:
        os.environ.pop("SKIP_MOSAIK_WEBVIS", None)
        print("🌐 WebVis enabled at http://localhost:8002")

    # シミュレーションパラメータを環境変数で渡す
    os.environ["SIM_STEPS"] = str(args.steps)
    os.environ["SIM_RT_FACTOR"] = str(args.rt_factor)
    os.environ["SIM_DELAY"] = str(args.delay)
    os.environ["SIM_JITTER"] = str(args.jitter)
    os.environ["SIM_PACKET_LOSS"] = str(args.packet_loss)

    print("-" * 50)

    try:
        simulation_main()
    except KeyboardInterrupt:
        print("\n⏹️  Simulation interrupted by user")
    except Exception as e:
        print(f"\n❌ Simulation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
