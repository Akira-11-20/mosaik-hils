"""
Mosaik HILS (Hardware-in-the-Loop Simulation) メインファイル

このファイルは、数値シミュレーション、ハードウェアインターフェース、データ収集、
および可視化コンポーネントを統合したMosaikシナリオの設定と実行を行います。

主な機能:
- 数値シミュレーション（正弦波生成）
- ハードウェアシミュレーション（センサー読取り・アクチュエータ制御）
- データ収集とHDF5保存
- WebVis によるリアルタイム可視化
"""

from datetime import datetime
from pathlib import Path

import mosaik
import mosaik.util


def main():
    """
    Mosaik co-simulation scenario with numerical simulation and hardware interface

    この関数は以下の手順でシミュレーションを実行します:
    1. WebVisカスタマイズの自動適用
    2. シミュレーター設定の定義
    3. Mosaikワールドの作成
    4. 各シミュレーターの起動
    5. エンティティの作成と接続
    6. データ収集とWebVis可視化の設定
    7. シミュレーションの実行
    """

    # WebVis用のローカルアセットを事前にデプロイ
    print("🔧 Deploying WebVis local assets...")
    try:
        from scripts.manage_webvis_assets import deploy_assets

        deploy_assets()
    except Exception as e:
        print(f"⚠️  Asset deployment failed: {e}")

    # Simulation configuration - 各シミュレーターの設定
    sim_config = {
        # 数値シミュレーター: 正弦波を生成する数学的モデル
        "NumericalSim": {
            "python": "src.simulators.numerical_simulator:NumericalSimulator",
            "api_version": "1",
        },
        # ハードウェアシミュレーター: 物理デバイスとのインターフェース
        "HardwareSim": {
            "python": "src.simulators.hardware_simulator:HardwareSimulator",
            "api_version": "1",
        },
        # 遅延シミュレーター: 通信遅延とジッターをモデル化
        "DelaySim": {
            "python": "src.simulators.delay_simulator:DelaySimulator",
        },
        # Web可視化ツール: mosaik-web公式 (ポート8004)
        "WebVis": {
            "cmd": "mosaik-web %(addr)s --serve=127.0.0.1:8004",
        },
        # データ収集器: シミュレーションデータをJSONファイルに保存
        "DataCollector": {
            "python": "src.simulators.data_collector:DataCollectorSimulator",
            "api_version": "1",
        },
        # HDF5データベース: より高度なデータ記録（コメントアウト中）
        "HDF5": {
            "cmd": "mosaik-hdf5 %(addr)s",
        },
    }

    # Prepare run directory - ログ出力用ディレクトリを作成
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path("logs") / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    # Create World - Mosaikワールドの作成（全シミュレーターを管理）
    # debug=True enables execution graph tracking required by util plotting helpers.
    world = mosaik.World(sim_config, debug=True)

    # Start simulators - 各シミュレーターの起動
    # step_size=1: 各シミュレーターは1秒間隔で実行される
    numerical_sim = world.start("NumericalSim", step_size=1)
    hardware_sim = world.start("HardwareSim", step_size=1)
    # 遅延シミュレーター: 高頻度実行で精密な遅延制御（1時間単位）
    delay_sim = world.start("DelaySim", step_size=1, time_resolution=1)
    webvis = world.start("WebVis", start_date="2024-01-01 00:00:00", step_size=1)

    # Create entities - 各シミュレーター内でエンティティ（モデル）を作成
    # 数値モデル: 初期値1.0、ステップサイズ0.5で正弦波を生成
    numerical_model = numerical_sim.NumericalModel(initial_value=1.0, step_size=0.5)
    # ハードウェアインターフェース: センサー01をシリアル接続でシミュレート
    hardware_interface = hardware_sim.HardwareInterface(
        device_id="sensor_01", connection_type="serial"
    )
    # 遅延ノード: 通信遅延3ステップ、ジッター1ステップ、パケットロス0.1%
    delay_node = delay_sim.DelayNode(
        base_delay=3,  # 3ステップ基本遅延
        jitter_std=1,  # 1ステップ標準偏差ジッター
        packet_loss_rate=0.001,  # 0.1%パケットロス
        preserve_order=True,  # パケット順序保持
    )

    # Connect entities - エンティティ間のデータフロー接続
    # 遅延ノードを経由した通信パス: numerical → delay_node → hardware
    world.connect(numerical_model, delay_node, ("output", "input"))
    world.connect(
        delay_node, hardware_interface, ("delayed_output", "actuator_command")
    )

    # Data recording setup - カスタムデータ収集器のセットアップ
    data_collector = world.start("DataCollector", step_size=1)
    collector = data_collector.DataCollector(output_dir=str(run_dir))

    # Data collection setup - HDF5形式でのデータ記録設定
    # 全シミュレーターのデータをカスタムコレクターに集約
    mosaik.util.connect_many_to_one(world, [numerical_model], collector, "output")
    mosaik.util.connect_many_to_one(
        world, [delay_node], collector, "delayed_output", "stats"
    )
    mosaik.util.connect_many_to_one(
        world, [hardware_interface], collector, "sensor_value", "actuator_command"
    )

    # Note: 現在はカスタムDataCollectorを使用してHDF5保存を実装
    # 遅延ノードの統計情報や複雑なデータ型にも対応済み

    # WebVis setup - ポート8002でWeb可視化を設定
    vis_topo = None
    if webvis is not None:
        vis_topo = webvis.Topology()

        # Connect to visualization using many_to_one pattern - 可視化への接続
        # 数値モデルの出力を可視化に接続
        mosaik.util.connect_many_to_one(world, [numerical_model], vis_topo, "output")
        # ハードウェアインターフェースのセンサー値を可視化に接続
        mosaik.util.connect_many_to_one(
            world, [hardware_interface], vis_topo, "sensor_value"
        )
        # 遅延ノードの統計データと遅延出力を可視化に接続
        mosaik.util.connect_many_to_one(
            world, [delay_node], vis_topo, "stats", "delayed_output"
        )
        # mosaik.util.connect_many_to_one(
        #     world, [hardware_interface], vis_topo, "actuator_command"
        # )

    # Set entity types for visualization - 可視化のためのエンティティタイプ設定
    if webvis is not None:
        webvis.set_etypes(
            {
                # 数値モデル: 負荷として表示、出力値を-2から2の範囲で表示
                "NumericalModel": {
                    "cls": "load",  # 負荷クラス（青色で表示）
                    "attr": "output",  # 表示する属性
                    "unit": "Signal",  # 単位
                    "default": 0,  # デフォルト値
                    "min": -2,  # 最小値
                    "max": 2,  # 最大値
                },
                # ハードウェアインターフェース: 発電機として表示、センサー値を0から2Vの範囲で表示
                "HardwareInterface": {
                    "cls": "gen",  # 発電機クラス（緑色で表示）
                    "attr": "sensor_value",  # 表示する属性
                    "unit": "Sensor [V]",  # 単位（ボルト）
                    "default": 1,  # デフォルト値
                    "min": 0,  # 最小値
                    "max": 2,  # 最大値
                },
                # 遅延ノード: 制御装置として表示、遅延出力を表示
                "DelayNode": {
                    "cls": "ctrl",  # 制御装置クラス（オレンジ色で表示）
                    "attr": "delayed_output",  # 表示する属性
                    "unit": "Delayed Signal",  # 単位
                    "default": 0.0,  # デフォルト値
                    "min": -2,  # 最小値
                    "max": 2,  # 最大値
                },
            }
        )

    # Run simulation with progress monitoring - シミュレーション実行とプログレス監視
    print("Starting mosaik co-simulation...")  # Mosaikコシミュレーション開始
    print("Numerical simulator generates sine wave")  # 数値シミュレーターが正弦波を生成
    print(
        "Hardware simulator provides sensor feedback"
    )  # ハードウェアシミュレーターがセンサーフィードバックを提供
    if webvis is not None:
        print(
            "Official WebVis enabled at: http://localhost:8004"
        )  # 公式WebVisがhttp://localhost:8004で有効
    else:
        print("Official WebVis skipped (SKIP_MOSAIK_WEBVIS=1)")
    print(
        "Running for 300 simulation steps in slow real-time (10x slower)..."
    )  # 300シミュレーションステップをスローリアルタイムで実行
    if webvis is not None:
        print(
            "Visit http://localhost:8004 to see official mosaik visualization!"
        )  # 公式mosaik可視化を見るためのURL
    print("Press Ctrl+C to stop the simulation")  # シミュレーション停止の方法

    # Use mosaik.util for connection patterns - 接続パターンのためのmosaik.util使用
    # 必要に応じてutil関数で複数エンティティを接続
    # world.connectはmosaik.util.connect_randomlyやconnect_many_to_oneで置き換え可能

    # シミュレーション実行: 300ステップまで、リアルタイムファクター0.5（約2倍速）
    world.run(until=300, rt_factor=0.5)

    print("Co-simulation completed successfully!")  # コシミュレーション成功完了
    print(
        f"Simulation data recorded to: {run_dir / 'simulation_data.h5'}"
    )  # シミュレーションデータのHDF5ファイル保存先
    print("You can replay/analyze the data later!")  # 後からデータを再生/分析可能

    # Optional: Generate mosaik.util visualizations if dependencies are available
    figures_dir = run_dir

    try:
        plot_kwargs = {"folder": str(figures_dir), "show_plot": False}
        mosaik.util.plot_dataflow_graph(world, **plot_kwargs)
        mosaik.util.plot_execution_graph(world, **plot_kwargs)
        mosaik.util.plot_execution_time(world, **plot_kwargs)
        mosaik.util.plot_execution_time_per_simulator(world, **plot_kwargs)
        print(f"Additional plots saved under {figures_dir}/")
    except ImportError as exc:
        print(f"Optional mosaik visualizations skipped (missing dependency): {exc}")
    except Exception as exc:  # noqa: BLE001 - diagnostics for optional tooling
        print(f"Could not generate mosaik visualizations: {exc}")


if __name__ == "__main__":
    main()
