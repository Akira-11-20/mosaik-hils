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

import os
from datetime import datetime
from pathlib import Path

import mosaik
import mosaik.util


def main():
    """
    Mosaik co-simulation scenario with numerical simulation and hardware interface

    この関数は以下の手順でシミュレーションを実行します:
    1. シミュレーター設定の定義
    2. Mosaikワールドの作成
    3. 各シミュレーターの起動
    4. エンティティの作成と接続
    5. データ収集とWebVis可視化の設定
    6. シミュレーションの実行
    """

    # Simulation configuration - 各シミュレーターの設定
    sim_config = {
        # 数値シミュレーター: 正弦波を生成する数学的モデル
        "NumericalSim": {
            "python": "numerical_simulator:NumericalSimulator",
        },
        # ハードウェアシミュレーター: 物理デバイスとのインターフェース
        "HardwareSim": {
            "python": "hardware_simulator:HardwareSimulator",
        },
        # Web可視化ツール: リアルタイムでシミュレーション結果を表示
        "WebVis": {
            "cmd": "mosaik-web %(addr)s --serve=127.0.0.1:8002",
        },
        # データ収集器: シミュレーションデータをJSONファイルに保存
        "DataCollector": {
            "python": "data_collector:DataCollectorSimulator",
        },
        # HDF5データベース: より高度なデータ記録（コメントアウト中）
        "HDF5": {
            "cmd": "mosaik-hdf5 %(addr)s",
        },
    }

    skip_official_webvis = os.getenv("SKIP_MOSAIK_WEBVIS", "0") == "1"
    if skip_official_webvis:
        sim_config.pop("WebVis")

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
    webvis = None
    if not skip_official_webvis:
        webvis = world.start("WebVis", start_date="2024-01-01 00:00:00", step_size=1)

    # Create entities - 各シミュレーター内でエンティティ（モデル）を作成
    # 数値モデル: 初期値1.0、ステップサイズ0.5で正弦波を生成
    numerical_model = numerical_sim.NumericalModel(initial_value=1.0, step_size=0.5)
    # ハードウェアインターフェース: センサー01をシリアル接続でシミュレート
    hardware_interface = hardware_sim.HardwareInterface(
        device_id="sensor_01", connection_type="serial"
    )

    # Connect entities - エンティティ間のデータフロー接続
    # 数値シミュレーションの出力をハードウェアのアクチュエータコマンドに接続
    # (循環参照を避けるため、一方向のみの接続)
    world.connect(numerical_model, hardware_interface, ("output", "actuator_command"))

    # Data recording setup - カスタムデータ収集器のセットアップ
    data_collector = world.start("DataCollector", step_size=1)
    collector = data_collector.DataCollector(output_dir=str(run_dir))

    # Connect all data to collector for recording - データ収集のための接続
    # 数値モデルの出力を収集器に接続
    mosaik.util.connect_many_to_one(world, [numerical_model], collector, "output")
    # ハードウェアインターフェースのセンサー値とアクチュエータコマンドを収集器に接続
    mosaik.util.connect_many_to_one(
        world, [hardware_interface], collector, "sensor_value", "actuator_command"
    )

    # Alternative: HDF5 database for more complex data recording
    # より複雑なデータ記録にはHDF5データベースも使用可能（現在はコメントアウト）
    # hdf5_db = world.start('HDF5', step_size=1, duration=300)
    # hdf5_output = hdf5_db.Database(filename='simulation_recording.hdf5')
    # mosaik.util.connect_many_to_one(world, [numerical_model], hdf5_output, 'output')
    # mosaik.util.connect_many_to_one(world, [hardware_interface], hdf5_output, 'sensor_value', 'actuator_command')

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
            "Official WebVis enabled at: http://localhost:8002"
        )  # 公式WebVisがhttp://localhost:8002で有効
    else:
        print("Official WebVis skipped (SKIP_MOSAIK_WEBVIS=1)")
    print(
        "Running for 300 simulation steps in slow real-time (10x slower)..."
    )  # 300シミュレーションステップをスローリアルタイムで実行
    if webvis is not None:
        print(
            "Visit http://localhost:8002 to see official mosaik visualization!"
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
