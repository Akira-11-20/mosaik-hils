"""
HILS Simulation Main Scenario - 1DOF版

模擬HILS構成:
    Controller → Bridge(cmd) → Plant → Bridge(sense) → Env → Controller (time-shifted)

特徴:
- 1ms時間解像度
- cmd/sense経路で非対称な遅延設定
- time-shifted接続で循環依存を解決
- 初期実装: 補償機能なし
"""

from datetime import datetime
from pathlib import Path

import mosaik
import mosaik.util


# === SIMULATION CONFIGURATION ===

# 通信遅延設定
CMD_DELAY = 50          # 制御指令経路の遅延 [ms]
CMD_JITTER = 10         # 制御指令経路のジッター標準偏差 [ms]
CMD_LOSS_RATE = 0.01    # 制御指令経路のパケットロス率（1%）

SENSE_DELAY = 100       # 測定経路の遅延 [ms]
SENSE_JITTER = 20       # 測定経路のジッター標準偏差 [ms]
SENSE_LOSS_RATE = 0.02  # 測定経路のパケットロス率（2%）

# シミュレーション設定
SIMULATION_TIME = 0.5   # シミュレーション時間 [秒] = 0.5秒（テスト用）
TIME_RESOLUTION = 0.001 # 時間解像度 [秒/step] = 1step = 1ms
SIMULATION_STEP = int(SIMULATION_TIME / TIME_RESOLUTION)  # シミュレーションステップ数（0.5秒 / 0.001 = 500ステップ）
RT_FACTOR = None        # 実時間比率（None = 最高速、1.0 = 実時間、0.5 = 2倍速）

# 制御パラメータ
CONTROL_PERIOD = 10     # 制御周期 [ms]
KP = 20.0                # 比例ゲイン
KD = 5.0  # 微分ゲイン
TARGET_POSITION = 5.0  # 目標位置 [m]
MAX_THRUST = 100.0  # 最大推力 [N]

# 宇宙機パラメータ
SPACECRAFT_MASS = 100.0  # 質量 [kg]


def main():
    """
    HILS メインシナリオ
    """
    print("=" * 70)
    print("HILS Simulation - 1DOF Configuration")
    print("=" * 70)

    # ログディレクトリの作成
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path("results") / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Log directory: {run_dir}")

    # シミュレーター構成
    sim_config = {
        "ControllerSim": {
            "python": "simulators.controller_simulator:ControllerSimulator",
        },
        "PlantSim": {
            "python": "simulators.plant_simulator:PlantSimulator",
        },
        "EnvSim": {
            "python": "simulators.env_simulator:EnvSimulator",
        },
        "BridgeSim": {
            "python": "simulators.bridge_simulator:BridgeSimulator",
        },
        "DataCollector": {
            "python": "simulators.data_collector:DataCollectorSimulator",
        },
    }

    # Worldの作成（1ms精度）
    print(
        f"\n🌍 Creating Mosaik World (time_resolution={TIME_RESOLUTION}s = {TIME_RESOLUTION * 1000}ms)"
    )
    world = mosaik.World(
        sim_config,
        time_resolution=TIME_RESOLUTION,
        debug=True,
    )

    # シミュレーターの起動
    print("\n🚀 Starting simulators...")

    controller_sim = world.start(
        "ControllerSim",
        step_size=CONTROL_PERIOD,
    )  # 10ms周期
    plant_sim = world.start("PlantSim", step_size=1)  # 1ms周期
    env_sim = world.start("EnvSim", step_size=1)  # 1ms周期
    bridge_cmd_sim = world.start("BridgeSim", step_size=1)  # 1ms周期
    bridge_sense_sim = world.start("BridgeSim", step_size=1)  # 1ms周期

    # エンティティの作成
    print("\n📦 Creating entities...")

    # 制御器
    controller = controller_sim.PDController(
        kp=KP,
        kd=KD,
        target_position=TARGET_POSITION,
        max_thrust=MAX_THRUST,
        thrust_duration=CONTROL_PERIOD,
    )

    # 推力測定器
    plant = plant_sim.ThrustStand(stand_id="stand_01")

    # 宇宙機環境
    spacecraft = env_sim.Spacecraft1DOF(
        mass=SPACECRAFT_MASS,
        initial_position=0.0,
        initial_velocity=0.0,
    )

    # 通信ブリッジ（cmd経路）
    bridge_cmd = bridge_cmd_sim.CommBridge(
        bridge_type="cmd",
        base_delay=CMD_DELAY,
        jitter_std=CMD_JITTER,
        packet_loss_rate=CMD_LOSS_RATE,
        preserve_order=True,
    )

    # 通信ブリッジ（sense経路）
    bridge_sense = bridge_sense_sim.CommBridge(
        bridge_type="sense",
        base_delay=SENSE_DELAY,
        jitter_std=SENSE_JITTER,
        packet_loss_rate=SENSE_LOSS_RATE,
        preserve_order=True,
    )

    # データフローの接続
    print("\n🔗 Connecting data flows...")

    # 1. Controller → Bridge(cmd) - 制御指令経路（通常接続）
    world.connect(
        controller,
        bridge_cmd,
        ("command", "input"),
    )

    # 2. Bridge(cmd) → Plant - 遅延後の制御指令（パッケージ化コマンド）
    world.connect(
        bridge_cmd,
        plant,
        ("delayed_output", "command"),
    )

    # 3. Plant → Bridge(sense) - 測定値経路
    world.connect(
        plant,
        bridge_sense,
        ("measured_thrust", "input"),
    )

    # 4. Bridge(sense) → Env - 遅延後の測定値
    world.connect(
        bridge_sense,
        spacecraft,
        ("delayed_output", "force"),
    )

    # 5. Env → Controller - 状態フィードバック（time-shifted for circular dependency）
    print("   ⏱️  Using time-shifted connection for Env → Controller (resolves circular dependency)")
    world.connect(
        spacecraft,
        controller,
        "position",
        "velocity",
        time_shifted=True,
        initial_data={
            "position": 0.0,
            "velocity": 0.0,
        },
    )

    # 6. データ収集の設定
    print("\n📊 Setting up data collection...")
    data_collector_sim = world.start("DataCollector", step_size=1)
    collector = data_collector_sim.Collector(output_dir=str(run_dir))

    # 全エンティティからデータを収集
    mosaik.util.connect_many_to_one(
        world,
        [controller],
        collector,
        "command",
        "error",
    )
    mosaik.util.connect_many_to_one(
        world,
        [bridge_cmd],
        collector,
        "stats",
    )
    mosaik.util.connect_many_to_one(
        world,
        [plant],
        collector,
        "measured_thrust",
        "status",
    )
    mosaik.util.connect_many_to_one(
        world,
        [bridge_sense],
        collector,
        "stats",
    )
    mosaik.util.connect_many_to_one(
        world,
        [spacecraft],
        collector,
        "position",
        "velocity",
        "acceleration",
        "force",
    )

    print("\n✅ Data flow configured:")
    print("   Controller → Bridge(cmd) → Plant → Bridge(sense) → Env")
    print("   Env → Controller (time-shifted: resolves circular dependency + step_size sync)")
    print("   All data → DataCollector → HDF5")
    print("   ℹ️  Command format: JSON/dict {thrust, duration}")

    # シミュレーション実行
    print(
        f"\n▶️  Running simulation until {SIMULATION_TIME}s ({SIMULATION_STEP} steps)..."
    )
    print("=" * 70)

    world.run(until=SIMULATION_STEP, rt_factor=RT_FACTOR)

    print("=" * 70)
    print("✅ Simulation completed successfully!")

    # 実行グラフの生成（オプション）
    print(f"\n📊 Generating execution graphs...")
    try:
        plot_kwargs = {
            "folder": str(run_dir),
            "show_plot": False,
        }
        mosaik.util.plot_dataflow_graph(world, **plot_kwargs)
        # mosaik.util.plot_execution_graph(world, **plot_kwargs)
        mosaik.util.plot_execution_time(world, **plot_kwargs)
        print(f"   Graphs saved to {run_dir}/")
    except Exception as e:
        print(f"   ⚠️  Graph generation failed: {e}")

    print("\n" + "=" * 70)
    print("HILS Simulation Finished")
    print("=" * 70)


if __name__ == "__main__":
    main()
