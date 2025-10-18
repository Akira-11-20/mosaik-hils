"""
HILS Simulation with Inverse Compensation - 1DOF版

This script demonstrates command inverse compensation in HILS setup.
It extends main_hils.py by adding an InverseCompensator in the command path.

模擬HILS構成（逆補償あり）:
    Env → Controller → [Inverse Comp] → Bridge(cmd) → Plant → Bridge(sense) → Env

特徴:
- Command inverse compensation: 制御指令の遅延を事前補償
- 従来のmain_hils.pyと比較可能な独立スクリプト
- 補償ゲインはパラメータで調整可能
- 1ms時間解像度、非対称な遅延設定をサポート
"""

import json
import os
from datetime import datetime
from pathlib import Path

import mosaik
import mosaik.util
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def get_env_float(key: str, default: float) -> float:
    """Get float value from environment variable"""
    value = os.getenv(key)
    if value is None or value == "None":
        return default
    return float(value)


# === SIMULATION CONFIGURATION (loaded from .env) ===

# Communication delays
CMD_DELAY = get_env_float("CMD_DELAY", 20)  # Command path delay [ms]
CMD_JITTER = get_env_float("CMD_JITTER", 0)  # Command path jitter std [ms]
CMD_LOSS_RATE = get_env_float("CMD_LOSS_RATE", 0.0)  # Command path packet loss rate

SENSE_DELAY = get_env_float("SENSE_DELAY", 30)  # Sensing path delay [ms]
SENSE_JITTER = get_env_float("SENSE_JITTER", 0.0)  # Sensing path jitter std [ms]
SENSE_LOSS_RATE = get_env_float("SENSE_LOSS_RATE", 0.0)  # Sensing path packet loss rate

# Inverse compensation parameters
ENABLE_INVERSE_COMP = os.getenv("ENABLE_INVERSE_COMP", "True").lower() == "true"
INVERSE_COMP_GAIN = get_env_float("INVERSE_COMP_GAIN", 15.0)  # Compensation gain

# Simulation settings
SIMULATION_TIME = get_env_float("SIMULATION_TIME", 2)  # Simulation time [s]
TIME_RESOLUTION = get_env_float("TIME_RESOLUTION", 0.0001)  # Time resolution [s/step]
SIMULATION_STEPS = int(SIMULATION_TIME / TIME_RESOLUTION)
RT_FACTOR_STR = os.getenv("RT_FACTOR", "None")
RT_FACTOR = None if RT_FACTOR_STR == "None" else float(RT_FACTOR_STR)

# Control parameters
CONTROL_PERIOD = get_env_float("CONTROL_PERIOD", 10)  # Control period [ms]
KP = get_env_float("KP", 15.0)  # Proportional gain
KD = get_env_float("KD", 5.0)  # Derivative gain
TARGET_POSITION = get_env_float("TARGET_POSITION", 5.0)  # Target position [m]
MAX_THRUST = get_env_float("MAX_THRUST", 100.0)  # Maximum thrust [N]

# Simulator periods [steps]
ENV_SIM_PERIOD_MS = get_env_float("ENV_SIM_PERIOD", 10)  # [ms]
PLANT_SIM_PERIOD_MS = get_env_float("PLANT_SIM_PERIOD", 10)  # [ms]
ENV_SIM_PERIOD = int(ENV_SIM_PERIOD_MS / 1000 / TIME_RESOLUTION)  # Convert ms to steps
PLANT_SIM_PERIOD = int(PLANT_SIM_PERIOD_MS / 1000 / TIME_RESOLUTION)  # Convert ms to steps

# Spacecraft parameters
SPACECRAFT_MASS = get_env_float("SPACECRAFT_MASS", 1.0)  # Mass [kg]
INITIAL_POSITION = get_env_float("INITIAL_POSITION", 0.0)  # Initial position [m]
INITIAL_VELOCITY = get_env_float("INITIAL_VELOCITY", 10.0)  # Initial velocity [m/s]
GRAVITY = get_env_float("GRAVITY", 9.81)  # Gravity acceleration [m/s^2]


def save_simulation_config(output_dir: Path):
    """
    シミュレーション設定をJSON形式で保存

    Args:
        output_dir: 出力ディレクトリ
    """
    config = {
        "simulation": {
            "simulation_time_s": SIMULATION_TIME,
            "time_resolution_s": TIME_RESOLUTION,
            "simulation_steps": SIMULATION_STEPS,
            "rt_factor": RT_FACTOR,
        },
        "communication": {
            "cmd_delay_s": CMD_DELAY / 1000.0,  # ms → s
            "cmd_jitter_s": CMD_JITTER / 1000.0,  # ms → s
            "cmd_loss_rate": CMD_LOSS_RATE,
            "sense_delay_s": SENSE_DELAY / 1000.0,  # ms → s
            "sense_jitter_s": SENSE_JITTER / 1000.0,  # ms → s
            "sense_loss_rate": SENSE_LOSS_RATE,
        },
        "inverse_compensation": {
            "enabled": ENABLE_INVERSE_COMP,
            "gain": INVERSE_COMP_GAIN,
            "type": "command",
            "note": "Applied before command delay to pre-shape signal",
        },
        "control": {
            "control_period_s": CONTROL_PERIOD / 1000.0,  # ms → s
            "kp": KP,
            "kd": KD,
            "target_position_m": TARGET_POSITION,
            "max_thrust_N": MAX_THRUST,
        },
        "simulators": {
            "env_sim_period_s": ENV_SIM_PERIOD * TIME_RESOLUTION,  # steps → s
            "plant_sim_period_s": PLANT_SIM_PERIOD * TIME_RESOLUTION,  # steps → s
        },
        "spacecraft": {
            "mass_kg": SPACECRAFT_MASS,
        },
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "description": "HILS 1-DOF Spacecraft Control with Command Inverse Compensation",
            "note": "All time units are in seconds (s)",
        },
    }

    config_path = output_dir / "simulation_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"💾 Configuration saved: {config_path}")
    return config_path


def main():
    """
    HILS メインシナリオ（逆補償あり）
    """
    print("=" * 70)
    print("HILS Simulation - 1DOF with Inverse Compensation")
    print("=" * 70)

    # ログディレクトリの作成
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path("results") / f"{timestamp}_inverse_comp"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Log directory: {run_dir}")

    # シミュレーション設定の保存
    save_simulation_config(run_dir)

    # シミュレーター構成
    sim_config = {
        "ControllerSim": {
            "python": "simulators.controller_simulator:ControllerSimulator",
        },
        "InverseCompSim": {
            "python": "simulators.inverse_compensator_simulator:InverseCompensatorSimulator",
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
        debug=False,  # デバッグモード無効化（高速化）
    )

    # シミュレーターの起動
    print("\n🚀 Starting simulators...")

    controller_sim = world.start(
        "ControllerSim",
        step_size=int(CONTROL_PERIOD / 1000 / TIME_RESOLUTION),  # 10ms → steps
    )

    # Inverse compensator (only if enabled)
    if ENABLE_INVERSE_COMP:
        # Compensator should run at same rate as controller to capture signal changes
        inverse_comp_sim = world.start(
            "InverseCompSim",
            step_size=int(CONTROL_PERIOD / 1000 / TIME_RESOLUTION)  # Same as controller
        )
        print(f"   ✨ Inverse Compensator enabled (gain={INVERSE_COMP_GAIN}, period={CONTROL_PERIOD}ms)")

    plant_sim = world.start("PlantSim", step_size=PLANT_SIM_PERIOD)
    env_sim = world.start("EnvSim", step_size=ENV_SIM_PERIOD)
    bridge_cmd_sim = world.start(
        "BridgeSim", step_size=1, log_dir=str(run_dir)
    )
    bridge_sense_sim = world.start(
        "BridgeSim", step_size=1, log_dir=str(run_dir)
    )  # 1ms周期

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

    # 逆補償器（有効な場合のみ）
    if ENABLE_INVERSE_COMP:
        inverse_comp = inverse_comp_sim.InverseCompensator(
            comp_id="cmd_compensator",
            gain=INVERSE_COMP_GAIN,
            comp_type="command",
        )

    # 推力測定器
    plant = plant_sim.ThrustStand(stand_id="stand_01")

    # 宇宙機環境
    spacecraft = env_sim.Spacecraft1DOF(
        mass=SPACECRAFT_MASS,
        initial_position=INITIAL_POSITION,
        initial_velocity=INITIAL_VELOCITY,
        gravity=GRAVITY,
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

    if ENABLE_INVERSE_COMP:
        # 【逆補償あり】Controller → InverseComp → Bridge(cmd) → Plant
        print(
            "   ⏱️  Controller → Inverse Compensator: time-shifted connection"
        )
        world.connect(
            controller,
            inverse_comp,
            ("command", "input"),
            time_shifted=True,
            initial_data={"command": {"thrust": 0.0, "duration": CONTROL_PERIOD}},
        )

        print(
            "   ✨ Inverse Compensator → Bridge(cmd): compensated command path"
        )
        world.connect(
            inverse_comp,
            bridge_cmd,
            ("compensated_output", "input"),
        )
    else:
        # 【逆補償なし】Controller → Bridge(cmd) → Plant (従来通り)
        print(
            "   ⏱️  Controller → Bridge(cmd): time-shifted connection (no compensation)"
        )
        world.connect(
            controller,
            bridge_cmd,
            ("command", "input"),
            time_shifted=True,
            initial_data={"command": {"thrust": 0.0, "duration": CONTROL_PERIOD}},
        )

    # 2. Bridge(cmd) → Plant - 遅延後の制御指令
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

    # 5. Env → Controller - 状態フィードバック（同一ステップで送信）
    print("   📡 Env → Controller: same-step connection (state feedback)")
    world.connect(
        spacecraft,
        controller,
        "position",
        "velocity",
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

    if ENABLE_INVERSE_COMP:
        mosaik.util.connect_many_to_one(
            world,
            [inverse_comp],
            collector,
            "compensated_output",
            "stats",
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
    print("   Env → Controller (same step)")
    if ENABLE_INVERSE_COMP:
        print(
            "   Controller → [Inverse Comp] → Bridge(cmd) → Plant (time-shifted)"
        )
    else:
        print("   Controller → Bridge(cmd) → Plant (time-shifted)")
    print("   Plant → Bridge(sense) → Env")
    print("   All data → DataCollector → HDF5")
    print("   ℹ️  Command format: JSON/dict {thrust, duration}")

    # シミュレーション実行
    print(
        f"\n▶️  Running simulation until {SIMULATION_TIME}s ({SIMULATION_STEPS} steps)..."
    )
    print("=" * 70)

    world.run(until=SIMULATION_STEPS, rt_factor=RT_FACTOR)

    print("=" * 70)
    print("✅ Simulation completed successfully!")

    # 実行グラフの生成（オプション）
    print(f"\n📊 Generating execution graphs...")
    try:
        from utils.plot_utils import (
            plot_execution_graph_with_data_only,
            plot_dataflow_graph_custom,
        )

        plot_kwargs = {
            "folder": str(run_dir),
            "show_plot": False,
        }

        # データフローグラフ（カスタム版 - サイズ調整可能）
        plot_dataflow_graph_custom(
            world,
            folder=str(run_dir),
            show_plot=False,
            dpi=600,
            format="png",
            # カスタマイズパラメータ
            node_size=150,  # ノードサイズ（デフォルト: 100）
            node_label_size=12,  # ノードラベルサイズ（デフォルト: 8）
            edge_label_size=8,  # エッジラベルサイズ（デフォルト: 6）
            node_color="tab:blue",  # ノード色
            node_alpha=0.8,  # ノード透明度
            label_alpha=0.8,  # ラベル透明度
            edge_alpha=0.6,  # エッジ透明度
            arrow_size=25,  # 矢印サイズ（デフォルト: 20）
            figsize=(6, 5),  # 図のサイズ
            exclude_nodes=["DataCollector"],  # DataCollectorを非表示
        )

        # 実行時間グラフ
        mosaik.util.plot_execution_time(world, **plot_kwargs)

        print(f"   Graphs saved to {run_dir}/")
    except Exception as e:
        print(f"   ⚠️  Graph generation failed: {e}")

    print("\n" + "=" * 70)
    print("HILS Simulation with Inverse Compensation Finished")
    print("=" * 70)


if __name__ == "__main__":
    main()
