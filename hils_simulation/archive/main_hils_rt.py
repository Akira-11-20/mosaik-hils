"""
RT (Real-Time) Simulation Main Scenario - 1DOF版 (比較用)

単一ノード構成（通信遅延なし）:
    Env → Controller → Plant → Env

特徴:
- 1ms時間解像度（HILSと同じ）
- 通信ブリッジなし（遅延、ジッタ、パケットロスなし）
- Controller → Plant間で直接接続（同一ノード内の動作を想定）
- HILSシステムとの性能比較用

用途:
- HILSシステム（通信遅延あり）との制御性能比較
- 理想的な制御ループの性能ベースライン
- 通信遅延の影響を定量的に評価
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

# Simulation settings
SIMULATION_TIME = get_env_float("SIMULATION_TIME", 2)  # Simulation time [s]
TIME_RESOLUTION = get_env_float("TIME_RESOLUTION", 0.0001)  # Time resolution [s/step]
SIMULATION_STEPS = int(SIMULATION_TIME / TIME_RESOLUTION)
RT_FACTOR_STR = os.getenv("RT_FACTOR", "None")
RT_FACTOR = None if RT_FACTOR_STR == "None" else float(RT_FACTOR_STR)

# Control parameters (same as HILS)
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
            "type": "RT (Real-Time, No Delay)",
        },
        "communication": {
            "cmd_delay_s": 0.0,  # No delay in RT version
            "cmd_jitter_s": 0.0,
            "cmd_loss_rate": 0.0,
            "sense_delay_s": 0.0,
            "sense_jitter_s": 0.0,
            "sense_loss_rate": 0.0,
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
            "description": "RT 1-DOF Spacecraft Control Simulation (No Communication Delay)",
            "note": "All time units are in seconds (s). This is a baseline simulation without communication delays for comparison with HILS.",
        },
    }

    config_path = output_dir / "simulation_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"💾 Configuration saved: {config_path}")
    return config_path


def main():
    """
    RT (Real-Time) メインシナリオ - 通信遅延なし版
    """
    print("=" * 70)
    print("RT Simulation - 1DOF Configuration (No Delay)")
    print("=" * 70)

    # ログディレクトリの作成
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path("results_rt") / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Log directory: {run_dir}")

    # シミュレーション設定の保存
    save_simulation_config(run_dir)

    # シミュレーター構成（BridgeSimなし）
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
        step_size=int(CONTROL_PERIOD / 1000 / TIME_RESOLUTION),  # 10ms → steps (100)
    )
    # Plant と Env は毎ステップ実行（Pure Pythonと同じ）
    plant_sim = world.start("PlantSim", step_size=1)  # 0.1ms = 1 step
    env_sim = world.start("EnvSim", step_size=1)  # 0.1ms = 1 step

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
        initial_position=INITIAL_POSITION,
        initial_velocity=INITIAL_VELOCITY,
        gravity=GRAVITY,
    )

    # データフローの接続（通信ブリッジなし）
    print("\n🔗 Connecting data flows (direct connections, no delay)...")

    # 1. Controller → Plant - 制御指令経路（time_shiftedで循環を回避）
    print("   ⚡ Controller → Plant: 1-step shifted (to break cycle)")
    world.connect(
        controller,
        plant,
        ("command", "command"),
        time_shifted=True,  # 循環依存回避のため必須
        initial_data={"command": {"thrust": 0.0, "duration": CONTROL_PERIOD}},
    )

    # 2. Plant → Env - 測定値経路（直接接続）
    print("   ⚡ Plant → Env: direct connection (no delay)")
    world.connect(
        plant,
        spacecraft,
        ("measured_thrust", "force"),
    )

    # 3. Env → Controller - 状態フィードバック（同一ステップで送信）
    print("   📡 Env → Controller: same-step connection (state feedback)")
    world.connect(
        spacecraft,
        controller,
        "position",
        "velocity",
    )

    # 4. データ収集の設定
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
        [plant],
        collector,
        "measured_thrust",
        "status",
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
    print("   Env → Controller (every 10ms)")
    print("   Controller → Plant (1-step = 0.1ms shifted, to break cycle)")
    print("   Plant → Env (every 0.1ms)")
    print("   All data → DataCollector → HDF5")
    print("   ℹ️  Command format: JSON/dict {thrust, duration}")
    print("   ⚡ Controller: 10ms period, Plant/Env: 0.1ms period (like Pure Python)")
    print("   ⚠️  Note: 1-step shift = 0.1ms delay (minimal overhead)")

    # シミュレーション実行
    print(f"\n▶️  Running simulation until {SIMULATION_TIME}s ({SIMULATION_STEPS} steps)...")
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
            node_color="tab:green",  # ノード色（RTはグリーン）
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
    print("RT Simulation Finished")
    print("=" * 70)


if __name__ == "__main__":
    main()
