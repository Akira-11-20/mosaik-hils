"""
Pure Python Simulation - 1DOF版 (Mosaikなし)

理想的な連続時間シミュレーション:
    連続的な制御ループ（離散化による誤差を最小化）

特徴:
- Mosaikフレームワークを使用しない素のPythonシミュレーション
- 数値積分による連続時間シミュレーション
- 通信遅延なし
- 離散化誤差を最小化したベースライン

用途:
- Mosaikフレームワークのオーバーヘッド評価
- 理論的な最適制御性能のベースライン
- HILSシステムとの性能比較
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
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

# Control parameters (same as HILS)
CONTROL_PERIOD = get_env_float("CONTROL_PERIOD", 10)  # Control period [ms]
CONTROL_PERIOD_S = CONTROL_PERIOD / 1000.0  # Convert to seconds
KP = get_env_float("KP", 15.0)  # Proportional gain
KD = get_env_float("KD", 5.0)  # Derivative gain
TARGET_POSITION = get_env_float("TARGET_POSITION", 5.0)  # Target position [m]
MAX_THRUST = get_env_float("MAX_THRUST", 100.0)  # Maximum thrust [N]

# Simulator periods
ENV_SIM_PERIOD_MS = get_env_float("ENV_SIM_PERIOD", 10)  # [ms]
PLANT_SIM_PERIOD_MS = get_env_float("PLANT_SIM_PERIOD", 10)  # [ms]

# Spacecraft parameters
SPACECRAFT_MASS = get_env_float("SPACECRAFT_MASS", 1.0)  # Mass [kg]
INITIAL_POSITION = get_env_float("INITIAL_POSITION", 0.0)  # Initial position [m]
INITIAL_VELOCITY = get_env_float("INITIAL_VELOCITY", 10.0)  # Initial velocity [m/s]
GRAVITY = get_env_float("GRAVITY", 9.81)  # Gravity acceleration [m/s^2]


class Spacecraft1DOF:
    """1自由度宇宙機の運動シミュレータ"""

    def __init__(
        self,
        mass: float,
        initial_position: float,
        initial_velocity: float,
        gravity: float,
    ):
        self.mass = mass
        self.position = initial_position
        self.velocity = initial_velocity
        self.gravity = gravity
        self.force = 0.0
        self.acceleration = 0.0

    def step(self, dt: float, force: float):
        """
        運動方程式を数値積分（Explicit Euler法、Mosaikと同じ）

        Args:
            dt: 時間ステップ [s]
            force: 推力 [N]
        """
        self.force = force

        # 運動方程式: F = ma → a = F/m - g
        self.acceleration = (self.force / self.mass) - self.gravity

        # Explicit Euler法による積分（Mosaikと同じ順序）
        # x(t+dt) = x(t) + v(t) * dt （先に位置を更新、古い速度を使用）
        self.position += self.velocity * dt

        # v(t+dt) = v(t) + a * dt （後で速度を更新）
        self.velocity += self.acceleration * dt


class PDController:
    """PDコントローラ"""

    def __init__(self, kp: float, kd: float, target_position: float, max_thrust: float):
        self.kp = kp
        self.kd = kd
        self.target_position = target_position
        self.max_thrust = max_thrust
        self.error = 0.0

    def compute_control(self, position: float, velocity: float) -> float:
        """
        制御入力を計算

        Args:
            position: 現在位置 [m]
            velocity: 現在速度 [m/s]

        Returns:
            推力 [N]
        """
        # 位置誤差
        self.error = self.target_position - position

        # PD制御則
        thrust = self.kp * self.error - self.kd * velocity

        # 推力制限なし（負の推力も許容）
        # max_thrustパラメータは残すが、制限は行わない

        return thrust


class ThrustStand:
    """推力測定器（理想的なセンサー）"""

    def __init__(self):
        self.measured_thrust = 0.0

    def measure(self, thrust: float) -> float:
        """
        推力を測定（理想的なセンサーなのでそのまま返す）

        Args:
            thrust: 入力推力 [N]

        Returns:
            測定推力 [N]
        """
        self.measured_thrust = thrust
        return self.measured_thrust


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
            "type": "Pure Python (No Mosaik)",
        },
        "communication": {
            "cmd_delay_s": 0.0,
            "cmd_jitter_s": 0.0,
            "cmd_loss_rate": 0.0,
            "sense_delay_s": 0.0,
            "sense_jitter_s": 0.0,
            "sense_loss_rate": 0.0,
        },
        "control": {
            "control_period_s": CONTROL_PERIOD_S,
            "kp": KP,
            "kd": KD,
            "target_position_m": TARGET_POSITION,
            "max_thrust_N": MAX_THRUST,
        },
        "simulators": {
            "env_sim_period_s": ENV_SIM_PERIOD_MS / 1000.0,
            "plant_sim_period_s": PLANT_SIM_PERIOD_MS / 1000.0,
        },
        "spacecraft": {
            "mass_kg": SPACECRAFT_MASS,
        },
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "description": "Pure Python 1-DOF Spacecraft Control Simulation (No Mosaik Framework)",
            "note": "All time units are in seconds (s). This is a baseline simulation without any framework overhead.",
        },
    }

    config_path = output_dir / "simulation_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"💾 Configuration saved: {config_path}")
    return config_path


def save_data_to_hdf5(output_dir: Path, data: Dict[str, List]):
    """
    シミュレーション結果をHDF5形式で保存

    Args:
        output_dir: 出力ディレクトリ
        data: データ辞書
    """
    h5_path = output_dir / "hils_data.h5"

    with h5py.File(h5_path, "w") as f:
        # データグループを作成
        data_group = f.create_group("data")

        # 各データセットを保存
        for key, values in data.items():
            data_group.create_dataset(key, data=np.array(values), compression="gzip")

    print(f"📁 HDF5 data saved: {h5_path}")
    return h5_path


def run_simulation():
    """
    純粋なPythonシミュレーションを実行
    """
    print("=" * 70)
    print("Pure Python Simulation - 1DOF Configuration (No Mosaik)")
    print("=" * 70)

    # ログディレクトリの作成
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path("results_pure") / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Log directory: {run_dir}")

    # シミュレーション設定の保存
    save_simulation_config(run_dir)

    # コンポーネントの初期化
    print("\n📦 Creating components...")
    spacecraft = Spacecraft1DOF(
        mass=SPACECRAFT_MASS,
        initial_position=INITIAL_POSITION,
        initial_velocity=INITIAL_VELOCITY,
        gravity=GRAVITY,
    )
    controller = PDController(
        kp=KP,
        kd=KD,
        target_position=TARGET_POSITION,
        max_thrust=MAX_THRUST,
    )
    plant = ThrustStand()

    print(
        f"   Spacecraft: mass={SPACECRAFT_MASS}kg, x0={INITIAL_POSITION}m, v0={INITIAL_VELOCITY}m/s"
    )
    print(f"   Controller: Kp={KP}, Kd={KD}, target={TARGET_POSITION}m")

    # データ記録用リスト
    data = {
        "time_s": [],
        "time_ms": [],
        "position_Spacecraft": [],
        "velocity_Spacecraft": [],
        "acceleration_Spacecraft": [],
        "force_Spacecraft": [],
        "command_Controller_thrust": [],
        "error_Controller": [],
        "measured_thrust_Plant": [],
    }

    # シミュレーション実行
    print(f"\n▶️  Running simulation until {SIMULATION_TIME}s ({SIMULATION_STEPS} steps)...")
    print("=" * 70)

    control_period_steps = int(CONTROL_PERIOD_S / TIME_RESOLUTION)
    thrust = 0.0  # 初期推力
    log_interval_steps = int(1.0 / TIME_RESOLUTION)  # 1秒ごとのログ出力

    for step in range(SIMULATION_STEPS):
        time_s = step * TIME_RESOLUTION
        time_ms = time_s * 1000

        # 制御周期ごとに制御入力を計算（Mosaikと同じく Step 0 から開始）
        if step % control_period_steps == 0:
            thrust = controller.compute_control(spacecraft.position, spacecraft.velocity)
            measured_thrust = plant.measure(thrust)

            # 定期的にログ出力
            if step % log_interval_steps == 0:  # 1秒ごと
                print(
                    f"[t={time_ms:.0f}ms] pos={spacecraft.position:.3f}m, "
                    f"vel={spacecraft.velocity:.3f}m/s, error={controller.error:.3f}m, "
                    f"thrust={thrust:.3f}N"
                )

        # データ記録（制御計算後、物理更新前 - Mosaikと同じタイミング）
        data["time_s"].append(time_s)
        data["time_ms"].append(time_ms)
        data["position_Spacecraft"].append(spacecraft.position)
        data["velocity_Spacecraft"].append(spacecraft.velocity)
        data["acceleration_Spacecraft"].append(spacecraft.acceleration)
        data["force_Spacecraft"].append(spacecraft.force)
        data["command_Controller_thrust"].append(thrust)
        data["error_Controller"].append(controller.error)
        data["measured_thrust_Plant"].append(plant.measured_thrust)

        # 宇宙機の運動を更新
        spacecraft.step(TIME_RESOLUTION, thrust)

    print("=" * 70)
    print("✅ Simulation completed successfully!")

    # データ保存
    print("\n💾 Saving data...")
    save_data_to_hdf5(run_dir, data)

    print(f"\n📊 Final state:")
    print(f"   Position: {spacecraft.position:.3f} m (target: {TARGET_POSITION} m)")
    print(f"   Velocity: {spacecraft.velocity:.3f} m/s")
    print(f"   Error: {controller.error:.3f} m")

    print("\n" + "=" * 70)
    print("Pure Python Simulation Finished")
    print("=" * 70)

    return run_dir


if __name__ == "__main__":
    run_simulation()
