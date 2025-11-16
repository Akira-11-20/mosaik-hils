"""
OrbitalControllerSimulator - 軌道制御器シミュレーター

ThrustModelを使用して推力指令を計算。
OrbitalEnvからの状態量（位置・速度）を入力とし、
OrbitalPlantへの推力指令を出力する。
"""

import sys
from pathlib import Path

import mosaik_api
import numpy as np

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.thrust_model import HCWThrustModel, HohmannThrustModel, PDThrustModel, ThrustModel

meta = {
    "type": "time-based",
    "models": {
        "OrbitalController": {
            "public": True,
            "params": [
                "target_position",  # 目標位置 [x, y, z] [m]
                "target_velocity",  # 目標速度 [vx, vy, vz] [m/s] (PD制御用)
                "control_gain",  # 制御ゲイン
                "controller_type",  # 制御タイプ: "zero", "pd", "hohmann"
                "mu",  # 重力定数（ホーマン遷移用）
                "initial_altitude",  # 初期軌道高度（ホーマン遷移用）
                "target_altitude",  # 目標軌道高度（ホーマン遷移用）
                "radius_body",  # 天体半径（ホーマン遷移用）
                "spacecraft_mass",  # 衛星質量
                "max_thrust",  # 最大推力
                "start_time",  # 遷移開始時刻（ホーマン遷移用）
            ],
            "attrs": [
                # 入力（from OrbitalEnv）
                "position_x",
                "position_y",
                "position_z",
                "velocity_x",
                "velocity_y",
                "velocity_z",
                # 入力（from Chief/Target - HCW用）
                "chief_position_x",
                "chief_position_y",
                "chief_position_z",
                "chief_velocity_x",
                "chief_velocity_y",
                "chief_velocity_z",
                # 出力（to OrbitalPlant）
                "thrust_command_x",
                "thrust_command_y",
                "thrust_command_z",
            ],
        },
    },
}


class OrbitalControllerSimulator(mosaik_api.Simulator):
    """
    軌道制御器シミュレーター

    入力:
        position_x, position_y, position_z: 位置 [m]
        velocity_x, velocity_y, velocity_z: 速度 [m/s]

    出力:
        thrust_command_x, thrust_command_y, thrust_command_z: 推力指令 [N]
    """

    def __init__(self):
        super().__init__(meta)
        self.entities = {}
        self.step_size = 1
        self.time = 0
        self.time_resolution = 1.0

    def init(self, sid, time_resolution=1.0, step_size=1):
        """
        初期化

        Args:
            sid: シミュレーターID
            time_resolution: 時間解像度 [s]
            step_size: ステップサイズ [時間単位]
        """
        self.sid = sid
        self.time_resolution = time_resolution
        self.step_size = step_size
        return self.meta

    def create(
        self,
        num,
        model,
        target_position=None,
        target_velocity=None,
        control_gain=1.0,
        controller_type="zero",
        mu=3.986004418e14,
        initial_altitude=400e3,
        target_altitude=600e3,
        radius_body=6378137.0,
        spacecraft_mass=500.0,
        max_thrust=1.0,
        start_time=10.0,
    ):
        """
        軌道制御器エンティティの作成

        Args:
            num: 作成数
            model: モデル名
            target_position: 目標位置 [x, y, z] [m]
            target_velocity: 目標速度 [vx, vy, vz] [m/s] (PD制御用)
            control_gain: 制御ゲイン
            controller_type: 制御タイプ ("zero", "pd", "hohmann")
            mu: 重力定数 [m³/s²]
            initial_altitude: 初期軌道高度 [m]
            target_altitude: 目標軌道高度 [m]
            radius_body: 天体半径 [m]
            spacecraft_mass: 衛星質量 [kg]
            max_thrust: 最大推力 [N]
            start_time: 遷移開始時刻 [s]
        """
        entities = []

        # デフォルト目標: 原点（地球中心）
        if target_position is None:
            target_position = [0.0, 0.0, 0.0]

        for i in range(num):
            eid = f"{model}_{i}"

            # 制御タイプに応じてThrustModelを初期化
            if controller_type == "hohmann":
                # ホーマン遷移モデル
                thrust_model = HohmannThrustModel(
                    mu=mu,
                    initial_altitude=initial_altitude,
                    target_altitude=target_altitude,
                    radius_body=radius_body,
                    spacecraft_mass=spacecraft_mass,
                    max_thrust=max_thrust,
                    start_time=start_time,
                )
                print(f"[OrbitalControllerSim] Created {eid} with Hohmann transfer:")
                print(f"  Initial altitude: {initial_altitude / 1e3:.2f} km")
                print(f"  Target altitude: {target_altitude / 1e3:.2f} km")
                print(f"  Start time: {start_time:.2f} s")

            elif controller_type == "pd":
                # PD制御モデル
                thrust_model = PDThrustModel(
                    target_position=target_position,
                    target_velocity=target_velocity,
                    kp=control_gain * 1e-3,
                    kd=control_gain * 1e-2,
                    max_thrust=max_thrust,
                )
                print(f"[OrbitalControllerSim] Created {eid} with PD control:")
                print(f"  Target position: {target_position} m")
                print(f"  Target velocity: {target_velocity} m/s")
                print(f"  Control gain: {control_gain}")

            elif controller_type == "hcw":
                # HCW編隊飛行制御
                # target_positionは相対位置として使用
                thrust_model = HCWThrustModel(
                    target_relative_position=target_position if target_position else [0, 0, 0],
                    target_relative_velocity=target_velocity if target_velocity else [0, 0, 0],
                    chief_position=None,  # シナリオで設定
                    chief_velocity=None,  # シナリオで設定
                    mu=mu,
                    kp_x=control_gain * 0.001,
                    kp_y=control_gain * 0.001,
                    kp_z=control_gain * 0.001,
                    kd_x=control_gain * 0.01,
                    kd_y=control_gain * 0.01,
                    kd_z=control_gain * 0.01,
                    max_thrust=max_thrust,
                )
                print(f"[OrbitalControllerSim] Created {eid} with HCW control:")
                print(f"  Target relative position (LVLH): {target_position} m")
                print(f"  Control gain: {control_gain}")

            else:
                # ゼロ推力モデル（デフォルト）
                thrust_model = ThrustModel(target_position=target_position, control_gain=control_gain)
                print(f"[OrbitalControllerSim] Created {eid} (zero thrust):")
                print(f"  Target position: {target_position} m")
                print(f"  Control gain: {control_gain}")

            self.entities[eid] = {
                "thrust_model": thrust_model,
                "position": np.zeros(3),
                "velocity": np.zeros(3),
                "thrust_command": np.zeros(3),
            }

            entities.append({"eid": eid, "type": model})

        return entities

    def step(self, time, inputs, max_advance=None):
        """
        シミュレーションステップ

        Args:
            time: 現在時刻 [時間単位]
            inputs: 入力データ
        """
        self.time = time

        for eid, entity in self.entities.items():
            # 入力: 位置・速度の受信
            position = np.zeros(3)
            velocity = np.zeros(3)
            chief_position = None
            chief_velocity = None

            if eid in inputs:
                # Deputy（自機）の位置
                for axis, attr in enumerate(["position_x", "position_y", "position_z"]):
                    if attr in inputs[eid]:
                        value = list(inputs[eid][attr].values())[0]
                        position[axis] = value if value is not None else 0.0

                # Deputy（自機）の速度
                for axis, attr in enumerate(["velocity_x", "velocity_y", "velocity_z"]):
                    if attr in inputs[eid]:
                        value = list(inputs[eid][attr].values())[0]
                        velocity[axis] = value if value is not None else 0.0

                # Chief（目標機）の位置（HCW用）
                chief_pos_attrs = ["chief_position_x", "chief_position_y", "chief_position_z"]
                if any(attr in inputs[eid] for attr in chief_pos_attrs):
                    chief_position = np.zeros(3)
                    for axis, attr in enumerate(chief_pos_attrs):
                        if attr in inputs[eid]:
                            value = list(inputs[eid][attr].values())[0]
                            chief_position[axis] = value if value is not None else 0.0

                # Chief（目標機）の速度（HCW用）
                chief_vel_attrs = ["chief_velocity_x", "chief_velocity_y", "chief_velocity_z"]
                if any(attr in inputs[eid] for attr in chief_vel_attrs):
                    chief_velocity = np.zeros(3)
                    for axis, attr in enumerate(chief_vel_attrs):
                        if attr in inputs[eid]:
                            value = list(inputs[eid][attr].values())[0]
                            chief_velocity[axis] = value if value is not None else 0.0

            # 状態を更新
            entity["position"] = position
            entity["velocity"] = velocity

            # 推力指令の計算
            thrust_model = entity["thrust_model"]
            current_time = time * self.time_resolution  # 時間単位を秒に変換

            # HCWモデルの場合、Chief位置・速度を引数として渡す
            if isinstance(thrust_model, HCWThrustModel):
                thrust_command = thrust_model.calculate_thrust(
                    position, velocity,
                    chief_position=chief_position,
                    chief_velocity=chief_velocity,
                    time=current_time
                )
            else:
                thrust_command = thrust_model.calculate_thrust(position, velocity, time=current_time)

            entity["thrust_command"] = thrust_command

        return time + self.step_size

    def get_data(self, outputs):
        """
        データ取得

        Args:
            outputs: 要求データ仕様
        """
        data = {}

        for eid, attrs in outputs.items():
            if eid not in self.entities:
                continue

            entity = self.entities[eid]
            data[eid] = {}

            thrust_command = entity["thrust_command"]

            # データマッピング
            attr_map = {
                "thrust_command_x": thrust_command[0],
                "thrust_command_y": thrust_command[1],
                "thrust_command_z": thrust_command[2],
                "position_x": entity["position"][0],
                "position_y": entity["position"][1],
                "position_z": entity["position"][2],
                "velocity_x": entity["velocity"][0],
                "velocity_y": entity["velocity"][1],
                "velocity_z": entity["velocity"][2],
            }

            for attr in attrs:
                data[eid][attr] = attr_map.get(attr, None)

        return data


if __name__ == "__main__":
    mosaik_api.start_simulator(OrbitalControllerSimulator())
