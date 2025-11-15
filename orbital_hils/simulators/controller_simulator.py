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

from models.thrust_model import ThrustModel

meta = {
    "type": "time-based",
    "models": {
        "OrbitalController": {
            "public": True,
            "params": [
                "target_position",  # 目標位置 [x, y, z] [m]
                "control_gain",  # 制御ゲイン
            ],
            "attrs": [
                # 入力（from OrbitalEnv）
                "position_x",
                "position_y",
                "position_z",
                "velocity_x",
                "velocity_y",
                "velocity_z",
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

    def create(self, num, model, target_position=None, control_gain=1.0):
        """
        軌道制御器エンティティの作成

        Args:
            num: 作成数
            model: モデル名
            target_position: 目標位置 [x, y, z] [m]
            control_gain: 制御ゲイン
        """
        entities = []

        # デフォルト目標: 原点（地球中心）
        if target_position is None:
            target_position = [0.0, 0.0, 0.0]

        for i in range(num):
            eid = f"{model}_{i}"

            # ThrustModelの初期化
            thrust_model = ThrustModel(target_position=target_position, control_gain=control_gain)

            self.entities[eid] = {
                "thrust_model": thrust_model,
                "position": np.zeros(3),
                "velocity": np.zeros(3),
                "thrust_command": np.zeros(3),
            }

            entities.append({"eid": eid, "type": model})

            print(f"[OrbitalControllerSim] Created {eid}:")
            print(f"  Target position: {target_position} m")
            print(f"  Control gain: {control_gain}")

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

            if eid in inputs:
                # 位置
                for axis, attr in enumerate(["position_x", "position_y", "position_z"]):
                    if attr in inputs[eid]:
                        value = list(inputs[eid][attr].values())[0]
                        position[axis] = value if value is not None else 0.0

                # 速度
                for axis, attr in enumerate(["velocity_x", "velocity_y", "velocity_z"]):
                    if attr in inputs[eid]:
                        value = list(inputs[eid][attr].values())[0]
                        velocity[axis] = value if value is not None else 0.0

            # 状態を更新
            entity["position"] = position
            entity["velocity"] = velocity

            # 推力指令の計算
            thrust_model = entity["thrust_model"]
            thrust_command = thrust_model.calculate_thrust(position, velocity)
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
