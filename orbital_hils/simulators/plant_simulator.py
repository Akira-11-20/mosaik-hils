"""
OrbitalPlantSimulator - 3軸推力計測デバイスシミュレーター

推力スタンドの物理モデル:
- 1次遅れ系: τ * ḟ + f = f_cmd
- 計測ノイズ: f_measured = f + N(0, σ²)
- 3軸独立動作（X, Y, Z）
"""

import mosaik_api
import numpy as np

meta = {
    "type": "time-based",
    "models": {
        "OrbitalThrustStand": {
            "public": True,
            "params": [
                "time_constant",  # 時定数 [s]
                "noise_std",  # 計測ノイズ標準偏差 [N]
            ],
            "attrs": [
                # 入力（from OrbitalController）
                "command_x",
                "command_y",
                "command_z",
                # 出力（to OrbitalEnv）
                "measured_force_x",
                "measured_force_y",
                "measured_force_z",
                "norm_measured_force",
                "alpha",
            ],
        },
    },
}


class OrbitalPlantSimulator(mosaik_api.Simulator):
    """
    3軸推力計測デバイスシミュレーター

    物理モデル:
        1次遅れ系: τ * ḟ + f = f_cmd
        離散化: f[n+1] = f[n] + (dt/τ)*(f_cmd[n] - f[n])

        計測ノイズ: f_measured = f + N(0, σ²)

    入力:
        command_x, command_y, command_z: 推力指令 [N]

    出力:
        measured_force_x, measured_force_y, measured_force_z: 計測推力 [N]
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

    def create(self, num, model, time_constant=10.0, noise_std=0.01):
        """
        推力計測デバイスエンティティの作成

        Args:
            num: 作成数
            model: モデル名
            time_constant: 時定数 [s]（1次遅れ系の応答速度）
            noise_std: 計測ノイズ標準偏差 [N]
        """
        entities = []

        for i in range(num):
            eid = f"{model}_{i}"

            self.entities[eid] = {
                "time_constant": time_constant,
                "noise_std": noise_std,
                "command": np.zeros(3),  # 指令推力 [N]
                "force": np.zeros(3),  # 実推力（1次遅れ） [N]
                "measured_force": np.zeros(3),  # 計測推力 [N]
            }

            entities.append({"eid": eid, "type": model})

            print(f"[OrbitalPlantSim] Created {eid}:")
            print(f"  Time constant: {time_constant} s")
            print(f"  Noise std: {noise_std} N")

        return entities

    def step(self, time, inputs, max_advance=None):
        """
        シミュレーションステップ

        Args:
            time: 現在時刻 [時間単位]
            inputs: 入力データ
        """
        self.time = time
        dt = self.step_size * self.time_resolution

        for eid, entity in self.entities.items():
            # 入力: 推力指令の受信
            command = np.zeros(3)

            if eid in inputs:
                for axis, attr in enumerate(["command_x", "command_y", "command_z"]):
                    if attr in inputs[eid]:
                        value = list(inputs[eid][attr].values())[0]
                        command[axis] = value if value is not None else 0.0

            entity["command"] = command

            # 1次遅れ系の更新
            tau = entity["time_constant"]
            force = entity["force"]

            norm_command = np.linalg.norm(command)
            norm_force = np.linalg.norm(force)

            # 離散化: f[n+1] = f[n] + (dt/τ)*(f_cmd[n] - f[n])

            alpha = tau / dt
            if alpha > 1.0:
                new_norm_force = norm_force + (norm_command - norm_force) / alpha
                if norm_command > 0:
                    new_force = new_norm_force * (command / norm_command)
                else:
                    new_force = new_norm_force * (force / norm_force) if norm_force > 0 else np.zeros(3)
            else:
                # τ=0の場合は即座に追従
                new_force = command

            entity["force"] = new_force

            # 計測ノイズの付加
            noise_std = entity["noise_std"]
            noise = np.random.normal(0, noise_std, 3)
            measured_force = new_force + noise

            entity["measured_force"] = measured_force
            entity["alpha"] = alpha

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

            measured_force = entity["measured_force"]

            # データマッピング
            attr_map = {
                "measured_force_x": measured_force[0],
                "measured_force_y": measured_force[1],
                "measured_force_z": measured_force[2],
                "command_x": entity["command"][0],
                "command_y": entity["command"][1],
                "command_z": entity["command"][2],
                "norm_measured_force": np.linalg.norm(measured_force),
                "alpha": entity["alpha"],
            }

            for attr in attrs:
                data[eid][attr] = attr_map.get(attr, None)

        return data


if __name__ == "__main__":
    mosaik_api.start_simulator(OrbitalPlantSimulator())
