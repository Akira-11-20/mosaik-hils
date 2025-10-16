"""
EnvSimulator - 1DOF環境シミュレーター

宇宙機の1自由度運動をシミュレート。
推力入力から位置・速度を計算（ニュートンの運動方程式）。

初期実装: 1次元並進運動、外乱なし
"""

import mosaik_api


meta = {
    "type": "time-based",
    "models": {
        "Spacecraft1DOF": {
            "public": True,
            "params": [
                "mass",
                "initial_position",
                "initial_velocity",
                "gravity",
            ],
            "attrs": [
                "force",  # 入力: 推力 [N]
                "position",  # 出力: 位置 [m]
                "velocity",  # 出力: 速度 [m/s]
                "acceleration",  # 出力: 加速度 [m/s^2]
            ],
        },
    },
}


class EnvSimulator(mosaik_api.Simulator):
    """
    環境シミュレーター（1DOF版）

    運動方程式（重力項を含む）:
        F_total = F_thrust + F_gravity
        F_gravity = -m * g  （下向きを負とする）
        a = F_total / m = (F_thrust / m) - g
        v(t+dt) = v(t) + a * dt
        x(t+dt) = x(t) + v(t) * dt

    入力:
        force: 推力 [N]

    出力:
        position: 位置 [m]
        velocity: 速度 [m/s]
        acceleration: 加速度 [m/s^2]
    """

    def __init__(self):
        super().__init__(meta)
        self.entities = {}
        self.step_size = 1  # デフォルト: 1ms
        self.time = 0

    def init(
        self,
        sid,
        time_resolution=0.001,
        step_size=1,
    ):
        """
        初期化

        Args:
            sid: シミュレーターID
            time_resolution: 時間解像度（0.001 = 1ms）
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
        mass=100.0,
        initial_position=0.0,
        initial_velocity=0.0,
        gravity=0.0,
    ):
        """
        宇宙機エンティティの作成

        Args:
            num: 作成数
            model: モデル名
            mass: 質量 [kg]
            initial_position: 初期位置 [m]
            initial_velocity: 初期速度 [m/s]
            gravity: 重力加速度 [m/s^2] (デフォルト: 0.0=宇宙空間, 地球なら9.81)
        """
        entities = []

        for i in range(num):
            eid = f"{model}_{i}"

            self.entities[eid] = {
                "mass": mass,
                "position": initial_position,
                "velocity": initial_velocity,
                "acceleration": 0.0,
                "force": 0.0,
                "gravity": gravity,
            }

            entities.append({"eid": eid, "type": model})
            gravity_str = f", g={gravity}m/s²" if gravity != 0.0 else ""
            print(
                f"[EnvSim] Created {eid} (mass={mass}kg, x0={initial_position}m, v0={initial_velocity}m/s{gravity_str})"
            )

        return entities

    def step(self, time, inputs, max_advance=None):
        """
        シミュレーションステップ（運動方程式の積分）

        Args:
            time: 現在時刻 [ms]
            inputs: 入力データ
        """
        self.time = time

        # 実時間での時間刻み [秒]
        dt = self.step_size * self.time_resolution

        for (
            eid,
            entity,
        ) in self.entities.items():
            # 入力: 推力の受信
            if eid in inputs and "force" in inputs[eid]:
                force_value = list(inputs[eid]["force"].values())[0]
                entity["force"] = force_value if force_value is not None else 0.0
            else:
                # 入力がない場合は推力ゼロ
                entity["force"] = 0.0

            # 運動方程式（重力項を含む）:
            # F_total = F_thrust + F_gravity
            # F_gravity = -m * g
            # a = F_total / m = (F_thrust / m) - g
            thrust_acceleration = entity["force"] / entity["mass"]
            gravity_acceleration = -entity["gravity"]  # 下向きを負とする
            entity["acceleration"] = thrust_acceleration + gravity_acceleration

            # オイラー法による積分
            # x(t+dt) = x(t) + v(t) * dt （先に位置を更新、古い速度を使用）
            entity["position"] += entity["velocity"] * dt

            # v(t+dt) = v(t) + a * dt （後で速度を更新）
            entity["velocity"] += entity["acceleration"] * dt

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

            data[eid] = {}
            entity = self.entities[eid]

            for attr in attrs:
                if attr in entity:
                    data[eid][attr] = entity[attr]
                else:
                    data[eid][attr] = None

        return data


if __name__ == "__main__":
    mosaik_api.start_simulator(EnvSimulator())
