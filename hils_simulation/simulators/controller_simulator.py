"""
ControllerSimulator - シンプルなPD制御器

位置制御を行うコントローラー。
目標位置との誤差をもとに推力指令を生成。

初期実装: 1DOF、シンプルなPD制御
"""

import mosaik_api


meta = {
    "type": "time-based",
    "models": {
        "PDController": {
            "public": True,
            "params": [
                "kp",
                "kd",
                "target_position",
                "max_thrust",
                "thrust_duration",
            ],
            "attrs": [
                "position",  # 入力: 現在位置 [m]
                "velocity",  # 入力: 現在速度 [m/s]
                "command",  # 出力: 制御コマンド（辞書: {thrust, duration}）
                "error",  # 出力: 位置誤差 [m]
            ],
        },
    },
}


class ControllerSimulator(mosaik_api.Simulator):
    """
    PD制御器シミュレーター（1DOF版）

    制御則:
        error = target_position - current_position
        thrust = Kp * error - Kd * velocity

    入力:
        position: 現在位置 [m]
        velocity: 現在速度 [m/s]

    出力:
        command: 制御コマンド（辞書）
            {
                "thrust": 推力指令 [N],
                "duration": 持続時間 [ms]
            }
        error: 位置誤差 [m]
    """

    def __init__(self):
        super().__init__(meta)
        self.entities = {}
        self.step_size = 10  # デフォルト: 10ms（制御周期）
        self.time = 0

    def init(
        self,
        sid,
        time_resolution=0.001,
        step_size=10,
    ):
        """
        初期化

        Args:
            sid: シミュレーターID
            time_resolution: 時間解像度（0.001 = 1ms）
            step_size: ステップサイズ [時間単位]（制御周期）
        """
        self.sid = sid
        self.time_resolution = time_resolution
        self.step_size = step_size
        return self.meta

    def create(
        self,
        num,
        model,
        kp=1.0,
        kd=0.5,
        target_position=10.0,
        max_thrust=50.0,
        thrust_duration=10,
    ):
        """
        制御器エンティティの作成

        Args:
            num: 作成数
            model: モデル名
            kp: 比例ゲイン
            kd: 微分ゲイン
            target_position: 目標位置 [m]
            max_thrust: 最大推力 [N]
            thrust_duration: 推力持続時間 [ms]
        """
        entities = []

        for i in range(num):
            eid = f"{model}_{i}"

            self.entities[eid] = {
                "kp": kp,
                "kd": kd,
                "target_position": target_position,
                "max_thrust": max_thrust,
                "thrust_duration": thrust_duration,
                "position": 0.0,
                "velocity": 0.0,
                "command": {
                    "thrust": 0.0,
                    "duration": thrust_duration,
                },  # パッケージ化
                "error": target_position,
            }

            entities.append({"eid": eid, "type": model})
            print(
                f"[ControllerSim] Created {eid} (Kp={kp}, Kd={kd}, target={target_position}m)"
            )

        return entities

    def step(self, time, inputs, max_advance=None):
        """
        制御演算ステップ

        Args:
            time: 現在時刻 [ms]
            inputs: 入力データ
        """
        self.time = time

        for (
            eid,
            entity,
        ) in self.entities.items():
            # 入力: 状態量（位置・速度）の受信
            if eid in inputs:
                if "position" in inputs[eid]:
                    pos_values = inputs[eid]["position"].values()
                    if pos_values:
                        entity["position"] = list(pos_values)[0]

                if "velocity" in inputs[eid]:
                    vel_values = inputs[eid]["velocity"].values()
                    if vel_values:
                        entity["velocity"] = list(vel_values)[0]

            # PD制御則
            error = entity["target_position"] - entity["position"]
            entity["error"] = error

            # 推力指令計算: F = Kp * error - Kd * velocity
            thrust = entity["kp"] * error - entity["kd"] * entity["velocity"]

            # 推力制限（飽和処理）
            if thrust > entity["max_thrust"]:
                thrust = entity["max_thrust"]
            elif thrust < -entity["max_thrust"]:
                thrust = -entity["max_thrust"]

            # コマンドをパッケージ化
            entity["command"] = {
                "thrust": thrust,
                "duration": entity["thrust_duration"],
            }

            # デバッグ出力（10制御周期に1回）
            if time % (self.step_size * 10) == 0:
                print(
                    f"[ControllerSim] t={time}ms: pos={entity['position']:.3f}m, "
                    f"vel={entity['velocity']:.3f}m/s, error={error:.3f}m, thrust={thrust:.3f}N"
                )

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
    mosaik_api.start_simulator(ControllerSimulator())
