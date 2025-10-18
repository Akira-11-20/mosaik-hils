"""
PlantSimulator - 1DOF推力測定器シミュレーター

スラストスタンドを模擬し、制御指令（推力・持続時間）を受け取り、
実際に発生する推力を出力する。

初期実装: 1自由度、ノイズなし、理想的な応答
"""

import mosaik_api


meta = {
    "type": "time-based",
    "models": {
        "ThrustStand": {
            "public": True,
            "params": ["stand_id"],
            "attrs": [
                "command",  # 入力: 制御コマンド（辞書: {thrust, duration}）
                "measured_thrust",  # 出力: 測定された推力 [N]
                "status",  # 状態: "idle", "thrusting"
            ],
        },
    },
}


class PlantSimulator(mosaik_api.Simulator):
    """
    推力測定器シミュレーター（1DOF版）

    入力:
        command: 制御コマンド（辞書）
            {
                "thrust": 推力指令値 [N],
                "duration": 持続時間 [ms]
            }

    出力:
        measured_thrust: 実際に測定される推力 [N]
        status: 測定器の状態
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
        stand_id="thrust_stand_01",
    ):
        """
        推力測定器エンティティの作成

        Args:
            num: 作成数
            model: モデル名
            stand_id: 測定器ID
        """
        entities = []

        for i in range(num):
            eid = f"{model}_{i}"

            self.entities[eid] = {
                "stand_id": stand_id,
                "thrust_cmd": 0.0,  # 現在の推力指令
                "duration_cmd": 0.0,  # 持続時間指令
                "measured_thrust": 0.0,  # 測定値（出力）
                "status": "idle",  # 初期状態
                "thrust_start_time": None,  # 推力開始時刻
                "thrust_end_time": None,  # 推力終了時刻
            }

            entities.append({"eid": eid, "type": model})
            print(f"[PlantSim] Created {eid} (ID: {stand_id})")

        return entities

    def step(self, time, inputs, max_advance=None):
        """
        シミュレーションステップ

        Args:
            time: 現在時刻 [ms]
            inputs: 入力データ
        """
        self.time = time

        for (
            eid,
            entity,
        ) in self.entities.items():
            # 入力: パッケージ化された制御コマンドの受信
            if eid in inputs and "command" in inputs[eid]:
                cmd = list(inputs[eid]["command"].values())[0]

                # コマンドが辞書型で、かつNoneでない場合に処理
                if cmd is not None and isinstance(cmd, dict):
                    thrust = cmd.get("thrust", 0.0)
                    duration = cmd.get("duration", 0.0)

                    entity["thrust_cmd"] = thrust
                    entity["duration_cmd"] = duration

                    # 新しい推力指令が来たら、開始・終了時刻を設定
                    # 負の推力も許容するため、thrust != 0 で判定
                    if thrust != 0 and duration > 0:
                        entity["thrust_start_time"] = time
                        entity["thrust_end_time"] = time + duration
                        entity["status"] = "thrusting"

            # 推力測定ロジック（理想応答、ノイズなし）
            if entity["status"] == "thrusting":
                # 推力持続時間内かチェック
                if (
                    entity["thrust_end_time"] is not None
                    and time < entity["thrust_end_time"]
                ):
                    # 指令通りの推力を出力（理想的な応答）
                    entity["measured_thrust"] = entity["thrust_cmd"]
                else:
                    # 持続時間終了
                    entity["measured_thrust"] = 0.0
                    entity["status"] = "idle"
                    entity["thrust_start_time"] = None
                    entity["thrust_end_time"] = None
            else:
                # アイドル状態
                entity["measured_thrust"] = 0.0

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
    mosaik_api.start_simulator(PlantSimulator())
