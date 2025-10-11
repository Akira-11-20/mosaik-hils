"""
ハードウェアシミュレーター (Hardware Simulator)

物理デバイスとのインターフェースをシミュレートするMosaikシミュレーター。
センサーデータの読み取りとアクチュエーターの制御をシミュレートします。

主な機能:
- センサーデータのシミュレーション（ランダム値）
- 数値シミュレーターからのアクチュエーターコマンド受信
- デバイス状態管理
- シリアル接続のシミュレーション
"""

import mosaik_api
import time
import random


# Mosaikシミュレーターメタデータ - ハードウェアインターフェースの仕様を定義
meta = {
    "type": "time-based",  # 時間ベースのシミュレーション
    "models": {
        "HardwareInterface": {  # ハードウェアインターフェースモデル
            "public": True,  # 他のシミュレーターから利用可能
            "params": [
                "device_id",
                "connection_type",
            ],  # 初期化パラメータ（デバイスID、接続タイプ）
            "attrs": [
                "sensor_value",
                "actuator_command",
                "hardware_input",
                "status",
            ],  # 公開属性
        },
    },
}


class HardwareSimulator(mosaik_api.Simulator):
    """
    ハードウェアシミュレーター実装クラス

    物理デバイスとのインターフェースをシミュレートし、
    センサーデータの生成とアクチュエーター制御を行います。
    """

    def __init__(self):
        """シミュレーターの初期化"""
        super().__init__(meta)
        self.eid_prefix = "HW_"  # エンティティID接頭辞（ハードウェア）
        self.entities = {}  # エンティティデータ保存用辞書
        self.step_size = 1  # デフォルトステップサイズ

    def init(self, sid, time_resolution=1.0, step_size=1):
        """
        シミュレーター初期化メソッド

        Args:
            sid: シミュレーターID
            time_resolution: 時間解像度（デフォルト1.0）
            step_size: ステップサイズ（デフォルト1）

        Returns:
            meta: シミュレーターメタデータ
        """
        self.step_size = step_size
        return meta

    def create(self, num, model, device_id="default", connection_type="serial"):
        """
        ハードウェアインターフェースエンティティの作成

        Args:
            num: 作成するエンティティ数
            model: モデルタイプ
            device_id: デバイスID（デフォルトは"default"）
            connection_type: 接続タイプ（デフォルトは"serial"）

        Returns:
            entities: 作成されたエンティティのリスト
        """
        entities = []
        for i in range(num):
            eid = f"{self.eid_prefix}{i}"  # ユニークなエンティティID生成
            # ハードウェアインターフェースの状態データを初期化
            self.entities[eid] = {
                "device_id": device_id,  # デバイス識別子
                "connection_type": connection_type,  # 接続方式（シリアル等）
                "sensor_value": 0.0,  # センサー読み取り値
                "actuator_command": 0.0,  # アクチュエーターへのコマンド
                "hardware_input": 1.0,  # ハードウェア入力値
                "status": "connected",  # 接続状態
                "last_update": 0,  # 最終更新時刻
            }
            entities.append({"eid": eid, "type": model})
        return entities

    def step(self, time, inputs, max_advance=None):
        """
        ハードウェアシミュレーションステップの実行

        センサーデータの読み取りと数値シミュレーターからの
        アクチュエーターコマンド処理を行います。

        Args:
            time: 現在のシミュレーション時刻
            inputs: 他のシミュレーターからの入力データ
            max_advance: 最大進行時間（オプション）

        Returns:
            next_step: 次のステップ時刻
        """
        for eid, entity in self.entities.items():
            # Simulate hardware sensor readings - ハードウェアセンサー読み取りのシミュレーション
            entity["sensor_value"] = (
                self._read_sensor_data()
            )  # ランダムなセンサー値を生成
            entity["hardware_input"] = entity[
                "sensor_value"
            ]  # センサー値をハードウェア入力として設定

            # Process commands from numerical simulation - 数値シミュレーションからのコマンド処理
            if eid in inputs:
                for attr, values in inputs[eid].items():
                    if attr == "actuator_command":  # アクチュエーターコマンドを受信
                        # Receive numerical simulation output as actuator command - 数値シミュレーション出力をアクチュエーターコマンドとして受信
                        sim_output = list(values.values())[
                            0
                        ]  # シミュレーション出力値を取得
                        entity["actuator_command"] = (
                            sim_output  # アクチュエーターコマンドとして設定
                        )
                        self._send_actuator_command(
                            entity["actuator_command"]
                        )  # アクチュエーターにコマンド送信（シミュレート）

            entity["last_update"] = time  # 最終更新時刻を記録
            entity["status"] = (
                "active" if time > 0 else "initializing"
            )  # ステータスを更新

        return time + self.step_size  # 次のステップ時刻を返す

    def get_data(self, outputs):
        """
        エンティティのデータを取得

        他のシミュレーターからリクエストされた属性の値を返します。

        Args:
            outputs: リクエストされたエンティティと属性の辞書

        Returns:
            data: エンティティのデータ辞書
        """
        data = {}
        for eid, attrs in outputs.items():  # 各エンティティに対して
            data[eid] = {}
            for attr in attrs:  # リクエストされた各属性に対して
                if attr in self.entities[eid]:  # 属性が存在する場合
                    data[eid][attr] = self.entities[eid][attr]  # 属性値をコピー
        return data

    def _read_sensor_data(self):
        """
        センサーデータの読み取りシミュレーション

        実際の実装では、ここで物理ハードウェアとインターフェースします。
        現在は0.5から1.5の範囲でランダム値を生成しています。

        Returns:
            float: シミュレートされたセンサー値
        """
        # Simulate hardware sensor reading - ハードウェアセンサー読み取りのシミュレーション
        # In real implementation, this would interface with actual hardware - 実装時は実際のハードウェアとインターフェース
        return random.uniform(0.5, 1.5)  # 0.5Vから1.5Vの範囲でランダムセンサー値

    def _send_actuator_command(self, command):
        """
        アクチュエーターへのコマンド送信シミュレーション

        実際の実装では、ここで物理アクチュエーターを制御します。
        現在はコンソールにコマンド値を出力しています。

        Args:
            command: アクチュエーターへのコマンド値
        """
        # Simulate sending command to hardware actuator - ハードウェアアクチュエーターへのコマンド送信シミュレーション
        # In real implementation, this would control actual hardware - 実装時は実際のハードウェアを制御
        print(
            f"Hardware actuator command: {command:.3f}"
        )  # コマンド値をコンソールに表示


# スタンドアロン実行時の処理 - シミュレーターを直接起動
if __name__ == "__main__":
    # Mosaik APIを使用してハードウェアシミュレーターを開始
    mosaik_api.start_simulator(HardwareSimulator())
