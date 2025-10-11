"""
数値シミュレーター (Numerical Simulator)

正弦波を生成する数学的モデルを実装したMosaikシミュレーター。
ハードウェアシミュレーターからのフィードバックを受信して、
出力値を調整する機能も持っています。

主な機能:
- 時間に応じた正弦波の生成
- ハードウェアからの入力に基づく出力調整
- Mosaikエコシステムとの連携
"""

import mosaik_api


# Mosaikシミュレーターメタデータ - シミュレーターの仕様を定義
meta = {
    "type": "time-based",  # 時間ベースのシミュレーション
    "models": {
        "NumericalModel": {  # 数値モデルの定義
            "public": True,  # 他のシミュレーターから利用可能
            "params": ["initial_value", "step_size"],  # 初期化パラメータ
            "attrs": ["value", "output", "hardware_input"],  # 公開属性
        },
    },
}


class NumericalSimulator(mosaik_api.Simulator):
    """
    数値シミュレーター実装クラス

    Mosaikフレームワークと連携して正弦波生成と
    ハードウェアフィードバック処理を行います。
    """

    def __init__(self):
        """シミュレーターの初期化"""
        super().__init__(meta)
        self.eid_prefix = "NumSim_"  # エンティティID接頭辞
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

    def create(self, num, model, initial_value=0.0, step_size=1.0):
        """
        モデルエンティティの作成

        Args:
            num: 作成するエンティティ数
            model: モデルタイプ
            initial_value: 初期値（デフォルト0.0）
            step_size: ステップサイズ（デフォルト1.0）

        Returns:
            entities: 作成されたエンティティのリスト
        """
        entities = []
        for i in range(num):
            eid = f"{self.eid_prefix}{i}"  # ユニークなエンティティID生成
            # エンティティの状態データを初期化
            self.entities[eid] = {
                "value": initial_value,  # 現在の値
                "step_size": step_size,  # ステップサイズ
                "time": 0,  # 現在時刻
                "hardware_input": 1.0,  # ハードウェア入力（初期値1.0）
            }
            entities.append({"eid": eid, "type": model})
        return entities

    def step(self, time, inputs, max_advance=None):
        """
        シミュレーションステップの実行

        各エンティティに対して正弦波を計算し、
        ハードウェアからの入力を処理します。

        Args:
            time: 現在のシミュレーション時刻
            inputs: 他のシミュレーターからの入力データ
            max_advance: 最大進行時間（オプション）

        Returns:
            next_step: 次のステップ時刻
        """
        for eid, entity in self.entities.items():
            # Simple numerical simulation: sine wave generation - 簡単な数値シミュレーション：正弦波生成
            import math

            entity["time"] = time  # 現在時刻を更新
            # 正弦波計算：sin(time * step_size * 0.1) で緩やかな正弦波を生成
            entity["value"] = math.sin(time * entity["step_size"] * 0.1)
            entity["output"] = entity["value"]  # 出力値を設定

            # Process inputs from hardware simulator - ハードウェアシミュレーターからの入力処理
            if eid in inputs:
                for attr, values in inputs[eid].items():
                    if attr == "hardware_input":
                        # Modify simulation based on hardware feedback - ハードウェアフィードバックに基づいてシミュレーションを修正
                        hardware_value = list(values.values())[
                            0
                        ]  # ハードウェア値を取得
                        entity["value"] = (
                            entity["value"] * hardware_value
                        )  # 出力値をハードウェア値で乗算

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


# スタンドアロン実行時の処理 - シミュレーターを直接起動
if __name__ == "__main__":
    # Mosaik APIを使用してシミュレーターを開始
    mosaik_api.start_simulator(NumericalSimulator())
