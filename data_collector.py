"""
データコレクターシミュレーター (Data Collector Simulator)

シミュレーション実行中に各シミュレーターからのデータを収集し、
JSONファイルに保存するMosaikシミュレーター。

主な機能:
- 数値シミュレーターの出力データ収集
- ハードウェアシミュレーターのセンサー値・アクチュエーターコマンド収集
- 時系列データのJSONファイル出力
- リアルタイムコンソール表示
"""

import mosaik_api
import json
import time


# Mosaikシミュレーターメタデータ - データコレクターの仕様を定義
meta = {
    "type": "time-based",                                        # 時間ベースのシミュレーション
    "models": {
        "DataCollector": {                                       # データコレクターモデル
            "public": True,                                       # 他のシミュレーターから利用可能
            "params": [],                                        # 初期化パラメータ（なし）
            "attrs": ["data", "output", "sensor_value", "actuator_command"],  # 収集可能な属性
        },
    },
}


class DataCollectorSimulator(mosaik_api.Simulator):
    """
    データコレクターシミュレーター実装クラス
    
    他のシミュレーターからのデータを収集し、
    JSONファイルとして保存します。
    """
    
    def __init__(self):
        """シミュレーターの初期化"""
        super().__init__(meta)
        self.eid_prefix = "DataCollector_"    # エンティティID接頭辞
        self.entities = {}                     # エンティティデータ保存用辞書
        self.step_size = 1                     # デフォルトステップサイズ
        self.data_log = []                     # 収集したデータのログ

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

    def create(self, num, model):
        """
        データコレクターエンティティの作成
        
        Args:
            num: 作成するエンティティ数
            model: モデルタイプ
            
        Returns:
            entities: 作成されたエンティティのリスト
        """
        entities = []
        for i in range(num):
            eid = f"{self.eid_prefix}{i}"         # ユニークなエンティティID生成
            # データコレクターの状態データを初期化
            self.entities[eid] = {
                "data": [],                      # 収集したデータのリスト
                "current_time": 0,               # 現在のシミュレーション時刻
            }
            entities.append({"eid": eid, "type": model})
        return entities

    def step(self, time, inputs, max_advance=None):
        """
        データ収集ステップの実行
        
        各シミュレーターからの入力データを収集し、
        ログに記録してコンソールに表示します。
        
        Args:
            time: 現在のシミュレーション時刻
            inputs: 他のシミュレーターからの入力データ
            max_advance: 最大進行時間（オプション）
            
        Returns:
            next_step: 次のステップ時刻
        """
        for eid, entity in self.entities.items():
            entity["current_time"] = time              # 現在時刻を更新

            # Collect all input data - 全ての入力データを収集
            if eid in inputs:
                data_point = {"time": time}           # 時刻情報を含むデータポイントを作成
                for attr, values in inputs[eid].items():  # 各属性に対して
                    for source_eid, value in values.items():  # 各ソースからの値に対して
                        data_point[f"{attr}_{source_eid}"] = value  # "属性_ソースID"の形式でデータを保存
                        # リアルタイムデータ表示（コンソールログ）
                        print(f"Time {time}: {attr} from {source_eid} = {value:.3f}")

                entity["data"].append(data_point)     # エンティティのデータリストに追加
                self.data_log.append(data_point)      # グローバルデータログに追加

        return time + self.step_size                  # 次のステップ時刻を返す

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
        for eid, attrs in outputs.items():              # 各エンティティに対して
            data[eid] = {}
            for attr in attrs:                          # リクエストされた各属性に対して
                if attr in self.entities[eid]:         # 属性が存在する場合
                    data[eid][attr] = self.entities[eid][attr]  # 属性値をコピー
        return data

    def finalize(self):
        """
        シミュレーション終了時の処理
        
        収集した全データをJSONファイルに保存します。
        このメソッドはMosaikがシミュレーション終了時に自動的に呼び出します。
        """
        # Save collected data to file - 収集データをファイルに保存
        with open("simulation_data.json", "w") as f:
            json.dump(self.data_log, f, indent=2)  # 美しいフォーマットでJSON出力
        print(f"Saved {len(self.data_log)} data points to simulation_data.json")  # 保存結果をコンソールに表示


# スタンドアロン実行時の処理 - シミュレーターを直接起動
if __name__ == "__main__":
    # Mosaik APIを使用してデータコレクターシミュレーターを開始
    mosaik_api.start_simulator(DataCollectorSimulator())
