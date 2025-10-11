"""
データコレクターシミュレーター (Data Collector Simulator)

シミュレーション実行中に各シミュレーターからのデータを収集し、
HDF5 形式で保存する mosaik シミュレーター。

主な機能:
- 数値シミュレーターの出力データ収集
- ハードウェアシミュレーターのセンサー値・アクチュエーターコマンド収集
- 時系列データを HDF5 ファイルにエクスポート
- リアルタイムコンソール表示
"""

from datetime import datetime
from pathlib import Path

import mosaik_api

try:
    import h5py
except ImportError:  # h5py is an optional dependency
    h5py = None


# Mosaikシミュレーターメタデータ - データコレクターの仕様を定義
meta = {
    "type": "time-based",  # 時間ベースのシミュレーション
    "models": {
        "DataCollector": {  # データコレクターモデル
            "public": True,  # 他のシミュレーターから利用可能
            "params": ["output_dir"],  # 出力先ディレクトリ（オプション）
            "attrs": [
                "data",
                "output",
                "sensor_value",
                "actuator_command",
            ],  # 収集可能な属性
        },
    },
}


class DataCollectorSimulator(mosaik_api.Simulator):
    """
    データコレクターシミュレーター実装クラス

    他のシミュレーターからのデータを収集し、
    HDF5 ファイルとして保存します。
    """

    def __init__(self):
        """シミュレーターの初期化"""
        super().__init__(meta)
        self.eid_prefix = "DataCollector_"  # エンティティID接頭辞
        self.entities = {}  # エンティティデータ保存用辞書
        self.step_size = 1  # デフォルトステップサイズ
        self.data_log = []  # 収集したデータのログ

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

    def create(self, num, model, output_dir=None):
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
            eid = f"{self.eid_prefix}{i}"  # ユニークなエンティティID生成
            target_dir = Path(output_dir) if output_dir else Path.cwd()
            target_dir.mkdir(parents=True, exist_ok=True)
            # データコレクターの状態データを初期化
            self.entities[eid] = {
                "data": [],  # 収集したデータのリスト
                "current_time": 0,  # 現在のシミュレーション時刻
                "output_dir": target_dir,
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
            entity["current_time"] = time  # 現在時刻を更新

            # Collect all input data - 全ての入力データを収集
            if eid in inputs:
                data_point = {"time": time}  # 時刻情報を含むデータポイントを作成
                for attr, values in inputs[eid].items():  # 各属性に対して
                    for source_eid, value in values.items():  # 各ソースからの値に対して
                        data_point[f"{attr}_{source_eid}"] = (
                            value  # "属性_ソースID"の形式でデータを保存
                        )
                        # リアルタイムデータ表示（コンソールログ）
                        print(f"Time {time}: {attr} from {source_eid} = {value:.3f}")

                entity["data"].append(data_point)  # エンティティのデータリストに追加
                self.data_log.append(data_point)  # グローバルデータログに追加

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

    def finalize(self):
        """
        シミュレーション終了時の処理

        収集した全データを HDF5 ファイルに保存します。
        このメソッドは mosaik がシミュレーション終了時に自動的に呼び出します。
        """
        if not self.entities:
            return

        if h5py is None:
            print(
                "h5py not available; skipped HDF5 export."
            )  # 依存欠如時にスキップを通知
            return

        if not self.data_log:
            print("No data collected; nothing to write to HDF5.")
            return

        # Save collected data to file - 収集データをファイルに保存
        first_entity = next(iter(self.entities.values()))
        output_dir: Path = first_entity["output_dir"]
        
        # ディレクトリが存在することを確認
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / "simulation_data.h5"

        keys = sorted({key for entry in self.data_log for key in entry.keys()})

        with h5py.File(output_path, "w") as h5_file:
            steps_group = h5_file.create_group("steps")
            for key in keys:
                column = []
                for entry in self.data_log:
                    value = entry.get(key)
                    column.append(float("nan") if value is None else float(value))
                steps_group.create_dataset(name=key, data=column)

            steps_group.attrs["created_at"] = datetime.utcnow().isoformat() + "Z"
            steps_group.attrs["num_steps"] = len(self.data_log)

        print(
            f"Saved {len(self.data_log)} data points to {output_path}"
        )  # 保存結果をコンソールに表示


# スタンドアロン実行時の処理 - シミュレーターを直接起動
if __name__ == "__main__":
    # Mosaik APIを使用してデータコレクターシミュレーターを開始
    mosaik_api.start_simulator(DataCollectorSimulator())
