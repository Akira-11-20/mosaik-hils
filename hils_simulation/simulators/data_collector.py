"""
DataCollectorSimulator - HILS用データ収集器

シミュレーション中の全データを収集し、HDF5形式で保存する。

収集データ:
- Controller: command, error, position, velocity
- Bridge(cmd): stats
- Plant: measured_thrust, status
- Bridge(sense): stats
- Env: position, velocity, acceleration, force
"""
import json
from datetime import datetime
from pathlib import Path

import mosaik_api
import numpy as np

try:
    import h5py
except ImportError:
    h5py = None

meta = {
    "type": "time-based",
    "models": {
        "Collector": {
            "public": True,
            "params": ["output_dir"],
            "attrs": [],  # 任意の属性を受け入れる（動的収集）
            "any_inputs": True,  # 任意の入力を受け入れる
        },
    },
}


class DataCollectorSimulator(mosaik_api.Simulator):
    """
    データ収集器シミュレーター（HILS版）

    全シミュレーターからデータを収集し、HDF5ファイルに保存。
    """

    def __init__(self):
        super().__init__(meta)
        self.entities = {}
        self.step_size = 1  # 1ms毎に記録
        self.data_log = []

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
            step_size: ステップサイズ
        """
        self.sid = sid
        self.time_resolution = time_resolution
        self.step_size = step_size
        return self.meta

    def create(self, num, model, output_dir=None):
        """
        データコレクターエンティティの作成

        Args:
            num: 作成数
            model: モデル名
            output_dir: 出力ディレクトリ
        """
        entities = []

        for i in range(num):
            eid = f"{model}_{i}"
            target_dir = Path(output_dir) if output_dir else Path.cwd()
            target_dir.mkdir(parents=True, exist_ok=True)

            self.entities[eid] = {
                "data": [],
                "current_time": 0,
                "output_dir": target_dir,
            }

            entities.append({"eid": eid, "type": model})
            print(f"[DataCollector] Created {eid} -> {target_dir}")

        return entities

    def step(self, time, inputs, max_advance=None):
        """
        データ収集ステップ

        Args:
            time: 現在時刻 [ms]
            inputs: 入力データ
        """
        # 実時間 [秒]
        real_time = time * self.time_resolution

        for (
            eid,
            entity,
        ) in self.entities.items():
            entity["current_time"] = time

            if eid in inputs:
                # データポイントの作成
                data_point = {
                    "time_ms": time,  # シミュレーション時刻 [ms]
                    "time_s": real_time,  # 実時間 [s]
                }

                # 全入力データを収集
                for attr, values in inputs[eid].items():
                    for (
                        source_eid,
                        value,
                    ) in values.items():
                        # 属性名とソースIDでキーを作成
                        key = f"{attr}_{source_eid}"

                        # 値の型に応じて処理
                        if isinstance(value, dict):
                            # 辞書型（例: command）
                            # JSON文字列として保存
                            import json

                            data_point[key] = json.dumps(value)

                            # 各要素も個別に記録（プロット用）
                            for (
                                k,
                                v,
                            ) in value.items():
                                data_point[f"{key}_{k}"] = v

                        elif isinstance(
                            value,
                            (int, float),
                        ):
                            # 数値型
                            data_point[key] = value

                        elif value is None:
                            # None値
                            data_point[key] = float("nan")

                        else:
                            # その他（文字列等）
                            data_point[key] = str(value)

                entity["data"].append(data_point)
                self.data_log.append(data_point)

        return time + self.step_size

    def get_data(self, outputs):
        """
        データ取得（通常は使用しない）
        """
        return {}

    def finalize(self):
        """
        シミュレーション終了時の処理

        収集した全データをHDF5ファイルに保存。
        """
        if not self.entities:
            print("[DataCollector] No entities to finalize.")
            return

        if h5py is None:
            print(
                "[DataCollector] ⚠️  h5py not available; skipped HDF5 export."
            )
            return

        if not self.data_log:
            print(
                "[DataCollector] No data collected; nothing to write."
            )
            return

        # 出力先ディレクトリ
        first_entity = next(iter(self.entities.values()))
        output_dir: Path = first_entity["output_dir"]
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / "hils_data.h5"

        print(
            f"\n[DataCollector] 💾 Saving {len(self.data_log)} data points to HDF5..."
        )

        # 全キーを収集
        all_keys = sorted(
            {key for entry in self.data_log for key in entry.keys()}
        )

        with h5py.File(output_path, "w") as h5_file:
            # メタデータ
            h5_file.attrs["created_at"] = (
                datetime.utcnow().isoformat() + "Z"
            )
            h5_file.attrs["num_samples"] = len(self.data_log)
            h5_file.attrs["time_resolution"] = self.time_resolution

            # データグループ
            data_group = h5_file.create_group("data")

            # 各属性ごとにデータセットを作成
            for key in all_keys:
                column = []
                for entry in self.data_log:
                    value = entry.get(key)

                    if value is None:
                        column.append(float("nan"))
                    elif isinstance(value, str):
                        # 文字列はそのまま
                        column.append(value)
                    elif isinstance(value, (int, float)):
                        column.append(float(value))
                    else:
                        column.append(str(value))

                # データ型に応じてデータセットを作成
                if column and isinstance(column[0], str):
                    # 文字列データ
                    data_group.create_dataset(
                        name=key,
                        data=column,
                        dtype=h5py.string_dtype(),
                    )
                else:
                    # 数値データ
                    data_group.create_dataset(
                        name=key,
                        data=column,
                    )

            # データセット一覧を表示
            print(f"[DataCollector] ✅ Saved datasets:")
            for key in sorted(data_group.keys()):
                dataset = data_group[key]
                print(f"  - {key}: {dataset.shape} {dataset.dtype}")

            # data_with_time_s グループを作成（time_s を x軸として対応付け）
            if "time_s" in data_group:
                print(f"\n[DataCollector] 📊 Creating data_with_time_s group...")
                time_s_data = data_group["time_s"][:]

                data_with_time = h5_file.create_group("data_with_time_s")
                data_with_time.attrs["description"] = "Datasets paired with time_s axis"
                data_with_time.attrs["source_group"] = "data"

                skip_keys = ["time_s", "time_ms"]
                created_count = 0

                for key in all_keys:
                    if key in skip_keys:
                        continue

                    # object型（文字列）はスキップ
                    dataset = data_group[key]
                    try:
                        # 数値型かチェック
                        if not np.issubdtype(dataset.dtype, np.number):
                            continue
                    except TypeError:
                        continue

                    # 2次元配列として作成: (N, 2) where [:, 0]=time_s, [:, 1]=value
                    combined_data = np.column_stack((time_s_data, dataset[:]))

                    data_with_time.create_dataset(
                        name=key,
                        data=combined_data,
                        dtype=np.float64
                    )
                    data_with_time[key].attrs["columns"] = "time_s, value"
                    data_with_time[key].attrs["unit"] = key.split("_")[0]
                    created_count += 1

                print(f"[DataCollector] ✅ Created {created_count} datasets in data_with_time_s/ (Nx2 format)")

                # 説明を更新
                data_with_time.attrs["description"] = "Datasets with time_s as x-axis (Nx2 arrays)"
                data_with_time.attrs["format"] = "Each dataset is (N, 2) where [:, 0]=time_s, [:, 1]=value"

        print(f"[DataCollector] 📁 Output: {output_path}")


if __name__ == "__main__":
    mosaik_api.start_simulator(DataCollectorSimulator())
