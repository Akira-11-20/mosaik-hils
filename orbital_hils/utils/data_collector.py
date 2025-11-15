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

try:
    import h5py
except ImportError:
    h5py = None

meta = {
    "type": "time-based",
    "models": {
        "Collector": {
            "public": True,
            "params": ["output_dir", "minimal_mode"],
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
        self.time_resolution = 0.001
        self.step_ms = self.time_resolution * 1000
        self.all_keys = set()  # キーを効率的に追跡

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
        self.step_ms = self.time_resolution * 1000 if self.time_resolution else 1.0
        return self.meta

    def create(self, num, model, output_dir=None, minimal_mode=False):
        """
        データコレクターエンティティの作成

        Args:
            num: 作成数
            model: モデル名
            output_dir: 出力ディレクトリ
            minimal_mode: 最小限のデータのみ記録（time, position, velocity）
        """
        entities = []

        for i in range(num):
            eid = f"{model}_{i}"
            target_dir = Path(output_dir) if output_dir else Path.cwd()
            target_dir.mkdir(parents=True, exist_ok=True)

            self.entities[eid] = {
                "current_time": 0,
                "output_dir": target_dir,
                "minimal_mode": minimal_mode,
            }

            entities.append({"eid": eid, "type": model})
            mode_str = " (minimal mode)" if minimal_mode else ""
            print(f"[DataCollector] Created {eid} -> {target_dir}{mode_str}")

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
        time_ms = time * self.step_ms

        for (
            eid,
            entity,
        ) in self.entities.items():
            entity["current_time"] = time

            if eid in inputs:
                # Check if minimal mode is enabled
                minimal_mode = entity.get("minimal_mode", False)

                # データポイントの作成（生データのまま保存、JSON変換は最後に実行）
                data_point = {
                    "time_ms": time_ms,  # シミュレーション時刻 [ms]
                    "time_s": real_time,  # 実時間 [s]
                }

                # 全入力データを収集（最小限の処理）
                for attr, values in inputs[eid].items():
                    # Minimal mode: only collect position and velocity from Env
                    if minimal_mode:
                        if attr not in ["position", "velocity"]:
                            continue

                    for source_eid, value in values.items():
                        # Minimal mode: only Env data
                        if minimal_mode and "Env" not in source_eid:
                            continue

                        # 属性名とソースIDでキーを作成
                        key = f"{attr}_{source_eid}"

                        # キーを記録
                        self.all_keys.add(key)

                        # 生データをそのまま保存
                        data_point[key] = value

                        # 辞書型の場合、各要素も記録（プロット用）
                        # Minimal mode: skip dict expansion to save time
                        if isinstance(value, dict) and not minimal_mode:
                            for k, v in value.items():
                                subkey = f"{key}_{k}"
                                data_point[subkey] = v
                                self.all_keys.add(subkey)

                # data_logにのみ保存（重複を避ける）
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
            print("[DataCollector] ⚠️  h5py not available; skipped HDF5 export.")
            return

        if not self.data_log:
            print("[DataCollector] No data collected; nothing to write.")
            return

        # 出力先ディレクトリ
        first_entity = next(iter(self.entities.values()))
        output_dir: Path = first_entity["output_dir"]
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / "hils_data.h5"

        print(f"\n[DataCollector] 💾 Saving {len(self.data_log)} data points to HDF5...")

        # 全キーを収集（ステップ中に追跡したキーを使用）
        self.all_keys.add("time_ms")
        self.all_keys.add("time_s")
        all_keys = sorted(self.all_keys)

        with h5py.File(output_path, "w") as h5_file:
            # メタデータ
            h5_file.attrs["created_at"] = datetime.utcnow().isoformat() + "Z"
            h5_file.attrs["num_samples"] = len(self.data_log)
            h5_file.attrs["time_resolution"] = self.time_resolution

            # 時刻データは共通グループに
            time_group = h5_file.create_group("time")

            # ノードごとにグループを作成
            node_groups = {}

            # 各属性をノードごとに分類
            import re

            for key in all_keys:
                if key in ["time_ms", "time_s"]:
                    # 時刻データは特別扱い
                    target_group = time_group
                    dataset_name = key
                else:
                    # キーをパース: attr_SimName-ID.EntityID[_suffix]
                    # 例1: buffer_size_BridgeSim-0.CommBridge_0 -> node=BridgeSim-0.CommBridge_0, attr=buffer_size
                    # 例2: compensated_output_InverseCompSim-0.cmd_compensator -> node=InverseCompSim-0.cmd_compensator, attr=compensated_output
                    # 例3: command_ControllerSim-0.PIDController_0_thrust -> node=ControllerSim-0.PIDController_0, attr=command_thrust

                    # Match: attr_prefix + SimName-ID. + rest
                    sim_match = re.match(r"([a-z_]+)_([A-Z][a-zA-Z]*Sim-\d+)\.(.+)", key)

                    if sim_match:
                        attr_prefix = sim_match.group(1)
                        sim_name = sim_match.group(2)
                        rest = sim_match.group(3)

                        # Parse 'rest' to separate entity_id from sub-attribute suffixes
                        # Known sub-attribute suffixes (from dict expansion)
                        known_suffixes = ["thrust", "duration"]

                        parts = rest.split("_")
                        suffix_idx = None

                        # Find if any known suffix exists
                        for i, part in enumerate(parts):
                            if part in known_suffixes:
                                suffix_idx = i
                                break

                        if suffix_idx is not None:
                            # Entity ID is everything before the suffix
                            entity_id = "_".join(parts[:suffix_idx])
                            attr_suffix = "_".join(parts[suffix_idx:])
                            attr_name = f"{attr_prefix}_{attr_suffix}"
                        else:
                            # No known suffix - entire rest is entity_id
                            entity_id = rest
                            attr_name = attr_prefix

                        node_name = f"{sim_name}.{entity_id}"

                        # ノードグループを作成（初回のみ）
                        if node_name not in node_groups:
                            # グループ名に使えない文字を置換
                            safe_node_name = node_name.replace(".", "_")
                            node_groups[node_name] = h5_file.create_group(safe_node_name)

                        target_group = node_groups[node_name]
                        dataset_name = attr_name
                    else:
                        # パースできない場合はルートに配置
                        target_group = h5_file
                        dataset_name = key

                # データ収集
                column = []
                for entry in self.data_log:
                    value = entry.get(key)

                    if value is None:
                        column.append(float("nan"))
                    elif isinstance(value, dict):
                        # 辞書型はJSON文字列に変換（finalize時のみ）
                        column.append(json.dumps(value))
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
                    target_group.create_dataset(
                        name=dataset_name,
                        data=column,
                        dtype=h5py.string_dtype(),
                    )
                else:
                    # 数値データ
                    target_group.create_dataset(
                        name=dataset_name,
                        data=column,
                    )

            # データセット一覧を表示（ノードごと）
            print("[DataCollector] ✅ Saved datasets by node:")

            # 時刻データ
            print("\n  [time/]")
            for key in sorted(time_group.keys()):
                dataset = time_group[key]
                print(f"    - {key}: {dataset.shape} {dataset.dtype}")

            # ノードデータ
            for node_name in sorted(node_groups.keys()):
                node_group = node_groups[node_name]
                print(f"\n  [{node_name}/]")
                for key in sorted(node_group.keys()):
                    dataset = node_group[key]
                    print(f"    - {key}: {dataset.shape} {dataset.dtype}")

        print(f"[DataCollector] 📁 Output: {output_path}")


if __name__ == "__main__":
    mosaik_api.start_simulator(DataCollectorSimulator())
