"""
BridgeSimulator - 通信遅延シミュレーター

制御指令経路（cmd）と測定経路（sense）の通信遅延・ジッターを模擬。
各経路は別々のインスタンスとして動作し、非対称な遅延設定が可能。

初期実装: 遅延・ジッター・パケットロスのみ（補償機能なし）
"""

import random

import mosaik_api

try:
    import sys
    from pathlib import Path

    # プロジェクトルートをパスに追加
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

    from common_utils.event_logger import DataTag, EventLogger
except ImportError:
    EventLogger = None
    DataTag = None


meta = {
    "type": "time-based",
    "models": {
        "CommBridge": {
            "public": True,
            "params": [
                "bridge_type",  # "cmd" or "sense"
                "base_delay",  # 基本遅延 [ms]
                "jitter_std",  # ジッター標準偏差 [ms]
                "packet_loss_rate",  # パケットロス率 [0.0-1.0]
                "preserve_order",  # パケット順序保持
                "compensate_time_shifted",  # time_shifted接続の遅延を補償するか
                "time_shifted_delay_ms",  # time_shifted接続による遅延量 [ms]
            ],
            "attrs": [
                "input",  # 入力: 任意のデータ
                "delayed_output",  # 出力: 遅延後のデータ
                "stats",  # 統計情報
                "packet_receive_time",  # パケット受信時刻 [ms]
                "packet_send_time",  # パケット送信時刻 [ms]
                "packet_actual_delay",  # 実際の遅延時間 [ms]
                # Debug attributes
                "buffer_size",  # バッファー内のパケット数
                "buffer_content",  # バッファー内容（JSON文字列）
                "oldest_packet_time",  # 最古パケットの到着時刻 [ms]
                "newest_packet_time",  # 最新パケットの到着時刻 [ms]
            ],
        },
    },
}


class BridgeSimulator(mosaik_api.Simulator):
    """
    通信ブリッジシミュレーター

    データの遅延・ジッター・パケットロスを模擬する。
    補償機能は将来実装予定。
    """

    def __init__(self):
        super().__init__(meta)
        self.entities = {}
        self.step_size = 1  # デフォルト: 1ms（高頻度実行）
        self.time = 0
        self.loggers = {}  # エンティティごとのロガー
        self.output_dir = None
        self.time_resolution = 0.001
        self.step_ms = self.time_resolution * 1000

    def init(
        self,
        sid,
        time_resolution=0.001,
        step_size=1,
        log_dir=None,
    ):
        """
        初期化

        Args:
            sid: シミュレーターID
            time_resolution: 時間解像度（0.001 = 1ms）
            step_size: ステップサイズ [時間単位]
            log_dir: ログ出力ディレクトリ（オプション）
        """
        self.sid = sid
        self.time_resolution = time_resolution
        self.step_size = step_size
        self.output_dir = log_dir
        self.step_ms = self.time_resolution * 1000 if self.time_resolution else 1.0
        return self.meta

    def _ms_to_steps(self, value_ms: float) -> float:
        """Convert milliseconds to simulator steps."""
        if not self.step_ms:
            return value_ms
        return value_ms / self.step_ms

    def _steps_to_ms(self, steps: float) -> float:
        """Convert simulator steps to milliseconds."""
        return steps * self.step_ms

    def create(
        self,
        num,
        model,
        bridge_type="cmd",
        base_delay=50,
        jitter_std=10,
        packet_loss_rate=0.01,
        preserve_order=True,
        compensate_time_shifted=False,
        time_shifted_delay_ms=0.0,
    ):
        """
        通信ブリッジエンティティの作成

        Args:
            num: 作成数
            model: モデル名
            bridge_type: ブリッジタイプ（"cmd" or "sense"）
            base_delay: 基本遅延 [ms]
            jitter_std: ジッター標準偏差 [ms]
            packet_loss_rate: パケットロス率（0.0～1.0）
            preserve_order: パケット順序保持フラグ
            compensate_time_shifted: time_shifted接続の遅延を補償するか
            time_shifted_delay_ms: time_shifted接続による遅延量 [ms]（通常は制御周期）

        Note:
            time_resolutionはinit()で設定された値を使用します
            compensate_time_shifted=Trueの場合、実際の遅延は
            base_delay - time_shifted_delay_ms となります
        """
        entities = []

        for i in range(num):
            eid = f"{model}_{i}"

            # time_shifted補償の計算
            actual_delay = base_delay
            if compensate_time_shifted and time_shifted_delay_ms > 0:
                actual_delay = max(0, base_delay - time_shifted_delay_ms)
                print("[BridgeSim] Compensating for time_shifted delay:")
                print(f"  Configured delay: {base_delay}ms")
                print(f"  time_shifted delay: {time_shifted_delay_ms}ms")
                print(f"  Actual Bridge delay: {actual_delay}ms")
                print(f"  Total delay: {base_delay}ms")

            self.entities[eid] = {
                "bridge_type": bridge_type,
                "base_delay_ms": actual_delay,
                "base_delay_steps": self._ms_to_steps(actual_delay),
                "jitter_std_ms": jitter_std,
                "jitter_std_steps": self._ms_to_steps(jitter_std),
                "packet_loss_rate": packet_loss_rate,
                "preserve_order": preserve_order,
                "compensate_time_shifted": compensate_time_shifted,
                "configured_delay_ms": base_delay,  # 設定値を保存
                # パケットバッファ: (arrival_time, data, scheduled_output_time, seq_num)
                "packet_buffer": [],
                "sequence_counter": 0,
                "current_output": None,
                "input": None,
                # 統計情報
                "stats": {
                    "packets_received": 0,
                    "packets_sent": 0,
                    "packets_dropped": 0,
                    "avg_delay": actual_delay,
                },
                # パケット追跡用データ（最後に送信したパケットの情報）
                "last_packet_receive_time": None,
                "last_packet_send_time": None,
                "last_packet_actual_delay": None,
            }

            entities.append({"eid": eid, "type": model})
            if compensate_time_shifted:
                print(
                    f"[BridgeSim] Created {eid} ({bridge_type}): configured_delay={base_delay}ms (actual={actual_delay}ms), jitter={jitter_std}ms, loss={packet_loss_rate * 100:.1f}%"
                )
            else:
                print(
                    f"[BridgeSim] Created {eid} ({bridge_type}): delay={actual_delay}ms, jitter={jitter_std}ms, loss={packet_loss_rate * 100:.1f}%"
                )

            # イベントロガーの作成
            if EventLogger and self.output_dir:
                logger_name = f"Bridge_{bridge_type}_{i}"
                self.loggers[eid] = EventLogger(self.output_dir, logger_name)

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
            # 1. 新しいデータの受信処理
            if eid in inputs and "input" in inputs[eid]:
                input_data = list(inputs[eid]["input"].values())[0]
                source_eid = list(inputs[eid]["input"].keys())[0]
                entity["input"] = input_data

                # データタグを作成/抽出
                if DataTag:
                    existing_tag = DataTag.extract(input_data)
                    if not existing_tag:
                        # タグがない場合は新規作成
                        data_tag = DataTag.create(
                            sender=source_eid,
                            send_time_ms=self._steps_to_ms(time),
                            sequence_num=entity["sequence_counter"],
                            data_type=entity["bridge_type"],
                        )
                    else:
                        # 既存のタグを使用（シーケンス番号のみ更新）
                        data_tag = existing_tag.copy()
                        data_tag["bridge_seq"] = entity["sequence_counter"]
                else:
                    data_tag = {}

                # ジッター計算（ガウス分布）
                jitter_std_steps = entity["jitter_std_steps"]
                jitter = (
                    random.gauss(
                        0,
                        jitter_std_steps,
                    )
                    if jitter_std_steps > 0
                    else 0
                )
                scheduled_time = time + entity["base_delay_steps"] + jitter

                # パケットロス判定
                if random.random() >= entity["packet_loss_rate"]:
                    # パケットをバッファに追加
                    packet = (
                        time,
                        input_data,
                        scheduled_time,
                        entity["sequence_counter"],
                        data_tag,  # タグを追加
                    )
                    entity["packet_buffer"].append(packet)
                    entity["sequence_counter"] += 1
                    entity["stats"]["packets_received"] += 1

                    # ログ: 受信イベント
                    if eid in self.loggers:
                        self.loggers[eid].log_receive(
                            time_ms=self._steps_to_ms(time),
                            source=source_eid,
                            value=input_data,
                            data_tag=data_tag,
                        )
                else:
                    # パケットロス
                    entity["stats"]["packets_dropped"] += 1

                    # ログ: ドロップイベント
                    if eid in self.loggers:
                        self.loggers[eid].log_drop(
                            time_ms=self._steps_to_ms(time),
                            reason="packet_loss",
                            value=input_data,
                            data_tag=data_tag,
                        )

            # 2. 出力準備ができたパケットの処理
            ready_packets = [pkt for pkt in entity["packet_buffer"] if pkt[2] <= time]

            if ready_packets:
                selected_packet = None
                if entity["preserve_order"]:
                    # 順序保持: 最も古いシーケンス番号のパケットを出力
                    selected_packet = min(
                        ready_packets,
                        key=lambda x: x[3],
                    )
                    entity["current_output"] = selected_packet[1]

                    # そのパケットより古いものを全て削除
                    entity["packet_buffer"] = [pkt for pkt in entity["packet_buffer"] if pkt[3] > selected_packet[3]]
                else:
                    # 順序無視: 最新のパケットを出力
                    selected_packet = max(
                        ready_packets,
                        key=lambda x: x[2],
                    )
                    entity["current_output"] = selected_packet[1]

                    # 準備完了パケットを全て削除
                    for pkt in ready_packets:
                        entity["packet_buffer"].remove(pkt)

                    entity["stats"]["packets_sent"] += 1

                # パケット追跡データを記録
                if selected_packet:
                    arrival_time, data, scheduled_time, _seq_num, data_tag = selected_packet
                    actual_delay_steps = time - arrival_time
                    actual_delay_ms = self._steps_to_ms(actual_delay_steps)

                    entity["last_packet_receive_time"] = self._steps_to_ms(arrival_time)
                    entity["last_packet_send_time"] = self._steps_to_ms(time)
                    entity["last_packet_actual_delay"] = actual_delay_ms

                # ログ: 送信イベント
                if eid in self.loggers and selected_packet:
                    arrival_time, data, scheduled_time, _seq_num, data_tag = selected_packet
                    actual_delay_steps = time - arrival_time
                    actual_delay_ms = self._steps_to_ms(actual_delay_steps)
                    self.loggers[eid].log_send(
                        time_ms=self._steps_to_ms(time),
                        destination="downstream",
                        value=data,
                        data_tag={
                            **data_tag,
                            "actual_delay_ms": actual_delay_ms,
                            "scheduled_time_ms": self._steps_to_ms(scheduled_time),
                        },
                    )

            # 3. 統計情報の更新
            if entity["packet_buffer"]:
                total_delay = sum(pkt[2] - pkt[0] for pkt in entity["packet_buffer"])
                entity["stats"]["avg_delay"] = (total_delay / len(entity["packet_buffer"])) * self.step_ms

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
                if attr == "delayed_output":
                    data[eid][attr] = entity["current_output"]
                elif attr == "stats":
                    # 統計情報を数値として出力（WebVis対応）
                    data[eid][attr] = entity["stats"]["avg_delay"]
                elif attr == "input":
                    data[eid][attr] = entity["input"]
                elif attr == "packet_receive_time":
                    data[eid][attr] = entity["last_packet_receive_time"]
                elif attr == "packet_send_time":
                    data[eid][attr] = entity["last_packet_send_time"]
                elif attr == "packet_actual_delay":
                    data[eid][attr] = entity["last_packet_actual_delay"]
                # Debug attributes
                elif attr == "buffer_size":
                    data[eid][attr] = len(entity["packet_buffer"])
                elif attr == "buffer_content":
                    # バッファー内容をJSON文字列として返す
                    import json

                    buffer_info = []
                    for pkt in entity["packet_buffer"]:
                        arrival_time, pkt_data, scheduled_time, seq_num, _tag = pkt
                        buffer_info.append(
                            {
                                "seq": seq_num,
                                "arrival_ms": self._steps_to_ms(arrival_time),
                                "scheduled_ms": self._steps_to_ms(scheduled_time),
                                "data": str(pkt_data)[:50],  # データの一部のみ
                            }
                        )
                    data[eid][attr] = json.dumps(buffer_info)
                elif attr == "oldest_packet_time":
                    if entity["packet_buffer"]:
                        oldest = min(entity["packet_buffer"], key=lambda x: x[0])
                        data[eid][attr] = self._steps_to_ms(oldest[0])
                    else:
                        data[eid][attr] = None
                elif attr == "newest_packet_time":
                    if entity["packet_buffer"]:
                        newest = max(entity["packet_buffer"], key=lambda x: x[0])
                        data[eid][attr] = self._steps_to_ms(newest[0])
                    else:
                        data[eid][attr] = None
                else:
                    data[eid][attr] = None

        return data

    def finalize(self):
        """
        シミュレーション終了時の処理

        イベントログをファイルに保存
        """
        for logger in self.loggers.values():
            logger.finalize()


if __name__ == "__main__":
    mosaik_api.start_simulator(BridgeSimulator())
