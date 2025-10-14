"""
BridgeSimulator - 通信遅延シミュレーター

制御指令経路（cmd）と測定経路（sense）の通信遅延・ジッターを模擬。
各経路は別々のインスタンスとして動作し、非対称な遅延設定が可能。

初期実装: 遅延・ジッター・パケットロスのみ（補償機能なし）
"""

import random
from typing import Any, List, Tuple

import mosaik_api


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
            ],
            "attrs": [
                "input",  # 入力: 任意のデータ
                "delayed_output",  # 出力: 遅延後のデータ
                "stats",  # 統計情報
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
        bridge_type="cmd",
        base_delay=50,
        jitter_std=10,
        packet_loss_rate=0.01,
        preserve_order=True,
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
        """
        entities = []

        for i in range(num):
            eid = f"{model}_{i}"

            self.entities[eid] = {
                "bridge_type": bridge_type,
                "base_delay": base_delay,
                "jitter_std": jitter_std,
                "packet_loss_rate": packet_loss_rate,
                "preserve_order": preserve_order,
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
                    "avg_delay": base_delay,
                },
            }

            entities.append({"eid": eid, "type": model})
            print(
                f"[BridgeSim] Created {eid} ({bridge_type}): "
                f"delay={base_delay}ms, jitter={jitter_std}ms, loss={packet_loss_rate * 100:.1f}%"
            )

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
                entity["input"] = input_data

                # ジッター計算（ガウス分布）
                jitter = (
                    random.gauss(
                        0,
                        entity["jitter_std"],
                    )
                    if entity["jitter_std"] > 0
                    else 0
                )
                scheduled_time = time + entity["base_delay"] + jitter

                # パケットロス判定
                if random.random() >= entity["packet_loss_rate"]:
                    # パケットをバッファに追加
                    packet = (
                        time,
                        input_data,
                        scheduled_time,
                        entity["sequence_counter"],
                    )
                    entity["packet_buffer"].append(packet)
                    entity["sequence_counter"] += 1
                    entity["stats"]["packets_received"] += 1
                else:
                    # パケットロス
                    entity["stats"]["packets_dropped"] += 1

            # 2. 出力準備ができたパケットの処理
            ready_packets = [pkt for pkt in entity["packet_buffer"] if pkt[2] <= time]

            if ready_packets:
                if entity["preserve_order"]:
                    # 順序保持: 最も古いシーケンス番号のパケットを出力
                    oldest_packet = min(
                        ready_packets,
                        key=lambda x: x[3],
                    )
                    entity["current_output"] = oldest_packet[1]

                    # そのパケットより古いものを全て削除
                    entity["packet_buffer"] = [
                        pkt
                        for pkt in entity["packet_buffer"]
                        if pkt[3] > oldest_packet[3]
                    ]
                else:
                    # 順序無視: 最新のパケットを出力
                    latest_packet = max(
                        ready_packets,
                        key=lambda x: x[2],
                    )
                    entity["current_output"] = latest_packet[1]

                    # 準備完了パケットを全て削除
                    for pkt in ready_packets:
                        entity["packet_buffer"].remove(pkt)

                entity["stats"]["packets_sent"] += 1

            # 3. 統計情報の更新
            if entity["packet_buffer"]:
                total_delay = sum(pkt[2] - pkt[0] for pkt in entity["packet_buffer"])
                entity["stats"]["avg_delay"] = total_delay / len(
                    entity["packet_buffer"]
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
                if attr == "delayed_output":
                    data[eid][attr] = entity["current_output"]
                elif attr == "stats":
                    # 統計情報を数値として出力（WebVis対応）
                    data[eid][attr] = entity["stats"]["avg_delay"]
                elif attr == "input":
                    data[eid][attr] = entity["input"]
                else:
                    data[eid][attr] = None

        return data


if __name__ == "__main__":
    mosaik_api.start_simulator(BridgeSimulator())
