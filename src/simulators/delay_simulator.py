"""
Enhanced Delay Simulator for Mosaik HILS
高頻度実行とジッター機能を持つ遅延シミュレーター

主な機能:
- 高精度タイミング制御（サブ秒精度）
- ガウシアンジッター実装
- パケットロス/順序制御
- リアルタイム遅延計算
"""

import collections
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import mosaik_api


META = {
    "type": "time-based",
    "models": {
        "DelayNode": {
            "public": True,
            "params": [
                "base_delay",
                "jitter_std",
                "packet_loss_rate",
                "preserve_order",
                "max_buffer_size",
            ],
            "attrs": ["input", "delayed_output", "stats"],
        },
    },
}


class DelaySimulator(mosaik_api.Simulator):
    """
    Enhanced delay simulator with jitter and packet management
    """

    def __init__(self):
        super().__init__(META)
        self.entities = {}
        self.time = 0

    def init(self, sid, time_resolution=1, **sim_params):
        """
        Initialize the delay simulator

        Args:
            sid: Simulator ID
            time_resolution: Time resolution in time units (default: 1)
        """
        self.sid = sid
        self.time_resolution = time_resolution
        return self.meta

    def create(self, num, model, **model_params):
        """
        Create delay node entities

        Args:
            num: Number of entities to create
            model: Model type (should be "DelayNode")
            **model_params: Model parameters
        """
        entities = []

        for i in range(num):
            eid = f"{model}_{i}"

            # デフォルトパラメータ
            config = {
                "base_delay": model_params.get("base_delay", 0.1),  # 100ms
                "jitter_std": model_params.get("jitter_std", 0.02),  # 20ms std
                "packet_loss_rate": model_params.get("packet_loss_rate", 0.0),  # 0%
                "preserve_order": model_params.get("preserve_order", True),
                "max_buffer_size": model_params.get("max_buffer_size", 1000),
            }

            self.entities[eid] = DelayNode(eid, **config)
            entities.append({"eid": eid, "type": model})

        return entities

    def step(self, time, inputs, max_advance=None):
        """
        Perform one simulation step

        Args:
            time: Current simulation time
            inputs: Input data from connected simulators
            max_advance: Maximum time advance (unused)
        """
        self.time = time

        # 各遅延ノードを更新
        for eid, entity in self.entities.items():
            entity_inputs = inputs.get(eid, {})
            entity.step(time, entity_inputs)

        # 次のステップまでの時間（高頻度実行）
        return self.time + self.time_resolution

    def get_data(self, outputs):
        """
        Get output data from entities

        Args:
            outputs: Requested output attributes
        """
        data = {}

        for eid, attrs in outputs.items():
            if eid in self.entities:
                data[eid] = {}
                entity = self.entities[eid]

                for attr in attrs:
                    if attr == "delayed_output":
                        data[eid][attr] = entity.get_output()
                    elif attr == "stats":
                        # WebVis用に平均遅延を数値として出力
                        stats = entity.get_stats()
                        if isinstance(stats, dict):
                            # 平均遅延を主要な数値として出力
                            data[eid][attr] = float(stats.get('avg_delay', 0.0))
                        else:
                            data[eid][attr] = 0.0
                    elif attr in ["input"]:  # パススルー属性
                        data[eid][attr] = getattr(entity, attr, None)

        return data


class DelayNode:
    """
    Individual delay node with jitter and packet management
    """

    def __init__(
        self,
        eid: str,
        base_delay: float = 0.1,
        jitter_std: float = 0.02,
        packet_loss_rate: float = 0.0,
        preserve_order: bool = True,
        max_buffer_size: int = 1000,
    ):
        """
        Initialize delay node

        Args:
            eid: Entity ID
            base_delay: Base delay in seconds
            jitter_std: Jitter standard deviation in seconds
            packet_loss_rate: Packet loss probability (0.0-1.0)
            preserve_order: Whether to preserve packet order
            max_buffer_size: Maximum buffer size
        """
        self.eid = eid
        self.base_delay = base_delay
        self.jitter_std = jitter_std
        self.packet_loss_rate = packet_loss_rate
        self.preserve_order = preserve_order
        self.max_buffer_size = max_buffer_size

        # パケットバッファ: (arrival_time, data, scheduled_output_time, sequence_num)
        self.packet_buffer: List[Tuple[float, Any, float, int]] = []
        self.sequence_counter = 0

        # 統計情報
        self.stats = {
            "packets_received": 0,
            "packets_sent": 0,
            "packets_dropped_loss": 0,
            "packets_dropped_buffer": 0,
            "current_buffer_size": 0,
            "avg_delay": 0.0,
            "last_jitter": 0.0,
        }

        # 現在の出力値
        self.current_output = None
        self.input = None

    def step(self, current_time: float, inputs: Dict[str, Any]):
        """
        Process one time step

        Args:
            current_time: Current simulation time
            inputs: Input data dictionary
        """
        # 1. 新しいデータの受信処理
        self._process_inputs(current_time, inputs)

        # 2. 出力準備ができたパケットの処理
        self._process_output_queue(current_time)

        # 3. バッファの整理（古いパケットや溢れた分の削除）
        self._cleanup_buffer(current_time)

    def _process_inputs(self, current_time: float, inputs: Dict[str, Any]):
        """Process incoming data and add to buffer with jitter"""
        for attr, values in inputs.items():
            if attr == "input" and values:
                # 最新の値を取得
                input_data = (
                    list(values.values())[0] if isinstance(values, dict) else values
                )
                self.input = input_data

                # ジッター計算
                jitter = random.gauss(0, self.jitter_std) if self.jitter_std > 0 else 0
                scheduled_output_time = current_time + self.base_delay + jitter

                # パケットロス判定
                if random.random() >= self.packet_loss_rate:
                    # バッファサイズチェック
                    if len(self.packet_buffer) >= self.max_buffer_size:
                        # 最古のパケットを削除
                        self.packet_buffer.pop(0)
                        self.stats["packets_dropped_buffer"] += 1

                    # パケットをバッファに追加
                    packet = (
                        current_time,
                        input_data,
                        scheduled_output_time,
                        self.sequence_counter,
                    )
                    self.packet_buffer.append(packet)
                    self.sequence_counter += 1

                    self.stats["packets_received"] += 1
                    self.stats["last_jitter"] = jitter
                else:
                    self.stats["packets_dropped_loss"] += 1

    def _process_output_queue(self, current_time: float):
        """Process packets ready for output"""
        # 出力準備ができたパケットを抽出
        ready_packets = [pkt for pkt in self.packet_buffer if pkt[2] <= current_time]

        if ready_packets:
            if self.preserve_order:
                # 到着順序を保持する場合、最も早い到着時間のパケットを選択
                oldest_packet = min(ready_packets, key=lambda x: x[3])  # sequence_num順
                self.current_output = oldest_packet[1]

                # そのパケットと、それより古いシーケンス番号のパケットを全て削除
                self.packet_buffer = [
                    pkt for pkt in self.packet_buffer if pkt[3] > oldest_packet[3]
                ]
            else:
                # 順序を保持しない場合、最新のパケットを出力
                latest_packet = max(
                    ready_packets, key=lambda x: x[2]
                )  # scheduled_output_time順
                self.current_output = latest_packet[1]

                # 準備ができた全てのパケットを削除
                for pkt in ready_packets:
                    self.packet_buffer.remove(pkt)

            self.stats["packets_sent"] += len(ready_packets)

    def _cleanup_buffer(self, current_time: float):
        """Clean up old packets and update statistics"""
        # 統計情報更新
        self.stats["current_buffer_size"] = len(self.packet_buffer)

        if self.stats["packets_sent"] > 0:
            total_delay = sum(pkt[2] - pkt[0] for pkt in self.packet_buffer)
            self.stats["avg_delay"] = (
                total_delay / len(self.packet_buffer)
                if self.packet_buffer
                else self.base_delay
            )

        # 非常に古いパケット（10倍の基本遅延を超えたもの）を削除
        max_age = current_time - (self.base_delay * 10)
        old_packets = [pkt for pkt in self.packet_buffer if pkt[0] < max_age]
        if old_packets:
            self.packet_buffer = [
                pkt for pkt in self.packet_buffer if pkt[0] >= max_age
            ]
            self.stats["packets_dropped_buffer"] += len(old_packets)

    def get_output(self) -> Any:
        """Get current output value"""
        return self.current_output

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics dictionary"""
        return self.stats.copy()


if __name__ == "__main__":
    # テスト実行
    mosaik_api.start_simulation(DelaySimulator())
