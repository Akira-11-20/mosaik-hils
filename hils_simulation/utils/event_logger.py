"""
イベントロガー

シミュレーション中の詳細なイベント（送信・受信・処理）をタイムスタンプ付きで記録。
データの識別タグを使って、遅延やパケットロスを追跡可能にする。
"""

import json
from pathlib import Path
from datetime import datetime


class EventLogger:
    """
    シミュレーションイベントの詳細ログを記録
    """

    def __init__(self, output_dir: str, simulator_name: str):
        """
        Args:
            output_dir: ログ出力ディレクトリ
            simulator_name: シミュレーター名
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.simulator_name = simulator_name
        self.events = []

        # ログファイル名
        self.log_file = self.output_dir / f"{simulator_name}_events.jsonl"

    def log_event(
        self,
        time_ms: int,
        event_type: str,
        data: dict,
        data_tag: dict = None,
    ):
        """
        イベントをログに記録

        Args:
            time_ms: シミュレーション時刻 [ms]
            event_type: イベントタイプ（"send", "receive", "process", "drop"）
            data: イベントデータ
            data_tag: データ識別タグ（タイムスタンプ、シーケンス番号など）
        """
        event = {
            "timestamp_ms": time_ms,
            "timestamp_s": time_ms * 0.001,
            "simulator": self.simulator_name,
            "event_type": event_type,
            "data": data,
            "data_tag": data_tag or {},
        }
        self.events.append(event)

    def log_send(
        self,
        time_ms: int,
        destination: str,
        value: any,
        data_tag: dict,
    ):
        """
        送信イベントをログ

        Args:
            time_ms: 送信時刻 [ms]
            destination: 送信先
            value: 送信データ
            data_tag: データタグ
        """
        self.log_event(
            time_ms=time_ms,
            event_type="send",
            data={
                "destination": destination,
                "value": value,
            },
            data_tag=data_tag,
        )

    def log_receive(
        self,
        time_ms: int,
        source: str,
        value: any,
        data_tag: dict,
    ):
        """
        受信イベントをログ

        Args:
            time_ms: 受信時刻 [ms]
            source: 送信元
            value: 受信データ
            data_tag: データタグ
        """
        self.log_event(
            time_ms=time_ms,
            event_type="receive",
            data={
                "source": source,
                "value": value,
            },
            data_tag=data_tag,
        )

    def log_process(
        self,
        time_ms: int,
        operation: str,
        input_value: any = None,
        output_value: any = None,
        data_tag: dict = None,
    ):
        """
        処理イベントをログ

        Args:
            time_ms: 処理時刻 [ms]
            operation: 操作内容
            input_value: 入力値
            output_value: 出力値
            data_tag: データタグ
        """
        self.log_event(
            time_ms=time_ms,
            event_type="process",
            data={
                "operation": operation,
                "input": input_value,
                "output": output_value,
            },
            data_tag=data_tag,
        )

    def log_drop(
        self,
        time_ms: int,
        reason: str,
        value: any,
        data_tag: dict,
    ):
        """
        ドロップイベントをログ（パケットロスなど）

        Args:
            time_ms: ドロップ時刻 [ms]
            reason: ドロップ理由
            value: ドロップされたデータ
            data_tag: データタグ
        """
        self.log_event(
            time_ms=time_ms,
            event_type="drop",
            data={
                "reason": reason,
                "value": value,
            },
            data_tag=data_tag,
        )

    def finalize(self):
        """
        ログをファイルに書き出し
        """
        if not self.events:
            return

        # JSONL形式で保存（1行1イベント）
        with open(self.log_file, "w") as f:
            for event in self.events:
                f.write(json.dumps(event) + "\n")

        print(
            f"[EventLogger] 📝 {self.simulator_name}: {len(self.events)} events logged to {self.log_file}"
        )


class DataTag:
    """
    データ識別タグのヘルパークラス
    """

    @staticmethod
    def create(
        sender: str,
        send_time_ms: int,
        sequence_num: int,
        data_type: str = "data",
    ) -> dict:
        """
        データタグを生成

        Args:
            sender: 送信元シミュレーター名
            send_time_ms: 送信時刻 [ms]
            sequence_num: シーケンス番号
            data_type: データタイプ（"command", "measurement", "state"など）

        Returns:
            データタグ辞書
        """
        return {
            "sender": sender,
            "send_time_ms": send_time_ms,
            "sequence_num": sequence_num,
            "data_type": data_type,
        }

    @staticmethod
    def extract(data: any) -> dict:
        """
        データからタグを抽出

        Args:
            data: データ（辞書の場合は"_tag"キーを探す）

        Returns:
            データタグ辞書（存在しない場合は空辞書）
        """
        if isinstance(data, dict) and "_tag" in data:
            return data["_tag"]
        return {}

    @staticmethod
    def attach(data: any, tag: dict) -> dict:
        """
        データにタグを付与

        Args:
            data: 元のデータ
            tag: データタグ

        Returns:
            タグ付きデータ（辞書形式）
        """
        if isinstance(data, dict):
            # 辞書の場合は_tagキーを追加
            tagged_data = data.copy()
            tagged_data["_tag"] = tag
            return tagged_data
        else:
            # 非辞書の場合はラップ
            return {
                "value": data,
                "_tag": tag,
            }

    @staticmethod
    def unwrap(data: any) -> any:
        """
        タグ付きデータから元のデータを取り出す

        Args:
            data: タグ付きデータ

        Returns:
            元のデータ
        """
        if isinstance(data, dict):
            if "_tag" in data:
                # タグを除いたデータを返す
                unwrapped = {k: v for k, v in data.items() if k != "_tag"}
                # valueキーのみの場合は値を直接返す
                if len(unwrapped) == 1 and "value" in unwrapped:
                    return unwrapped["value"]
                return unwrapped
        return data
