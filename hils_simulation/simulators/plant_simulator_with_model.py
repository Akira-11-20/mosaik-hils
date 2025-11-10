"""
PlantSimulator with Time Constant Model - 時定数モデル統合版

時定数が推力変化率や熱的影響により動的に変化するバージョン
"""

import sys
from pathlib import Path

import mosaik_api
import numpy as np

# models モジュールをインポート
sys.path.insert(0, str(Path(__file__).parent.parent))
from models import create_time_constant_model

meta = {
    "type": "time-based",
    "models": {
        "ThrustStand": {
            "public": True,
            "params": [
                "stand_id",
                "time_constant",
                "time_constant_std",
                "time_constant_noise",
                "enable_lag",
                "tau_model_type",  # 新規: 時定数モデルのタイプ
                "tau_model_params",  # 新規: モデルパラメータ（辞書）
            ],
            "attrs": [
                "command",
                "measured_thrust",
                "actual_thrust",
                "status",
                "time_constant",  # 動的に変化する時定数
            ],
        },
    },
}


class PlantSimulator(mosaik_api.Simulator):
    """
    推力測定器シミュレーター（時定数モデル統合版）

    新機能:
    - 推力変化率依存の時定数
    - 熱的影響による長期ドリフト
    - 複数のモデルタイプから選択可能
    """

    def __init__(self):
        super().__init__(meta)
        self.entities = {}
        self.step_size = 1
        self.time = 0
        self.time_resolution = 0.001

    def init(self, sid, time_resolution=0.001, step_size=1):
        """初期化"""
        self.sid = sid
        self.time_resolution = time_resolution
        self.step_size = step_size
        return self.meta

    def create(
        self,
        num,
        model,
        stand_id="thrust_stand_01",
        time_constant=50.0,
        time_constant_std=0.0,
        time_constant_noise=0.0,
        enable_lag=True,
        tau_model_type="linear",  # "constant", "linear", "saturation", "thermal", "hybrid"
        tau_model_params=None,  # モデル固有のパラメータ
    ):
        """
        推力測定器エンティティの作成

        Args:
            tau_model_type: 時定数モデルのタイプ
                - "constant": 固定時定数（従来通り）
                - "linear": 推力変化率に線形依存
                - "saturation": 飽和特性あり
                - "thermal": 熱的影響
                - "hybrid": 複合モデル（推力変化率 + 熱）
                - "stochastic": 確率的変動
            tau_model_params: モデルパラメータ（辞書）
                例: {"sensitivity": 0.5} for linear model
        """
        entities = []

        if tau_model_params is None:
            tau_model_params = {}

        for i in range(num):
            eid = f"{model}_{i}"

            # ベース時定数のばらつき適用
            if time_constant_std > 0:
                actual_time_constant = max(0.1, np.random.normal(time_constant, time_constant_std))
            else:
                actual_time_constant = time_constant

            # 時定数モデルの作成
            tau_model = create_time_constant_model(tau_model_type, **tau_model_params)

            self.entities[eid] = {
                "stand_id": stand_id,
                "thrust_cmd": 0.0,
                "duration_cmd": 0.0,
                "measured_thrust": 0.0,
                "actual_thrust": 0.0,
                "status": "idle",
                "thrust_start_time": None,
                "thrust_end_time": None,
                # 時定数関連
                "time_constant_base": actual_time_constant,
                "time_constant": actual_time_constant,
                "time_constant_noise": time_constant_noise,
                "enable_lag": enable_lag,
                # 新規: 時定数モデル
                "tau_model": tau_model,
                "tau_model_type": tau_model_type,
            }

            entities.append({"eid": eid, "type": model})

            lag_status = "enabled" if enable_lag else "disabled"
            print(
                f"[PlantSim] Created {eid} (ID: {stand_id}, "
                f"τ_base={actual_time_constant:.2f}ms, "
                f"model={tau_model_type}, "
                f"lag={lag_status})"
            )

        return entities

    def step(self, time, inputs, max_advance=None):
        """シミュレーションステップ"""
        self.time = time

        for eid, entity in self.entities.items():
            # 1. 制御コマンドの受信
            if eid in inputs and "command" in inputs[eid]:
                cmd = list(inputs[eid]["command"].values())[0]

                if cmd is not None and isinstance(cmd, dict):
                    thrust = cmd.get("thrust", 0.0)
                    duration = cmd.get("duration", 0.0)

                    entity["thrust_cmd"] = thrust
                    entity["duration_cmd"] = duration

                    if thrust != 0 and duration > 0:
                        entity["thrust_start_time"] = time
                        entity["thrust_end_time"] = time + duration
                        entity["status"] = "thrusting"

            # 2. 理想的な推力測定
            if entity["status"] == "thrusting":
                if entity["thrust_end_time"] is not None and time < entity["thrust_end_time"]:
                    entity["measured_thrust"] = entity["thrust_cmd"]
                else:
                    entity["measured_thrust"] = 0.0
                    entity["status"] = "idle"
                    entity["thrust_start_time"] = None
                    entity["thrust_end_time"] = None
            else:
                entity["measured_thrust"] = 0.0

            # 3. 一次遅延ダイナミクス（時定数モデル統合）
            if entity["enable_lag"]:
                dt = self.step_size * self.time_resolution * 1000  # [ms]
                tau_base = entity["time_constant_base"]

                # 時定数モデルから動的な時定数を取得
                tau_model = entity["tau_model"]
                measured_thrust = entity["measured_thrust"]

                # 時定数を計算（推力変化率や熱的影響を考慮）
                tau_dynamic = tau_model.get_time_constant(
                    thrust=measured_thrust,
                    base_tau=tau_base,
                    dt=dt,
                )

                # 時間変動ノイズの追加（オプション）
                if entity["time_constant_noise"] > 0:
                    noise = np.random.normal(0, entity["time_constant_noise"])
                    tau = max(0.1, tau_dynamic + noise)
                else:
                    tau = tau_dynamic

                # 一次遅延の計算
                u = measured_thrust
                y = entity["actual_thrust"]

                # サブステップ分割
                sub_steps = max(1, int(dt / 0.1))
                dt_sub = dt / sub_steps

                for _ in range(sub_steps):
                    y = y + (dt_sub / tau) * (u - y)

                # y = y + (dt / tau) * (u - y)

                entity["actual_thrust"] = y
                entity["time_constant"] = tau  # 記録用
            else:
                entity["actual_thrust"] = entity["measured_thrust"]

        return time + self.step_size

    def get_data(self, outputs):
        """データ取得"""
        data = {}

        for eid, attrs in outputs.items():
            if eid not in self.entities:
                continue

            data[eid] = {}
            entity = self.entities[eid]

            for attr in attrs:
                if attr in entity:
                    data[eid][attr] = entity[attr]
                else:
                    data[eid][attr] = None

        return data


if __name__ == "__main__":
    # デモ: 異なるモデルタイプのテスト
    import matplotlib.pyplot as plt

    print("=== PlantSimulator with Time Constant Model Demo ===\n")

    # 簡易シミュレーション
    models_to_test = [
        ("constant", {}),
        ("linear", {"sensitivity": 0.5}),
        (
            "hybrid",
            {"thrust_sensitivity": 0.3, "heating_rate": 0.001, "cooling_rate": 0.01, "thermal_sensitivity": 0.05},
        ),
    ]

    time_steps = 200
    dt = 10.0  # ms
    thrust_profile = np.zeros(time_steps)
    thrust_profile[10:100] = 75.0  # 10ms-1000msで75N

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    for model_type, params in models_to_test:
        tau_model = create_time_constant_model(model_type, **params)
        base_tau = 100.0

        taus = []
        y = 0.0  # actual_thrust
        y_history = []

        for F in thrust_profile:
            # 時定数計算
            tau = tau_model.get_time_constant(F, base_tau, dt=dt)
            taus.append(tau)

            # 一次遅延計算
            sub_steps = max(1, int(dt / 0.1))
            dt_sub = dt / sub_steps
            for _ in range(sub_steps):
                y = y + (dt_sub / tau) * (F - y)

            y_history.append(y)

        time_axis = np.arange(time_steps) * dt

        # 時定数のプロット
        axes[0].plot(time_axis, taus, linewidth=2, label=f"{model_type}")

        # 実推力のプロット
        axes[1].plot(time_axis, y_history, linewidth=2, label=f"{model_type}")

        # モデルリセット
        if hasattr(tau_model, "reset"):
            tau_model.reset()

    # 推力プロファイルも表示
    axes[1].plot(time_axis, thrust_profile, "k--", linewidth=1, alpha=0.5, label="Commanded")

    axes[0].set_xlabel("Time [ms]")
    axes[0].set_ylabel("Time Constant [ms]")
    axes[0].set_title("Dynamic Time Constant")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Time [ms]")
    axes[1].set_ylabel("Thrust [N]")
    axes[1].set_title("Actual Thrust Response")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("/tmp/plant_sim_with_model_demo.png", dpi=150)
    print("Demo plot saved to /tmp/plant_sim_with_model_demo.png")

    mosaik_api.start_simulator(PlantSimulator())
