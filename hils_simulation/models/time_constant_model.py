"""
Time Constant Model - 時定数の動的変化モデル

推力測定器の時定数が、入力推力や動作状態によって変化する現象をモデル化。
物理的な背景:
- アクチュエーターの非線形性（推力が大きいと応答が遅くなる等）
- 熱の影響（連続動作で温度が上がり、応答特性が変化）
- 機械的な摩擦や慣性の非線形効果
"""

from abc import ABC, abstractmethod

import numpy as np


class TimeConstantModel(ABC):
    """時定数モデルの抽象基底クラス"""

    @abstractmethod
    def get_time_constant(
        self,
        thrust: float,
        base_tau: float,
        **kwargs,
    ) -> float:
        """
        現在の推力に基づいて時定数を計算

        Args:
            thrust: 現在の推力 [N]
            base_tau: ベース時定数 [ms]
            **kwargs: モデル固有のパラメータ

        Returns:
            時定数 [ms]
        """
        pass


class ConstantTimeConstantModel(TimeConstantModel):
    """定数時定数モデル（参照実装）"""

    def get_time_constant(
        self,
        thrust: float,
        base_tau: float,
        **kwargs,
    ) -> float:
        """
        常に一定の時定数を返す

        Args:
            thrust: 現在の推力 [N]（未使用）
            base_tau: ベース時定数 [ms]

        Returns:
            base_tau
        """
        return base_tau


class LinearThrustDependentModel(TimeConstantModel):
    """
    推力変動に線形依存する時定数モデル

    τ(dF/dt) = τ_base + k * |dF/dt|

    物理的解釈:
    - 推力の変化率が大きいほど、アクチュエーターの動的な応答遅れが増加
    - 急激な変化に対しては、慣性や摩擦の影響で応答が遅くなる
    - 静的な推力値ではなく、動的な変化に依存
    """

    def __init__(self, sensitivity: float = 0.1):
        """
        Args:
            sensitivity: 推力変化率感度 [ms/(N/ms)] = [s/N]
                正: 変化率が大きいと時定数が増加（応答が遅くなる）
        """
        self.sensitivity = sensitivity
        self.previous_thrust = 0.0

    def get_time_constant(
        self,
        thrust: float,
        base_tau: float,
        dt: float = 0.1,  # デフォルトは0.1ms
        **kwargs,
    ) -> float:
        """
        推力変化率に比例して時定数が変化

        Args:
            thrust: 現在の推力 [N]
            base_tau: ベース時定数 [ms]
            dt: 時間ステップ [ms]

        Returns:
            調整後の時定数 [ms]
        """
        # 推力変化率を計算 [N/ms]
        if dt > 0:
            thrust_rate = abs(thrust - self.previous_thrust) / dt
        else:
            thrust_rate = 0.0

        self.previous_thrust = thrust

        # 変化率に基づく時定数の調整
        delta_tau = self.sensitivity * thrust_rate
        tau = base_tau + delta_tau

        # 時定数が負にならないようにクリップ
        return max(0.1, tau)

    def reset(self):
        """内部状態をリセット"""
        self.previous_thrust = 0.0


class SaturationModel(TimeConstantModel):
    """
    飽和特性を持つ時定数モデル（推力変化率依存）

    τ(dF/dt) = τ_base + Δτ_max * tanh(k * |dF/dt|)

    物理的解釈:
    - 低変化率域では線形的に変化
    - 高変化率域では飽和して一定値に近づく
    - 実際のアクチュエーターの飽和特性を模擬（急激な変化に対する限界）
    """

    def __init__(
        self,
        max_delta_tau: float = 20.0,
        saturation_rate: float = 0.5,  # [ms/N] → 変化率に対する飽和率
    ):
        """
        Args:
            max_delta_tau: 最大時定数変化量 [ms]
            saturation_rate: 飽和率 [ms/N]
                大きいほど早く飽和する
        """
        self.max_delta_tau = max_delta_tau
        self.saturation_rate = saturation_rate
        self.previous_thrust = 0.0

    def get_time_constant(
        self,
        thrust: float,
        base_tau: float,
        dt: float = 0.1,
        **kwargs,
    ) -> float:
        """
        飽和特性を持つ時定数変化（推力変化率ベース）

        Args:
            thrust: 現在の推力 [N]
            base_tau: ベース時定数 [ms]
            dt: 時間ステップ [ms]

        Returns:
            調整後の時定数 [ms]
        """
        # 推力変化率を計算 [N/ms]
        if dt > 0:
            thrust_rate = abs(thrust - self.previous_thrust) / dt
        else:
            thrust_rate = 0.0

        self.previous_thrust = thrust

        # 飽和特性（tanh）を適用
        delta_tau = self.max_delta_tau * np.tanh(self.saturation_rate * thrust_rate)
        tau = base_tau + delta_tau

        return max(0.1, tau)

    def reset(self):
        """内部状態をリセット"""
        self.previous_thrust = 0.0


class ThermalModel(TimeConstantModel):
    """
    熱的影響を考慮した時定数モデル

    推力の累積エネルギー（熱）により時定数が変化する。
    連続動作で温度が上がり、応答特性が変化する現象を模擬。

    dT/dt = α * |F|^2 - β * T
    τ(T) = τ_base * (1 + γ * T)

    物理的解釈:
    - 推力の二乗に比例して熱が発生（ジュール熱）
    - 環境への放熱により温度が減少
    - 温度が高いと粘性が変化し、応答が遅くなる
    """

    def __init__(
        self,
        heating_rate: float = 0.001,
        cooling_rate: float = 0.01,
        thermal_sensitivity: float = 0.05,
    ):
        """
        Args:
            heating_rate: 加熱率 α [K/(N^2·ms)]
            cooling_rate: 冷却率 β [1/ms]
            thermal_sensitivity: 熱感度 γ [1/K]
        """
        self.heating_rate = heating_rate
        self.cooling_rate = cooling_rate
        self.thermal_sensitivity = thermal_sensitivity
        self.temperature = 0.0  # 基準温度からの偏差 [K]

    def get_time_constant(
        self,
        thrust: float,
        base_tau: float,
        dt: float = 0.1,  # デフォルトは0.1ms
        **kwargs,
    ) -> float:
        """
        熱的影響を考慮した時定数計算

        Args:
            thrust: 現在の推力 [N]
            base_tau: ベース時定数 [ms]
            dt: 時間ステップ [ms]

        Returns:
            調整後の時定数 [ms]
        """
        # 温度の更新（1次微分方程式の離散化）
        heating = self.heating_rate * thrust**2
        cooling = self.cooling_rate * self.temperature
        dT = (heating - cooling) * dt

        self.temperature = max(0.0, self.temperature + dT)

        # 時定数の計算
        tau = base_tau * (1.0 + self.thermal_sensitivity * self.temperature)

        return max(0.1, tau)

    def reset(self):
        """温度をリセット"""
        self.temperature = 0.0


class HybridModel(TimeConstantModel):
    """
    複合モデル - 推力変化率依存 + 熱的影響

    τ(dF/dt, T) = (τ_base + k * |dF/dt|) * (1 + γ * T)

    最もリアルな動作を模擬:
    - 瞬時的な推力変化率依存性（動的応答）
    - 累積的な熱的影響（時間積分効果）
    """

    def __init__(
        self,
        thrust_sensitivity: float = 0.1,
        heating_rate: float = 0.001,
        cooling_rate: float = 0.01,
        thermal_sensitivity: float = 0.05,
    ):
        """
        Args:
            thrust_sensitivity: 推力変化率感度 [s/N]
            heating_rate: 加熱率 [K/(N^2·ms)]
            cooling_rate: 冷却率 [1/ms]
            thermal_sensitivity: 熱感度 [1/K]
        """
        self.thrust_sensitivity = thrust_sensitivity
        self.heating_rate = heating_rate
        self.cooling_rate = cooling_rate
        self.thermal_sensitivity = thermal_sensitivity
        self.temperature = 0.0
        self.previous_thrust = 0.0

    def get_time_constant(
        self,
        thrust: float,
        base_tau: float,
        dt: float = 0.1,
        **kwargs,
    ) -> float:
        """
        推力変化率依存 + 熱的影響を考慮した時定数計算

        Args:
            thrust: 現在の推力 [N]
            base_tau: ベース時定数 [ms]
            dt: 時間ステップ [ms]

        Returns:
            調整後の時定数 [ms]
        """
        # 推力変化率を計算 [N/ms]
        if dt > 0:
            thrust_rate = abs(thrust - self.previous_thrust) / dt
        else:
            thrust_rate = 0.0

        self.previous_thrust = thrust

        # 温度の更新
        heating = self.heating_rate * thrust**2
        cooling = self.cooling_rate * self.temperature
        dT = (heating - cooling) * dt
        self.temperature = max(0.0, self.temperature + dT)

        # 推力変化率依存項
        tau_base_modified = base_tau + self.thrust_sensitivity * thrust_rate

        # 熱的影響
        tau = tau_base_modified * (1.0 + self.thermal_sensitivity * self.temperature)

        return max(0.1, tau)

    def reset(self):
        """温度と推力履歴をリセット"""
        self.temperature = 0.0
        self.previous_thrust = 0.0


class StochasticModel(TimeConstantModel):
    """
    確率的変動モデル（推力変化率依存）

    τ(dF/dt) = τ_base + k * |dF/dt| + N(0, σ^2)

    物理的解釈:
    - 測定ノイズや環境外乱による時定数の変動
    - 推力変化率依存性 + ホワイトノイズ
    """

    def __init__(
        self,
        thrust_sensitivity: float = 0.1,
        noise_std: float = 5.0,
    ):
        """
        Args:
            thrust_sensitivity: 推力変化率感度 [s/N]
            noise_std: ノイズ標準偏差 [ms]
        """
        self.thrust_sensitivity = thrust_sensitivity
        self.noise_std = noise_std
        self.previous_thrust = 0.0

    def get_time_constant(
        self,
        thrust: float,
        base_tau: float,
        dt: float = 0.1,
        **kwargs,
    ) -> float:
        """
        確率的変動を含む時定数計算（推力変化率ベース）

        Args:
            thrust: 現在の推力 [N]
            base_tau: ベース時定数 [ms]
            dt: 時間ステップ [ms]

        Returns:
            調整後の時定数 [ms]
        """
        # 推力変化率を計算 [N/ms]
        if dt > 0:
            thrust_rate = abs(thrust - self.previous_thrust) / dt
        else:
            thrust_rate = 0.0

        self.previous_thrust = thrust

        # 推力変化率依存項
        delta_tau = self.thrust_sensitivity * thrust_rate

        # ノイズ項
        noise = np.random.normal(0, self.noise_std)

        tau = base_tau + delta_tau + noise

        return max(0.1, tau)

    def reset(self):
        """内部状態をリセット"""
        self.previous_thrust = 0.0


# ファクトリー関数
def create_time_constant_model(
    model_type: str,
    **params,
) -> TimeConstantModel:
    """
    時定数モデルを作成するファクトリー関数

    Args:
        model_type: モデルタイプ
            - "constant": 定数モデル
            - "linear": 線形推力依存モデル
            - "saturation": 飽和モデル
            - "thermal": 熱モデル
            - "hybrid": 複合モデル
            - "stochastic": 確率モデル
        **params: モデル固有のパラメータ

    Returns:
        TimeConstantModel インスタンス

    Examples:
        >>> model = create_time_constant_model("linear", sensitivity=0.1)
        >>> tau = model.get_time_constant(thrust=50.0, base_tau=100.0)

        >>> model = create_time_constant_model(
        ...     "hybrid",
        ...     thrust_sensitivity=0.1,
        ...     heating_rate=0.001,
        ...     cooling_rate=0.01,
        ...     thermal_sensitivity=0.05
        ... )
    """
    models = {
        "constant": ConstantTimeConstantModel,
        "linear": LinearThrustDependentModel,
        "saturation": SaturationModel,
        "thermal": ThermalModel,
        "hybrid": HybridModel,
        "stochastic": StochasticModel,
    }

    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. Available types: {list(models.keys())}")

    model_class = models[model_type]
    return model_class(**params)


if __name__ == "__main__":
    # デモンストレーション
    import matplotlib.pyplot as plt

    print("=== Time Constant Model Demo ===\n")

    # 推力範囲
    thrusts = np.linspace(0, 100, 100)

    # 各モデルでの時定数を計算
    base_tau = 100.0  # ms

    models_to_test = {
        "Constant": create_time_constant_model("constant"),
        "Linear (k=0.2)": create_time_constant_model("linear", sensitivity=0.2),
        "Saturation": create_time_constant_model("saturation", max_delta_tau=30.0, saturation_rate=0.05),
    }

    plt.figure(figsize=(12, 8))

    # 1. 推力依存性のプロット
    plt.subplot(2, 2, 1)
    for name, model in models_to_test.items():
        taus = [model.get_time_constant(F, base_tau) for F in thrusts]
        plt.plot(thrusts, taus, label=name, linewidth=2)

    plt.xlabel("Thrust [N]")
    plt.ylabel("Time Constant [ms]")
    plt.title("Time Constant vs Thrust")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 2. 熱モデルの時間応答
    plt.subplot(2, 2, 2)
    thermal_model = create_time_constant_model(
        "thermal",
        heating_rate=0.001,
        cooling_rate=0.01,
        thermal_sensitivity=0.05,
    )

    time_steps = 1000
    dt = 10.0  # ms
    thrust_profile = [50.0 if i < 500 else 0.0 for i in range(time_steps)]
    taus_thermal = []

    for F in thrust_profile:
        tau = thermal_model.get_time_constant(F, base_tau, dt=dt)
        taus_thermal.append(tau)

    time_axis = np.arange(time_steps) * dt / 1000  # convert to seconds
    plt.plot(time_axis, taus_thermal, label="Thermal Model", linewidth=2)
    plt.xlabel("Time [s]")
    plt.ylabel("Time Constant [ms]")
    plt.title("Thermal Model Response (50N → 0N at 5s)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 3. 複合モデル
    plt.subplot(2, 2, 3)
    hybrid_model = create_time_constant_model(
        "hybrid",
        thrust_sensitivity=0.1,
        heating_rate=0.001,
        cooling_rate=0.01,
        thermal_sensitivity=0.05,
    )

    taus_hybrid = []
    for F in thrust_profile:
        tau = hybrid_model.get_time_constant(F, base_tau, dt=dt)
        taus_hybrid.append(tau)

    plt.plot(time_axis, taus_hybrid, label="Hybrid Model", linewidth=2, color="purple")
    plt.xlabel("Time [s]")
    plt.ylabel("Time Constant [ms]")
    plt.title("Hybrid Model Response")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 4. 確率モデルの分布
    plt.subplot(2, 2, 4)
    stochastic_model = create_time_constant_model(
        "stochastic",
        thrust_sensitivity=0.1,
        noise_std=5.0,
    )

    # 固定推力でのサンプリング
    samples = 1000
    test_thrust = 50.0
    tau_samples = [stochastic_model.get_time_constant(test_thrust, base_tau) for _ in range(samples)]

    plt.hist(tau_samples, bins=50, alpha=0.7, edgecolor="black")
    plt.xlabel("Time Constant [ms]")
    plt.ylabel("Frequency")
    plt.title(f"Stochastic Model Distribution (F={test_thrust}N)")
    plt.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("/tmp/time_constant_models.png", dpi=150)
    print("Plot saved to /tmp/time_constant_models.png")

    # 数値例を表示
    print("\n=== Numerical Examples ===")
    print(f"Base time constant: {base_tau} ms\n")

    for name, model in models_to_test.items():
        tau_0 = model.get_time_constant(0.0, base_tau)
        tau_50 = model.get_time_constant(50.0, base_tau)
        tau_100 = model.get_time_constant(100.0, base_tau)
        print(f"{name}:")
        print(f"  τ(0N)   = {tau_0:.2f} ms")
        print(f"  τ(50N)  = {tau_50:.2f} ms")
        print(f"  τ(100N) = {tau_100:.2f} ms")
        print()
