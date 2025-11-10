"""
推力変化率依存モデルのテスト

推力の絶対値ではなく、変化率に依存することを確認
"""

import matplotlib.pyplot as plt
import numpy as np
from time_constant_model import create_time_constant_model


def test_thrust_rate_dependency():
    """推力変化率依存性のテスト"""

    print("=== 推力変化率依存性テスト ===\n")

    # モデル作成
    model = create_time_constant_model("linear", sensitivity=1.0)  # 1.0 [s/N]
    base_tau = 100.0  # ms
    dt = 10.0  # ms

    # テストケース
    print("1. 静的な推力（変化なし）")
    print("   推力: 0 → 50 → 50 → 50 N")

    thrusts = [0, 50, 50, 50]
    taus = []

    for i, F in enumerate(thrusts):
        tau = model.get_time_constant(F, base_tau, dt=dt)
        taus.append(tau)

        if i > 0:
            dF = F - thrusts[i - 1]
            dF_dt = dF / dt
            print(f"   Step {i}: F={F:5.1f}N, dF/dt={dF_dt:6.3f}N/ms, τ={tau:.2f}ms")

    print("\n2. 動的な推力（急激な変化）")
    print("   推力: 0 → 100 → 0 → 50 N")

    model.reset()
    thrusts = [0, 100, 0, 50]
    taus = []

    for i, F in enumerate(thrusts):
        tau = model.get_time_constant(F, base_tau, dt=dt)
        taus.append(tau)

        if i > 0:
            dF = F - thrusts[i - 1]
            dF_dt = dF / dt
            print(f"   Step {i}: F={F:5.1f}N, dF/dt={dF_dt:6.3f}N/ms, τ={tau:.2f}ms")

    print("\n3. 徐々に増加する推力")
    print("   推力: 0 → 25 → 50 → 75 N")

    model.reset()
    thrusts = [0, 25, 50, 75]
    taus = []

    for i, F in enumerate(thrusts):
        tau = model.get_time_constant(F, base_tau, dt=dt)
        taus.append(tau)

        if i > 0:
            dF = F - thrusts[i - 1]
            dF_dt = dF / dt
            print(f"   Step {i}: F={F:5.1f}N, dF/dt={dF_dt:6.3f}N/ms, τ={tau:.2f}ms")


def visualize_models():
    """各モデルの特性を可視化"""

    print("\n=== モデル可視化 ===\n")

    base_tau = 100.0  # ms
    dt = 10.0  # ms

    # ステップ応答シミュレーション
    # 0 → 75N にステップ入力
    time_steps = 100
    thrust_profile = np.zeros(time_steps)
    thrust_profile[10:] = 75.0  # ステップ10で75Nに

    # モデル定義
    models = {
        "Constant": create_time_constant_model("constant"),
        "Linear (k=0.5)": create_time_constant_model("linear", sensitivity=0.5),
        "Saturation": create_time_constant_model("saturation", max_delta_tau=30.0, saturation_rate=0.5),
        "Hybrid": create_time_constant_model(
            "hybrid", thrust_sensitivity=0.3, heating_rate=0.001, cooling_rate=0.01, thermal_sensitivity=0.05
        ),
    }

    # プロット
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 推力プロファイル
    ax = axes[0, 0]
    time_axis = np.arange(time_steps) * dt
    ax.plot(time_axis, thrust_profile, linewidth=2, color="black")
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Thrust [N]")
    ax.set_title("Thrust Profile (Step Input)")
    ax.grid(True, alpha=0.3)
    ax.axvline(x=100, color="red", linestyle="--", alpha=0.5, label="Step change")
    ax.legend()

    # 2. 時定数の時間応答
    ax = axes[0, 1]

    for name, model in models.items():
        taus = []
        for F in thrust_profile:
            tau = model.get_time_constant(F, base_tau, dt=dt)
            taus.append(tau)

        ax.plot(time_axis, taus, linewidth=2, label=name)

        # リセット
        if hasattr(model, "reset"):
            model.reset()

    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Time Constant [ms]")
    ax.set_title("Time Constant Response to Step Input")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axvline(x=100, color="red", linestyle="--", alpha=0.5)

    # 3. 推力変化率 vs 時定数
    ax = axes[1, 0]

    thrust_rates = np.linspace(0, 10, 100)  # [N/ms]

    # 各モデルでの時定数を計算（変化率に対する応答）
    for name, model in models.items():
        if hasattr(model, "reset"):
            model.reset()

        taus = []
        for rate in thrust_rates:
            # 推力変化率をシミュレート
            # previous_thrust = 0, current_thrust = rate * dt
            F_prev = 0
            F_current = rate * dt

            # まず前の推力で初期化
            if hasattr(model, "previous_thrust"):
                model.previous_thrust = F_prev

            # 現在の推力で時定数を計算
            tau = model.get_time_constant(F_current, base_tau, dt=dt)
            taus.append(tau)

        ax.plot(thrust_rates, taus, linewidth=2, label=name)

    ax.set_xlabel("Thrust Rate [N/ms]")
    ax.set_ylabel("Time Constant [ms]")
    ax.set_title("Time Constant vs Thrust Rate")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. パルス応答（複数のステップ変化）
    ax = axes[1, 1]

    # パルス列の推力プロファイル
    pulse_profile = np.zeros(time_steps)
    pulse_profile[10:20] = 50.0
    pulse_profile[30:40] = 75.0
    pulse_profile[50:60] = 100.0

    # Hybridモデルでの応答
    hybrid_model = create_time_constant_model(
        "hybrid", thrust_sensitivity=0.5, heating_rate=0.002, cooling_rate=0.01, thermal_sensitivity=0.08
    )

    taus_hybrid = []
    for F in pulse_profile:
        tau = hybrid_model.get_time_constant(F, base_tau, dt=dt)
        taus_hybrid.append(tau)

    ax2 = ax.twinx()
    ax.plot(time_axis, pulse_profile, linewidth=2, color="black", alpha=0.5, label="Thrust")
    ax2.plot(time_axis, taus_hybrid, linewidth=2, color="purple", label="τ (Hybrid)")

    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Thrust [N]", color="black")
    ax2.set_ylabel("Time Constant [ms]", color="purple")
    ax.set_title("Hybrid Model: Pulse Response with Thermal Memory")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig("/tmp/thrust_rate_models.png", dpi=150)
    print("Plot saved to /tmp/thrust_rate_models.png")

    plt.show()


if __name__ == "__main__":
    test_thrust_rate_dependency()
    visualize_models()
