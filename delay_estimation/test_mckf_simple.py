"""
簡易版MCKF (遅延なし) のテストスクリプト

論文: "Modified Kalman and Maximum Correntropy Kalman Filters for Systems
       With Bernoulli Distribution k-step Random Delay and Packet Loss"

このスクリプトは遅延処理を省略し、MCKFのコアである
「最大コレンロピー基準による外れ値抑制」の効果を検証します。
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

from estimators.mckf_simple import SimpleMCKF
from estimators.kalman_filter import KalmanFilter


def create_spacecraft_system(dt: float = 0.1, inertia: float = 1.0):
    """
    1自由度宇宙機の状態空間モデル

    状態: x = [θ, ω]' (姿勢角, 角速度)
    制御入力: u = トルク
    観測: y = θ (姿勢角のみ)
    """
    A = np.array([
        [1.0, dt],
        [0.0, 1.0]
    ])
    B = np.array([[0.5 * dt**2 / inertia], [dt / inertia]])
    C = np.array([[1.0, 0.0]])

    return A, B, C


def add_non_gaussian_noise(
    measurement: np.ndarray,
    std: float,
    outlier_prob: float = 0.1,
    outlier_scale: float = 10.0
) -> np.ndarray:
    """
    非ガウスノイズ (外れ値を含む)

    論文と同じく、混合ガウス分布:
    ν(n) ~ 0.9*N(0, std²) + 0.1*N(0, (outlier_scale*std)²)
    """
    if np.random.rand() < outlier_prob:
        # 外れ値 (10倍の標準偏差)
        noise = np.random.randn(*measurement.shape) * std * outlier_scale
    else:
        # 通常ノイズ
        noise = np.random.randn(*measurement.shape) * std

    return measurement + noise


def run_comparison(
    total_time: float = 20.0,
    dt: float = 0.1,
    measurement_noise_std: float = 0.1,
    outlier_prob: float = 0.1,
    kernel_bandwidth: float = 2.0,
    seed: int = 42
):
    """
    SimpleMCKFと標準KFの比較シミュレーション (遅延なし)
    """
    np.random.seed(seed)

    num_steps = int(total_time / dt)
    time = np.arange(num_steps) * dt

    # システム作成
    A, B, C = create_spacecraft_system(dt=dt)

    # ノイズ共分散
    Q = np.diag([0.001, 0.01])  # プロセスノイズ
    R = np.array([[measurement_noise_std**2]])  # 観測ノイズ

    # 初期状態
    x_true = np.array([0.0, 0.0])
    x0_est = np.array([0.0, 0.0])
    P0 = np.diag([1.0, 1.0])

    # フィルタ作成
    kf = KalmanFilter(A, B, C, Q, R, x0_est, P0)
    mckf = SimpleMCKF(
        A, B, C, Q, R, x0_est, P0,
        kernel_bandwidth=kernel_bandwidth,
        max_iterations=10
    )

    # 結果格納
    results = {
        'time': time,
        'true_state': np.zeros((num_steps, 2)),
        'measurement': np.zeros(num_steps),
        'is_outlier': np.zeros(num_steps, dtype=bool),
        'kf_estimate': np.zeros((num_steps, 2)),
        'mckf_estimate': np.zeros((num_steps, 2)),
        'mckf_iterations': np.zeros(num_steps, dtype=int),
    }

    print("=" * 60)
    print("Simple MCKF Test (No Delay)")
    print("=" * 60)
    print(f"Total time: {total_time} s")
    print(f"Time step: {dt} s")
    print(f"Steps: {num_steps}")
    print(f"Measurement noise: {measurement_noise_std} rad")
    print(f"Outlier probability: {outlier_prob*100}%")
    print(f"Kernel bandwidth η: {kernel_bandwidth}")
    print("=" * 60)

    for k in range(num_steps):
        t = time[k]

        # 制御入力 (正弦波 + ステップ)
        u = np.array([0.5 * np.sin(0.5 * t)])
        if t > 10.0:
            u += 1.0

        # 真の状態更新
        process_noise = np.random.multivariate_normal([0, 0], Q)
        x_true = A @ x_true + B.flatten() * u + process_noise

        # 真の観測
        y_true = C @ x_true

        # 非ガウスノイズ付加
        is_outlier = np.random.rand() < outlier_prob
        if is_outlier:
            noise = np.random.randn() * measurement_noise_std * 10.0
        else:
            noise = np.random.randn() * measurement_noise_std

        y_meas = y_true + noise

        # 結果保存
        results['true_state'][k] = x_true
        results['measurement'][k] = y_meas[0]
        results['is_outlier'][k] = is_outlier

        # 標準KF
        x_kf, P_kf, _ = kf.step(y_meas, u)

        # SimpleMCKF
        x_mckf, P_mckf, info = mckf.step(y_meas, u)

        results['kf_estimate'][k] = x_kf
        results['mckf_estimate'][k] = x_mckf
        results['mckf_iterations'][k] = info['num_iterations']

        if (k + 1) % 50 == 0:
            print(f"Step {k+1}/{num_steps} ({(k+1)/num_steps*100:.1f}%)")

    # 誤差統計
    kf_error = results['true_state'] - results['kf_estimate']
    mckf_error = results['true_state'] - results['mckf_estimate']

    print("\n" + "=" * 60)
    print("Results:")
    print("=" * 60)
    print("\nStandard Kalman Filter:")
    print(f"  Angle RMSE:    {np.sqrt(np.mean(kf_error[:, 0]**2)):.4f} rad")
    print(f"  Velocity RMSE: {np.sqrt(np.mean(kf_error[:, 1]**2)):.4f} rad/s")

    print("\nSimple MCKF:")
    print(f"  Angle RMSE:    {np.sqrt(np.mean(mckf_error[:, 0]**2)):.4f} rad")
    print(f"  Velocity RMSE: {np.sqrt(np.mean(mckf_error[:, 1]**2)):.4f} rad/s")
    print(f"  Avg iterations: {mckf.get_avg_iterations():.2f}")

    return results


def plot_results(results: dict, save_path: Path = None):
    """結果のプロット"""
    fig, axes = plt.subplots(4, 1, figsize=(12, 10))
    time = results['time']

    # 1. 姿勢角推定
    ax = axes[0]
    ax.plot(time, results['true_state'][:, 0], 'k-', linewidth=2, label='True', alpha=0.7)
    ax.plot(time, results['kf_estimate'][:, 0], 'b--', linewidth=1.5, label='Standard KF')
    ax.plot(time, results['mckf_estimate'][:, 0], 'r-', linewidth=1.5, label='Simple MCKF')

    # 外れ値をマーク
    outlier_idx = results['is_outlier']
    ax.scatter(time[outlier_idx], results['measurement'][outlier_idx],
               c='orange', s=30, marker='x', alpha=0.7, label='Outlier', zorder=5)

    ax.set_ylabel('Angle θ [rad]')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_title('Spacecraft Attitude Estimation with Simple MCKF (No Delay)')

    # 2. 角速度推定
    ax = axes[1]
    ax.plot(time, results['true_state'][:, 1], 'k-', linewidth=2, label='True', alpha=0.7)
    ax.plot(time, results['kf_estimate'][:, 1], 'b--', linewidth=1.5, label='Standard KF')
    ax.plot(time, results['mckf_estimate'][:, 1], 'r-', linewidth=1.5, label='Simple MCKF')
    ax.set_ylabel('Angular Velocity ω [rad/s]')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # 3. 推定誤差 (対数スケール)
    ax = axes[2]
    kf_error = np.abs(results['true_state'][:, 0] - results['kf_estimate'][:, 0])
    mckf_error = np.abs(results['true_state'][:, 0] - results['mckf_estimate'][:, 0])
    ax.plot(time, kf_error, 'b--', linewidth=1.5, label='Standard KF', alpha=0.7)
    ax.plot(time, mckf_error, 'r-', linewidth=1.5, label='Simple MCKF')
    ax.set_ylabel('|Angle Error| [rad]')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # 4. MCKF反復回数
    ax = axes[3]
    iter_idx = results['mckf_iterations'] > 0
    ax.plot(time[iter_idx], results['mckf_iterations'][iter_idx],
            'ro', markersize=3, label='MCKF Iterations')
    ax.set_ylabel('Iterations')
    ax.set_xlabel('Time [s]')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")

    return fig


def main():
    """メイン実行"""
    results_dir = Path("results")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = results_dir / f"mckf_simple_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)

    # シミュレーション実行
    results = run_comparison(
        total_time=20.0,
        dt=0.1,
        measurement_noise_std=0.1,
        outlier_prob=0.1,  # 10%の外れ値
        kernel_bandwidth=2.0,  # η=2.0 (論文より小さめで外れ値抑制強化)
        seed=42
    )

    # プロット
    fig = plot_results(results, save_path=save_dir / "mckf_simple_comparison.png")
    plt.show()

    print(f"\nResults saved to: {save_dir}")


if __name__ == "__main__":
    main()
