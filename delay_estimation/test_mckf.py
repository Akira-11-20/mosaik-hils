"""
MCKFのテスト・デモンストレーションスクリプト

1自由度宇宙機の姿勢制御システムにおいて、
通信遅延・パケット損失・非ガウスノイズがある環境で
MCKFの性能を検証します。

システムダイナミクス:
    姿勢角 θ と角速度 ω の2状態システム
    θ̇ = ω
    ω̇ = u/I  (I: 慣性モーメント)

シナリオ:
    - ランダム通信遅延 (0-5 steps)
    - パケット損失 (5%)
    - 非ガウスノイズ (外れ値を含む)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

from estimators.mckf import MaximumCorrentropyKalmanFilter
from estimators.kalman_filter import KalmanFilter


def create_spacecraft_system(dt: float = 0.1, inertia: float = 1.0):
    """
    1自由度宇宙機の状態空間モデルを作成

    状態: x = [θ, ω]' (姿勢角, 角速度)
    制御入力: u = トルク [Nm]
    観測: y = θ (姿勢角のみ観測)

    離散化:
        x(k+1) = A*x(k) + B*u(k)
        y(k) = C*x(k)

    Args:
        dt: サンプリング時間 [s]
        inertia: 慣性モーメント [kg·m²]

    Returns:
        A, B, C: システム行列
    """
    # 連続時間システム:
    # dx/dt = [0, 1; 0, 0]*x + [0; 1/I]*u
    Ac = np.array([
        [0.0, 1.0],
        [0.0, 0.0]
    ])
    Bc = np.array([[0.0], [1.0/inertia]])

    # 離散化 (ゼロ次ホールド)
    # A = I + Ac*dt + (Ac*dt)^2/2 (2次まで)
    A = np.eye(2) + Ac * dt + (Ac @ Ac) * (dt**2 / 2)
    B = Bc * dt

    # 観測行列 (姿勢角のみ)
    C = np.array([[1.0, 0.0]])

    return A, B, C


def generate_control_input(t: float) -> float:
    """
    制御入力の生成 (シミュレーション用)

    正弦波 + ステップ変化で姿勢変更を模擬

    Args:
        t: 時刻 [s]

    Returns:
        u: トルク [Nm]
    """
    # 正弦波ベース + ステップ変化
    u = 0.5 * np.sin(0.5 * t)

    # 時刻10秒でステップ変化
    if t > 10.0:
        u += 1.0

    return u


def add_non_gaussian_noise(
    measurement: np.ndarray,
    std: float,
    outlier_prob: float = 0.05,
    outlier_scale: float = 10.0
) -> np.ndarray:
    """
    非ガウスノイズの付加 (外れ値を含む)

    通常: ガウスノイズ N(0, std^2)
    外れ値: 確率 outlier_prob で N(0, (outlier_scale*std)^2)

    Args:
        measurement: 元の観測値
        std: 標準的なノイズの標準偏差
        outlier_prob: 外れ値の発生確率
        outlier_scale: 外れ値のスケール (通常の何倍か)

    Returns:
        noisy_measurement: ノイズを付加した観測値
    """
    if np.random.rand() < outlier_prob:
        # 外れ値
        noise = np.random.randn(*measurement.shape) * std * outlier_scale
    else:
        # 通常のガウスノイズ
        noise = np.random.randn(*measurement.shape) * std

    return measurement + noise


class DelayedNetwork:
    """
    通信遅延・パケット損失をシミュレートするネットワークモデル
    """

    def __init__(
        self,
        max_delay: int = 5,
        delay_probs: np.ndarray = None,
        packet_loss_prob: float = 0.05
    ):
        """
        Args:
            max_delay: 最大遅延ステップ数
            delay_probs: 各遅延の発生確率 [P(0), P(1), ..., P(K)]
            packet_loss_prob: パケット損失確率
        """
        self.max_delay = max_delay
        self.packet_loss_prob = packet_loss_prob

        if delay_probs is None:
            # 均等分布 (0遅延が最も高確率)
            self.delay_probs = np.array([0.5] + [0.5/(max_delay)] * max_delay)
            self.delay_probs = self.delay_probs / self.delay_probs.sum()
        else:
            self.delay_probs = delay_probs

        # 遅延バッファ: (観測値, 受信予定時刻)
        self.buffer = []

    def transmit(self, measurement: np.ndarray, current_time: int):
        """
        観測値を送信 (遅延・損失を考慮)

        Args:
            measurement: 観測値
            current_time: 現在時刻
        """
        # パケット損失判定
        if np.random.rand() < self.packet_loss_prob:
            return  # 送信せず (損失)

        # 遅延のサンプリング
        delay = np.random.choice(self.max_delay + 1, p=self.delay_probs)

        # バッファに追加 (受信時刻 = 現在時刻 + 遅延)
        receive_time = current_time + delay
        self.buffer.append((measurement.copy(), receive_time, delay))

    def receive(self, current_time: int):
        """
        現在時刻に受信できる観測値を取得

        Args:
            current_time: 現在時刻

        Returns:
            measurement: 受信した観測値 (なければNone)
            actual_delay: 実際の遅延ステップ数
        """
        # 受信可能な観測を検索
        received = None
        actual_delay = 0

        for i, (meas, recv_time, delay) in enumerate(self.buffer):
            if recv_time == current_time:
                received = meas
                actual_delay = delay
                # バッファから削除
                del self.buffer[i]
                break

        return received, actual_delay


def run_comparison_simulation(
    total_time: float = 20.0,
    dt: float = 0.1,
    max_delay: int = 5,
    measurement_noise_std: float = 0.1,
    outlier_prob: float = 0.1,
    seed: int = 42
):
    """
    MCKFと標準KFの比較シミュレーション

    Args:
        total_time: シミュレーション時間 [s]
        dt: サンプリング時間 [s]
        max_delay: 最大遅延ステップ数
        measurement_noise_std: 観測ノイズ標準偏差 [rad]
        outlier_prob: 外れ値発生確率
        seed: 乱数シード

    Returns:
        results: シミュレーション結果の辞書
    """
    np.random.seed(seed)

    # タイムステップ数
    num_steps = int(total_time / dt)
    time = np.arange(num_steps) * dt

    # システム作成
    A, B, C = create_spacecraft_system(dt=dt, inertia=1.0)

    # ノイズ共分散
    # プロセスノイズ (モデル化誤差、微小外乱)
    Q = np.diag([0.001, 0.01])  # [θ, ω]

    # 観測ノイズ (センサノイズ)
    R = np.array([[measurement_noise_std**2]])

    # 初期状態
    x_true = np.array([0.0, 0.0])  # [θ, ω]
    x0_est = np.array([0.0, 0.0])
    P0 = np.diag([1.0, 1.0])

    # フィルタの作成
    # 1. 標準カルマンフィルタ
    kf = KalmanFilter(A, B, C, Q, R, x0_est, P0)

    # 2. MCKF
    mckf = MaximumCorrentropyKalmanFilter(
        A, B, C, Q, R, x0_est, P0,
        max_delay=max_delay,
        kernel_bandwidth=2.0,  # η = 2.0 (外れ値抑制)
        max_iterations=10,
        convergence_threshold=1e-4
    )

    # 遅延ネットワーク
    network = DelayedNetwork(
        max_delay=max_delay,
        packet_loss_prob=0.05
    )

    # 結果格納
    results = {
        'time': time,
        'true_state': np.zeros((num_steps, 2)),
        'measurement': np.zeros(num_steps),
        'measurement_available': np.zeros(num_steps, dtype=bool),
        'delay': np.zeros(num_steps, dtype=int),
        'kf_estimate': np.zeros((num_steps, 2)),
        'mckf_estimate': np.zeros((num_steps, 2)),
        'kf_covariance': np.zeros((num_steps, 2, 2)),
        'mckf_covariance': np.zeros((num_steps, 2, 2)),
        'mckf_iterations': np.zeros(num_steps, dtype=int),
    }

    print("=" * 60)
    print("MCKF Simulation Starting...")
    print("=" * 60)
    print(f"Total time: {total_time} s")
    print(f"Time step: {dt} s")
    print(f"Number of steps: {num_steps}")
    print(f"Max delay: {max_delay} steps ({max_delay*dt} s)")
    print(f"Measurement noise std: {measurement_noise_std} rad")
    print(f"Outlier probability: {outlier_prob*100}%")
    print("=" * 60)

    # シミュレーションループ
    for k in range(num_steps):
        t = time[k]

        # 制御入力
        u = np.array([generate_control_input(t)])

        # 真の状態更新 (システムダイナミクス)
        process_noise = np.random.multivariate_normal([0, 0], Q)
        x_true = A @ x_true + B.flatten() * u + process_noise

        # 真の観測 (ノイズなし)
        y_true = C @ x_true

        # ノイズを付加した観測
        y_noisy = add_non_gaussian_noise(
            y_true,
            measurement_noise_std,
            outlier_prob=outlier_prob,
            outlier_scale=10.0
        )

        # ネットワーク送信
        network.transmit(y_noisy, k)

        # ネットワーク受信
        y_received, actual_delay = network.receive(k)

        # 結果保存
        results['true_state'][k] = x_true
        results['measurement'][k] = y_noisy[0] if y_noisy is not None else np.nan
        results['measurement_available'][k] = (y_received is not None)
        results['delay'][k] = actual_delay if y_received is not None else -1

        # フィルタ更新
        if y_received is not None:
            # 標準KF
            kf.predict(u)
            x_kf, P_kf, _ = kf.update(y_received)

            # MCKF
            x_mckf, P_mckf, info = mckf.step(y_received, k, u)

            results['mckf_iterations'][k] = info['num_iterations']
        else:
            # 観測なし (予測のみ)
            x_kf, P_kf = kf.predict(u)
            x_mckf, P_mckf, info = mckf.step(None, k, u)

        results['kf_estimate'][k] = x_kf
        results['mckf_estimate'][k] = x_mckf
        results['kf_covariance'][k] = P_kf
        results['mckf_covariance'][k] = P_mckf

        # 進捗表示
        if (k + 1) % 50 == 0:
            print(f"Step {k+1}/{num_steps} ({(k+1)/num_steps*100:.1f}%)")

    print("\nSimulation completed!")

    # 誤差統計
    kf_error = results['true_state'] - results['kf_estimate']
    mckf_error = results['true_state'] - results['mckf_estimate']

    print("\n" + "=" * 60)
    print("Estimation Performance:")
    print("=" * 60)
    print("\nStandard Kalman Filter:")
    print(f"  Angle RMSE:    {np.sqrt(np.mean(kf_error[:, 0]**2)):.4f} rad")
    print(f"  Velocity RMSE: {np.sqrt(np.mean(kf_error[:, 1]**2)):.4f} rad/s")

    print("\nMaximum Correntropy Kalman Filter:")
    print(f"  Angle RMSE:    {np.sqrt(np.mean(mckf_error[:, 0]**2)):.4f} rad")
    print(f"  Velocity RMSE: {np.sqrt(np.mean(mckf_error[:, 1]**2)):.4f} rad/s")
    print(f"  Avg iterations: {np.mean(results['mckf_iterations'][results['measurement_available']]):.2f}")

    return results


def plot_results(results: dict, save_path: Path = None):
    """
    シミュレーション結果のプロット

    Args:
        results: run_comparison_simulation()の戻り値
        save_path: 保存先パス
    """
    fig, axes = plt.subplots(4, 1, figsize=(12, 10))
    time = results['time']

    # 1. 姿勢角の推定
    ax = axes[0]
    ax.plot(time, results['true_state'][:, 0], 'k-', linewidth=2, label='True', alpha=0.7)
    ax.plot(time, results['kf_estimate'][:, 0], 'b--', linewidth=1.5, label='Standard KF')
    ax.plot(time, results['mckf_estimate'][:, 0], 'r-', linewidth=1.5, label='MCKF')

    # 観測点をマーク
    meas_idx = results['measurement_available']
    ax.scatter(time[meas_idx], results['measurement'][meas_idx],
               c='gray', s=10, alpha=0.3, label='Measurement (delayed)')

    ax.set_ylabel('Angle θ [rad]')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_title('Spacecraft Attitude Estimation with MCKF')

    # 2. 角速度の推定
    ax = axes[1]
    ax.plot(time, results['true_state'][:, 1], 'k-', linewidth=2, label='True', alpha=0.7)
    ax.plot(time, results['kf_estimate'][:, 1], 'b--', linewidth=1.5, label='Standard KF')
    ax.plot(time, results['mckf_estimate'][:, 1], 'r-', linewidth=1.5, label='MCKF')
    ax.set_ylabel('Angular Velocity ω [rad/s]')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # 3. 推定誤差
    ax = axes[2]
    kf_error = np.abs(results['true_state'][:, 0] - results['kf_estimate'][:, 0])
    mckf_error = np.abs(results['true_state'][:, 0] - results['mckf_estimate'][:, 0])
    ax.plot(time, kf_error, 'b--', linewidth=1.5, label='Standard KF', alpha=0.7)
    ax.plot(time, mckf_error, 'r-', linewidth=1.5, label='MCKF')
    ax.set_ylabel('|Angle Error| [rad]')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # 4. 通信遅延とMCKF反復回数
    ax = axes[3]
    ax2 = ax.twinx()

    # 遅延
    ax.bar(time[meas_idx], results['delay'][meas_idx],
           width=time[1]-time[0], alpha=0.3, color='gray', label='Delay')
    ax.set_ylabel('Delay [steps]', color='gray')
    ax.tick_params(axis='y', labelcolor='gray')

    # MCKF反復回数
    iter_idx = results['mckf_iterations'] > 0
    ax2.plot(time[iter_idx], results['mckf_iterations'][iter_idx],
             'ro', markersize=3, label='MCKF Iterations')
    ax2.set_ylabel('MCKF Iterations', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    ax.set_xlabel('Time [s]')
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")

    return fig


def main():
    """メイン実行"""
    # 結果保存ディレクトリ
    results_dir = Path("results")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = results_dir / f"mckf_test_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)

    # シミュレーション実行
    results = run_comparison_simulation(
        total_time=20.0,
        dt=0.1,
        max_delay=5,
        measurement_noise_std=0.1,
        outlier_prob=0.1,  # 10%の確率で外れ値
        seed=42
    )

    # プロット
    fig = plot_results(results, save_path=save_dir / "mckf_comparison.png")
    plt.show()

    print(f"\nResults saved to: {save_dir}")


if __name__ == "__main__":
    main()
