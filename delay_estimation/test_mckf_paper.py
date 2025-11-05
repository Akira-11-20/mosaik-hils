"""
論文の実験条件でMCKFをテスト

論文: "Modified Kalman and Maximum Correntropy Kalman Filters for
Systems With Bernoulli Distribution k-step Random Delay and Packet Loss" (2024)

Section 5: EXPERIMENTAL TESTING の条件を再現
"""

import numpy as np
import matplotlib.pyplot as plt
from estimators.mckf import MaximumCorrentropyKalmanFilter

# 論文のシステムパラメータ（Section 5）
T = 0.3  # サンプリング時間 [s]

# 状態遷移行列 F (4x4)
# [x1, x2, x3, x4]^T = [横位置, 縦位置, 横速度, 縦速度]^T
F = np.array([
    [1, 0, T, 0],
    [0, 1, 0, T],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])

# 制御入力なし（B = 0）
B = np.zeros((4, 1))

# 観測行列 H (2x4)
H = np.array([
    [-1, 0, -1, 0],
    [0, -1, 0, -1]
])

# プロセスノイズ共分散 Q
Q = np.diag([0.001, 0.001, 0.001, 0.001])

# 観測ノイズ共分散 R
R = np.diag([10.0, 10.0])

# 初期状態
x0 = np.array([0.0, 0.0, 0.0, 0.0])
P0 = np.eye(4) * 1.0

# シミュレーションパラメータ
N = 100  # 時間ステップ数
max_delay = 2  # 2-step random delay
kernel_bandwidth = 4.0  # η = 4 (論文のパラメータ)

# 遅延確率（論文の条件）
# - 遅延またはパケット損失の確率: 0.3
# - 遅延の確率: 0.3 × 0.8 = 0.24
# - 2-step遅延の確率: 0.3 × 0.8 × 0.5 = 0.12
delay_prob = 0.3
one_step_delay_prob = 0.24
two_step_delay_prob = 0.12

print("=" * 70)
print("MCKF Simulation - Paper Conditions (Two-step Random Delay)")
print("=" * 70)
print(f"\nSystem Parameters:")
print(f"  State dimension: {F.shape[0]}")
print(f"  Observation dimension: {H.shape[0]}")
print(f"  Sampling time T: {T} s")
print(f"  Simulation steps: {N}")
print(f"\nDelay Parameters:")
print(f"  Max delay: {max_delay} steps")
print(f"  Delay/loss probability: {delay_prob}")
print(f"  One-step delay probability: {one_step_delay_prob}")
print(f"  Two-step delay probability: {two_step_delay_prob}")
print(f"\nMCKF Parameters:")
print(f"  Kernel bandwidth η: {kernel_bandwidth}")
print(f"  Max iterations: 10")
print()

# MCKFの初期化
mckf = MaximumCorrentropyKalmanFilter(
    A=F,
    B=B,
    C=H,
    Q=Q,
    R=R,
    x0=x0,
    P0=P0,
    max_delay=max_delay,
    kernel_bandwidth=kernel_bandwidth,
    max_iterations=10,
    convergence_threshold=1e-4
)

# 真の状態とノイズの生成
np.random.seed(42)

# 真の状態の生成（単純な運動）
x_true = np.zeros((N, 4))
x_true[0] = x0.copy()

for k in range(1, N):
    # プロセスノイズ（ガウシアン）
    w = np.random.multivariate_normal(np.zeros(4), Q)
    x_true[k] = F @ x_true[k-1] + w

# 観測値の生成（遅延とパケット損失を含む）
y_measurements = []  # (観測値, タイムステップ) のリスト
y_true = np.zeros((N, 2))

for k in range(N):
    # 真の観測値
    v = np.random.multivariate_normal(np.zeros(2), R)
    y_true[k] = H @ x_true[k] + v

    # 遅延のシミュレーション
    rand_val = np.random.random()

    if rand_val < (1 - delay_prob):
        # 遅延なし（確率 0.7）
        y_measurements.append((y_true[k].copy(), k))
    elif rand_val < (1 - delay_prob + one_step_delay_prob):
        # 1-step遅延（確率 0.24）- 次のステップで届く
        if k + 1 < N:
            y_measurements.append((y_true[k].copy(), k + 1))
    elif rand_val < (1 - delay_prob + one_step_delay_prob + two_step_delay_prob):
        # 2-step遅延（確率 0.12）- 2ステップ後に届く
        if k + 2 < N:
            y_measurements.append((y_true[k].copy(), k + 2))
    # else: パケット損失（確率 0.06）

# 観測値を時刻順にソート
y_measurements.sort(key=lambda x: x[1])

# MCKFの実行
x_est = np.zeros((N, 4))
P_est = np.zeros((N, 4, 4))
iterations = []

print("Running MCKF simulation...")
print(f"Total measurements: {len(y_measurements)} / {N} ({len(y_measurements)/N*100:.1f}%)")
print()

for k in range(N):
    # この時刻に届いた観測を探す
    measurement = None
    for y_meas, t_meas in y_measurements:
        if t_meas == k:
            measurement = y_meas
            break

    # MCKFステップ
    x_k, P_k, info = mckf.step(measurement, k)

    x_est[k] = x_k
    P_est[k] = P_k
    iterations.append(info['num_iterations'])

    if k % 20 == 0:
        print(f"Step {k:3d}: Meas={measurement is not None}, "
              f"Iter={info['num_iterations']}, "
              f"Buffer={info['buffer_size']}")

print(f"\nSimulation completed!")
print(f"Average iterations: {np.mean(iterations):.2f}")
print(f"Max iterations: {np.max(iterations)}")

# 推定誤差の計算
errors = x_est - x_true
rmse = np.sqrt(np.mean(errors**2, axis=0))

print(f"\n{'='*70}")
print("Results:")
print(f"{'='*70}")
print(f"RMSE for each state:")
print(f"  s1 (x position): {rmse[0]:.4f}")
print(f"  s2 (y position): {rmse[1]:.4f}")
print(f"  s3 (x velocity): {rmse[2]:.4f}")
print(f"  s4 (y velocity): {rmse[3]:.4f}")
print()

# プロット
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('MCKF with Two-step Random Delay and Packet Loss', fontsize=14, fontweight='bold')

states = ['x position', 'y position', 'x velocity', 'y velocity']
time = np.arange(N) * T

for i, (ax, state_name) in enumerate(zip(axes.flat, states)):
    ax.plot(time, x_true[:, i], 'b-', label='True', linewidth=2)
    ax.plot(time, x_est[:, i], 'r--', label='MCKF Estimate', linewidth=1.5)
    ax.fill_between(time,
                     x_est[:, i] - 2*np.sqrt(P_est[:, i, i]),
                     x_est[:, i] + 2*np.sqrt(P_est[:, i, i]),
                     alpha=0.3, color='red', label='95% Confidence')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel(state_name)
    ax.set_title(f'{state_name} (RMSE: {rmse[i]:.4f})')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('delay_estimation/mckf_paper_test.png', dpi=150)
print(f"Plot saved to: delay_estimation/mckf_paper_test.png")

# 収束回数のヒストグラム
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.hist(iterations, bins=range(1, max(iterations)+2), edgecolor='black', alpha=0.7)
ax.set_xlabel('Number of Iterations')
ax.set_ylabel('Frequency')
ax.set_title('MCKF Convergence Iterations Distribution')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('delay_estimation/mckf_iterations.png', dpi=150)
print(f"Iteration plot saved to: delay_estimation/mckf_iterations.png")

print("\nTest completed successfully!")
