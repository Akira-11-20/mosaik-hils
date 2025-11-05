"""
論文の実験条件でMCKFを100回実行（モンテカルロシミュレーション）

論文: "Modified Kalman and Maximum Correntropy Kalman Filters for
Systems With Bernoulli Distribution k-step Random Delay and Packet Loss" (2024)

Table 3の結果を再現
"""

import numpy as np
import matplotlib.pyplot as plt
from estimators.mckf import MaximumCorrentropyKalmanFilter

# 論文のシステムパラメータ
T = 0.3  # サンプリング時間 [s]

F = np.array([
    [1, 0, T, 0],
    [0, 1, 0, T],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])

B = np.zeros((4, 1))
H = np.array([
    [-1, 0, -1, 0],
    [0, -1, 0, -1]
])

Q = np.diag([0.001, 0.001, 0.001, 0.001])
R = np.diag([10.0, 10.0])

x0 = np.array([0.0, 0.0, 0.0, 0.0])
P0 = np.eye(4) * 1.0

# シミュレーションパラメータ
N = 100
max_delay = 2
kernel_bandwidth = 4.0
num_monte_carlo = 100

# 遅延確率
delay_prob = 0.3
one_step_delay_prob = 0.24
two_step_delay_prob = 0.12

print("=" * 70)
print("MCKF Monte Carlo Simulation - 100 runs")
print("=" * 70)
print(f"Parameters: N={N}, max_delay={max_delay}, η={kernel_bandwidth}")
print(f"Monte Carlo runs: {num_monte_carlo}")
print()

# 結果を格納
all_rmse = np.zeros((num_monte_carlo, 4))

for mc_run in range(num_monte_carlo):
    if mc_run % 10 == 0:
        print(f"Progress: {mc_run}/{num_monte_carlo} ({mc_run/num_monte_carlo*100:.0f}%)")
    # 各実行で異なるシード
    np.random.seed(mc_run)

    # MCKFの初期化
    mckf = MaximumCorrentropyKalmanFilter(
        A=F, B=B, C=H, Q=Q, R=R,
        x0=x0, P0=P0,
        max_delay=max_delay,
        kernel_bandwidth=kernel_bandwidth,
        max_iterations=10,
        convergence_threshold=1e-4
    )

    # 真の状態の生成
    x_true = np.zeros((N, 4))
    x_true[0] = x0.copy()

    for k in range(1, N):
        w = np.random.multivariate_normal(np.zeros(4), Q)
        x_true[k] = F @ x_true[k-1] + w

    # 観測値の生成（遅延あり）
    y_measurements = []
    y_true = np.zeros((N, 2))

    for k in range(N):
        v = np.random.multivariate_normal(np.zeros(2), R)
        y_true[k] = H @ x_true[k] + v

        rand_val = np.random.random()

        if rand_val < (1 - delay_prob):
            # 遅延なし
            y_measurements.append((y_true[k].copy(), k))
        elif rand_val < (1 - delay_prob + one_step_delay_prob):
            # 1-step遅延
            if k + 1 < N:
                y_measurements.append((y_true[k].copy(), k + 1))
        elif rand_val < (1 - delay_prob + one_step_delay_prob + two_step_delay_prob):
            # 2-step遅延
            if k + 2 < N:
                y_measurements.append((y_true[k].copy(), k + 2))
        # else: パケット損失

    y_measurements.sort(key=lambda x: x[1])

    # MCKFの実行
    x_est = np.zeros((N, 4))

    for k in range(N):
        measurement = None
        for y_meas, t_meas in y_measurements:
            if t_meas == k:
                measurement = y_meas
                break

        x_k, P_k, info = mckf.step(measurement, k)
        x_est[k] = x_k

    # 誤差の計算
    errors = x_est - x_true
    rmse = np.sqrt(np.mean(errors**2, axis=0))
    all_rmse[mc_run] = rmse

# 平均RMSEの計算
mean_rmse = np.mean(all_rmse, axis=0)
std_rmse = np.std(all_rmse, axis=0)

print()
print("=" * 70)
print("Results (Mean ± Std over 100 Monte Carlo runs):")
print("=" * 70)
print(f"s1 (x position): {mean_rmse[0]:.4f} ± {std_rmse[0]:.4f}")
print(f"s2 (y position): {mean_rmse[1]:.4f} ± {std_rmse[1]:.4f}")
print(f"s3 (x velocity): {mean_rmse[2]:.4f} ± {std_rmse[2]:.4f}")
print(f"s4 (y velocity): {mean_rmse[3]:.4f} ± {std_rmse[3]:.4f}")
print()
print("Paper results (Table 3 - MCKF):")
print("s1: 0.7274, s2: 0.7172, s3: 0.1731, s4: 0.1533")
print()

# 論文との差
paper_rmse = np.array([0.7274, 0.7172, 0.1731, 0.1533])
diff = mean_rmse - paper_rmse
relative_diff = (mean_rmse - paper_rmse) / paper_rmse * 100

print("Difference from paper:")
print(f"s1: {diff[0]:+.4f} ({relative_diff[0]:+.1f}%)")
print(f"s2: {diff[1]:+.4f} ({relative_diff[1]:+.1f}%)")
print(f"s3: {diff[2]:+.4f} ({relative_diff[2]:+.1f}%)")
print(f"s4: {diff[3]:+.4f} ({relative_diff[3]:+.1f}%)")

# RMSEの分布をプロット
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
states = ['s1 (x position)', 's2 (y position)', 's3 (x velocity)', 's4 (y velocity)']

for i, (ax, state_name) in enumerate(zip(axes.flat, states)):
    ax.hist(all_rmse[:, i], bins=20, edgecolor='black', alpha=0.7, label='Our MCKF')
    ax.axvline(mean_rmse[i], color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_rmse[i]:.4f}')
    ax.axvline(paper_rmse[i], color='green', linestyle='--', linewidth=2, label=f'Paper: {paper_rmse[i]:.4f}')
    ax.set_xlabel('RMSE')
    ax.set_ylabel('Frequency')
    ax.set_title(state_name)
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('delay_estimation/mckf_monte_carlo.png', dpi=150)
print(f"\nPlot saved to: delay_estimation/mckf_monte_carlo.png")
