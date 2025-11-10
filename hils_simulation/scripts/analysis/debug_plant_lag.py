"""
一次遅延の動作確認スクリプト
100ステップ目での actual_thrust の値が理論値と一致するか検証
"""

import h5py
import numpy as np

# HDF5ファイルを読み込み
h5_path = "/home/akira/mosaik-hils/hils_simulation/results/20251108-141007/hils_data.h5"

with h5py.File(h5_path, "r") as f:
    print("=== HDF5ファイル構造 ===")
    print("Groups:", list(f.keys()))
    print()

    # 時間データ
    time_ms = f["time"]["time_ms"][:]
    time_s = f["time"]["time_s"][:]

    # PlantSimのデータ
    plant_group = None
    for key in f.keys():
        if "PlantSim" in key:
            plant_group = key
            break

    if plant_group is None:
        print("PlantSim group not found!")
        exit(1)

    print(f"PlantSim group: {plant_group}")
    print(f"Attributes: {list(f[plant_group].keys())}")
    print()

    # データ読み込み
    measured_thrust = f[plant_group]["measured_thrust"][:]
    actual_thrust = f[plant_group]["actual_thrust"][:]
    time_constant = f[plant_group]["time_constant"][:]

    # 100ステップ目を確認
    step = 100
    print(f"=== ステップ {step} の詳細 ===")
    print(f"Time: {time_ms[step]:.3f} ms ({time_s[step]:.6f} s)")
    print(f"Measured thrust (u): {measured_thrust[step]:.6f} N")
    print(f"Actual thrust (y): {actual_thrust[step]:.6f} N")
    print(f"Time constant (τ): {time_constant[step]:.6f} ms")
    print()

    # 前のステップからの変化を確認
    print(f"=== ステップ {step - 1} → {step} の遷移 ===")
    print(f"y[{step - 1}] = {actual_thrust[step - 1]:.6f} N")
    print(f"y[{step}] = {actual_thrust[step]:.6f} N")
    print(f"u[{step - 1}] = {measured_thrust[step - 1]:.6f} N")
    print(f"u[{step}] = {measured_thrust[step]:.6f} N")
    print()

    # 時定数とステップサイズから期待値を計算
    dt = time_ms[step] - time_ms[step - 1]
    tau = time_constant[step]

    print("=== 一次遅延の理論計算 ===")
    print(f"dt = {dt:.6f} ms")
    print(f"τ = {tau:.6f} ms")
    print(f"dt/τ = {dt / tau:.6f}")
    print()

    # 1ステップでの期待値（単純計算）
    y_prev = actual_thrust[step - 1]
    u_current = measured_thrust[step - 1]  # step関数内ではstep-1の入力を使う

    # 期待される変化量
    expected_y_simple = y_prev + (dt / tau) * (u_current - y_prev)
    print("単純計算 (1ステップ):")
    print(f"  y[{step}] = y[{step - 1}] + (dt/τ) * (u[{step - 1}] - y[{step - 1}])")
    print(f"  y[{step}] = {y_prev:.6f} + ({dt:.6f}/{tau:.6f}) * ({u_current:.6f} - {y_prev:.6f})")
    print(f"  y[{step}] = {y_prev:.6f} + {dt / tau:.6f} * {u_current - y_prev:.6f}")
    print(f"  y[{step}] = {y_prev:.6f} + {(dt / tau) * (u_current - y_prev):.6f}")
    print(f"  y[{step}] = {expected_y_simple:.6f} N")
    print()

    # サブステップ計算（実装に合わせる）
    sub_steps = max(1, int(dt / 0.1))
    dt_sub = dt / sub_steps

    print("サブステップ計算 (実装準拠):")
    print(f"  サブステップ数: {sub_steps}")
    print(f"  dt_sub = {dt_sub:.6f} ms")

    y = y_prev
    for i in range(sub_steps):
        y_old = y
        y = y + (dt_sub / tau) * (u_current - y)
        if i < 3 or i >= sub_steps - 3:  # 最初と最後の数ステップを表示
            print(
                f"    サブステップ {i}: y = {y_old:.6f} + ({dt_sub:.6f}/{tau:.6f}) * ({u_current:.6f} - {y_old:.6f}) = {y:.6f}"
            )

    print(f"  最終値: y[{step}] = {y:.6f} N")
    print()

    # 実際の値と比較
    print("=== 比較 ===")
    print(f"実際の値:       {actual_thrust[step]:.6f} N")
    print(f"理論値 (単純):  {expected_y_simple:.6f} N")
    print(f"理論値 (サブ):  {y:.6f} N")
    print(f"差分 (実際-サブ): {actual_thrust[step] - y:.9f} N")
    print()

    # 0ステップから100ステップまでの履歴を確認
    print("=== ステップ 0-20, 95-105 の履歴 ===")
    print("Step | Time[ms] | u (measured) | y (actual) | τ [ms]")
    print("-" * 70)

    for i in list(range(21)) + list(range(95, min(106, len(time_ms)))):
        print(
            f"{i:4d} | {time_ms[i]:8.3f} | {measured_thrust[i]:12.6f} | {actual_thrust[i]:10.6f} | {time_constant[i]:6.2f}"
        )

    print()

    # ゼロから75.33Nにステップ入力が入った場合の理論的な応答
    print("=== 理論的なステップ応答 (0 → 75.33 N) ===")
    u_step = 75.33
    tau_ms = 100.0

    # 各時刻での応答を計算
    print("Time[ms] | 理論値 y(t) = u * (1 - exp(-t/τ))")
    print("-" * 50)

    for t_ms in [0, 10, 20, 50, 100, 200, 300, 500]:
        if t_ms <= time_ms[-1]:
            y_theory = u_step * (1 - np.exp(-t_ms / tau_ms))
            print(f"{t_ms:8.1f} | {y_theory:10.6f} N ({y_theory / u_step * 100:.2f}%)")
