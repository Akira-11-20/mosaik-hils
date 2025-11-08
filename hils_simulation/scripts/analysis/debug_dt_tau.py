"""
dt/τ の計算を詳細に確認
"""

import h5py

# HDF5ファイルを読み込み
h5_path = "/home/akira/mosaik-hils/hils_simulation/results/20251108-141007/hils_data.h5"

with h5py.File(h5_path, 'r') as f:
    # 時間データ
    time_ms = f['time']['time_ms'][:]
    time_s = f['time']['time_s'][:]

    # PlantSimのデータ
    plant_group = 'PlantSim-0_ThrustStand_0'
    measured_thrust = f[plant_group]['measured_thrust'][:]
    actual_thrust = f[plant_group]['actual_thrust'][:]
    time_constant = f[plant_group]['time_constant'][:]

    print("=== パラメータ確認 ===")
    print(f"Time resolution: {time_s[1] - time_s[0]:.10f} s = {(time_s[1] - time_s[0])*1000:.10f} ms")
    print(f"Time constant τ: {time_constant[100]:.10f} ms")
    print()

    # ステップ0からステップ100への遷移を解析
    step_0 = 0
    step_100 = 100

    y_0 = actual_thrust[step_0]
    y_100 = actual_thrust[step_100]
    u_100 = measured_thrust[step_100]  # 現在の実装では u[k] を使っている

    time_0 = time_ms[step_0]
    time_100 = time_ms[step_100]

    dt = time_100 - time_0
    tau = time_constant[step_100]

    print("=== ステップ0 → ステップ100 の遷移 ===")
    print(f"y[0] = {y_0:.10f} N")
    print(f"y[100] = {y_100:.10f} N")
    print(f"u[100] = {u_100:.10f} N")
    print(f"time[0] = {time_0:.10f} ms")
    print(f"time[100] = {time_100:.10f} ms")
    print(f"dt = {dt:.10f} ms")
    print(f"τ = {tau:.10f} ms")
    print()

    # PlantSimulator内部の計算を再現
    print("=== PlantSimulator内部の計算再現 ===")

    # step_size と time_resolution から dt を計算
    # plant_simulator.py:195 より
    # dt = self.step_size * self.time_resolution * 1000  # [ms]

    # step_size を推定（PlantSimは10msごとに動作）
    step_size = 100  # 10ms / 0.1ms = 100 steps
    time_resolution = 0.0001  # 0.1ms

    dt_calc = step_size * time_resolution * 1000
    print(f"step_size = {step_size}")
    print(f"time_resolution = {time_resolution:.10f} s")
    print(f"dt_calc = step_size * time_resolution * 1000 = {dt_calc:.10f} ms")
    print()

    # サブステップ計算
    sub_steps = max(1, int(dt_calc / 0.1))
    dt_sub = dt_calc / sub_steps

    print(f"sub_steps = max(1, int({dt_calc:.3f} / 0.1)) = {sub_steps}")
    print(f"dt_sub = {dt_calc:.10f} / {sub_steps} = {dt_sub:.10f} ms")
    print()

    # alpha = τ / dt_sub の計算
    alpha = tau / dt_sub
    print(f"α = τ / dt_sub = {tau:.10f} / {dt_sub:.10f} = {alpha:.10f}")
    print(f"1/α = dt_sub / τ = {1/alpha:.10f}")
    print()

    # 一次遅延の計算を手動で実行
    print("=== 一次遅延計算の手動実行 ===")
    y = y_0  # 0.0
    u = u_100  # 75.328480

    print(f"初期値: y = {y:.10f}")
    print(f"入力: u = {u:.10f}")
    print()

    for i in range(min(5, sub_steps)):  # 最初の5サブステップを表示
        y_old = y
        delta = (dt_sub / tau) * (u - y)
        y = y + delta
        print(f"サブステップ {i}: y = {y_old:.10f} + ({dt_sub:.10f}/{tau:.10f}) * ({u:.10f} - {y_old:.10f})")
        print(f"             = {y_old:.10f} + {dt_sub/tau:.10f} * {u - y_old:.10f}")
        print(f"             = {y_old:.10f} + {delta:.10f}")
        print(f"             = {y:.10f}")
        print()

    if sub_steps > 5:
        print(f"... ({sub_steps - 5} サブステップ省略) ...")
        print()
        # 残りを計算
        for i in range(5, sub_steps):
            y = y + (dt_sub / tau) * (u - y)

    print(f"最終値: y[100] = {y:.10f} N")
    print(f"実測値: y[100] = {y_100:.10f} N")
    print(f"差分: {abs(y - y_100):.15f} N")
    print()

    # 期待値の計算（dt/τ = 0.1 の場合）
    print("=== 期待値の計算 ===")
    print("もし dt/τ = 0.1 であれば:")

    dt_expected = 10.0  # ms
    tau_expected = 100.0  # ms
    ratio_expected = dt_expected / tau_expected  # 0.1

    y_expected = y_0 + ratio_expected * (u_100 - y_0)
    print(f"y[100] = {y_0} + {ratio_expected:.10f} * ({u_100:.10f} - {y_0})")
    print(f"y[100] = {y_0} + {ratio_expected:.10f} * {u_100:.10f}")
    print(f"y[100] = {y_expected:.10f} N")
    print()

    # 実際の dt/τ を逆算
    # y[100] = y[0] + (dt/τ) * (u[100] - y[0])
    # 7.171863 = 0 + (dt/τ) * (75.328480 - 0)
    # dt/τ = 7.171863 / 75.328480

    actual_ratio = y_100 / u_100
    print("実際の dt/τ を逆算:")
    print(f"dt/τ = y[100] / u[100] = {y_100:.10f} / {u_100:.10f} = {actual_ratio:.10f}")
    print("期待値: dt/τ = 10/100 = 0.1000000000")
    print(f"差分: {abs(actual_ratio - 0.1):.15f}")
    print()

    # サブステップを考慮した場合の理論値
    # y = u * (1 - (1 - dt_sub/τ)^n) where n = sub_steps
    print("=== サブステップを考慮した理論値 ===")
    ratio_sub = dt_sub / tau
    y_theory = u_100 * (1 - (1 - ratio_sub)**sub_steps)
    print("y = u * (1 - (1 - dt_sub/τ)^n)")
    print(f"  = {u_100:.10f} * (1 - (1 - {ratio_sub:.10f})^{sub_steps})")
    print(f"  = {u_100:.10f} * (1 - {(1 - ratio_sub)**sub_steps:.10f})")
    print(f"  = {y_theory:.10f} N")
    print(f"実測値: {y_100:.10f} N")
    print(f"差分: {abs(y_theory - y_100):.15f} N")
