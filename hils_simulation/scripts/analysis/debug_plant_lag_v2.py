"""
一次遅延の動作確認スクリプト v2
10msステップでの動作を考慮
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

    print("=== シミュレーション設定 ===")
    print(f"Time resolution: {time_s[1] - time_s[0]:.6f} s = {time_ms[1] - time_ms[0]:.3f} ms")
    print(f"Time constant τ: {time_constant[100]:.2f} ms")
    print()

    # PlantSimが動作するステップを特定（measured_thrustが変化するステップ）
    plant_steps = []
    for i in range(1, len(measured_thrust)):
        if measured_thrust[i] != measured_thrust[i-1]:
            plant_steps.append(i)

    print("=== PlantSimが動作するステップ（最初の10個）===")
    for step in plant_steps[:10]:
        print(f"Step {step:4d}: time={time_ms[step]:8.3f} ms, u={measured_thrust[step]:10.6f} N")
    print()

    # ステップ100の時刻を確認
    step_100_time = time_ms[100]
    print(f"ステップ100の時刻: {step_100_time:.3f} ms")

    # ステップ100の前後でPlantSimが動作したステップを探す
    prev_plant_step = None
    next_plant_step = None

    for step in plant_steps:
        if step <= 100:
            prev_plant_step = step
        if step > 100 and next_plant_step is None:
            next_plant_step = step
            break

    print(f"ステップ100の直前のPlant動作: ステップ {prev_plant_step} ({time_ms[prev_plant_step]:.3f} ms)")
    print(f"ステップ100の直後のPlant動作: ステップ {next_plant_step} ({time_ms[next_plant_step]:.3f} ms)")
    print()

    # Plant動作ステップでの詳細を確認
    print("=== Plant動作ステップの詳細（ステップ0-1000）===")
    print("Step | Time[ms] | dt[ms] | u (measured) | y (actual) | Δy | 理論Δy")
    print("-" * 90)

    for i, step in enumerate(plant_steps[:11]):
        if step == 0:
            dt = 0
            prev_y = 0
            prev_u = 0
        else:
            prev_step = plant_steps[i-1]
            dt = time_ms[step] - time_ms[prev_step]
            prev_y = actual_thrust[prev_step]
            prev_u = measured_thrust[prev_step]

        current_y = actual_thrust[step]
        current_u = measured_thrust[step]
        delta_y = current_y - prev_y

        # 一次遅延の理論値を計算
        # y[k+1] = y[k] + (dt/τ) * (u[k] - y[k])
        tau = time_constant[step]
        if dt > 0:
            theory_delta_y = (dt / tau) * (prev_u - prev_y)
            theory_y = prev_y + theory_delta_y

            # サブステップ計算
            sub_steps = max(1, int(dt / 0.1))
            dt_sub = dt / sub_steps
            y_sub = prev_y
            for _ in range(sub_steps):
                y_sub = y_sub + (dt_sub / tau) * (prev_u - y_sub)

            print(f"{step:4d} | {time_ms[step]:8.3f} | {dt:6.2f} | {current_u:12.6f} | {current_y:10.6f} | {delta_y:7.4f} | {theory_delta_y:7.4f} (単純) {y_sub - prev_y:7.4f} (サブ)")
        else:
            print(f"{step:4d} | {time_ms[step]:8.3f} | {dt:6.2f} | {current_u:12.6f} | {current_y:10.6f} | {delta_y:7.4f} | -")

    print()

    # ステップ100が最初の推力印加ステップかを確認
    print("=== ステップ100の詳細解析 ===")
    step = 100

    # ステップ100での入力と出力
    u_100 = measured_thrust[100]
    y_100 = actual_thrust[100]

    print("ステップ100:")
    print(f"  Time: {time_ms[100]:.3f} ms")
    print(f"  Measured thrust (u): {u_100:.6f} N")
    print(f"  Actual thrust (y): {y_100:.6f} N")
    print()

    # 前のPlant動作ステップ（ステップ0）からの計算
    if prev_plant_step == 0:
        print("ステップ0からステップ100への遷移:")
        y_0 = actual_thrust[0]
        u_0 = measured_thrust[0]
        dt = time_ms[100] - time_ms[0]
        tau = time_constant[100]

        print(f"  y[0] = {y_0:.6f} N")
        print(f"  u[0] = {u_0:.6f} N")
        print(f"  dt = {dt:.3f} ms")
        print(f"  τ = {tau:.2f} ms")
        print()

        # 一次遅延計算（サブステップ）
        sub_steps = max(1, int(dt / 0.1))
        dt_sub = dt / sub_steps

        print("サブステップ計算:")
        print(f"  サブステップ数: {sub_steps}")
        print(f"  dt_sub = {dt_sub:.3f} ms")

        y = y_0
        for i in range(sub_steps):
            y = y + (dt_sub / tau) * (u_0 - y)

        print(f"  最終値: y[100] = {y:.6f} N")
        print(f"  実際の値: {y_100:.6f} N")
        print(f"  差分: {y_100 - y:.9f} N")
        print()

        # 理論値との比較（連続時間の解）
        # y(t) = u * (1 - exp(-t/τ)) （初期値0、ステップ入力の場合）
        # ただし、u[0]=0なので、y(t)=0が正しい

        print("問題の診断:")
        print(f"  u[0] = {u_0:.6f} N （入力は0）")
        print("  理論値: y(10ms) = 0 N （入力が0なので出力も0）")
        print(f"  実際の値: y[100] = {y_100:.6f} N （≠ 0!）")
        print()
        print("→ 入力u[0]=0なのに、出力y[100]≠0 となっている")
        print("→ これは、ステップ100でu[100]=75.33Nを使って計算している可能性がある")

    print()
    print("=== 仮説検証: ステップ100でu[100]を使っているか？ ===")

    # もしu[100]=75.33を使った場合の計算
    u_100_input = measured_thrust[100]  # 75.328480
    y_0 = actual_thrust[0]  # 0
    dt = time_ms[100] - time_ms[0]  # 10ms
    tau = time_constant[100]  # 100ms

    # サブステップ計算
    sub_steps = max(1, int(dt / 0.1))
    dt_sub = dt / sub_steps

    y = y_0
    for i in range(sub_steps):
        y = y + (dt_sub / tau) * (u_100_input - y)

    print(f"もし u[100]={u_100_input:.6f}N を入力として使った場合:")
    print(f"  計算値: {y:.6f} N")
    print(f"  実際の値: {y_100:.6f} N")
    print(f"  差分: {abs(y - y_100):.9f} N")

    if abs(y - y_100) < 1e-6:
        print()
        print("✓ 一致しました！")
        print("→ PlantSimは現在のステップの入力 u[k] を使って y[k] を計算している")
        print("→ 本来は u[k-1] を使うべき（一次遅延は過去の入力に応答）")
