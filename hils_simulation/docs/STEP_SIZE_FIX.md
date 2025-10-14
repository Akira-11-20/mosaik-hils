# Step Size ミスマッチ問題の解決

## 日時
2025-10-13

## 問題の発見

シミュレーション実行後、HDF5 データを確認すると：
- **Position と Velocity は正しく変化していた**（Env が正常動作）
- **でも Error は常に 10.000 のまま**（Controller が状態を受け取っていない）

```
Time [ms] | Position [m] | Velocity [m/s] | Error [m]
------------------------------------------------------------
       0 |     0.000000 |       0.000000 | 10.000000
      20 |     0.000000 |       0.000000 | 10.000000
     ...
     180 |     0.000555 |       0.014800 | 10.000000  ← Position は変化しているのに Error は固定！
```

## 原因

### 1. 接続構文の問題

**間違った接続:**
```python
world.connect(
    spacecraft,
    controller,
    ("position", "velocity"),  # タプルとして渡していた
    time_shifted=True,
    ...
)
```

Mosaik の `world.connect()` では、複数の属性を接続する場合、**タプルではなく個別の引数**として渡す必要がありました。

**正しい接続:**
```python
world.connect(
    spacecraft,
    controller,
    "position",     # 個別の引数として渡す
    "velocity",     # 個別の引数として渡す
    time_shifted=True,
    ...
)
```

### 2. Step Size のミスマッチ

- **Controller の step_size = 10ms**（制御周期）
- **Env の step_size = 1ms**（高頻度実行）

`time_shifted=True` により、Controller は「前のステップ」の Env の値を受け取ります。しかし、Bridge の遅延（~100-150ms）により、最初の 150-200ms では Controller が初期値しか受け取れていませんでした。

### 3. 遅延の累積効果

データフローでの遅延：
1. **Controller → Bridge(cmd)**: 約 50ms + jitter
2. **Bridge(cmd) → Plant**: 即座
3. **Plant → Bridge(sense)**: 約 100ms + jitter
4. **Bridge(sense) → Env**: 即座
5. **Env → Controller**: time_shifted により 1 Controller step (10ms)

**合計遅延: 約 150-170ms**

そのため、Controller が更新された値を受け取るのは約 200ms 以降になります。

## 解決策

### 1. 接続構文の修正

```python
# main_hils.py line 177-187
world.connect(
    spacecraft,
    controller,
    "position",      # ✅ 個別引数
    "velocity",      # ✅ 個別引数
    time_shifted=True,
    initial_data={
        "position": 0.0,
        "velocity": 0.0,
    },
)
```

### 2. シミュレーション時間の延長

遅延効果を観測するため、`SIMULATION_TIME` を 0.5秒（500ステップ）に延長しました。

```python
# main_hils.py line 33-35
SIMULATION_TIME = 0.5   # シミュレーション時間 [秒] = 0.5秒（テスト用）
TIME_RESOLUTION = 0.001 # 時間解像度 [秒/step] = 1step = 1ms
SIMULATION_STEP = int(SIMULATION_TIME / TIME_RESOLUTION)  # シミュレーションステップ数（500ステップ）
```

**重要な修正:**
- 以前は `SIMULATION_STEP = SIMULATION_TIME` としていたが、これは偶然動いていただけ
- 正しくは `SIMULATION_STEP = SIMULATION_TIME / TIME_RESOLUTION` で計算すべき
- `world.run(until=SIMULATION_STEP)` の `until` はステップ数を指定する

### 3. Controller の inputs 処理の改善

```python
# controller_simulator.py line 146-155
# 入力: 状態量（位置・速度）の受信
if eid in inputs:
    if "position" in inputs[eid]:
        pos_values = inputs[eid]["position"].values()
        if pos_values:  # 空でないことを確認
            entity["position"] = list(pos_values)[0]

    if "velocity" in inputs[eid]:
        vel_values = inputs[eid]["velocity"].values()
        if vel_values:  # 空でないことを確認
            entity["velocity"] = list(vel_values)[0]
```

## 結果

### 修正後のデータ

```
Time [ms] | Position [m] | Velocity [m/s] | Error [m] | Thrust [N]
--------------------------------------------------------------------------------
       0 |     0.000000 |       0.000000 | 10.000000 |     20.000
      50 |     0.000000 |       0.000000 | 10.000000 |     20.000
     100 |     0.000000 |       0.000000 | 10.000000 |     20.000
     150 |     0.000141 |       0.007400 |  9.999867 |     19.964  ← 初めて変化！
     200 |     0.000766 |       0.017400 |  9.999252 |     19.913
     250 |     0.001891 |       0.027397 |  9.998137 |     19.860
     300 |     0.003515 |       0.037377 |  9.996522 |     19.807
     350 |     0.005638 |       0.047329 |  9.994410 |     19.753
     400 |     0.008257 |       0.057257 |  9.991800 |     19.698
     450 |     0.011373 |       0.067155 |  9.988694 |     19.643
```

✅ **Error が正しく変化している！**
✅ **Thrust が正しく調整されている！**
✅ **遅延効果が正しくシミュレートされている！**

## 重要な学び

### 1. Mosaik の接続構文

複数属性を接続する場合：
```python
# ❌ 間違い
world.connect(src, dst, ("attr1", "attr2"))

# ✅ 正しい
world.connect(src, dst, "attr1", "attr2")
```

### 2. time_shifted 接続の意味

`time_shifted=True` は：
- 「このステップの出力を次のステップで使う」
- つまり、**受信側の step_size 分の遅延**が入る
- Controller の step_size=10ms なので、10ms の遅延

### 3. 遅延の累積効果

HILS システムでは、複数の遅延が累積します：
- Bridge(cmd): ~50ms
- Bridge(sense): ~100ms
- time_shifted: ~10ms
- **合計: ~160ms**

そのため、Controller が制御を開始するのは約 200ms 以降になります。

### 4. SIMULATION_STEP の正しい計算

```python
# ❌ 間違い（偶然動いていただけ）
SIMULATION_STEP = SIMULATION_TIME  # 500 [ms] を直接代入

# ✅ 正しい
SIMULATION_STEP = int(SIMULATION_TIME / TIME_RESOLUTION)  # 0.5 [秒] / 0.001 = 500 [ステップ]
```

`world.run(until=SIMULATION_STEP)` の `until` は**ステップ数**を指定するため、時間単位から変換が必要です。

## デバッグのヒント

### inputs の内容を確認する

Controller に以下のデバッグ出力を追加すると、inputs の構造を確認できます：

```python
if eid in inputs:
    if time % (self.step_size * 10) == 0:
        print(f"[DEBUG] t={time}ms, inputs for {eid}: {inputs[eid]}")
```

出力例：
```
[DEBUG] t=0ms, inputs for PDController_0: {'position': {'EnvSim-0.Spacecraft1DOF_0': 0.0}, 'velocity': {'EnvSim-0.Spacecraft1DOF_0': 0.0}}
[DEBUG] t=100ms, inputs for PDController_0: {'position': {'EnvSim-0.Spacecraft1DOF_0': 0.0}, 'velocity': {'EnvSim-0.Spacecraft1DOF_0': 0.0}}
[DEBUG] t=200ms, inputs for PDController_0: {'position': {'EnvSim-0.Spacecraft1DOF_0': 0.0005402}, 'velocity': {'EnvSim-0.Spacecraft1DOF_0': 0.0146}}
```

### HDF5 データの確認

```python
import h5py

with h5py.File('results/YYYYMMDD-HHMMSS/hils_data.h5', 'r') as f:
    time_s = f['data/time_s'][:]
    position = f['data/position_EnvSim-0.Spacecraft1DOF_0'][:]
    error = f['data/error_ControllerSim-0.PDController_0'][:]

    # Position は変化しているのに Error が変化していない → Controller が inputs を受け取っていない
    for i in range(0, len(time_s), 50):
        print(f"t={time_s[i]*1000:.0f}ms: pos={position[i]:.6f}m, error={error[i]:.6f}m")
```

## 関連ファイル

- [main_hils.py](main_hils.py:177-187) - 接続設定の修正箇所
- [controller_simulator.py](simulators/controller_simulator.py:146-155) - inputs 処理の改善
- [TIME_SHIFTED_FIX.md](TIME_SHIFTED_FIX.md) - time_shifted 接続の説明

## まとめ

**問題:** Controller が Env からの状態を受け取れず、error が常に 10.000 のまま

**原因:**
1. `world.connect()` の接続構文が間違っていた（タプルではなく個別引数が必要）
2. Step size のミスマッチと time_shifted による遅延
3. Bridge の遅延（~160ms）により、初期の 200ms は初期値しか届かない

**解決策:**
1. 接続構文を修正（個別引数で渡す）
2. シミュレーション時間を延長（0.5秒）
3. SIMULATION_STEP の計算を修正

**結果:** ✅ Controller が正しく動作し、error と thrust が正しく変化するようになった！
