# Time Constant Recording in HDF5

## 概要

Plant time constantは以下の3箇所に記録されます：

### 1. 実行時ログ（コンソール出力）

```
[PlantSim] Created ThrustStand_0 (ID: stand_01, τ=58.15ms (mean=50.0ms, std=10.0ms, noise=2.0ms), lag=enabled)
```

- **τ=58.15ms**: 実際にサンプリングされたベース時定数（個体差反映後）
- **mean=50.0ms**: 設定した平均時定数
- **std=10.0ms**: 個体差の標準偏差
- **noise=2.0ms**: 時間変動ノイズの標準偏差

### 2. simulation_config.json（設定値）

```json
{
  "plant": {
    "time_constant_s": 0.05,           // 平均時定数（設定値）[s]
    "time_constant_std_s": 0.01,       // 個体差の標準偏差（設定値）[s]
    "time_constant_noise_s": 0.002,    // 時間変動ノイズの標準偏差（設定値）[s]
    "enable_lag": true
  }
}
```

→ **ユーザーが設定した値**が記録される（実際のサンプリング値ではない）

### 3. HDF5ファイル（各ステップの実測値）✨

```
PlantSim-0_ThrustStand_0/time_constant
  Shape: (N_steps,)
  dtype: float64
```

→ **各ステップで実際に使用された時定数**が記録される

## 時定数の記録方法

### ケース1: 個体差のみ（`std` > 0, `noise` = 0）

```python
DelayConfig(
    plant_time_constant=50.0,
    plant_time_constant_std=10.0,
    plant_time_constant_noise=0.0
)
```

**HDF5の記録内容:**
- 全ステップで**同じ値**（例: 58.15ms）
- この値は個体差によりサンプリングされた固定値
- シミュレーション実行ごとに異なる値

**例:**
```
time_constant[0] = 58.15
time_constant[1] = 58.15
time_constant[2] = 58.15
...
time_constant[999] = 58.15
```

### ケース2: 時間変動のみ（`std` = 0, `noise` > 0）

```python
DelayConfig(
    plant_time_constant=50.0,
    plant_time_constant_std=0.0,
    plant_time_constant_noise=5.0
)
```

**HDF5の記録内容:**
- 各ステップで**異なる値**
- ホワイトノイズにより変動
- 平均は設定値（50ms）付近

**例:**
```
time_constant[0] = 47.68
time_constant[1] = 52.34
time_constant[2] = 48.91
...
time_constant[999] = 51.23
```

### ケース3: 個体差 + 時間変動（`std` > 0, `noise` > 0）

```python
DelayConfig(
    plant_time_constant=50.0,
    plant_time_constant_std=10.0,
    plant_time_constant_noise=2.0
)
```

**HDF5の記録内容:**
- 各ステップで**異なる値**
- ベース値（個体差反映）+ ホワイトノイズ
- 平均はサンプリングされたベース値付近

**例:**（ベース値が58.15msの場合）
```
time_constant[0] = 58.15  (初期値)
time_constant[1] = 60.23  (58.15 + noise)
time_constant[2] = 56.47  (58.15 + noise)
...
time_constant[999] = 59.81  (58.15 + noise)
```

## データの確認方法

### Python で確認

```python
import h5py
import numpy as np

with h5py.File('results/YYYYMMDD-HHMMSS/hils_data.h5', 'r') as f:
    tau = f['PlantSim-0_ThrustStand_0/time_constant'][:]

    print(f"Mean: {tau.mean():.2f} ms")
    print(f"Std:  {tau.std():.2f} ms")
    print(f"Min:  {tau.min():.2f} ms")
    print(f"Max:  {tau.max():.2f} ms")
    print(f"Unique values: {len(set(tau))}/{len(tau)}")
```

### グラフ化

```python
import h5py
import matplotlib.pyplot as plt

with h5py.File('results/YYYYMMDD-HHMMSS/hils_data.h5', 'r') as f:
    time_s = f['time/time_s'][:]
    tau = f['PlantSim-0_ThrustStand_0/time_constant'][:]

    plt.figure(figsize=(12, 4))
    plt.plot(time_s, tau, alpha=0.7)
    plt.axhline(50.0, color='r', linestyle='--', label='Mean (50ms)')
    plt.xlabel('Time [s]')
    plt.ylabel('Time Constant τ [ms]')
    plt.title('Time-Varying Time Constant')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('time_constant_plot.png')
```

## 時定数の計算方法（内部動作）

### 初期化時（create）

```python
# 個体差の適用
if time_constant_std > 0:
    tau_base = max(0.1, np.random.normal(time_constant, time_constant_std))
else:
    tau_base = time_constant

entity["time_constant"] = tau_base  # ベース値を保存
```

### 各ステップ（step）

```python
if enable_lag:
    # ベース値を取得
    tau_base = entity["time_constant"]

    # 時間変動ノイズを追加
    if time_constant_noise > 0:
        noise = np.random.normal(0, time_constant_noise)
        tau = max(0.1, tau_base + noise)
    else:
        tau = tau_base

    # 1次遅延の計算で使用
    # ...

    # 実際に使用された値を記録（HDF5に保存される）
    entity["time_constant"] = tau
```

## 注意事項

1. **下限値**: 時定数は0.1ms以下にはならない（数値安定性のため）
2. **ランダム性**:
   - 個体差（`std`）: シミュレーション開始時に1回だけサンプリング
   - 時間変動（`noise`）: 各ステップでサンプリング
3. **lag無効時**: `enable_lag=False`の場合、ベース時定数が記録される（変動なし）

## 期待される統計値

### 個体差のみ（`std`）

```
Unique values: 1/N_steps  (全ステップで同じ値)
Std: 0.00 ms  (変動なし)
```

### 時間変動のみ（`noise`）

```
Unique values: ≈ N_steps  (ほぼ全て異なる値)
Std: ≈ noise  (設定したノイズレベル)
Mean: ≈ time_constant  (設定した平均値)
```

### 両方（`std` + `noise`）

```
Unique values: ≈ N_steps  (ほぼ全て異なる値)
Std: ≈ noise  (ノイズによる変動)
Mean: ≈ sampled_base  (サンプリングされたベース値)
```

## トラブルシューティング

### Q: 全ステップで同じ値になる

**A:** `time_constant_noise`が0の場合、個体差のみが反映されるため、全ステップで同じ値になります。これは正常動作です。

### Q: 値が0.1msに頻繁にクリップされる

**A:** `time_constant - 3*noise`が負になる可能性があります。ノイズレベルを下げるか、平均時定数を上げてください。

推奨: `noise < time_constant / 3`

### Q: HDF5に`time_constant`データセットがない

**A:** シナリオファイルでDataCollectorの接続を確認してください：

```python
mosaik.util.connect_many_to_one(
    self.world,
    [self.plant],
    self.collector,
    "time_constant",  # ← これが必要
)
```

## 関連ファイル

- [simulators/plant_simulator.py](../simulators/plant_simulator.py) - 時定数の記録実装
- [scenarios/hils_scenario.py](../scenarios/hils_scenario.py) - DataCollector設定
- [PLANT_VARIABILITY.md](PLANT_VARIABILITY.md) - ばらつき機能の詳細
