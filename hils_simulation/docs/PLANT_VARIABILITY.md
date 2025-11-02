# Plant Time Constant Variability

プラント（アクチュエーター）の1次遅延時定数にばらつきを与える方法について説明します。

## 概要

実際のアクチュエーターには、以下のような不確実性があります：
- **個体差**: 製造ばらつきによる時定数の違い
- **時間変動**: 温度変化、劣化、外乱による時定数の変動

これらをシミュレーションで再現するため、2種類のばらつきパラメータを実装しています。

## ばらつきの種類

### 1. 時定数の個体差 (`PLANT_TIME_CONSTANT_STD`)

各Plantインスタンスが**異なる固定の時定数**を持つ（ガウス分布）。

```python
# .env 設定例
PLANT_TIME_CONSTANT=50.0        # 平均時定数 [ms]
PLANT_TIME_CONSTANT_STD=5.0     # 標準偏差 [ms]
```

**効果**:
- 各Plantインスタンス作成時に、時定数がガウス分布からサンプリングされる
- サンプリングされた時定数は、そのPlantの実行中は一定
- 複数のPlantを作成した場合、それぞれ異なる時定数を持つ

**使用例**:
```python
# 個体差のみを与える場合
DelayConfig(
    cmd_delay=20.0,
    sense_delay=30.0,
    plant_time_constant=50.0,
    plant_time_constant_std=5.0,  # 個体差: σ=5ms
    use_inverse_comp=True
)
```

**数式**:
```
τ_i ~ N(τ_mean, σ²)
τ_i = max(0.1, τ_i)  # 負の値を防ぐ
```

### 2. 時定数の時間変動 (`PLANT_TIME_CONSTANT_NOISE`)

時定数が**各ステップでランダムに変化**する（ホワイトノイズ）。

```python
# .env 設定例
PLANT_TIME_CONSTANT=50.0          # 平均時定数 [ms]
PLANT_TIME_CONSTANT_NOISE=2.0     # 時間変動ノイズ [ms]
```

**効果**:
- シミュレーションの各ステップで、時定数にホワイトノイズが追加される
- 時間変動する外乱（温度変化、振動など）を模擬
- より現実的なアクチュエーター挙動を再現

**使用例**:
```python
# 時間変動ノイズのみを与える場合
DelayConfig(
    cmd_delay=20.0,
    sense_delay=30.0,
    plant_time_constant=50.0,
    plant_time_constant_noise=2.0,  # 時間変動: σ=2ms
    use_inverse_comp=True
)
```

**数式**:
```
τ(t) = τ_base + N(0, σ_noise²)
τ(t) = max(0.1, τ(t))  # 負の値を防ぐ
```

### 3. 両方を組み合わせる

個体差と時間変動を同時に与えることも可能。

```python
# .env 設定例
PLANT_TIME_CONSTANT=50.0          # 平均時定数 [ms]
PLANT_TIME_CONSTANT_STD=5.0       # 個体差: σ=5ms
PLANT_TIME_CONSTANT_NOISE=2.0     # 時間変動: σ=2ms
```

**使用例**:
```python
# 個体差 + 時間変動
DelayConfig(
    cmd_delay=20.0,
    sense_delay=30.0,
    plant_time_constant=50.0,
    plant_time_constant_std=5.0,      # 個体差
    plant_time_constant_noise=2.0,    # 時間変動
    use_inverse_comp=True
)
```

**数式**:
```
# 初期化時（個体差）
τ_base ~ N(τ_mean, σ_std²)

# 各ステップ（時間変動）
τ(t) = τ_base + N(0, σ_noise²)
τ(t) = max(0.1, τ(t))
```

## 実験例

### 例1: 個体差の影響を調査

```python
configs = [
    # ばらつきなし（ベースライン）
    DelayConfig(0.0, 0.0, plant_time_constant=50.0, plant_enable_lag=True),

    # 小さな個体差
    DelayConfig(0.0, 0.0, plant_time_constant=50.0, plant_time_constant_std=2.0, plant_enable_lag=True),

    # 中程度の個体差
    DelayConfig(0.0, 0.0, plant_time_constant=50.0, plant_time_constant_std=5.0, plant_enable_lag=True),

    # 大きな個体差
    DelayConfig(0.0, 0.0, plant_time_constant=50.0, plant_time_constant_std=10.0, plant_enable_lag=True),
]
```

### 例2: 時間変動の影響を調査

```python
configs = [
    # ばらつきなし（ベースライン）
    DelayConfig(0.0, 0.0, plant_time_constant=50.0, plant_enable_lag=True),

    # 小さな時間変動
    DelayConfig(0.0, 0.0, plant_time_constant=50.0, plant_time_constant_noise=1.0, plant_enable_lag=True),

    # 中程度の時間変動
    DelayConfig(0.0, 0.0, plant_time_constant=50.0, plant_time_constant_noise=3.0, plant_enable_lag=True),

    # 大きな時間変動
    DelayConfig(0.0, 0.0, plant_time_constant=50.0, plant_time_constant_noise=5.0, plant_enable_lag=True),
]
```

### 例3: 逆補償の効果検証（ばらつきあり）

```python
configs = [
    # HILS（逆補償なし）+ ばらつき
    DelayConfig(
        cmd_delay=30.0,
        sense_delay=30.0,
        plant_time_constant=50.0,
        plant_time_constant_std=5.0,
        plant_time_constant_noise=2.0,
        use_inverse_comp=False
    ),

    # 逆補償あり + ばらつき
    DelayConfig(
        cmd_delay=30.0,
        sense_delay=30.0,
        plant_time_constant=50.0,
        plant_time_constant_std=5.0,
        plant_time_constant_noise=2.0,
        use_inverse_comp=True,
        comp_gain=15.0
    ),
]
```

## スクリプトの実行

### 基本的な使い方

```bash
cd hils_simulation

# 個体差のみ
PLANT_TIME_CONSTANT=50.0 PLANT_TIME_CONSTANT_STD=5.0 uv run python main.py i

# 時間変動のみ
PLANT_TIME_CONSTANT=50.0 PLANT_TIME_CONSTANT_NOISE=2.0 uv run python main.py i

# 両方
PLANT_TIME_CONSTANT=50.0 PLANT_TIME_CONSTANT_STD=5.0 PLANT_TIME_CONSTANT_NOISE=2.0 uv run python main.py i
```

### パラメータスイープ

```bash
cd hils_simulation
uv run python scripts/sweeps/test_plant_sweep.py
```

## パラメータ選択のガイドライン

### 個体差 (`PLANT_TIME_CONSTANT_STD`)

| 標準偏差 | 時定数の変動幅 (±3σ) | 用途 |
|---------|-------------------|------|
| 0.0 ms  | なし               | ベースライン |
| 2.0 ms  | ±6 ms (±12%)      | 小さな製造ばらつき |
| 5.0 ms  | ±15 ms (±30%)     | 中程度の製造ばらつき |
| 10.0 ms | ±30 ms (±60%)     | 大きな製造ばらつき |

### 時間変動 (`PLANT_TIME_CONSTANT_NOISE`)

| 標準偏差 | 瞬間的な変動幅 (±3σ) | 用途 |
|---------|---------------------|------|
| 0.0 ms  | なし                 | ベースライン |
| 1.0 ms  | ±3 ms (±6%)         | 小さな外乱 |
| 2.0 ms  | ±6 ms (±12%)        | 中程度の外乱 |
| 5.0 ms  | ±15 ms (±30%)       | 大きな外乱 |

## 結果の確認

シミュレーション実行時に、Plantの時定数が表示されます：

```
[PlantSim] Created ThrustStand_0 (ID: stand_01,
  τ=48.23ms (mean=50.0ms, std=5.0ms, noise=2.0ms), lag=enabled)
```

HDF5ファイルには、実際に使用された時定数が記録されます（個体差のベース値）。

## 注意事項

1. **時定数の下限**: 数値安定性のため、時定数は0.1ms以下にはならない
2. **ランダムシード**: 再現性のため、必要に応じて`np.random.seed()`を設定
3. **計算コスト**: ノイズを追加しても計算コストはほぼ変わらない
4. **データ保存**: 時間変動ノイズの瞬時値はHDF5に記録されない（ベース値のみ）

## 関連ファイル

- [config/parameters.py](../config/parameters.py) - パラメータ定義
- [simulators/plant_simulator.py](../simulators/plant_simulator.py) - 実装
- [scenarios/hils_scenario.py](../scenarios/hils_scenario.py) - HILS シナリオ
- [scenarios/inverse_comp_scenario.py](../scenarios/inverse_comp_scenario.py) - 逆補償シナリオ
- [scripts/sweeps/run_delay_sweep_advanced.py](../scripts/sweeps/run_delay_sweep_advanced.py) - パラメータスイープ
