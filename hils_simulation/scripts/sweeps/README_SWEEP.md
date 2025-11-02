# Plant Parameter Sweep - Quick Start Guide

このガイドでは、`run_sweep.py`を使ってplant time constantのばらつき実験を実行する方法を説明します。

## 🚀 クイックスタート

```bash
cd hils_simulation
uv run python scripts/sweeps/run_sweep.py
```

## 📋 利用可能なシナリオ

### Scenario 1: Basic Time Constant Sweep
時定数τの異なる値をテスト（ばらつきなし）

```python
configs = SCENARIO_1_BASIC
```

**テスト内容:**
- τ = 30ms (補償ゲイン: 4)
- τ = 90ms (補償ゲイン: 10)
- τ = 150ms (補償ゲイン: 16)
- τ = 300ms (補償ゲイン: 31)
- lag無効 (理想的)

**用途:** 基本的な時定数の影響を理解する

---

### Scenario 2: Individual Variability
個体差（製造ばらつき）の影響をテスト

```python
configs = SCENARIO_2_VARIABILITY
```

**テスト内容:**
- σ = 0ms (ばらつきなし)
- σ = 2ms (±6ms @ 3σ, 小さなばらつき)
- σ = 5ms (±15ms @ 3σ, 中程度のばらつき)
- σ = 10ms (±30ms @ 3σ, 大きなばらつき)

**用途:** 製造ばらつきが制御性能に与える影響を評価

---

### Scenario 3: Time-Varying Noise
時間変動するノイズ（外乱）の影響をテスト

```python
configs = SCENARIO_3_NOISE
```

**テスト内容:**
- σ = 0ms (ノイズなし)
- σ = 1ms (小さなノイズ)
- σ = 3ms (中程度のノイズ)
- σ = 5ms (大きなノイズ)

**用途:** 温度変化、振動などの環境外乱の影響を評価

---

### Scenario 4: Combined Effects
個体差と時間変動を組み合わせてテスト

```python
configs = SCENARIO_4_COMBINED
```

**テスト内容:**
1. ばらつきなし (ベースライン)
2. 個体差のみ (σ_std = 5ms)
3. 時間変動のみ (σ_noise = 2ms)
4. 両方を組み合わせ (σ_std = 5ms, σ_noise = 2ms)

**用途:** 複合的な不確実性の影響を評価

---

### Scenario 5: Realistic Scenario
通信遅延 + 逆補償 + ばらつきの組み合わせ

```python
configs = SCENARIO_5_REALISTIC
```

**テスト内容:**
1. HILS (遅延30ms) + ばらつき, 補償なし
2. HILS (遅延30ms) + ばらつき, 補償あり (ゲイン: 15)

**用途:** 実際のシステムに近い条件での性能評価

## ⚙️ シナリオの選択方法

`run_sweep.py`を編集して、使いたいシナリオのコメントを外してください：

```python
# ============================================================================
# Select scenario to run
# ============================================================================
# Uncomment one of the following:
# configs = SCENARIO_1_BASIC
configs = SCENARIO_2_VARIABILITY  # ← この行のコメントを外す
# configs = SCENARIO_3_NOISE
# configs = SCENARIO_4_COMBINED
# configs = SCENARIO_5_REALISTIC
```

## 📊 カスタムシナリオの作成

独自のテストケースを追加する場合：

```python
# Custom scenario example
SCENARIO_MY_TEST = [
    DelayConfig(
        cmd_delay=20.0,              # コマンド遅延 [ms]
        sense_delay=30.0,            # センシング遅延 [ms]
        plant_time_constant=50.0,   # 平均時定数 [ms]
        plant_time_constant_std=5.0,    # 個体差の標準偏差 [ms]
        plant_time_constant_noise=2.0,  # 時間変動ノイズの標準偏差 [ms]
        plant_enable_lag=True,       # 1次遅延を有効化
        use_inverse_comp=True,       # 逆補償を使用
        comp_gain=15.0               # 補償ゲイン
    ),
]

# シナリオを選択
configs = SCENARIO_MY_TEST
```

## 🎯 パラメータ選択のガイド

### 個体差 (`plant_time_constant_std`)

| 標準偏差 | 変動幅 (±3σ) | 用途 |
|---------|-------------|------|
| 0 ms    | なし         | ベースライン |
| 2 ms    | ±6 ms       | 小さな製造ばらつき |
| 5 ms    | ±15 ms      | 中程度の製造ばらつき |
| 10 ms   | ±30 ms      | 大きな製造ばらつき |

### 時間変動ノイズ (`plant_time_constant_noise`)

| 標準偏差 | 瞬間的な変動 | 用途 |
|---------|-------------|------|
| 0 ms    | なし         | ベースライン |
| 1 ms    | ±3 ms       | 小さな外乱 |
| 3 ms    | ±9 ms       | 中程度の外乱 |
| 5 ms    | ±15 ms      | 大きな外乱 |

## 📈 実行例

```bash
# Scenario 2を実行
$ cd hils_simulation
$ uv run python scripts/sweeps/run_sweep.py

======================================================================
Plant Time Constant Sweep
======================================================================
Total configurations: 4

Testing DelayConfig with plant parameters:

1. DelayConfig(cmd=0.0ms, sense=0.0ms, no_comp, plant_tau=50.0ms, plant_lag=True)
   Label: delay0ms_nocomp_tau50ms
   Delays: cmd=0.0ms, sense=0.0ms
   Plant τ: 50.0ms
   Plant lag enabled: True
   Inverse compensation: False

2. DelayConfig(cmd=0.0ms, sense=0.0ms, no_comp, plant_tau=50.0ms, plant_tau_std=2.0ms, plant_lag=True)
   Label: delay0ms_nocomp_tau50ms_std2ms
   Delays: cmd=0.0ms, sense=0.0ms
   Plant τ: 50.0ms
   Plant τ std: 2.0ms (±6.0ms @ 3σ)
   Plant lag enabled: True
   Inverse compensation: False

...

======================================================================
Proceed with simulations? [y/N]: y

======================================================================
Running simulations...
======================================================================

[PlantSim] Created ThrustStand_0 (ID: stand_01, τ=48.23ms (mean=50.0ms, std=2.0ms), lag=enabled)
...
```

## 📁 結果の確認

シミュレーション結果は `results/` ディレクトリに保存されます：

```
results/
├── YYYYMMDD-HHMMSS/
│   ├── hils_data.h5              # データファイル
│   ├── simulation_config.json    # 設定
│   └── *.png                     # グラフ
```

結果を可視化：

```bash
uv run python scripts/analysis/visualize_results.py results/YYYYMMDD-HHMMSS/hils_data.h5
```

## 💡 よくある使い方

### 製造ばらつきの影響を調査
```python
configs = SCENARIO_2_VARIABILITY
```
→ 個体差が0, 2, 5, 10msの場合の制御性能を比較

### 外乱ロバスト性を評価
```python
configs = SCENARIO_3_NOISE
```
→ 時間変動ノイズが0, 1, 3, 5msの場合の性能を比較

### 逆補償の効果を検証（ばらつきあり）
```python
configs = SCENARIO_5_REALISTIC
```
→ 遅延 + ばらつきがある状態で、逆補償の有無を比較

## 🔗 関連ドキュメント

- [PLANT_VARIABILITY.md](../../docs/PLANT_VARIABILITY.md) - ばらつき機能の詳細
- [README.md](../README.md) - スクリプト全体のドキュメント
- [CLAUDE.md](../../../.claude/CLAUDE.md) - 開発コマンド
