# Plant Time Constant Variability - ログ確認ガイド

## ✅ ばらつき機能は正常に動作しています

`plant_time_constant_std`を指定すると、以下のようなログが出力されます：

### ログ例

```bash
$ cd hils_simulation
$ uv run python scripts/sweeps/run_sweep.py
```

**期待されるログ出力：**

```
[PlantSim] Created ThrustStand_0 (ID: stand_01, τ=54.23ms (mean=50.0ms, std=10.0ms), lag=enabled)
```

### ログの見方

| 要素 | 説明 |
|------|------|
| `τ=54.23ms` | **実際にサンプリングされた時定数**（個体差反映後） |
| `mean=50.0ms` | 設定した平均時定数 |
| `std=10.0ms` | 設定した標準偏差 |
| `lag=enabled` | 1次遅延が有効 |

### ばらつきがない場合

```
[PlantSim] Created ThrustStand_0 (ID: stand_01, τ=50.0ms, lag=enabled)
```

→ `mean`や`std`の表示なし（シンプルな表示）

## 🔍 ログが出力されない場合のチェックリスト

### 1. シミュレーションが実行されているか確認

ログは **Plant作成時** に出力されます。以下のタイミングで表示：

```bash
# シミュレーション実行
uv run python scripts/sweeps/run_sweep.py

# 実行開始後、以下のような出力が続く：
🚀 Starting simulators...
[PlantSim] Created ThrustStand_0 (ID: stand_01, τ=XX.XXms ...)  ← ここ！
```

### 2. 正しいパラメータが設定されているか確認

**run_sweep.py の設定例：**

```python
DelayConfig(
    cmd_delay=0.0,
    sense_delay=0.0,
    plant_time_constant=50.0,
    plant_time_constant_std=10.0,  # ← これが設定されているか
    plant_enable_lag=True,           # ← これがTrueか
    use_inverse_comp=True
)
```

### 3. ログが流れて見逃していないか

多くのログが出力されるため、ばらつき関連のログだけをフィルタ：

```bash
uv run python scripts/sweeps/run_sweep.py 2>&1 | grep PlantSim
```

## 📊 実際の動作確認

### 簡単なテスト

```bash
cd hils_simulation

# Test 1: ばらつきなし
PLANT_TIME_CONSTANT=50.0 PLANT_TIME_CONSTANT_STD=0.0 uv run python main.py i 2>&1 | grep PlantSim

# 出力例:
# [PlantSim] Created ThrustStand_0 (ID: stand_01, τ=50.0ms, lag=enabled)

# Test 2: ばらつきあり
PLANT_TIME_CONSTANT=50.0 PLANT_TIME_CONSTANT_STD=10.0 uv run python main.py i 2>&1 | grep PlantSim

# 出力例:
# [PlantSim] Created ThrustStand_0 (ID: stand_01, τ=54.97ms (mean=50.0ms, std=10.0ms), lag=enabled)
```

## 💡 よくある質問

### Q: 毎回同じ時定数になる
A: 個体差（`std`）は **シミュレーション開始時** にサンプリングされます。同じ設定で複数回実行すると、ランダムシードが異なるため毎回違う値になります。

### Q: ログに`noise`が表示されない
A: `time_constant_noise`（時間変動ノイズ）は **各ステップ** で追加されるため、Plant作成時のログには表示されません。ノイズの効果はシミュレーション結果のHDF5データを見て確認してください。

### Q: ログに`std`が表示されるのに、いつも同じ値に見える
A: それぞれのシミュレーションで新しいPlantが作成され、毎回異なる値がサンプリングされています。複数の設定を連続実行すると、各ケースで異なる値が表示されます。

## 🎯 正常動作の確認方法

以下のコマンドで、4つの異なる時定数を確認できます：

```bash
cd hils_simulation

# 4回実行して、それぞれ異なる時定数が表示されることを確認
for i in {1..4}; do
  echo "=== Run $i ==="
  PLANT_TIME_CONSTANT_STD=10.0 uv run python main.py i 2>&1 | grep "PlantSim.*Created"
  sleep 1
done
```

**期待される出力：**
```
=== Run 1 ===
[PlantSim] Created ThrustStand_0 (ID: stand_01, τ=54.97ms (mean=50.0ms, std=10.0ms), lag=enabled)
=== Run 2 ===
[PlantSim] Created ThrustStand_0 (ID: stand_01, τ=48.62ms (mean=50.0ms, std=10.0ms), lag=enabled)
=== Run 3 ===
[PlantSim] Created ThrustStand_0 (ID: stand_01, τ=43.88ms (mean=50.0ms, std=10.0ms), lag=enabled)
=== Run 4 ===
[PlantSim] Created ThrustStand_0 (ID: stand_01, τ=61.25ms (mean=50.0ms, std=10.0ms), lag=enabled)
```

→ **毎回異なる値** になれば正常動作！

## 📁 結果の確認

シミュレーション後、実際に使われた時定数は `simulation_config.json` に記録されます：

```bash
cat results/YYYYMMDD-HHMMSS/simulation_config.json | grep -A 3 "plant"
```

```json
"plant": {
  "time_constant_s": 0.05497,    // 実際の値（秒単位）
  "time_constant_std_s": 0.01,   // 標準偏差（秒単位）
  "enable_lag": true
}
```
