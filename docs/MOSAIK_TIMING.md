# Mosaikの時間とデータ交換の仕組み

このドキュメントは、Mosaikシミュレーションにおける時間管理とデータ交換のタイミングについてまとめたものです。

## 時間の単位と解像度

### time_resolution（時間解像度）
- **ワールド全体**の時間の最小単位
- `world = mosaik.World(time_resolution=1)` → **1ステップ = 1秒**
- すべてのシミュレーターがこの単位を共有

### step_size（ステップサイズ）
- **各シミュレーター**がtime_resolutionの何倍で実行されるか
- `step_size=10` → **10 × time_resolution間隔**で実行

### 例: 異なる実行頻度の設定
```python
world = mosaik.World(time_resolution=1)  # 1ステップ = 1秒

# 各シミュレーターの実行頻度
numerical_sim = world.start("NumericalSim", step_size=10)  # 10秒毎
hardware_sim = world.start("HardwareSim", step_size=10)    # 10秒毎
delay_sim = world.start("DelaySim", step_size=1)           # 1秒毎
```

**実行タイミング:**
- **DelaySimの実行**: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10秒...
- **NumericalSim/HardwareSimの実行**: 0, 10, 20, 30秒...

### world.run(until=N)の意味
- `until`パラメータは**ステップ数**を指定（時間ではない）
- `world.run(until=300)` → 300ステップまで実行
- 実際の時間 = 300ステップ × time_resolution

## 1ステップ内でのデータ交換タイミング

### データ交換は「最初に一気に」行われる

Mosaikの1ステップは以下の順序で実行されます:

```
1. 📥 入力データ収集 (step()実行前)
   ↓ 他のシミュレーターのキャッシュから一括取得

2. 🔄 step()メソッド実行
   ↓ 収集済みデータで計算実行

3. 📤 出力データ配信 (step()実行後)
   ↓ 計算結果を他のシミュレーターのバッファに送信
```

### 詳細な実行フロー

#### Phase 1: 入力データ収集
```python
# 依存関係の確認
await wait_for_dependencies(sim, lazy_stepping)

# 入力データの収集（キャッシュから）
input_data = get_input_data(world, sim)
max_advance = get_max_advance(world, sim, until)
```

#### Phase 2: シミュレーション実行
```python
# step()メソッドを実行
await step(world, sim, input_data, max_advance)
```

#### Phase 3: 出力データ配信
```python
# 出力データの収集と配信
await get_outputs(world, sim)
push_output_data(...)  # 他のシミュレーターのバッファに送信
```

### 重要な特徴

1. **離散的データ交換**: step()実行中は通信しない
2. **スナップショット方式**: 入力データは実行前に「一気に」収集
3. **キャッシュ機能**: 異なるstep_sizeに対応するため出力をキャッシュ
4. **因果関係保証**: 依存関係を満たすまで実行を待機

### 具体例

10秒ステップで実行されるHardwareSimの場合:

```
時刻10秒でHardwareSimが実行される:

📥 入力収集: DelaySimの5秒時点の出力データをキャッシュから取得
🔄 step()実行: 取得済みデータで10秒間のシミュレーション
📤 出力配信: HardwareSimの結果を他のシミュレーターのバッファに送信
```

## 各シミュレーターのtime_resolution

各シミュレーターは`init()`メソッドで独自の`time_resolution`を受け取れます:

```python
def init(self, sid, time_resolution=1.0, **sim_params):
    self.time_resolution = time_resolution
    return self.meta
```

### 使用目的

技術的には不要だが、以下の理由で有用:

1. **可読性**: 「このシミュレーターは1ms精度」が明確
2. **設定の統一**: パラメータファイルで一元管理
3. **チーム開発**: 時間単位の誤解を防ぐ
4. **内部計算の精度**: シミュレーター内部でサブステップ処理する際に使用

### 内部サブステップ処理の例

```python
def step(self, time, inputs, max_advance):
    # Mosaikから time=10 (10秒) を受け取る
    real_time = time * self.time_resolution  # 実時間に変換

    # 内部で細かく刻んで高精度計算
    dt = 0.001  # 1ms刻み
    for i in range(int(self.time_resolution / dt)):
        self.integrate_dynamics(dt)
        self.update_control_loop(real_time + i * dt)

    return time + 1
```

## データフローの実例

### 通信遅延がある場合のタイムライン

設定:
- `TIME_RESOLUTION = 1` (1秒)
- `COMMUNICATION_DELAY = 5` (5秒)
- `NumericalSim`: step_size=10
- `HardwareSim`: step_size=10
- `DelaySim`: step_size=1

```
時刻  | NumericalSim | DelaySim        | HardwareSim
------|--------------|-----------------|-------------
0秒   | データ生成   | バッファ処理    | センサー読取り
      | → DelaySim   |                 |
1秒   | (待機)       | バッファ処理    | (待機)
2秒   | (待機)       | バッファ処理    | (待機)
3秒   | (待機)       | バッファ処理    | (待機)
4秒   | (待機)       | バッファ処理    | (待機)
5秒   | (待機)       | バッファ処理    | (待機)
      |              | → 遅延データ準備|
10秒  | データ生成   | バッファ処理    | センサー読取り
      | → DelaySim   |                 | ← 遅延データ受信
```

## シミュレーション設定のベストプラクティス

### 1. time_resolutionの選択
- 必要な時間精度に合わせて設定
- 細かすぎると計算負荷が増大
- 目安: 制御周期の1/10程度

### 2. step_sizeの調整
- シミュレーターの更新頻度に合わせる
- 高頻度更新が必要なノード: 小さいstep_size
- 低頻度で十分なノード: 大きいstep_size

### 3. リアルタイムファクター
```python
rt_factor = 0     # 最高速で実行（非リアルタイム）
rt_factor = 1     # リアルタイム実行
rt_factor = 0.5   # 2倍速実行
```

### 4. debugモードの注意
```python
world = mosaik.World(sim_config, debug=True)  # 実行グラフ記録
```
- 実行グラフの追跡でオーバーヘッド
- デバッグ時のみ有効化
- 本番実行では`debug=False`推奨

## トラブルシューティング

### "Simulation too slow for real-time factor"警告
**原因:**
- time_resolutionが細かすぎる
- debugモードが有効
- 計算負荷が高すぎる

**解決策:**
- time_resolutionを緩くする
- rt_factorを下げる（または0にする）
- debugモードを無効化

### "next step time must be int"エラー
**原因:**
- step()メソッドがfloat型を返している
- step_sizeとtime_resolutionの混同

**解決策:**
```python
# 間違い
return time + self.step_size  # floatになる可能性

# 正しい
return time + 1  # intを返す
```

## 参考資料

- Mosaik公式ドキュメント: https://mosaik.readthedocs.io/
- 本プロジェクトのCLAUDE.md: 開発コマンドとアーキテクチャ概要
