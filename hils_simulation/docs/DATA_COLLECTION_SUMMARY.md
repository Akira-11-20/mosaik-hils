# HDF5 データ収集機能 - 実装サマリー

## 実装日時
2025-10-13

## 概要

HILS シミュレーションの全データを HDF5 形式で自動記録する機能を実装しました。コマンドのタイミング、遅延効果、システムの挙動を詳細に解析できるようになりました。

## 実装内容

### 1. 新規ファイル

#### `simulators/data_collector.py`
- **目的**: 全シミュレーションデータを収集し、HDF5 形式で保存
- **主要機能**:
  - 任意の属性を動的に受け入れる（`any_inputs: True`）
  - dict/numeric/string/None 型を自動判別
  - dict 型（例: command）は JSON 文字列として保存し、各要素も個別に記録
  - 1ms 毎にデータポイントを収集
  - シミュレーション終了時に HDF5 ファイルを自動保存

#### `analyze_data.py`
- **目的**: HDF5 データの解析と可視化
- **主要機能**:
  - HDF5 ファイルの読み込み
  - サマリー統計の表示
  - 8つのグラフを自動生成:
    1. Spacecraft Position（位置の時系列）
    2. Spacecraft Velocity（速度の時系列）
    3. Spacecraft Acceleration（加速度の時系列）
    4. Thrust Command vs Measured（指令推力 vs 測定推力）
    5. Position Error（制御誤差の時系列）
    6. Applied Force（作用力の時系列）
    7. Phase Plane（位相平面：位置-速度）
    8. Bridge Statistics（通信統計）

#### `DATA_COLLECTION.md`
- **目的**: データ収集機能の詳細ドキュメント
- **内容**:
  - 機能説明
  - 使用方法
  - HDF5 ファイル構造
  - カスタム解析例
  - パラメータ設定
  - 注意事項

### 2. 変更ファイル

#### `main_hils.py`
- **変更内容**:
  - DataCollector の sim_config への追加
  - DataCollector の起動と設定（line 160-176）
  - 全エンティティから DataCollector へのデータ接続
  - SIMULATION_TIME を 500ms → 200ms に短縮（テスト用）

```python
# DataCollector の起動
data_collector_sim = world.start("DataCollector", step_size=1)
collector = data_collector_sim.Collector(output_dir=str(run_dir))

# 全エンティティからデータを収集
mosaik.util.connect_many_to_one(world, [controller], collector, "command", "error")
mosaik.util.connect_many_to_one(world, [bridge_cmd], collector, "stats")
mosaik.util.connect_many_to_one(world, [plant], collector, "measured_thrust", "status")
mosaik.util.connect_many_to_one(world, [bridge_sense], collector, "stats")
mosaik.util.connect_many_to_one(
    world, [spacecraft], collector, "position", "velocity", "acceleration", "force"
)
```

#### `README.md`
- **変更内容**:
  - ディレクトリ構成の更新（data_collector.py, analyze_data.py 追加）
  - データ解析セクションの追加
  - 各シミュレーターの説明更新（command 形式の明記）
  - DataCollectorSimulator の追加
  - 今後の拡張リストの更新（✓チェック追加）

## 記録されるデータ

### Controller
- `command`: 制御コマンド（JSON/dict: `{thrust, duration}`）
  - `command_*_thrust`: 推力値 [N]
  - `command_*_duration`: 持続時間 [ms]
- `error`: 位置誤差 [m]

### Bridge (cmd/sense)
- `stats`: 通信統計（遅延、パケット数等）

### Plant
- `measured_thrust`: 測定推力 [N]
- `status`: 動作状態（"idle", "thrusting"）

### Env
- `position`: 位置 [m]
- `velocity`: 速度 [m/s]
- `acceleration`: 加速度 [m/s²]
- `force`: 作用力 [N]

### Time
- `time_ms`: シミュレーション時刻 [ms]
- `time_s`: 実時間 [秒]

## 使用方法

### 1. シミュレーション実行
```bash
cd hils_simulation
uv run python main_hils.py
```

結果: `results/YYYYMMDD-HHMMSS/hils_data.h5` に全データが保存される

### 2. データ解析
```bash
# 統計情報のみ表示
uv run python analyze_data.py results/20251013-183849/hils_data.h5 --no-plot

# グラフを画面表示
uv run python analyze_data.py results/20251013-183849/hils_data.h5

# グラフをファイル保存
uv run python analyze_data.py results/20251013-183849/hils_data.h5 --save-plots
```

### 3. カスタム解析
```python
import h5py

with h5py.File('results/20251013-183849/hils_data.h5', 'r') as f:
    time_s = f['data/time_s'][:]
    position = f['data/position_EnvSim-0.Spacecraft1DOF_0'][:]
    thrust = f['data/command_ControllerSim-0.PDController_0_thrust'][:]

    # 自由に解析・プロット
```

## 実装の工夫

### 1. 動的属性受け入れ
`meta` に `any_inputs: True` を設定することで、任意の属性を受け入れ可能にした。これにより、新しいシミュレーターや属性を追加しても DataCollector の変更が不要。

### 2. dict 型の自動展開
制御コマンドのような dict 型データは:
- JSON 文字列として保存（`command_*`: object型）
- 各要素も個別に記録（`command_*_thrust`: float64, `command_*_duration`: float64）

これにより、JSON 全体とプロット用の数値データの両方にアクセス可能。

### 3. None 値の処理
None 値は `float('nan')` に変換して HDF5 に保存。パケットロス等で値が欠損した場合も適切に記録される。

### 4. メタデータの記録
HDF5 ファイルに以下のメタデータを付与:
- `created_at`: 作成日時（ISO 8601形式）
- `num_samples`: データポイント数
- `time_resolution`: 時間解像度 [秒]

## テスト結果

### シミュレーション実行
- **期間**: 200ms（0.2秒）
- **データポイント数**: 200
- **ファイルサイズ**: 約 49KB
- **生成時間**: 約 4秒

### 記録されたデータセット（14種類）
```
- acceleration_EnvSim-0.Spacecraft1DOF_0: (200,) float64
- command_ControllerSim-0.PDController_0: (200,) object
- command_ControllerSim-0.PDController_0_duration: (200,) float64
- command_ControllerSim-0.PDController_0_thrust: (200,) float64
- error_ControllerSim-0.PDController_0: (200,) float64
- force_EnvSim-0.Spacecraft1DOF_0: (200,) float64
- measured_thrust_PlantSim-0.ThrustStand_0: (200,) float64
- position_EnvSim-0.Spacecraft1DOF_0: (200,) float64
- stats_BridgeSim-0.CommBridge_0: (200,) float64
- stats_BridgeSim-1.CommBridge_0: (200,) float64
- status_PlantSim-0.ThrustStand_0: (200,) object
- time_ms: (200,) float64
- time_s: (200,) float64
- velocity_EnvSim-0.Spacecraft1DOF_0: (200,) float64
```

### サマリー統計
```
📍 Position: Final = 0.000416 m
🚀 Velocity: Final = 0.012800 m/s
⚡ Acceleration: Mean = 0.064000 m/s²
🔥 Thrust Command: Mean = 20.000 N, Max = 20.000 N
📉 Position Error: RMS = 10.000000 m
```

### 可視化
8つのグラフを含む `analysis_plots.png` を自動生成（242KB）

## トラブルシューティング

### 問題1: "the destination attribute does not exist" エラー
**原因**: DataCollector の meta に "stats" 属性が定義されていなかった

**解決策**: `any_inputs: True` を設定して任意の属性を受け入れ可能に
```python
meta = {
    "type": "time-based",
    "models": {
        "Collector": {
            "public": True,
            "params": ["output_dir"],
            "attrs": [],  # 空にして
            "any_inputs": True,  # これを追加
        },
    },
}
```

### 問題2: シミュレーションが遅い
**原因**: debug=True により実行グラフ追跡のオーバーヘッド

**対応**: SIMULATION_TIME を 500ms → 200ms に短縮してテスト時間を削減

## 今後の拡張案

1. **データ圧縮**: HDF5 の gzip 圧縮を有効化してファイルサイズを削減
   ```python
   data_group.create_dataset(name=key, data=column, compression="gzip")
   ```

2. **リアルタイムストリーミング**: 長時間シミュレーションのためにメモリ上に全データを保持せず、逐次書き込み

3. **パケットロスイベント記録**: パケットがドロップされた時刻とシーケンス番号を別途記録

4. **Jupyter Notebook 例**: 対話的データ解析の例を提供

5. **NetCDF4 サポート**: 気象データ等で使われる NetCDF4 形式にも対応

## 関連ドキュメント

- [DATA_COLLECTION.md](DATA_COLLECTION.md): 詳細な使用方法
- [README.md](README.md): プロジェクト全体の説明
- [COMMAND_PACKAGE.md](COMMAND_PACKAGE.md): コマンドパッケージ化の説明

## まとめ

HDF5 データ収集機能により、HILS シミュレーションの全データを効率的に記録・解析できるようになりました。

**主な成果**:
- ✅ 全シミュレーションデータの自動記録（14種類のデータセット）
- ✅ HDF5 形式での効率的な保存（200ms で 49KB）
- ✅ 解析スクリプトによる自動可視化（8種類のグラフ）
- ✅ サマリー統計の自動生成
- ✅ カスタム解析のためのシンプルな API

**次のステップ**:
- コマンドタイミングの詳細解析
- 遅延効果の定量評価
- 補償機能の設計と実装
