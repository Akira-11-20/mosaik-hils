# Formation Flying Parameter Sweep ガイド

## 概要

Formation Flying（編隊飛行）シミュレーションにおけるパラメータスイープを実行するためのガイドです。
既存の `run_parameter_sweep.py` を使用して、様々な制御パラメータや初期条件の影響を評価できます。

## クイックスタート

```bash
cd /home/akira/mosaik-hils/orbital_hils

# 1. Dry runで設定確認（実行しない）
uv run python scripts/sweeps/examples/sweep_formation_flying.py --dry-run

# 2. 実際にスイープ実行
uv run python scripts/sweeps/examples/sweep_formation_flying.py
```

## スイープ可能なパラメータ

### 制御系パラメータ
- `CONTROLLER_TYPE`: シナリオタイプ（"formation" 固定）
- `FORMATION_CONTROLLER_TYPE`: 制御方式
  - `"hcw"`: HCW (Hill-Clohessy-Wiltshire) 方程式ベースの編隊飛行制御
  - `"pd"`: 従来型PD制御
- `CONTROL_GAIN`: 制御ゲイン（例: 0.001, 0.01, 0.1, 1.0, 10.0）
- `MAX_THRUST`: 最大推力 [N]（例: 0.1, 0.5, 1.0, 5.0）

### 初期条件
- `FORMATION_OFFSET_X`: 初期相対位置（X軸）[m]（例: 50, 100, 200, 500）
- `FORMATION_OFFSET_Y`: 初期相対位置（Y軸）[m]（例: 0, 50, 100）
- `FORMATION_OFFSET_Z`: 初期相対位置（Z軸）[m]（例: 0, 50, 100）

### Plant動特性（アクチュエータ遅れ）
- `PLANT_TIME_CONSTANT`: 1次遅れ時定数 [s]（例: 0.0, 5.0, 10.0, 20.0, 50.0）
- `PLANT_NOISE_STD`: 計測ノイズ標準偏差（例: 0.0, 0.01, 0.05, 0.1）

### 逆補償（Inverse Compensation）
- `INVERSE_COMPENSATION`: 逆補償のON/OFF（True/False）
- `INVERSE_COMPENSATION_GAIN`: 補償ゲイン（例: 50.0, 100.0, 200.0）

### シミュレーション設定
- `SIMULATION_TIME`: シミュレーション時間 [s]（例: 1000.0, 2000.0, 5500.0）
- `TIME_RESOLUTION`: 時間解像度 [s]（例: 0.1, 1.0, 10.0）

## プリセットスイープ例

### Example 1: 制御方式の比較
HCW vs PD制御の性能比較

```python
sweep_params = {
    "CONTROLLER_TYPE": ["formation"],
    "FORMATION_CONTROLLER_TYPE": ["hcw", "pd"],
    "FORMATION_OFFSET_X": [50.0, 100.0, 200.0],
    "SIMULATION_TIME": [1000.0],
}
```

### Example 2: Plant遅れの影響
アクチュエータ遅れが編隊維持に与える影響

```python
sweep_params = {
    "CONTROLLER_TYPE": ["formation"],
    "FORMATION_CONTROLLER_TYPE": ["hcw"],
    "PLANT_TIME_CONSTANT": [0.0, 5.0, 10.0, 20.0, 50.0],
    "PLANT_NOISE_STD": [0.0, 0.01],
    "FORMATION_OFFSET_X": [100.0],
}
```

### Example 3: 逆補償の効果
Plant遅れに対する逆補償の有効性

```python
sweep_params = {
    "CONTROLLER_TYPE": ["formation"],
    "FORMATION_CONTROLLER_TYPE": ["hcw"],
    "PLANT_TIME_CONSTANT": [20.0],
    "INVERSE_COMPENSATION": [True, False],
    "INVERSE_COMPENSATION_GAIN": [50.0, 100.0, 200.0],
    "FORMATION_OFFSET_X": [100.0],
}
```

### Example 4: 制御ゲインのチューニング
最適な制御ゲインの探索

```python
sweep_params = {
    "CONTROLLER_TYPE": ["formation"],
    "FORMATION_CONTROLLER_TYPE": ["hcw"],
    "CONTROL_GAIN": [0.001, 0.01, 0.1, 1.0, 10.0],
    "FORMATION_OFFSET_X": [100.0],
    "PLANT_TIME_CONSTANT": [10.0],
}
```

## 結果の確認

### 1. 自動生成される出力

スイープ実行後、以下のファイルが自動生成されます：

```
results_sweep/YYYYMMDD-HHMMSS_sweep/
├── sweep_summary.txt                # スイープサマリー
├── 001_inv_comp=True_gain=50.0/     # 各シミュレーション結果
│   ├── hils_data.h5                 # データファイル
│   ├── formation_3d_orbits.png      # 3D軌道プロット
│   ├── formation_relative_position.png
│   ├── formation_relative_3d.png
│   └── ...
├── 002_inv_comp=True_gain=100.0/
└── ...
```

### 2. 比較可視化

成功したシミュレーションが2つ以上ある場合、自動的に比較プロットが生成されます：

```bash
# 手動で比較を実行する場合
uv run python scripts/analysis/compare_sweep_results.py results_sweep/YYYYMMDD-HHMMSS_sweep/
```

### 3. 個別の可視化

```bash
# Formation flying専用プロット
uv run python scripts/analysis/visualize_formation_flying.py results_sweep/.../001_.../hils_data.h5

# インタラクティブ3Dプロット
uv run python scripts/analysis/visualize_orbital_interactive.py results_sweep/.../001_.../hils_data.h5
```

## カスタムスイープの作成

### 方法1: 既存スクリプトを編集

`sweep_formation_flying.py` の `sweep_choice` を変更：

```python
sweep_choice = "gain_tuning"  # control_type, plant_lag, inverse_comp, etc.
```

### 方法2: 新規スクリプトを作成

```python
from scripts.sweeps.run_parameter_sweep import ParameterSweepConfig, run_sweep

# カスタムパラメータ定義
sweep_params = {
    "CONTROLLER_TYPE": ["formation"],
    "FORMATION_CONTROLLER_TYPE": ["hcw"],
    "FORMATION_OFFSET_X": [100.0, 200.0, 500.0],
    "FORMATION_OFFSET_Y": [0.0, 100.0],
    "CONTROL_GAIN": [0.1, 1.0],
    "PLANT_TIME_CONSTANT": [0.0, 10.0],
}

# スイープ設定
config = ParameterSweepConfig(
    sweep_params=sweep_params,
    base_env_file=".env",
    output_base_dir="results_sweep",
    description="Custom Formation Sweep",
)

# 実行
run_sweep(config, dry_run=False)
```

## 評価指標

Formation flyingの性能は以下の指標で評価できます：

### 相対位置誤差
- **収束時間**: 目標位置（通常[0,0,0]）に到達するまでの時間
- **定常偏差**: 定常状態での相対位置誤差
- **オーバーシュート**: 制御開始時の最大偏差

### 制御性能
- **推力使用量**: 積分推力（燃料消費の指標）
- **推力ピーク**: 最大推力値（ハードウェア制約の確認）
- **制御頻度**: ONとOFFの切り替え回数（チャタリング評価）

### HDF5データから取得できる変数
- `position_x/y/z`: Chaser/Targetの絶対位置 [m]
- `velocity_x/y/z`: Chaser/Targetの速度 [m/s]
- `measured_force_x/y/z`: 計測推力 [N]
- `thrust_command_x/y/z`: 制御指令 [N]
- `norm_force`, `norm_measured_force`: 推力ノルム [N]
- `altitude`, `semi_major_axis`, `eccentricity`: 軌道要素

## Tips

### 計算時間の削減
- `SIMULATION_TIME` を短くする（例: 1000秒）
- `TIME_RESOLUTION` を大きくする（例: 1.0秒）
- `MINIMAL_DATA_MODE=True` を設定（必要最小限のデータのみ記録）

### 詳細な解析
- `AUTO_VISUALIZE=False` にして可視化をスキップ（後でまとめて実行）
- `MOSAIK_DEBUG_MODE=False` でMosaikのデバッグ出力を抑制

### 並列実行
現状は逐次実行ですが、将来的にGNU parallelなどで並列化可能：
```bash
# 例（要実装）
cat sweep_configs.txt | parallel -j 4 "uv run python run_single_sim.py {}"
```

## トラブルシューティング

### エラー: "FormationFlyingScenario not found"
- `run_parameter_sweep.py` がFormationFlyingScenarioをimportしているか確認
- 273行目付近に `elif controller_type == "formation":` があるか確認

### エラー: "OUTPUT_DIR_OVERRIDE not set"
- 正常動作です。sweep実行時に自動設定されます

### 結果が見つからない
- `results_sweep/` ディレクトリを確認
- タイムスタンプ付きディレクトリ（YYYYMMDD-HHMMSS_sweep）を探す

## 参考

- [run_parameter_sweep.py](run_parameter_sweep.py) - コアスイープ機能
- [sweep_formation_flying.py](examples/sweep_formation_flying.py) - Formation flying用スイープ例
- [FormationFlyingScenario](../../scenarios/formation_flying_scenario.py) - Formation flyingシナリオ実装
- [orbital_parameters.py](../../config/orbital_parameters.py) - パラメータ定義

## 実行例

```bash
# 1. プロジェクトディレクトリへ移動
cd /home/akira/mosaik-hils/orbital_hils

# 2. Dry runで設定確認
uv run python scripts/sweeps/examples/sweep_formation_flying.py --dry-run

# 3. 実行（制御方式の比較）
uv run python scripts/sweeps/examples/sweep_formation_flying.py

# 4. 結果確認
ls -lh results_sweep/

# 5. 比較プロット確認
open results_sweep/YYYYMMDD-HHMMSS_sweep/comparison_*.png
```
