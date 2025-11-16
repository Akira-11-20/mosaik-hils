# Hohmann Transfer Parameter Sweep ガイド

## 概要

Hohmann Transfer（ホーマン遷移）シミュレーションにおけるパラメータスイープを実行するためのガイドです。
既存の `run_parameter_sweep.py` を使用して、様々な制御パラメータや遷移条件の影響を評価できます。

**重要な機能**:
- **自動ベースライン生成**: `PLANT_TIME_CONSTANT=0.0`（理想的な応答）のベースラインシミュレーションが自動的に追加されます
- これにより、Inverse compensationの効果を正確に評価できます（ベースラインとの比較）

## クイックスタート

```bash
cd /home/akira/mosaik-hils/orbital_hils

# 1. Dry runで設定確認（実行しない）
uv run python scripts/sweeps/examples/sweep_hohmann_inverse_comp.py --dry-run

# 2. 実際にスイープ実行
uv run python scripts/sweeps/examples/sweep_hohmann_inverse_comp.py
```

## スイープ可能なパラメータ

### 遷移パラメータ
- `HOHMANN_INITIAL_ALTITUDE_KM`: 初期高度 [km]（例: 408.0 = ISS高度）
- `HOHMANN_TARGET_ALTITUDE_KM`: 目標高度 [km]（例: 500.0, 600.0, 800.0）
- `HOHMANN_START_TIME`: 遷移開始時刻 [s]（例: 100.0, 200.0）

### Plant動特性（アクチュエータ遅れ）
- `PLANT_TIME_CONSTANT`: 1次遅れ時定数 [s]（例: 0.0, 10.0, 20.0, 50.0, 100.0）
- `PLANT_NOISE_STD`: 計測ノイズ標準偏差（例: 0.0, 0.01, 0.05, 0.1）

### 逆補償（Inverse Compensation）
- `INVERSE_COMPENSATION`: 逆補償のON/OFF（True/False）
- `INVERSE_COMPENSATION_GAIN`: 補償ゲイン（例: 10.0, 50.0, 100.0）

### シミュレーション設定
- `SIMULATION_TIME`: シミュレーション時間 [s]（例: 3000.0 = 50分）
- `TIME_RESOLUTION`: 時間解像度 [s]（例: 0.1, 1.0, 10.0）

## プリセットスイープ例

### Example 1: Inverse Compensationの効果
Plant遅れに対する逆補償の有効性

```python
sweep_params = {
    "CONTROLLER_TYPE": ["hohmann"],
    "PLANT_TIME_CONSTANT": [10.0, 50.0],
    "INVERSE_COMPENSATION": [True, False],
    "INVERSE_COMPENSATION_GAIN": [10.0],
    "SIMULATION_TIME": [3000.0],
    "HOHMANN_INITIAL_ALTITUDE_KM": [408.0],
    "HOHMANN_TARGET_ALTITUDE_KM": [500.0],
}
```

### Example 2: 高度変化の影響
異なる目標高度での遷移性能

```python
sweep_params = {
    "CONTROLLER_TYPE": ["hohmann"],
    "HOHMANN_INITIAL_ALTITUDE_KM": [408.0],
    "HOHMANN_TARGET_ALTITUDE_KM": [450.0, 500.0, 600.0, 800.0],
    "PLANT_TIME_CONSTANT": [20.0],
    "SIMULATION_TIME": [5000.0],
}
```

### Example 3: Plant遅れの影響
アクチュエータ遅れが遷移に与える影響

```python
sweep_params = {
    "CONTROLLER_TYPE": ["hohmann"],
    "PLANT_TIME_CONSTANT": [0.0, 10.0, 20.0, 50.0, 100.0],
    "PLANT_NOISE_STD": [0.0, 0.01],
    "HOHMANN_INITIAL_ALTITUDE_KM": [408.0],
    "HOHMANN_TARGET_ALTITUDE_KM": [500.0],
}
```

## 結果の確認

### 1. 自動生成される出力

スイープ実行後、以下のファイルが自動生成されます：

```
results_sweep/YYYYMMDD-HHMMSS_sweep/
├── sweep_summary.txt                # スイープサマリー
├── 001_baseline_tau=0.0/            # ベースライン（理想応答）
│   ├── hils_data.h5                 # データファイル
│   ├── hohmann_altitude.png         # 高度プロット
│   ├── hohmann_phases.png           # フェーズ別プロット
│   └── ...
├── 002_tau=10.0_inv_comp=True/      # Inverse Comp ON
├── 003_tau=10.0_inv_comp=False/     # Inverse Comp OFF
└── ...
```

### 2. 比較可視化

成功したシミュレーションが2つ以上ある場合、自動的に比較プロットが生成されます：

**重要**: ベースライン（`PLANT_TIME_CONSTANT=0.0`）は太い線で強調表示されます。
これにより、Inverse compensationの効果を視覚的に評価できます。

**Hohmann transfer専用の比較**（自動生成）:
- `hohmann_altitude_thrust_comparison.png` - 高度と推力の比較
- `hohmann_orbital_elements_comparison.png` - 軌道要素（半長軸、離心率、エネルギー）の比較
- `hohmann_baseline_difference.png` - **ベースラインとの差分**（高度と推力）
- `hohmann_baseline_orbital_difference.png` - **ベースラインとの差分**（軌道要素）

```bash
# 手動でHohmann transfer専用比較を実行する場合
uv run python scripts/analysis/compare_hohmann_sweep.py results_sweep/YYYYMMDD-HHMMSS_sweep/

# 手動で一般的な比較を実行する場合
uv run python scripts/analysis/compare_sweep_results.py results_sweep/YYYYMMDD-HHMMSS_sweep/
```

### 3. 個別の可視化

```bash
# Hohmann transfer専用プロット
uv run python scripts/analysis/visualize_hohmann_phases.py results_sweep/.../001_.../hils_data.h5

# インタラクティブ3Dプロット
uv run python scripts/analysis/visualize_orbital_interactive.py results_sweep/.../001_.../hils_data.h5
```

## カスタムスイープの作成

### 方法1: 既存スクリプトを編集

`sweep_hohmann_inverse_comp.py` のパラメータを変更：

```python
sweep_params = {
    "CONTROLLER_TYPE": ["hohmann"],
    "PLANT_TIME_CONSTANT": [10.0, 50.0],  # 変更
    "INVERSE_COMPENSATION": [True, False],
    "INVERSE_COMPENSATION_GAIN": [10.0],
    "HOHMANN_INITIAL_ALTITUDE_KM": [408.0],
    "HOHMANN_TARGET_ALTITUDE_KM": [500.0],
}
```

### 方法2: 新規スクリプトを作成

```python
from scripts.sweeps.run_parameter_sweep import ParameterSweepConfig, run_sweep

# カスタムパラメータ定義
sweep_params = {
    "CONTROLLER_TYPE": ["hohmann"],
    "HOHMANN_INITIAL_ALTITUDE_KM": [408.0],
    "HOHMANN_TARGET_ALTITUDE_KM": [450.0, 500.0, 600.0],
    "PLANT_TIME_CONSTANT": [0.0, 20.0, 50.0],
    "INVERSE_COMPENSATION": [True, False],
    "INVERSE_COMPENSATION_GAIN": [50.0],
}

# スイープ設定
config = ParameterSweepConfig(
    sweep_params=sweep_params,
    base_env_file=".env",
    output_base_dir="results_sweep",
    description="Custom Hohmann Transfer Sweep",
    include_baseline=True,  # ベースライン自動追加
)

# 実行
run_sweep(config, dry_run=False)
```

## 評価指標

Hohmann transferの性能は以下の指標で評価できます：

### 遷移性能
- **遷移時間**: 目標高度到達までの時間
- **高度誤差**: 目標高度との差（定常偏差）
- **軌道要素誤差**: 半長軸、離心率の目標値からのずれ

### 制御性能
- **推力使用量**: 積分推力（燃料消費の指標）
  - 第1バーン（加速）
  - 第2バーン（減速/昇軌道化）
- **推力ピーク**: 最大推力値（ハードウェア制約の確認）
- **バーンタイミング精度**: 計画バーン時刻との差

### HDF5データから取得できる変数
- `altitude`: 高度 [m]
- `semi_major_axis`: 軌道長半径 [m]
- `eccentricity`: 離心率 [-]
- `specific_energy`: 軌道エネルギー [J/kg]
- `position_x/y/z`: 位置 [m]
- `velocity_x/y/z`: 速度 [m/s]
- `norm_force`: 推力ノルム [N]（環境への入力）
- `norm_measured_force`: 計測推力 [N]（Plant出力）

## Tips

### 計算時間の削減
- `SIMULATION_TIME` を短くする（例: 2000秒）
- `TIME_RESOLUTION` を大きくする（例: 1.0秒、10.0秒）
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

### エラー: "HohmannTransferScenario not found"
- `run_parameter_sweep.py` がHohmannTransferScenarioをimportしているか確認
- 正しいCONTROLLER_TYPEが設定されているか確認

### エラー: "OUTPUT_DIR_OVERRIDE not set"
- 正常動作です。sweep実行時に自動設定されます

### 結果が見つからない
- `results_sweep/` ディレクトリを確認
- タイムスタンプ付きディレクトリ（YYYYMMDD-HHMMSS_sweep）を探す

### 遷移が完了しない
- `SIMULATION_TIME` を長くする（5000秒以上推奨）
- 目標高度と初期高度の差が大きい場合は、さらに長く

## 参考

- [run_parameter_sweep.py](run_parameter_sweep.py) - コアスイープ機能
- [sweep_hohmann_inverse_comp.py](examples/sweep_hohmann_inverse_comp.py) - Hohmann transfer用スイープ例
- [HohmannTransferScenario](../../scenarios/hohmann_transfer_scenario.py) - Hohmann transferシナリオ実装
- [orbital_parameters.py](../../config/orbital_parameters.py) - パラメータ定義

## 実行例

```bash
# 1. プロジェクトディレクトリへ移動
cd /home/akira/mosaik-hils/orbital_hils

# 2. Dry runで設定確認
uv run python scripts/sweeps/examples/sweep_hohmann_inverse_comp.py --dry-run

# 3. 実行（Inverse Compensationの効果検証）
uv run python scripts/sweeps/examples/sweep_hohmann_inverse_comp.py

# 4. 結果確認
ls -lh results_sweep/

# 5. 比較プロット確認
open results_sweep/YYYYMMDD-HHMMSS_sweep/comparison/*.png
```

## ベースライン比較の読み方

### ベースラインとの差分プロット

**`hohmann_baseline_difference.png`**:
- 上段: Δ Altitude（高度差）
  - 正の値 = ベースラインより高い高度
  - 負の値 = ベースラインより低い高度
  - Plant遅れや補償の影響で遷移軌道がずれる様子を確認
- 下段: Δ Thrust（推力差）
  - Inverse compensationがONの場合、ベースラインに近い推力プロファイルになる
  - OFFの場合、Plant遅れの影響で推力が遅れる

**`hohmann_baseline_orbital_difference.png`**:
- Semi-major Axis差: 軌道長半径のずれ → エネルギー投入の誤差
- Eccentricity差: 離心率のずれ → 円軌道からのずれ
- Specific Energy差: 軌道エネルギーのずれ → 燃料消費効率

### 評価のポイント
1. **Inverse compensation OFF**の場合、差分が大きい → Plant遅れの影響が顕著
2. **Inverse compensation ON**の場合、差分が小さい → 補償が有効に機能
3. ベースライン（太線）は理想応答なので、これに近いほど良い制御
