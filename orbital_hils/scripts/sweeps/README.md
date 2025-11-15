# Orbital HILS Parameter Sweep Scripts

軌道HILSシミュレーションの汎用パラメータスイープツール。

## 概要

`.env`ファイルの任意のパラメータをリストで指定して、全ての組み合わせでシミュレーションを自動実行します。

## 基本的な使い方

### 1. 汎用スイープスクリプト

`run_parameter_sweep.py`を直接編集して使用：

```python
from scripts.sweeps.run_parameter_sweep import ParameterSweepConfig, run_sweep

# スイープするパラメータを定義
sweep_params = {
    "PLANT_TIME_CONSTANT": [10.0, 20.0, 50.0],
    "INVERSE_COMPENSATION_GAIN": [1.0, 2.0, 5.0],
}

# 設定を作成
config = ParameterSweepConfig(
    sweep_params=sweep_params,
    base_env_file=".env",
    output_base_dir="results_sweep",
    description="My Custom Sweep",
)

# 実行
run_sweep(config, dry_run=False)
```

### 2. 設定確認（Dry Run）

実行前に設定を確認：

```bash
cd orbital_hils
uv run python scripts/sweeps/run_parameter_sweep.py --dry-run
```

または、Pythonコード内で：

```python
run_sweep(config, dry_run=True)
```

### 3. スイープ実行

```bash
cd orbital_hils
uv run python scripts/sweeps/run_parameter_sweep.py
```

## 使用例

### Example 1: Inverse Compensation効果の検証

```bash
uv run python scripts/sweeps/examples/sweep_inverse_comp.py
```

スイープ内容：
- Plant time constant: 10, 20, 50, 100 ms
- Inverse compensation: ON/OFF
- Compensation gain: 1.0, 2.0, 5.0, 10.0

### Example 2: Plant動特性パラメータの影響

```bash
uv run python scripts/sweeps/examples/sweep_plant_dynamics.py
```

スイープ内容：
- Plant time constant: 1, 5, 10, 20, 50, 100 ms
- Noise std: 0.0, 0.01, 0.05, 0.1 N

### Example 3: 制御ゲインの最適化

```bash
uv run python scripts/sweeps/examples/sweep_control_gain.py
```

スイープ内容：
- Control gain: 0.0001, 0.001, 0.01, 0.1, 1.0

### Example 4: Hohmann遷移でのInverse Compensation効果

```bash
uv run python scripts/sweeps/examples/sweep_hohmann_inverse_comp.py
```

スイープ内容：
- Controller type: hohmann (固定)
- Plant time constant: 10, 50 s
- Inverse compensation: ON/OFF
- Hohmann target altitude: 500 km

## 重要な注意事項

### Controller Typeの選択

スイープシステムは`CONTROLLER_TYPE`パラメータに基づいて適切なシナリオを自動選択します：

- `CONTROLLER_TYPE=zero` → `OrbitalScenario` (自由軌道運動、デフォルト)
- `CONTROLLER_TYPE=pd` → `OrbitalScenario` (PD制御)
- `CONTROLLER_TYPE=hohmann` → `HohmannScenario` (ホーマン遷移)

**重要**: Hohmann遷移をテストする場合、必ず`CONTROLLER_TYPE: ["hohmann"]`をスイープパラメータに含めてください。

## カスタムスイープの作成

### ステップ1: パラメータを定義

.envファイルの任意のパラメータをスイープ可能：

```python
sweep_params = {
    # Plant parameters
    "PLANT_TIME_CONSTANT": [10.0, 20.0, 50.0],
    "PLANT_NOISE_STD": [0.0, 0.01, 0.05],

    # Control parameters
    "CONTROL_GAIN": [0.01, 0.1],

    # Compensation parameters
    "INVERSE_COMPENSATION": [True, False],
    "INVERSE_COMPENSATION_GAIN": [1.0, 2.0],

    # Simulation parameters
    "SIMULATION_TIME": [100.0],
    "TIME_RESOLUTION": [1.0],

    # Orbital parameters
    "ALTITUDE_KM": [400.0, 500.0],
    "SPACECRAFT_MASS": [500.0, 1000.0],

    # その他
    "MINIMAL_DATA_MODE": [True],  # データ量削減
    "AUTO_VISUALIZE": [False],    # 自動可視化無効
}
```

### ステップ2: 設定を作成

```python
config = ParameterSweepConfig(
    sweep_params=sweep_params,
    base_env_file=".env",              # ベースとなる.envファイル
    output_base_dir="results_sweep",   # 結果の出力先
    description="Custom Parameter Sweep",  # スイープの説明
)
```

### ステップ3: 実行

```python
run_sweep(config, dry_run=False)
```

## 出力ディレクトリ構造

```
results_sweep/
└── 20251116-003000_sweep/
    ├── 001_tau=10.0_noise=0.0/
    │   └── hils_data.h5
    ├── 002_tau=10.0_noise=0.01/
    │   └── hils_data.h5
    ├── 003_tau=20.0_noise=0.0/
    │   └── hils_data.h5
    ├── ...
    └── sweep_summary.txt
```

### sweep_summary.txt の内容

```
Orbital HILS Parameter Sweep
======================================================================

Timestamp: 20251116-003000
Total configurations: 12
Successful: 12
Failed: 0

Sweep parameters:
  PLANT_TIME_CONSTANT: [10.0, 20.0, 50.0]
  PLANT_NOISE_STD: [0.0, 0.01, 0.05, 0.1]

Results:

1. tau=10.0_noise=0.0
   Status: success
   Directory: /path/to/results_sweep/20251116-003000_sweep/001_tau=10.0_noise=0.0
...
```

## パラメータ名の短縮

スイープ結果のディレクトリ名を簡潔にするため、よく使うパラメータは自動的に短縮されます：

| 元のパラメータ名 | 短縮名 |
|------------------|--------|
| PLANT_TIME_CONSTANT | tau |
| PLANT_NOISE_STD | noise |
| INVERSE_COMPENSATION | inv_comp |
| INVERSE_COMPENSATION_GAIN | gain |
| CONTROL_GAIN | Kp |
| SIMULATION_TIME | T |
| TIME_RESOLUTION | dt |
| SPACECRAFT_MASS | mass |
| ALTITUDE_KM | alt |

カスタムの短縮名を追加する場合は、`ParameterSweepConfig._shorten_key()`を編集してください。

## Tips

### 1. 計算時間の削減

```python
sweep_params = {
    "SIMULATION_TIME": [100.0],      # 短いシミュレーション時間
    "TIME_RESOLUTION": [1.0],        # 粗い時間解像度
    "MINIMAL_DATA_MODE": [True],     # 最小限のデータのみ保存
    "AUTO_VISUALIZE": [False],       # 可視化を無効化
}
```

### 2. ベースライン設定の追加

特定の基準設定を含めたい場合：

```python
# スイープパラメータ
sweep_params = {
    "PLANT_TIME_CONSTANT": [10.0, 20.0, 50.0],
}

# ベースライン用の固定値
baseline_override = {
    "PLANT_TIME_CONSTANT": 0.0,  # 理想的な応答（遅れなし）
}

# 手動でベースライン設定を追加
configs = config.configs
configs.insert(0, baseline_override)
```

### 3. 並列実行

現在のスクリプトは順次実行ですが、将来的にmultiprocessingで並列化可能です。

## 結果の比較可視化

スイープ実行後、自動的に比較可視化が生成されます（成功したシミュレーションが2つ以上ある場合）。

### 自動生成されるプロット

スイープ完了後、`comparison/`ディレクトリに以下が生成されます：

1. **altitude_thrust_comparison.png** - 高度と推力の時系列比較
2. **3d_trajectory_comparison.png** - 3D軌道の比較
3. **trajectory_interactive.html** - インタラクティブ3D軌道（ブラウザで開く）
4. **phase_comparison.png** - フェーズプロット（高度-速度、軌道要素など）

### 手動で比較可視化を生成

```bash
cd orbital_hils

# 全結果を比較
uv run python scripts/analysis/compare_sweep_results.py results_sweep/20251116-010424_sweep

# 特定の結果のみ比較
uv run python scripts/analysis/compare_sweep_results.py results_sweep/20251116-010424_sweep --indices 1 2 3

# フェーズプロットも生成
uv run python scripts/analysis/compare_sweep_results.py results_sweep/20251116-010424_sweep --with-phases

# 出力先を指定
uv run python scripts/analysis/compare_sweep_results.py results_sweep/20251116-010424_sweep --output-dir my_comparison
```

### 可視化の詳細

#### altitude_thrust_comparison.png
- 上段: 各設定の高度の時間変化
- 下段: 各設定の推力の時間変化
- パラメータの影響を一目で比較可能

#### 3d_trajectory_comparison.png
- 3D空間での軌道の重ね合わせ
- 地球（半透明の球）を中心に表示
- 開始点をマーカーで表示

#### trajectory_interactive.html
- ブラウザで開いてインタラクティブに操作可能
- 回転、ズーム、個別の軌道の表示/非表示
- マウスオーバーで詳細情報を表示

#### phase_comparison.png
- 左上: 高度 vs 速度
- 右上: 高度 vs 軌道長半径
- 左下: 高度 vs 離心率
- 右下: 時間 vs 高度

## トラブルシューティング

### エラー: "No .env file found"

`.env`ファイルが`orbital_hils/`ディレクトリに存在することを確認してください。

### エラー: メモリ不足

`MINIMAL_DATA_MODE=True`を設定してデータ量を削減してください。

### スイープが途中で止まる

各設定の実行状況は`sweep_summary.txt`に記録されます。失敗した設定を確認してください。

### インタラクティブHTMLが生成されない

plotlyがインストールされていない可能性があります：
```bash
uv add plotly
```
