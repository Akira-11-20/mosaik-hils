# ホーマン遷移（Hohmann Transfer）実装ガイド

## 概要

このドキュメントでは、orbital HILSシミュレーションにおけるホーマン遷移の実装と使用方法を説明します。

ホーマン遷移は、2つの円軌道間を最小のエネルギー（ΔV）で移動するための2インパルス軌道遷移手法です。

## 実装済み機能

### 1. ホーマン遷移モデル (`models/hohmann_transfer.py`)

#### `HohmannTransferModel`
2つの円軌道間の最適な2インパルス遷移を計算・実行します。

**主な機能**:
- ΔV計算（第1バーン、第2バーン）
- 遷移時間の計算
- バーン時間の計算（有限推力を仮定）
- 推力スケジューリング（第1バーン → コースト → 第2バーン）

**使用例**:
```python
from models.hohmann_transfer import HohmannTransferModel

# ホーマン遷移モデルを作成（400km → 600km）
hohmann = HohmannTransferModel(
    mu=3.986004418e14,         # 地球の重力定数
    initial_altitude=400e3,    # 初期軌道高度 [m]
    target_altitude=600e3,     # 目標軌道高度 [m]
    radius_body=6378137.0,     # 地球半径 [m]
    spacecraft_mass=500.0,     # 衛星質量 [kg]
    max_thrust=10.0,           # 最大推力 [N]
)

# 遷移を開始
hohmann.start_transfer(current_time=10.0)

# 推力を計算
thrust = hohmann.calculate_thrust(current_time, position, velocity)

# 遷移状態を取得
status = hohmann.get_status()
print(f"Total ΔV: {status['total_delta_v']:.2f} m/s")
print(f"Transfer time: {status['transfer_time'] / 60:.2f} min")
```

#### `LambertTransferModel`
Lambert問題を解いて、任意の位置から任意の位置への遷移を計算します。

**主な機能**:
- `lamberthub.izzo2015`を使用した高速なLambert問題の解法
- 任意の初期位置・目標位置への遷移
- 飛行時間の指定

**使用例**:
```python
from models.hohmann_transfer import LambertTransferModel
import numpy as np

# Lambertモデルを作成
lambert = LambertTransferModel(
    mu=3.986004418e14,
    spacecraft_mass=500.0,
    max_thrust=10.0,
)

# 初期位置・目標位置
r1 = np.array([6778137.0, 0.0, 0.0])  # 400km軌道上の位置
r2 = np.array([0.0, 6978137.0, 0.0])  # 600km軌道上の位置（90度先）

# 飛行時間
tof = 2838.0  # 秒

# 現在の速度
v1_current = np.array([0.0, 7668.56, 0.0])

# Lambert問題を解く
v1_lambert, v2_lambert = lambert.solve_lambert(r1, r2, tof, v1_current)

print(f"Required ΔV: {np.linalg.norm(v1_lambert - v1_current):.2f} m/s")
```

### 2. 推力計算モデル (`models/thrust_model.py`)

#### `ThrustModel`（基底クラス）
デフォルトではゼロ推力を返す（自由軌道運動）。

#### `PDThrustModel`
PD制御による推力計算。

**制御則**: `F = Kp*(r_target - r) + Kd*(v_target - v)`

#### `HohmannThrustModel`
`HohmannTransferModel`をラップして、Mosaikシミュレーションで使用可能にしたモデル。

**使用例**:
```python
from models.thrust_model import HohmannThrustModel

model = HohmannThrustModel(
    mu=3.986004418e14,
    initial_altitude=400e3,
    target_altitude=600e3,
    radius_body=6378137.0,
    spacecraft_mass=500.0,
    max_thrust=10.0,
    start_time=100.0,  # 100秒後に遷移開始
)

# 推力計算（時刻を渡す）
thrust = model.calculate_thrust(position, velocity, time=current_time)
```

### 3. コントローラーシミュレーター統合

`OrbitalControllerSimulator` (`simulators/controller_simulator.py`)は、以下の制御タイプをサポート：

1. **"zero"** - ゼロ推力（自由軌道運動）
2. **"pd"** - PD制御
3. **"hohmann"** - ホーマン遷移制御

**使用例**:
```python
# シナリオ内でホーマン遷移コントローラーを作成
controller = controller_sim.OrbitalController(
    controller_type="hohmann",
    mu=mu,
    initial_altitude=400e3,
    target_altitude=600e3,
    radius_body=radius_body,
    spacecraft_mass=mass,
    max_thrust=10.0,
    start_time=100.0,
)
```

## 使用方法

### 方法1: デモスクリプトで動作確認

最も簡単な方法は、デモスクリプトを実行することです：

```bash
cd orbital_hils
uv run python examples/demo_hohmann_transfer.py
```

このスクリプトは、ホーマン遷移のパラメータ計算と推力計算のテストを実行します。

### 方法2: 完全なHILSシミュレーション

ホーマン遷移を含む完全なMosaik HILSシミュレーションを実行：

```bash
cd orbital_hils
uv run python examples/run_hohmann_simulation.py
```

このスクリプトは以下を実行します：
- 400km → 600km への遷移シミュレーション
- 約2.8時間のシミュレーション時間
- HDF5形式での結果保存
- 自動プロット生成

### 方法3: カスタムシナリオの作成

独自のシナリオを作成する場合：

```python
from scenarios.orbital_scenario import OrbitalScenario
from config.orbital_parameters import OrbitalSimulationConfig

class MyHohmannScenario(OrbitalScenario):
    def setup_entities(self):
        """エンティティのセットアップをカスタマイズ"""

        # ... (position, velocity計算)

        # ホーマン遷移コントローラーを作成
        controller_sim = self.world.start("OrbitalControllerSim", ...)
        self.controller = controller_sim.OrbitalController(
            controller_type="hohmann",
            mu=self.config.orbit.mu,
            initial_altitude=400e3,
            target_altitude=600e3,
            spacecraft_mass=self.config.spacecraft.mass,
            max_thrust=self.config.spacecraft.max_thrust,
            start_time=100.0,
        )

        # ... (plant, env, collectorのセットアップ)

# 実行
config = OrbitalSimulationConfig.create_iss_config()
scenario = MyHohmannScenario(config=config)
scenario.run()
```

## ホーマン遷移のパラメータ

### ΔV（速度変化量）

2つの円軌道間のホーマン遷移に必要なΔVは以下の式で計算されます：

**第1バーン（初期円軌道 → 遷移楕円軌道）**:
```
ΔV1 = v_transfer_periapsis - v_circular_1
```

**第2バーン（遷移楕円軌道 → 目標円軌道）**:
```
ΔV2 = v_circular_2 - v_transfer_apoapsis
```

**総ΔV**:
```
Total ΔV = |ΔV1| + |ΔV2|
```

### 遷移時間

遷移楕円軌道の半周期：
```
T_transfer = π * sqrt(a_transfer^3 / μ)
```

ここで、`a_transfer = (r1 + r2) / 2` は遷移軌道の半長軸。

### バーン時間

有限推力を仮定した場合のバーン時間：
```
t_burn = ΔV * m / F_max
```

## 計算例：400km → 600km 遷移

**初期条件**:
- 初期軌道高度: 400 km
- 目標軌道高度: 600 km
- 衛星質量: 500 kg
- 最大推力: 10 N

**計算結果**:
- ΔV1: +55.54 m/s
- ΔV2: +55.14 m/s
- 総ΔV: 110.69 m/s
- 遷移時間: 47.31 分
- バーン1時間: 46.29 分（推力10Nの場合）
- バーン2時間: 45.95 分（推力10Nの場合）

**タイムライン**（遷移開始時刻を100秒とした場合）:
```
t = 100.0s - 2877.2s  : 第1バーン（速度増加）
t = 2877.2s - 2938.5s : コーストフェーズ（楕円遷移軌道）
t = 2938.5s - 5695.0s : 第2バーン（円軌道化）
t > 5695.0s           : 遷移完了（目標軌道到達）
```

## 推力の大きさによる影響

| 最大推力 | バーン1時間 | バーン2時間 | 総バーン時間 | 備考 |
|---------|------------|------------|-------------|------|
| 1 N     | 7.71 時間  | 7.66 時間  | 15.37 時間  | 小推力スラスタ |
| 10 N    | 46.3 分    | 46.0 分    | 1.54 時間   | 中推力スラスタ |
| 100 N   | 4.63 分    | 4.60 分    | 9.23 分     | 大推力スラスタ |

**注意**: 推力が大きいほどバーン時間は短くなりますが、インパルス近似の仮定が崩れます。

## データ収集と可視化

### HDF5ファイル構造

シミュレーション結果は以下のデータを含みます：

```
hils_data.h5
├── time/
│   ├── time_s
│   └── time_ms
├── OrbitalEnvSim-0_OrbitalSpacecraft_0/
│   ├── position_x, position_y, position_z
│   ├── velocity_x, velocity_y, velocity_z
│   ├── altitude
│   ├── semi_major_axis
│   └── eccentricity
├── OrbitalControllerSim-0_OrbitalController_0/
│   └── thrust_command_x, thrust_command_y, thrust_command_z
└── OrbitalPlantSim-0_OrbitalThrustStand_0/
    └── measured_force_x, measured_force_y, measured_force_z
```

### 可視化

```bash
cd orbital_hils
uv run python scripts/analysis/visualize_orbital_results.py results_orbital/YYYYMMDD-HHMMSS/hils_data.h5
```

以下のプロットが生成されます：
- 軌道高度の時系列
- 軌道長半径の時系列
- 離心率の時系列
- 3D軌道プロット
- 推力プロファイル

## トラブルシューティング

### 問題1: バーン時間が非常に長い

**原因**: 最大推力が小さすぎる

**解決策**: `max_thrust`パラメータを大きくする（例: 1N → 10N）

### 問題2: 遷移が完了しない

**原因**: シミュレーション時間が不足

**解決策**: `simulation_time`を以下より大きく設定：
```
simulation_time > start_time + transfer_time + burn1_duration + burn2_duration
```

### 問題3: 軌道が不安定

**原因**: 時間解像度が粗すぎる

**解決策**: `time_resolution`を小さくする（例: 10s → 1s）

## 参考文献

- **Lambert問題ソルバー**: [lamberthub](https://github.com/jorgepiloto/lamberthub)
  - Izzo, D. (2015). "Revisiting Lambert's problem". *Celestial Mechanics and Dynamical Astronomy*, 121(1), 1-15.

- **ホーマン遷移**:
  - Hohmann, W. (1925). "Die Erreichbarkeit der Himmelskörper".
  - Curtis, H. D. (2013). "Orbital Mechanics for Engineering Students" (3rd ed.). Butterworth-Heinemann.

## 次のステップ

1. **燃料消費の計算**: ツィオルコフスキーの式を使用した燃料消費量の計算
2. **多段ホーマン遷移**: 3つ以上の軌道を経由する遷移
3. **軌道面変更**: 軌道傾斜角を変更する遷移
4. **低推力遷移**: イオンエンジンなどの低推力連続推進
5. **MPC制御**: モデル予測制御によるロバストな軌道追従

## 質問・フィードバック

実装に関する質問やフィードバックは、プロジェクトのIssueまたはPull Requestで受け付けています。
