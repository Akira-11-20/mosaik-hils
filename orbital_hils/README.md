# Orbital HILS プロジェクト

6自由度（6DOF）軌道力学シミュレーションにおける制御器・推力計測デバイスの統合と、Mosaikフレームワークを用いたフィードバック制御の実現。

## プロジェクト概要

**目的**: 軌道制御システムのHardware-in-the-Loop Simulation（HILS）

**特徴**:
- 6DOF軌道力学（二体問題）
- RK4法による高精度数値積分
- 制御ループ（Controller → Plant → Environment）
- 3軸推力制御（X, Y, Z独立）
- HDF5形式でのデータ記録

## アーキテクチャ

### データフロー

```
┌─────────────────────┐
│  OrbitalController  │  制御器（推力指令計算）
│  [ThrustModel]      │
└──────────┬──────────┘
           │ thrust_command_x/y/z [N]
           ↓
┌─────────────────────┐
│   OrbitalPlant      │  推力計測デバイス
│  [3軸 1次遅れ系]    │  - 時定数: τ
│                     │  - 計測ノイズ: σ
└──────────┬──────────┘
           │ measured_force_x/y/z [N]
           ↓
┌─────────────────────┐
│    OrbitalEnv       │  軌道力学エンジン
│  [RK4積分・二体問題]│  - r̈ = -μ/r³ × r + F/m
└──────────┬──────────┘
           │ position_x/y/z, velocity_x/y/z
           ↓
           └─────────────────────┐
                                 │ フィードバック
                                 ↓
                        (Controller へ戻る)

         すべてのコンポーネント
                  ↓
         ┌──────────────────┐
         │  DataCollector   │
         │   [HDF5記録]     │
         └──────────────────┘
```

### コンポーネント

1. **OrbitalController** ([simulators/controller_simulator.py](simulators/controller_simulator.py))
   - 入力: 位置・速度ベクトル
   - 出力: 推力指令ベクトル [N]
   - モデル: ThrustModel（現在はゼロ推力）

2. **OrbitalPlant** ([simulators/plant_simulator.py](simulators/plant_simulator.py))
   - 入力: 推力指令 [N]
   - 出力: 計測推力 [N]
   - 物理モデル: 1次遅れ系 + 計測ノイズ
   - パラメータ: τ（時定数）、σ（ノイズ標準偏差）

3. **OrbitalEnv** ([simulators/env_simulator.py](simulators/env_simulator.py))
   - 入力: 推力ベクトル [N]
   - 出力: 位置・速度・加速度、軌道要素
   - 物理モデル: 二体問題（r̈ = -μ/r³ × r + F/m）
   - 積分手法: RK4法

## ディレクトリ構造

```
orbital_hils/
├── simulators/                 # Mosaikシミュレーター
│   ├── controller_simulator.py # 軌道制御器
│   ├── plant_simulator.py      # 3軸推力計測デバイス
│   └── env_simulator.py        # 軌道力学エンジン
│
├── scenarios/                  # シミュレーションシナリオ
│   └── orbital_scenario.py     # メインシナリオ
│
├── models/                     # 制御モデル
│   └── thrust_model.py         # 推力計算モデル
│
├── config/                     # 設定管理
│   └── orbital_parameters.py   # 軌道パラメータ
│
├── scripts/                    # ツールスクリプト
│   └── analysis/               # データ分析
│       ├── visualize_orbital_results.py
│       └── visualize_orbital_interactive.py
│
├── utils/                      # ユーティリティ
│   └── data_collector.py       # HDF5データ記録
│
├── results_orbital/            # シミュレーション結果
│
├── main.py                     # エントリーポイント
└── README.md                   # このファイル
```

## 使用方法

### 基本実行

```bash
cd /home/akira/mosaik-hils/orbital_hils
uv run python main.py
```

### 結果の確認

```bash
# 最新の結果ディレクトリを確認
ls -ltr results_orbital/

# プロット生成
uv run python scripts/analysis/visualize_orbital_results.py results_orbital/YYYYMMDD-HHMMSS/hils_data.h5

# インタラクティブ3Dプロット（ブラウザで開く）
uv run python scripts/analysis/visualize_orbital_interactive.py results_orbital/YYYYMMDD-HHMMSS/hils_data.h5
```

## 設定

### プリセット軌道設定

[config/orbital_parameters.py](config/orbital_parameters.py) で定義:

- `CONFIG_ISS` - ISS軌道（高度408km、傾斜角51.64°）
- `CONFIG_LEO_400` - 低軌道400km円軌道
- `CONFIG_LEO_600` - 低軌道600km円軌道
- `CONFIG_GEO` - 静止軌道

### カスタム設定例

```python
from config.orbital_parameters import OrbitalSimulationConfig

# カスタム軌道を作成
config = OrbitalSimulationConfig.create_leo_config(
    altitude_km=500.0,
    eccentricity=0.01,
    inclination_deg=45.0,
    simulation_time=3600.0,  # 1時間
    time_resolution=1.0,     # 1秒刻み
)

# シミュレーション実行
from scenarios.orbital_scenario import OrbitalScenario
scenario = OrbitalScenario(config=config)
scenario.run()
```

## HDF5データ構造

結果ファイル: `results_orbital/YYYYMMDD-HHMMSS/hils_data.h5`

### グループ構造

```
hils_data.h5
├── time/
│   ├── time_s                  # 時刻 [s]
│   └── time_ms                 # 時刻 [ms]
│
├── OrbitalControllerSim-0_OrbitalController_0/
│   ├── thrust_command_x
│   ├── thrust_command_y
│   └── thrust_command_z
│
├── OrbitalPlantSim-0_OrbitalThrustStand_0/
│   ├── measured_force_x
│   ├── measured_force_y
│   └── measured_force_z
│
└── OrbitalEnvSim-0_OrbitalSpacecraft_0/
    ├── position_x, position_y, position_z
    ├── velocity_x, velocity_y, velocity_z
    ├── altitude
    ├── semi_major_axis
    └── eccentricity
```

### データ読み込み例

```python
import h5py
import numpy as np

with h5py.File('results_orbital/YYYYMMDD-HHMMSS/hils_data.h5', 'r') as f:
    # 時刻
    time = f['time']['time_s'][:]

    # 軌道データ
    env = f['OrbitalEnvSim-0_OrbitalSpacecraft_0']
    position_x = env['position_x'][:]
    position_y = env['position_y'][:]
    position_z = env['position_z'][:]

    # プロット
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(position_x, position_y, position_z)
    plt.show()
```

## 物理モデル詳細

### 二体問題の運動方程式

```
r̈ = -μ/r³ × r + F/m

r: 位置ベクトル [m]
μ: 中心天体の重力定数 [m³/s²] (地球: 3.986×10¹⁴)
F: 推力ベクトル [N]
m: 衛星質量 [kg]
```

### Plant 1次遅れ系モデル

```
τ * ḟ + f = f_cmd

τ: 時定数 [s]
f: 実推力 [N]
f_cmd: 推力指令 [N]

離散化:
f[n+1] = f[n] + (dt/τ) * (f_cmd[n] - f[n])

計測ノイズ:
f_measured = f + N(0, σ²)
```

## 将来の拡張

### 制御アルゴリズム

- [ ] PD制御実装
- [ ] LQR (Linear Quadratic Regulator)
- [ ] MPC (Model Predictive Control)
- [ ] 軌道遷移制御

### 通信遅延モデル

- [ ] BridgeSimulator統合
- [ ] 指令経路遅延
- [ ] 計測経路遅延
- [ ] 逆補償器（Inverse Compensator）

### 軌道摂動

- [ ] J2摂動（地球扁平効果）
- [ ] 大気抵抗
- [ ] 三体問題（月・太陽の影響）
- [ ] 太陽輻射圧

### 可視化強化

- [ ] リアルタイムプロット
- [ ] 軌道要素の時間変化
- [ ] 制御入力vs状態量の相関分析
- [ ] エネルギー効率解析

## 技術スタック

- **シミュレーションフレームワーク**: Mosaik 3.x
- **数値計算**: NumPy
- **データ記録**: HDF5 (h5py)
- **可視化**: Matplotlib, Plotly
- **言語**: Python 3.10+

## 既存hilsとの関係

このプロジェクトは `hils_simulation`（1DOF HILS）とは完全に分離した独立プロジェクトです。

### 共通点
- Mosaikフレームワーク使用
- 同じ設計思想（Simulator-Scenario構造）
- HDF5データ記録

### 相違点

| 項目 | hils_simulation | orbital_hils |
|------|-----------------|--------------|
| 自由度 | 1DOF | 6DOF |
| 物理モデル | F=ma (直線運動) | 二体問題 (軌道運動) |
| 制御目標 | 位置制御 | 軌道制御 |
| 推力方向 | 1軸 | 3軸 |
| 積分手法 | Euler | RK4 |

## ライセンス

（プロジェクトのライセンスを記載）

## 作成者

Akira

## 更新履歴

- **2025-01-15**: プロジェクト作成
  - 基本アーキテクチャ実装
  - 制御ループ接続
  - HDF5データ記録
  - ゼロ推力モード（自由軌道運動）
