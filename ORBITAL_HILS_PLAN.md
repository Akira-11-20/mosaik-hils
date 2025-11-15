# Orbital HILS プロジェクト実装計画書

## プロジェクト概要

**プロジェクト名**: orbital_hils
**目的**: 6自由度（6DOF）軌道力学シミュレーションにおける制御器・推力計測デバイスの統合と、Mosaikフレームワークを用いたフィードバック制御の実現
**独立性**: 既存の`hils_simulation`（1DOF HILS）とは完全に分離した独立プロジェクトとして開発

---

## プロジェクト構造

```
mosaik-hils/
├── hils_simulation/                # 既存（1DOF HILS）- 変更なし
│   ├── simulators/
│   ├── scenarios/
│   ├── config/
│   └── ...
│
└── orbital_hils/                   # 新規プロジェクト
    ├── simulators/                 # Mosaikシミュレーター
    │   ├── __init__.py
    │   ├── controller_simulator.py # 軌道制御器
    │   ├── plant_simulator.py      # 3軸推力計測デバイス
    │   └── env_simulator.py        # 軌道力学エンジン
    │
    ├── scenarios/                  # シミュレーションシナリオ
    │   ├── __init__.py
    │   └── orbital_scenario.py     # メインシナリオ
    │
    ├── models/                     # 制御モデル
    │   ├── __init__.py
    │   └── thrust_model.py         # 推力計算モデル
    │
    ├── config/                     # 設定管理
    │   ├── __init__.py
    │   └── orbital_parameters.py   # 軌道パラメータ
    │
    ├── scripts/                    # ツールスクリプト
    │   └── analysis/               # データ分析
    │       ├── visualize_orbital_results.py
    │       └── visualize_orbital_interactive.py
    │
    ├── utils/                      # ユーティリティ
    │   ├── __init__.py
    │   ├── data_collector.py       # HDF5データ記録
    │   └── plot_utils.py           # プロットユーティリティ
    │
    ├── results_orbital/            # シミュレーション結果
    │
    ├── main.py                     # エントリーポイント
    └── README.md                   # プロジェクトドキュメント
```

---

## アーキテクチャ設計

### データフロー図

```
┌─────────────────────┐
│  OrbitalController  │  制御器
│  (推力指令計算)     │
└──────────┬──────────┘
           │ thrust_command_x/y/z
           ↓
┌─────────────────────┐
│   OrbitalPlant      │  推力計測デバイス
│  (3軸推力計測)      │
└──────────┬──────────┘
           │ measured_force_x/y/z
           ↓
┌─────────────────────┐
│    OrbitalEnv       │  軌道力学エンジン
│  (RK4積分・二体問題)│
└──────────┬──────────┘
           │ position_x/y/z, velocity_x/y/z
           ↓
           └─────────────────────┐
                                 │ フィードバック
                                 ↓
                        (Controller へ戻る)
```

### 詳細データフロー

1. **OrbitalEnv** → 状態量 → **OrbitalController**
   - `position_x`, `position_y`, `position_z` [m]
   - `velocity_x`, `velocity_y`, `velocity_z` [m/s]

2. **OrbitalController** → 推力指令 → **OrbitalPlant**
   - `thrust_command_x`, `thrust_command_y`, `thrust_command_z` [N]
   - ThrustModel内部で計算

3. **OrbitalPlant** → 計測推力 → **OrbitalEnv**
   - `measured_force_x`, `measured_force_y`, `measured_force_z` [N]
   - 1次遅れ系 + 計測ノイズ

4. **OrbitalEnv** → RK4積分 → 新しい状態
   - 二体問題の運動方程式: `r̈ = -μ/r³ × r + F/m`

---

## 実装タスク

### フェーズ1: プロジェクトセットアップ

#### タスク1.1: ディレクトリ作成
```bash
cd /home/akira/mosaik-hils
mkdir -p orbital_hils/{simulators,scenarios,models,config,scripts/analysis,utils,results_orbital}
```



---

### フェーズ2: 既存コンポーネントの移植

#### タスク2.1: OrbitalEnvSimulator移植
**移動元**: `hils_simulation/simulators/orbital_env_simulator.py`
**移動先**: `orbital_hils/simulators/env_simulator.py`
**変更**: importパスの調整のみ

#### タスク2.2: OrbitalParameters移植
**移動元**: `hils_simulation/config/orbital_parameters.py`
**移動先**: `orbital_hils/config/orbital_parameters.py`
**変更**: なし

#### タスク2.3: 可視化スクリプト移植
**移動元**:
- `hils_simulation/scripts/analysis/visualize_orbital_results.py`
- `hils_simulation/scripts/analysis/visualize_orbital_interactive.py`

**移動先**: `orbital_hils/scripts/analysis/`
**変更**: importパスの調整

#### タスク2.4: DataCollector移植
**移動元**: `hils_simulation/simulators/data_collector.py`
**移動先**: `orbital_hils/utils/data_collector.py`
**変更**: なし（汎用モジュール）

---

### フェーズ3: 新規コンポーネント実装

#### タスク3.1: ThrustModel作成
**ファイル**: `orbital_hils/models/thrust_model.py`

**クラス定義**:
```python
class ThrustModel:
    """軌道制御用推力計算モデル（プレースホルダー）"""

    def __init__(self, target_position=None, control_gain=1.0):
        self.target = target_position
        self.gain = control_gain

    def calculate_thrust(self, position, velocity):
        """
        Args:
            position: np.array([x, y, z]) [m]
            velocity: np.array([vx, vy, vz]) [m/s]

        Returns:
            thrust: np.array([Fx, Fy, Fz]) [N]
        """
        # TODO: 将来実装（PD制御、LQR、MPCなど）
        return np.zeros(3)  # 現在はゼロ推力
```

---

#### タスク3.2: OrbitalControllerSimulator作成
**ファイル**: `orbital_hils/simulators/controller_simulator.py`

**Mosaik meta定義**:
```python
meta = {
    "type": "time-based",
    "models": {
        "OrbitalController": {
            "public": True,
            "params": ["target_position", "control_gain"],
            "attrs": [
                # 入力（from OrbitalEnv）
                "position_x", "position_y", "position_z",
                "velocity_x", "velocity_y", "velocity_z",
                # 出力（to OrbitalPlant）
                "thrust_command_x", "thrust_command_y", "thrust_command_z",
            ],
        },
    },
}
```

**主要機能**:
- ThrustModelを使用した推力計算
- 状態量からの制御指令生成
- データ出力（HDF5記録用）

---

#### タスク3.3: OrbitalPlantSimulator作成
**ファイル**: `orbital_hils/simulators/plant_simulator.py`

**Mosaik meta定義**:
```python
meta = {
    "type": "time-based",
    "models": {
        "OrbitalThrustStand": {
            "public": True,
            "params": ["time_constant", "noise_std"],
            "attrs": [
                # 入力（from OrbitalController）
                "command_x", "command_y", "command_z",
                # 出力（to OrbitalEnv）
                "measured_force_x", "measured_force_y", "measured_force_z",
            ],
        },
    },
}
```

**物理モデル**:
- 1次遅れ系: `τ * ḟ + f = f_cmd`
- 計測ノイズ: `f_measured = f + N(0, σ²)`
- 3軸独立動作

---

#### タスク3.4: OrbitalScenario作成
**ファイル**: `orbital_hils/scenarios/orbital_scenario.py`

**主要メソッド**:

**setup_entities()**:
```python
def setup_entities(self):
    # Controller
    controller_sim = self.world.start("OrbitalControllerSim", ...)
    self.controller = controller_sim.OrbitalController(
        target_position=[0, 0, 0],
        control_gain=1.0
    )

    # Plant
    plant_sim = self.world.start("OrbitalPlantSim", ...)
    self.plant = plant_sim.OrbitalThrustStand(
        time_constant=10.0,
        noise_std=0.01
    )

    # Environment
    env_sim = self.world.start("OrbitalEnvSim", ...)
    self.spacecraft = env_sim.OrbitalSpacecraft(...)

    # Data Collector
    collector_sim = self.world.start("DataCollector", ...)
    self.collector = collector_sim.Collector(...)
```

**connect_entities()**:
```python
def connect_entities(self):
    # フィードバック: Env → Controller
    self.world.connect(
        self.spacecraft, self.controller,
        ("position_x", "position_x"),
        ("position_y", "position_y"),
        ("position_z", "position_z"),
        ("velocity_x", "velocity_x"),
        ("velocity_y", "velocity_y"),
        ("velocity_z", "velocity_z")
    )

    # 指令: Controller → Plant
    self.world.connect(
        self.controller, self.plant,
        ("thrust_command_x", "command_x"),
        ("thrust_command_y", "command_y"),
        ("thrust_command_z", "command_z")
    )

    # 計測: Plant → Env
    self.world.connect(
        self.plant, self.spacecraft,
        ("measured_force_x", "force_x"),
        ("measured_force_y", "force_y"),
        ("measured_force_z", "force_z")
    )
```

**setup_data_collection()**:
```python
def setup_data_collection(self):
    # Controller data
    self.world.connect(
        self.controller, self.collector,
        "thrust_command_x", "thrust_command_y", "thrust_command_z"
    )

    # Plant data
    self.world.connect(
        self.plant, self.collector,
        "measured_force_x", "measured_force_y", "measured_force_z"
    )

    # Environment data
    self.world.connect(
        self.spacecraft, self.collector,
        "position_x", "position_y", "position_z",
        "velocity_x", "velocity_y", "velocity_z",
        "altitude", "semi_major_axis", "eccentricity"
    )
```

---

#### タスク3.5: main.py作成
**ファイル**: `orbital_hils/main.py`

```python
"""
Orbital HILS Simulation - Main Entry Point

6DOF軌道力学シミュレーション
"""
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent))

from scenarios.orbital_scenario import OrbitalScenario
from config.orbital_parameters import CONFIG_ISS

def main():
    print("=" * 70)
    print("Orbital HILS Simulation")
    print("6DOF Orbital Dynamics with Control Feedback Loop")
    print("=" * 70)

    # ISS軌道でシミュレーション
    scenario = OrbitalScenario(config=CONFIG_ISS)
    scenario.run()

if __name__ == "__main__":
    main()
```

---

### フェーズ4: __init__.pyファイル作成

#### `orbital_hils/simulators/__init__.py`
```python
"""Orbital HILS Simulators"""
from .controller_simulator import OrbitalControllerSimulator
from .plant_simulator import OrbitalPlantSimulator
from .env_simulator import OrbitalEnvSimulator

__all__ = [
    "OrbitalControllerSimulator",
    "OrbitalPlantSimulator",
    "OrbitalEnvSimulator",
]
```

#### `orbital_hils/scenarios/__init__.py`
```python
"""Orbital HILS Scenarios"""
from .orbital_scenario import OrbitalScenario

__all__ = ["OrbitalScenario"]
```

#### `orbital_hils/models/__init__.py`
```python
"""Orbital Control Models"""
from .thrust_model import ThrustModel

__all__ = ["ThrustModel"]
```

---

## HDF5データ構造

### ファイル名
`results_orbital/YYYYMMDD-HHMMSS/orbital_data.h5`

### グループ構造
```
orbital_data.h5
├── time/
│   ├── time_s [dataset]         # 時刻 [s]
│   └── time_ms [dataset]        # 時刻 [ms]
│
├── OrbitalControllerSim-0_OrbitalController_0/
│   ├── thrust_command_x [dataset]
│   ├── thrust_command_y [dataset]
│   └── thrust_command_z [dataset]
│
├── OrbitalPlantSim-0_OrbitalThrustStand_0/
│   ├── measured_force_x [dataset]
│   ├── measured_force_y [dataset]
│   └── measured_force_z [dataset]
│
└── OrbitalEnvSim-0_OrbitalSpacecraft_0/
    ├── position_x [dataset]
    ├── position_y [dataset]
    ├── position_z [dataset]
    ├── position_norm [dataset]
    ├── velocity_x [dataset]
    ├── velocity_y [dataset]
    ├── velocity_z [dataset]
    ├── velocity_norm [dataset]
    ├── acceleration_x [dataset]
    ├── acceleration_y [dataset]
    ├── acceleration_z [dataset]
    ├── altitude [dataset]
    ├── semi_major_axis [dataset]
    ├── eccentricity [dataset]
    └── specific_energy [dataset]
```

---

## 実装スケジュール

### Week 1: セットアップ & 移植
- [ ] プロジェクトディレクトリ作成
- [ ] 既存コンポーネント移植（Env, Parameters, 可視化）
- [ ] 動作確認（自由軌道シミュレーション）

### Week 2: 制御器実装
- [ ] ThrustModel作成（プレースホルダー）
- [ ] OrbitalControllerSimulator実装
- [ ] 単体テスト

### Week 3: Plant実装
- [ ] OrbitalPlantSimulator実装
- [ ] 3軸遅れ系モデル検証
- [ ] 単体テスト

### Week 4: 統合テスト
- [ ] OrbitalScenario接続定義
- [ ] 完全な制御ループ動作確認
- [ ] HDF5データ検証
- [ ] プロット生成確認

---

## 実行方法

### 基本実行
```bash
cd /home/akira/mosaik-hils/orbital_hils
uv run python main.py
```

### 依存関係インストール
```bash
cd /home/akira/mosaik-hils/orbital_hils
uv sync
```

### 結果の確認
```bash
# 最新の結果ディレクトリを確認
ls -ltr results_orbital/

# プロット確認（ブラウザで開く）
firefox results_orbital/YYYYMMDD-HHMMSS/orbital_3d_interactive.html
```

---

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

---

## 既存hilsとの関係

### 完全分離
- **orbital_hils**: 6DOF軌道制御
- **hils_simulation**: 1DOF推力スタンド

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

---

## テスト計画

### 単体テスト
1. **OrbitalControllerSimulator**
   - ThrustModelの呼び出し検証
   - 入出力データ型チェック
   - ゼロ推力動作確認

2. **OrbitalPlantSimulator**
   - 1次遅れ系の応答検証
   - ノイズ統計検証
   - 3軸独立動作確認

3. **OrbitalEnvSimulator**
   - RK4積分精度検証
   - 軌道要素保存性確認
   - エネルギー保存則

### 統合テスト
1. **制御ループ接続**
   - データフロー確認
   - Mosaik接続検証
   - 時刻同期確認

2. **長時間シミュレーション**
   - 1軌道周期（90分）
   - 数値誤差の蓄積確認
   - メモリリーク検証

3. **データ記録**
   - HDF5ファイル生成
   - 全データ保存確認
   - プロット自動生成

---

## リスクと対策

### リスク1: Mosaik接続の複雑性
**対策**: 段階的な接続テスト（Env→Controller→Plant→Env）

### リスク2: RK4積分の数値誤差
**対策**: エネルギー保存則でモニタリング、必要に応じてRK45への変更

### リスク3: 制御不安定
**対策**: 初期はゼロ推力、徐々にゲイン増加

---

## 成功基準

### フェーズ1完了
- [ ] 自由軌道シミュレーションが動作
- [ ] HDF5データが正しく記録
- [ ] プロットが自動生成

### フェーズ2完了
- [ ] 制御ループが閉じる
- [ ] 全コンポーネントが通信
- [ ] データ収集が完全

### 最終目標
- [ ] 安定した制御動作
- [ ] 軌道要素の保存
- [ ] 再現可能な結果

---

## 参考資料

### Mosaik
- https://mosaik.offis.de/
- Mosaik API Documentation

### 軌道力学
- Vallado, "Fundamentals of Astrodynamics and Applications"
- Curtis, "Orbital Mechanics for Engineering Students"

### 制御理論
- Ogata, "Modern Control Engineering"
- LQR Theory for Spacecraft Control

---

**作成日**: 2025-01-15
**最終更新**: 2025-01-15
**バージョン**: 1.0
**ステータス**: 実装準備完了
