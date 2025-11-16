# Mosaik HILS (Hardware-in-the-Loop Simulation)

## 概要 (Overview)

このリポジトリは、MosaikフレームワークをベースとしたHILS（Hardware-in-the-Loop Simulation）システムの3つの独立したプロジェクトを含んでいます。

This repository contains three independent HILS (Hardware-in-the-Loop Simulation) projects based on the Mosaik framework:

### 📦 含まれるプロジェクト (Included Projects)

1. **`hils_simulation/`** - **1自由度宇宙機HILS** (メインプロジェクト)
   - 通信遅延・ジッター・パケットロスを含むリアルタイム制御シミュレーション
   - v2シナリオベースアーキテクチャで複数モード（HILS、RT、逆補償、Pure Python）を統一実行
   - 1-DOF spacecraft motion control with communication delay modeling
   - Scenario-based architecture (v2) for HILS, RT, Inverse Compensation modes

2. **`orbital_hils/`** - **6自由度軌道力学HILS**
   - RK4法による高精度軌道シミュレーション
   - ホーマン遷移制御（Hohmann transfer）の実装
   - 3軸推力制御（X, Y, Z独立）
   - 6-DOF orbital dynamics with RK4 integration
   - Hohmann transfer control implementation

3. **`delay_estimation/`** - **カルマンフィルタ遅延推定**
   - ネットワーク制御システムにおける測定遅延推定
   - カルマンフィルタベースの状態推定
   - Measurement delay estimation for networked control systems
   - Kalman filter-based state estimation

各プロジェクトは独立していますが、 `pyproject.toml` で依存関係を共有しています。

## ✨ 主な機能 (Key Features)

### 1-DOF HILS (hils_simulation/)

- 🚀 **1自由度宇宙機シミュレーション** - 推力制御による位置制御
- 🎮 **PID制御器** - 比例積分微分制御による目標位置追従
- 🔄 **通信遅延モデリング** - 制御指令経路・測定経路の独立した遅延設定
- 📡 **通信ブリッジ** - ジッター・パケットロス・順序保持機能
- 🔧 **逆補償機能** - 遅延補償アルゴリズムの評価
- 🎯 **シナリオベース設計** - 複数のシミュレーションモードを簡単に切り替え

### 6-DOF Orbital HILS (orbital_hils/)

- 🛰️ **6自由度軌道力学** - 二体問題の運動方程式をRK4法で積分
- 🎯 **ホーマン遷移制御** - 2つの円軌道間の最適遷移（Lambert solver使用）
- 🔧 **3軸推力制御** - X, Y, Z方向独立の推力制御
- 📊 **軌道要素計算** - 高度、離心率、軌道半長軸などをリアルタイム計算
- 📈 **インタラクティブ可視化** - Plotlyによる3D軌道可視化

### Delay Estimation (delay_estimation/)

- 📡 **遅延推定** - ネットワーク制御システムにおける測定遅延の推定
- 🎯 **カルマンフィルタ** - 遅延を考慮した状態推定
- 📊 **性能比較** - 標準カルマンフィルタとの性能比較

### 共通機能

- 📊 **HDF5データ収集** - 全シミュレーションデータの自動記録
- 📈 **データ解析ツール** - 統計情報とグラフの自動生成
- ⚙️ **環境変数設定** - `.env`ファイルによる柔軟なパラメータ管理

## 📁 プロジェクト構造 (Project Structure)

```text
mosaik-hils/
├── hils_simulation/              # 1-DOF HILS（メインプロジェクト）
│   ├── config/                   # 設定管理
│   ├── scenarios/                # シナリオ実装（HILS, RT, 逆補償等）
│   ├── simulators/               # Mosaikシミュレーター群
│   ├── scripts/                  # スイープ・解析スクリプト
│   ├── docs/                     # ドキュメント
│   ├── main.py                   # エントリーポイント
│   └── results/                  # 結果出力ディレクトリ
│
├── orbital_hils/                 # 6-DOF 軌道力学HILS
│   ├── config/                   # 軌道パラメータ設定
│   ├── scenarios/                # 軌道シナリオ
│   ├── simulators/               # 軌道制御シミュレーター
│   ├── models/                   # 推力モデル（PD, Hohmann等）
│   ├── scripts/                  # 可視化・解析スクリプト
│   ├── main.py                   # エントリーポイント
│   └── results_orbital/          # 結果出力ディレクトリ
│
├── delay_estimation/             # カルマンフィルタ遅延推定
│   ├── config/                   # 設定
│   ├── estimators/               # 遅延推定アルゴリズム
│   ├── scenarios/                # テストシナリオ
│   ├── simulators/               # システムシミュレーター
│   ├── main.py                   # エントリーポイント
│   └── results/                  # 結果出力ディレクトリ
│
├── pyproject.toml                # 依存関係管理（全プロジェクト共有）
├── uv.lock                       # 依存関係ロックファイル
├── README.md                     # このファイル
└── .claude/CLAUDE.md             # AI開発ガイド
```

各プロジェクトの詳細は各ディレクトリ内のREADME.mdを参照してください：

- [hils_simulation/README.md](hils_simulation/README.md) - 1-DOF HILS詳細
- [orbital_hils/README.md](orbital_hils/README.md) - 6-DOF Orbital HILS詳細
- [delay_estimation/README.md](delay_estimation/README.md) - 遅延推定詳細

## 🚀 クイックスタート (Quick Start)

### 1. 環境構築

```bash
# リポジトリをクローン
git clone <repository-url>
cd mosaik-hils

# 依存関係のインストール（全プロジェクト共通）
uv sync
```

### 2. シミュレーション実行

#### 2.1 1-DOF HILS (hils_simulation/)

```bash
cd hils_simulation

# HILSシミュレーション（通信遅延あり）
uv run python main.py h

# RTシミュレーション（通信遅延なし）- ベースライン比較用
uv run python main.py r

# 逆補償シミュレーション
uv run python main.py i

# デュアルフィードバック逆補償シミュレーション
uv run python main.py d

# Pure Pythonシミュレーション（Mosaikなし）
uv run python main.py p

# ヘルプ表示
uv run python main.py --help
```

#### 2.2 6-DOF Orbital HILS (orbital_hils/)

```bash
cd orbital_hils

# .envファイルで制御タイプを設定
# CONTROLLER_TYPE=zero      # 自由軌道運動（デフォルト）
# CONTROLLER_TYPE=pd        # PD制御
# CONTROLLER_TYPE=hohmann   # ホーマン遷移

# シミュレーション実行
uv run python main.py

# 結果可視化（基本プロット）
uv run python scripts/analysis/visualize_orbital_results.py results_orbital/YYYYMMDD-HHMMSS/hils_data.h5

# インタラクティブ3D可視化（ブラウザで開く）
uv run python scripts/analysis/visualize_orbital_interactive.py results_orbital/YYYYMMDD-HHMMSS/hils_data.h5

# ホーマン遷移フェーズ可視化
uv run python scripts/analysis/visualize_hohmann_phases.py results_orbital/YYYYMMDD-HHMMSS/hils_data.h5
```

#### 2.3 Delay Estimation (delay_estimation/)

```bash
cd delay_estimation

# 遅延推定シミュレーション実行
uv run python main.py
```

### 3. 結果確認

実行後、各プロジェクトの結果ディレクトリに出力が生成されます：

**1-DOF HILS結果**: `hils_simulation/results/YYYYMMDD-HHMMSS/`

- `hils_data.h5` - 全シミュレーションデータ（HDF5形式）
- `simulation_config.json` - シミュレーション設定
- `*_dataflowGraph_*.png` - データフローグラフ
- `*_executiontime.png` - 実行時間分析

**6-DOF Orbital HILS結果**: `orbital_hils/results_orbital/YYYYMMDD-HHMMSS/`

- `hils_data.h5` - 軌道データ（HDF5形式）
- `simulation_config.json` - シミュレーション設定
- `*.png` - 自動生成されたプロット

**Delay Estimation結果**: `delay_estimation/results/YYYYMMDD-HHMMSS/`

- 推定結果とプロット

## ⚙️ `.env` 設定とプラントモデルの切り替え

`hils_simulation/config/parameters.py` は `.env` を自動ロードし、環境変数で各種パラメータを上書きできます。  
時間定数モデルを利用する場合は以下のキーを設定してください。

| 変数名                   | デフォルト | 説明                                                                                                                 |
| ------------------------ | ---------- | -------------------------------------------------------------------------------------------------------------------- |
| `PLANT_TAU_MODEL_TYPE`   | `constant` | 使用する時定数モデルタイプ。 `constant` , `linear` , `saturation` , `thermal` , `hybrid` , `stochastic` を指定可能。 |
| `PLANT_TAU_MODEL_PARAMS` | `{}`       | モデル固有パラメータを JSON 文字列で指定。未設定または解析不可の場合は空辞書として扱います。                         |

`.env` 例:

```env
# Plant base dynamics
PLANT_TIME_CONSTANT=80.0
PLANT_TIME_CONSTANT_STD=5.0

# Dynamic tau model (hybrid = thrust-rate + thermal)
PLANT_TAU_MODEL_TYPE=hybrid
PLANT_TAU_MODEL_PARAMS={"thrust_sensitivity":0.25,"heating_rate":0.001,"cooling_rate":0.01,"thermal_sensitivity":0.04}
```

`.env` を編集した後は通常通り `uv run python main.py <scenario>` を実行するだけで、全シナリオ (HILS / RT / InverseComp) で新しいプラントモデル設定が反映されます。

## 🔧 システム構成 (System Architecture)

### v2 シナリオベースアーキテクチャ

```
main.py (統一エントリーポイント)
  ↓
scenarios/ (シナリオ選択)
  ├── base_scenario.py          # Template Methodパターン基底クラス
  ├── hils_scenario.py          # HILS構成
  ├── rt_scenario.py            # RT構成
  ├── inverse_comp_scenario.py  # 逆補償構成
  └── pure_python_scenario.py   # Pure Python構成
  ↓
config/ (パラメータ管理)
  ├── parameters.py             # .env対応の集中パラメータ管理
  └── sim_config.py             # Mosaikシミュレーター設定
```

### データフロー（HILSシナリオ）

```
Controller → Bridge(cmd) → Plant → Bridge(sense) → Env
     ↑                                                ↓
     └────────────── (same-step feedback) ────────────┘

All components → DataCollector → HDF5 File
```

**重要な接続**:
- **制御指令経路**: Controller → Bridge(cmd) [time-shifted] → Plant
- **測定経路**: Plant → Bridge(sense) → Env
- **状態フィードバック**: Env → Controller (same-step)
- **Time-shifted接続**: Controller→Bridge(cmd)で循環依存を解決

### 主要コンポーネント

#### 1. ControllerSimulator（制御器）

- **モデル**: PIDController
- **制御則**: `F = Kp * error + Ki * integral - Kd * velocity`
- **入力**: position, velocity
- **出力**: command（JSON: `{thrust, duration}`）, error, integral

#### 2. PlantSimulator（推力測定器）

- **モデル**: ThrustStand
- **機能**: 推力指令を受け取り、理想的な推力を出力
- **入力**: command（JSON: `{thrust, duration}`）
- **出力**: measured_thrust, status

#### 3. EnvSimulator（環境シミュレーター）

- **モデル**: Spacecraft1DOF
- **運動方程式**: `F = ma`, オイラー法で積分
- **入力**: force
- **出力**: position, velocity, acceleration

#### 4. BridgeSimulator（通信ブリッジ）

- **モデル**: CommBridge
- **機能**: 遅延・ジッター・パケットロスを模擬
- **入力**: input（任意のデータ）
- **出力**: delayed_output, stats

#### 5. InverseCompensatorSimulator（逆補償器）

- **モデル**: InverseCompensator
- **機能**: 遅延の影響を予測・補償
- **入力**: position, velocity, command
- **出力**: compensated_command

#### 6. DataCollectorSimulator（データ収集器）

- **モデル**: Collector
- **機能**: 全シミュレーションデータをHDF5形式で記録
- **入力**: 全シミュレーターからの全属性（動的）

## ⚙️ パラメータ設定

### `.env` ファイルによる設定（推奨）

`hils_simulation/` ディレクトリに `.env` ファイルを作成してパラメータを設定：

```bash
# hils_simulation/.env

# シミュレーション設定
SIMULATION_TIME=2.0          # シミュレーション時間 [s]
TIME_RESOLUTION=0.0001       # 時間解像度 [s]
RT_FACTOR=None               # リアルタイム係数（None=最速）

# 通信遅延パラメータ
CMD_DELAY=20.0               # 制御指令経路の遅延 [ms]
CMD_JITTER=0.0               # 制御指令経路のジッター [ms]
CMD_LOSS_RATE=0.0            # パケットロス率

SENSE_DELAY=30.0             # 測定経路の遅延 [ms]
SENSE_JITTER=0.0             # 測定経路のジッター [ms]
SENSE_LOSS_RATE=0.0          # パケットロス率

# 制御パラメータ
CONTROL_PERIOD=10.0          # 制御周期 [ms]
KP=15.0                      # 比例ゲイン
KI=0.5                       # 積分ゲイン
KD=5.0                       # 微分ゲイン
TARGET_POSITION=5.0          # 目標位置 [m]
MAX_THRUST=100.0             # 最大推力 [N]
INTEGRAL_LIMIT=100.0         # 積分項リミット

# 宇宙機パラメータ
SPACECRAFT_MASS=1.0          # 質量 [kg]
INITIAL_POSITION=0.0         # 初期位置 [m]
INITIAL_VELOCITY=10.0        # 初期速度 [m/s]
GRAVITY=9.81                 # 重力加速度 [m/s^2]

# 逆補償パラメータ
ENABLE_INVERSE_COMP=True     # 逆補償を有効化
INVERSE_COMP_GAIN=15.0       # 補償ゲイン
```

### プログラムによる設定

```python
from config.parameters import SimulationParameters
from scenarios import HILSScenario

# パラメータの読み込みとカスタマイズ
params = SimulationParameters.from_env()
params.simulation_time = 5.0
params.control.kp = 20.0
params.communication.cmd_delay = 50.0

# シナリオの実行
scenario = HILSScenario(params)
scenario.run()
```

## 📊 データ分析 (Data Analysis)

### HDF5データ構造

```python
import h5py

with h5py.File('hils_simulation/results/YYYYMMDD-HHMMSS/hils_data.h5', 'r') as f:
    # 時刻データ
    time = f['steps']['time_s'][:]

    # 位置・速度データ
    position = f['steps']['position_EnvSim-0.Spacecraft1DOF_0'][:]
    velocity = f['steps']['velocity_EnvSim-0.Spacecraft1DOF_0'][:]

    # 制御指令データ（JSON形式）
    command = f['steps']['command_ControllerSim-0.PIDController_0'][:]

    # 遅延統計（JSON形式）
    stats_cmd = f['steps']['stats_BridgeSim_cmd-0.CommBridge_0'][:]
```

### 統計情報の表示

```bash
cd hils_simulation
uv run python analyze_data.py results/YYYYMMDD-HHMMSS/hils_data.h5
```

出力例：
- 位置・速度の平均値、標準偏差、最小値、最大値
- 制御誤差の統計
- 推力指令の統計
- 遅延統計（パケットロス数、遅延時間など）

### グラフ生成

```bash
# プロット付きで解析
uv run python analyze_data.py results/YYYYMMDD-HHMMSS/hils_data.h5 --save-plots
```

生成されるグラフ：
- 位置・速度の時系列
- 制御誤差の時系列
- 推力指令の時系列
- 遅延統計のヒストグラム

## 🔬 実験例 (Experiment Examples)

### 実験1: ベースライン（遅延なし）

```bash
# RTシナリオを実行
cd hils_simulation
uv run python main.py r
```

### 実験2: 通信遅延の影響評価

```bash
# .envファイルで遅延を設定
# CMD_DELAY=20.0
# SENSE_DELAY=30.0

# HILSシナリオを実行
uv run python main.py h
```

### 実験3: 逆補償の効果検証

```bash
# .envファイルで逆補償を有効化
# ENABLE_INVERSE_COMP=True
# INVERSE_COMP_GAIN=15.0

# 逆補償シナリオを実行
uv run python main.py i
```

### 実験4: シナリオ間の比較

```bash
# 複数のシナリオを実行
uv run python main.py r   # ベースライン
uv run python main.py h   # 遅延あり
uv run python main.py i   # 逆補償

# 比較スクリプトを実行（archive/comparisons/にあり）
cd archive/comparisons
uv run python compare_positions.py
```

## 🛠️ 開発ガイド (Development Guide)

### コードフォーマット

```bash
# リンターの実行
uv run ruff check

# 自動修正
uv run ruff check --fix

# フォーマット
uv run ruff format
```

### 新しいシナリオの追加

1. `hils_simulation/scenarios/`に新しいファイルを作成
2. `BaseScenario`を継承してメソッドを実装
3. `scenarios/__init__.py`に追加
4. `main.py`のシナリオマッピングに追加

詳細は [.claude/CLAUDE.md](.claude/CLAUDE.md) を参照してください。

### 新しいシミュレーターの追加

1. `hils_simulation/simulators/`にファイル作成
2. `mosaik_api.Simulator`を継承
3. `config/sim_config.py`に追加
4. シナリオの`setup_entities()`と`connect_entities()`で使用

詳細は [.claude/CLAUDE.md](.claude/CLAUDE.md) を参照してください。

## 📝 使用技術 (Technologies)

- **Python 3.9+** - プログラミング言語
- **Mosaik 3.5+** - コシミュレーションフレームワーク
- **HDF5 / h5py** - データ保存・読み込み
- **matplotlib** - データ可視化
- **numpy** - 数値計算
- **python-dotenv** - 環境変数管理
- **uv** - 依存関係管理（高速）
- **ruff** - リンター・フォーマッター

## 🏆 実装済み機能と今後の拡張

### ✅ 実装済み

#### 1-DOF HILS (hils_simulation/)

- [x] シナリオベースアーキテクチャ（v2）
- [x] 通信遅延・ジッター・パケットロスモデリング
- [x] PID制御器（比例積分微分制御）
- [x] 逆補償機能（シングル＆デュアルフィードバック）
- [x] プラント時定数の動的モデル（constant, linear, saturation, thermal, hybrid, stochastic）
- [x] パラメータスイープ機能（遅延、ゲイン、プラント時定数）
- [x] データ収集とプロット機能（HDF5形式）
- [x] 環境変数による柔軟なパラメータ管理

#### 6-DOF Orbital HILS (orbital_hils/)

- [x] 6自由度軌道力学シミュレーション（RK4積分）
- [x] ホーマン遷移制御（Lambert solver使用）
- [x] PD位置・速度制御
- [x] 3軸推力制御（X, Y, Z独立）
- [x] インタラクティブ3D可視化（Plotly）
- [x] 軌道要素のリアルタイム計算

#### Delay Estimation (delay_estimation/)

- [x] カルマンフィルタベースの遅延推定
- [x] ネットワーク遅延モデリング
- [x] 標準カルマンフィルタとの性能比較

### 🚧 今後の拡張

#### 1-DOF HILS

- [ ] リアルタイムモニタリングダッシュボード
- [ ] 実機制御器との統合
- [ ] 先行送出・補間などの追加補償アルゴリズム
- [ ] Webベースのパラメータ設定UI

#### 6-DOF Orbital HILS

- [ ] BridgeSimulator統合（通信遅延モデリング）
- [ ] 逆補償器の統合
- [ ] LQR/MPC制御アルゴリズム
- [ ] 軌道面変更を含む遷移
- [ ] J2摂動、大気抵抗、三体問題などの摂動モデル

#### Delay Estimation

- [ ] より高度な遅延推定アルゴリズム
- [ ] HILSシミュレーションとの統合

## 📚 参考ドキュメント (Documentation)

### 全体

- [.claude/CLAUDE.md](.claude/CLAUDE.md) - AI開発ガイド（開発者向け詳細情報）
- [README.md](README.md) - このファイル（プロジェクト全体の概要）

### 1-DOF HILS (hils_simulation/)

- [hils_simulation/README.md](hils_simulation/README.md) - 1-DOF HILSの詳細ドキュメント
- [hils_simulation/docs/V2_ARCHITECTURE.md](hils_simulation/docs/V2_ARCHITECTURE.md) - v2アーキテクチャ詳細
- [hils_simulation/docs/README_scenarios.md](hils_simulation/docs/README_scenarios.md) - シナリオクイックスタート
- [hils_simulation/docs/PLANT_VARIABILITY.md](hils_simulation/docs/PLANT_VARIABILITY.md) - プラント時定数モデル詳細
- [hils_simulation/archive/README.md](hils_simulation/archive/README.md) - v1レガシー実装の説明

### 6-DOF Orbital HILS (orbital_hils/)

- [orbital_hils/README.md](orbital_hils/README.md) - 6-DOF Orbital HILSの詳細ドキュメント
- [orbital_hils/docs/HOHMANN_TRANSFER.md](orbital_hils/docs/HOHMANN_TRANSFER.md) - ホーマン遷移制御詳細

### Delay Estimation (delay_estimation/)

- [delay_estimation/README.md](delay_estimation/README.md) - 遅延推定の詳細ドキュメント
- [delay_estimation/docs/IMPLEMENTATION_GUIDE.md](delay_estimation/docs/IMPLEMENTATION_GUIDE.md) - 実装ガイド

## ❓ よくある質問 (FAQ)

### Q: どのプロジェクトを使えばいいですか？

**A**: 目的に応じて選択してください：

- **通信遅延の影響を研究したい** → `hils_simulation/` (1-DOF HILS)
- **軌道制御を研究したい** → `orbital_hils/` (6-DOF Orbital HILS)
- **遅延推定アルゴリズムを研究したい** → `delay_estimation/`

### Q: v1とv2の違いは？（1-DOF HILSについて）

**A**: v1は `main_hils.py` などの独立したスクリプトで各シナリオを実装していました。v2では `main.py` から統一されたインターフェースで全シナリオを実行でき、コードの重複が大幅に削減されています。

### Q: `.env` ファイルはどこに置く？

**A**: 各プロジェクトディレクトリ直下に配置してください：

- `hils_simulation/.env` - 1-DOF HILS用
- `orbital_hils/.env` - 6-DOF Orbital HILS用
- `delay_estimation/.env` - Delay Estimation用

### Q: 複数のシナリオを比較したい

**A**: 各プロジェクトには解析・比較スクリプトが用意されています：

- 1-DOF HILS: `scripts/analysis/compare_with_rt.py`
- 6-DOF Orbital HILS: `scripts/analysis/visualize_*.py`

### Q: 3つのプロジェクトは連携できますか？

**A**: 現在は独立していますが、将来的に統合予定です：

- 6-DOF Orbital HILSへの通信遅延モデリング統合
- Delay EstimationアルゴリズムのHILSへの統合

### Q: レガシーコード（v1）は使える？

**A**: はい。 `hils_simulation/archive/` に保存されています。ただし、新規開発ではv2を推奨します。

## 📜 ライセンス (License)

このプロジェクトはMITライセンスの下で公開されています。

## 🆕 更新履歴 (Changelog)

### 2025-11-16

- **orbital_hils**: ホーマン遷移制御実装、インタラクティブ可視化追加
- **README.md**: 3つのプロジェクト（1-DOF HILS, 6-DOF Orbital HILS, Delay Estimation）を統合した全体概要に更新

### 2025-10-18

- **hils_simulation v2.0**: シナリオベースアーキテクチャに移行、.env対応、PID制御への拡張

### 2025-10-16

- **v1.3**: プロジェクト構造整理、hils_simulationに統合

### 2025-10-13

- **v1.2**: コマンドパッケージ化、データフロー改善

### 初期リリース

- **v1.1**: データ収集機能追加
- **v1.0**: 1DOF HILS実装完了

---

**貢献・フィードバック歓迎！** Issue報告やPull Requestをお待ちしています。
