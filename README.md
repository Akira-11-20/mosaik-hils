# Mosaik HILS (Hardware-in-the-Loop Simulation)

## 概要 (Overview)

このプロジェクトは、MosaikフレームワークをベースとしたHILS（Hardware-in-the-Loop Simulation）システムです。1自由度の宇宙機運動制御を模擬した環境で、通信遅延・ジッター・パケットロスを含むリアルタイム制御シミュレーションを提供します。

**v2アーキテクチャ**では、シナリオベースの設計により、複数のシミュレーションモード（HILS、RT、逆補償、Pure Python）を統一されたインターフェースで実行できます。

This project implements a HILS (Hardware-in-the-Loop Simulation) system based on the Mosaik framework. It provides a 1-DOF spacecraft motion control simulation environment with communication delay, jitter, and packet loss modeling.

**v2 architecture** enables multiple simulation modes (HILS, RT, Inverse Compensation, Pure Python) through a unified scenario-based interface.

## ✨ 主な機能 (Key Features)

- 🚀 **1自由度宇宙機シミュレーション** - 推力制御による位置制御
- 🎮 **PID制御器** - 比例積分微分制御による目標位置追従
- 🔄 **通信遅延モデリング** - 制御指令経路・測定経路の独立した遅延設定
- 📡 **通信ブリッジ** - ジッター・パケットロス・順序保持機能
- 🔧 **逆補償機能** - 遅延補償アルゴリズムの評価
- 📊 **HDF5データ収集** - 全シミュレーションデータの自動記録
- 📈 **データ解析ツール** - 統計情報とグラフの自動生成
- 🎯 **シナリオベース設計** - 複数のシミュレーションモードを簡単に切り替え
- ⚙️ **環境変数設定** - `.env`ファイルによる柔軟なパラメータ管理

## 📁 プロジェクト構造 (Project Structure)

```
mosaik-hils/
├── hils_simulation/              # メインシミュレーションディレクトリ
│   ├── config/                   # v2: 設定管理
│   │   ├── parameters.py         # パラメータ管理（.env対応）
│   │   └── sim_config.py         # シミュレーター設定
│   ├── scenarios/                # v2: シナリオ実装
│   │   ├── base_scenario.py      # 基底クラス
│   │   ├── hils_scenario.py      # HILS（通信遅延あり）
│   │   ├── rt_scenario.py        # RT（通信遅延なし）
│   │   ├── inverse_comp_scenario.py  # 逆補償
│   │   └── pure_python_scenario.py   # Pure Python
│   ├── simulators/               # Mosaikシミュレーター群
│   │   ├── controller_simulator.py   # PID制御器
│   │   ├── plant_simulator.py        # 推力測定器
│   │   ├── env_simulator.py          # 環境シミュレーター（1DOF運動）
│   │   ├── bridge_simulator.py       # 通信ブリッジ
│   │   ├── inverse_compensator_simulator.py  # 逆補償器
│   │   └── data_collector.py         # データ収集器（HDF5）
│   ├── utils/                    # ユーティリティ
│   │   ├── plot_utils.py         # プロット関数
│   │   └── event_logger.py       # イベントログ
│   ├── docs/                     # ドキュメント
│   │   ├── V2_ARCHITECTURE.md    # v2アーキテクチャ詳細
│   │   ├── README_scenarios.md   # シナリオクイックスタート
│   │   └── その他の技術文書
│   ├── archive/                  # v1レガシーコード
│   │   ├── main_hils.py          # (旧) HILSエントリーポイント
│   │   ├── main_hils_rt.py       # (旧) RTエントリーポイント
│   │   └── その他v1ファイル
│   ├── results/                  # HILS/逆補償結果（自動生成）
│   ├── results_rt/               # RT結果（自動生成）
│   ├── results_pure/             # Pure Python結果（自動生成）
│   ├── main.py                   # v2: 統一エントリーポイント
│   ├── analyze_data.py           # データ解析スクリプト
│   └── visualize_results.py      # 結果可視化
├── inverse_comp_demo/            # 逆補償スタンドアロンデモ
├── pyproject.toml                # 依存関係管理
├── uv.lock                       # 依存関係ロックファイル
├── README.md                     # このファイル
└── .claude/CLAUDE.md             # AI開発ガイド
```

詳細は [hils_simulation/docs/V2_ARCHITECTURE.md](hils_simulation/docs/V2_ARCHITECTURE.md) を参照してください。

## 🚀 クイックスタート (Quick Start)

### 1. 環境構築

```bash
# リポジトリをクローン
git clone <repository-url>
cd mosaik-hils

# 依存関係のインストール
uv sync
```

### 2. シミュレーション実行（v2）

```bash
cd hils_simulation

# HILSシミュレーション（通信遅延あり）
uv run python main.py h

# RTシミュレーション（通信遅延なし）- ベースライン比較用
uv run python main.py r

# 逆補償シミュレーション
uv run python main.py i

# Pure Pythonシミュレーション（Mosaikなし）
uv run python main.py p

# ヘルプ表示
uv run python main.py --help
```

**利用可能なシナリオ**:
- `h` / `hils` - HILS（通信遅延あり）
- `r` / `rt` - RT（通信遅延なし）
- `i` / `inverse_comp` - 逆補償
- `p` / `pure_python` - Pure Python

### 3. 結果確認

実行後、各シナリオに対応したディレクトリに結果が生成されます：

**HILS / 逆補償**: `hils_simulation/results/YYYYMMDD-HHMMSS/`
- `hils_data.h5` - 全シミュレーションデータ（HDF5形式）
- `simulation_config.json` - シミュレーション設定
- `*_dataflowGraph_*.png` - データフローグラフ
- `*_executiontime.png` - 実行時間分析

**RT**: `hils_simulation/results_rt/YYYYMMDD-HHMMSS/`

**Pure Python**: `hils_simulation/results_pure/YYYYMMDD-HHMMSS/`

### 4. データ解析

```bash
# 統計情報とグラフを生成
cd hils_simulation
uv run python analyze_data.py results/YYYYMMDD-HHMMSS/hils_data.h5 --save-plots
```

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

### `.env`ファイルによる設定（推奨）

`hils_simulation/`ディレクトリに`.env`ファイルを作成してパラメータを設定：

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
- [x] シナリオベースアーキテクチャ（v2）
- [x] 通信遅延・ジッター・パケットロスモデリング
- [x] PID制御器（比例積分微分制御）
- [x] 逆補償機能
- [x] データ収集とプロット機能（HDF5形式）
- [x] データ解析スクリプト（analyze_data.py）
- [x] 環境変数による柔軟なパラメータ管理
- [x] 複数シナリオ実行・比較ツール

### 🚧 今後の拡張
- [ ] リアルタイムモニタリングダッシュボード
- [ ] 6DOF版への拡張（姿勢制御）
- [ ] 実機制御器との統合
- [ ] 先行送出・補間などの追加補償アルゴリズム
- [ ] Webベースのパラメータ設定UI
- [ ] 自動実験実行フレームワーク

## 📚 参考ドキュメント (Documentation)

- [.claude/CLAUDE.md](.claude/CLAUDE.md) - AI開発ガイド（開発者向け詳細情報）
- [hils_simulation/docs/V2_ARCHITECTURE.md](hils_simulation/docs/V2_ARCHITECTURE.md) - v2アーキテクチャ詳細
- [hils_simulation/docs/README_scenarios.md](hils_simulation/docs/README_scenarios.md) - シナリオクイックスタート
- [hils_simulation/archive/README.md](hils_simulation/archive/README.md) - v1レガシー実装の説明

## ❓ よくある質問 (FAQ)

### Q: v1とv2の違いは？
**A**: v1は`main_hils.py`などの独立したスクリプトで各シナリオを実装していました。v2では`main.py`から統一されたインターフェースで全シナリオを実行でき、コードの重複が大幅に削減されています。

### Q: `.env`ファイルはどこに置く？
**A**: `hils_simulation/`ディレクトリ直下に配置してください。

### Q: 複数のシナリオを比較したい
**A**: 各シナリオを実行後、`archive/comparisons/`内の比較スクリプトを使用するか、`analyze_data.py`で個別に解析してください。

### Q: レガシーコード（v1）は使える？
**A**: はい。`hils_simulation/archive/`に保存されています。ただし、新規開発ではv2を推奨します。

## 📜 ライセンス (License)

このプロジェクトはMITライセンスの下で公開されています。

## 🆕 更新履歴 (Changelog)

- **v2.0** (2025-10-18) - シナリオベースアーキテクチャに移行、.env対応、PID制御への拡張
- **v1.3** (2025-10-16) - プロジェクト構造整理、hils_simulationに統合
- **v1.2** (2025-10-13) - コマンドパッケージ化、データフロー改善
- **v1.1** - データ収集機能追加
- **v1.0** - 1DOF HILS実装完了

---

**貢献・フィードバック歓迎！** Issue報告やPull Requestをお待ちしています。
