# Mosaik HILS (Hardware-in-the-Loop Simulation)

## 概要 (Overview)

このプロジェクトは、MosaikフレームワークをベースとしたHILS（Hardware-in-the-Loop Simulation）システムです。1自由度の宇宙機運動制御を模擬した環境で、通信遅延・ジッター・パケットロスを含むリアルタイム制御シミュレーションを提供します。

This project implements a HILS (Hardware-in-the-Loop Simulation) system based on the Mosaik framework. It provides a 1-DOF spacecraft motion control simulation environment with communication delay, jitter, and packet loss modeling.

## ✨ 主な機能 (Key Features)

- 🚀 **1自由度宇宙機シミュレーション** - 推力制御による位置制御
- 🎮 **PD制御器** - 比例微分制御による目標位置追従
- 🔄 **通信遅延モデリング** - 制御指令経路・測定経路の独立した遅延設定
- 📡 **通信ブリッジ** - ジッター・パケットロス・順序保持機能
- 📊 **HDF5データ収集** - 全シミュレーションデータの自動記録
- 📈 **データ解析ツール** - 統計情報とグラフの自動生成

## 📁 プロジェクト構造 (Project Structure)

```
mosaik-hils/
├── hils_simulation/          # メインシミュレーションディレクトリ
│   ├── simulators/           # Mosaikシミュレーター群
│   │   ├── controller_simulator.py   # PD制御器
│   │   ├── plant_simulator.py        # 推力測定器（スラストスタンド）
│   │   ├── env_simulator.py          # 環境シミュレーター（1DOF運動）
│   │   ├── bridge_simulator.py       # 通信ブリッジ（遅延・ジッター）
│   │   └── data_collector.py         # データ収集器（HDF5）
│   ├── utils/                # ユーティリティ
│   │   ├── plot_utils.py     # プロット関数
│   │   └── event_logger.py   # イベントログ
│   ├── docs/                 # ドキュメント
│   │   ├── README.md         # 詳細なガイド
│   │   ├── DATA_COLLECTION.md
│   │   └── その他の技術文書
│   ├── results/              # シミュレーション結果（自動生成）
│   ├── main_hils.py          # メインシナリオ
│   └── analyze_data.py       # データ解析スクリプト
├── logs/                     # 古い実行ログ（アーカイブ）
├── pyproject.toml            # 依存関係管理
├── uv.lock                   # 依存関係ロックファイル
├── README.md                 # このファイル
└── CLAUDE.md                 # AI開発ガイド
```

詳細は [hils_simulation/docs/README.md](hils_simulation/docs/README.md) を参照してください。

## 🚀 クイックスタート (Quick Start)

### 1. 環境構築

```bash
# リポジトリをクローン
git clone <repository-url>
cd mosaik-hils

# 依存関係のインストール
uv sync
```

### 2. シミュレーション実行

```bash
cd hils_simulation
uv run python main_hils.py
```

### 3. 結果確認

実行後、`hils_simulation/results/YYYYMMDD-HHMMSS/` ディレクトリに以下のファイルが生成されます：

- `hils_data.h5` - 全シミュレーションデータ（HDF5形式）
- `*_dataflowGraph_*.png` - データフローグラフ
- `*_executionGraph.png` - 実行順序グラフ
- `*_executiontime.png` - 実行時間分析

### 4. データ解析

```bash
# 統計情報とグラフを生成
cd hils_simulation
uv run python analyze_data.py results/YYYYMMDD-HHMMSS/hils_data.h5 --save-plots
```

## 🔧 システム構成 (System Architecture)

### データフロー

```
Controller → Bridge(cmd) → Plant → Bridge(sense) → Env → Controller (time-shifted)
     ↓            ↓          ↓           ↓          ↓
     DataCollector ← ← ← ← ← ← ← ← ← ← ← ←
            ↓
       HDF5 File
```

### データフロー詳細

1. **制御指令経路（cmd）**:
   - Controller が推力指令（JSON形式）を生成
   - Bridge(cmd) で遅延・ジッター・パケットロスを模擬
   - Plant（推力測定器）が指令を受信

2. **測定経路（sense）**:
   - Plant が測定した推力を出力
   - Bridge(sense) で遅延・ジッター・パケットロスを模擬
   - Env（環境シミュレーター）が推力を受け取り、運動方程式を積分

3. **状態フィードバック経路**:
   - Env が位置・速度を出力
   - Controller が状態を受信（time-shifted接続で循環依存を解決）

### 主要コンポーネント

#### 1. ControllerSimulator（制御器）
- **モデル**: PDController
- **制御則**: `F = Kp * error - Kd * velocity`
- **ステップサイズ**: 10ms（制御周期）
- **入力**: position, velocity
- **出力**: command（JSON: `{thrust, duration}`）, error

#### 2. PlantSimulator（推力測定器）
- **モデル**: ThrustStand
- **機能**: 推力指令を受け取り、理想的な推力を出力
- **ステップサイズ**: 1ms
- **入力**: command（JSON: `{thrust, duration}`）
- **出力**: measured_thrust, status

#### 3. EnvSimulator（環境シミュレーター）
- **モデル**: Spacecraft1DOF
- **運動方程式**: `F = ma`, オイラー法で積分
- **ステップサイズ**: 1ms
- **入力**: force
- **出力**: position, velocity, acceleration

#### 4. BridgeSimulator（通信ブリッジ）
- **モデル**: CommBridge
- **機能**: 遅延・ジッター・パケットロスを模擬
- **ステップサイズ**: 1ms（高頻度実行）
- **入力**: input（任意のデータ）
- **出力**: delayed_output, stats

#### 5. DataCollectorSimulator（データ収集器）
- **モデル**: Collector
- **機能**: 全シミュレーションデータをHDF5形式で記録
- **ステップサイズ**: 1ms
- **入力**: 全シミュレーターからの全属性（動的）

## ⚙️ パラメータ設定

[hils_simulation/main_hils.py](hils_simulation/main_hils.py) の冒頭で以下のパラメータを変更できます：

### 通信遅延パラメータ

```python
CMD_DELAY = 50          # 制御指令経路の遅延 [ms]
CMD_JITTER = 10         # 制御指令経路のジッター [ms]
CMD_LOSS_RATE = 0.01    # パケットロス率（1%）

SENSE_DELAY = 100       # 測定経路の遅延 [ms]
SENSE_JITTER = 20       # 測定経路のジッター [ms]
SENSE_LOSS_RATE = 0.02  # パケットロス率（2%）
```

### 制御パラメータ

```python
CONTROL_PERIOD = 10     # 制御周期 [ms]
KP = 2.0                # 比例ゲイン
KD = 5.0                # 微分ゲイン
TARGET_POSITION = 10.0  # 目標位置 [m]
MAX_THRUST = 20.0       # 最大推力 [N]
```

### シミュレーション設定

```python
SIMULATION_TIME = 5000  # シミュレーション時間 [ms]
TIME_RESOLUTION = 0.001 # 時間解像度（1ms）
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

    # 制御指令データ
    thrust = f['steps']['command_ControllerSim-0.PDController_0'][:]

    # 遅延統計
    stats = f['steps']['stats_BridgeSim_cmd-0.CommBridge_0'][:]
```

### 統計情報の表示

```bash
cd hils_simulation
uv run python analyze_data.py results/YYYYMMDD-HHMMSS/hils_data.h5
```

統計情報の例：
- 位置・速度の平均値、標準偏差、最小値、最大値
- 制御誤差の統計
- 推力指令の統計
- 遅延統計（パケットロス数、遅延時間など）

## 🔬 実験例

### 実験1: 遅延なしベースライン

```python
CMD_DELAY = 0
CMD_JITTER = 0
SENSE_DELAY = 0
SENSE_JITTER = 0
```

### 実験2: 対称な遅延

```python
CMD_DELAY = 50
CMD_JITTER = 10
SENSE_DELAY = 50
SENSE_JITTER = 10
```

### 実験3: 非対称な遅延（sense経路が遅い）

```python
CMD_DELAY = 50
SENSE_DELAY = 150  # 3倍の遅延
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

### 新しいシミュレーター追加

1. `hils_simulation/simulators/` にファイル作成
2. `mosaik_api.Simulator` を継承
3. `hils_simulation/main_hils.py` の `sim_config` に追加
4. 接続を設定

詳細は [CLAUDE.md](CLAUDE.md) を参照してください。

## 📝 使用技術 (Technologies)

- **Python 3.9+** - プログラミング言語
- **Mosaik 3.5+** - コシミュレーションフレームワーク
- **HDF5 / h5py** - データ保存・読み込み
- **matplotlib** - データ可視化
- **numpy** - 数値計算
- **uv** - 依存関係管理
- **ruff** - リンター・フォーマッター

## 🏆 今後の拡張

- [ ] 補償機能の実装（先行送出、補間、Nowcasting）
- [ ] 6DOF版への拡張（姿勢制御）
- [x] データ収集とプロット機能（HDF5形式）
- [x] データ解析スクリプト（analyze_data.py）
- [ ] リアルタイムモニタリングダッシュボード
- [ ] 実機制御器との統合
- [ ] HDF5データ圧縮オプション

## 📚 参考

- [hils_simulation/docs/README.md](hils_simulation/docs/README.md) - 詳細なガイド
- [hils_simulation/docs/DATA_COLLECTION.md](hils_simulation/docs/DATA_COLLECTION.md) - データ収集の詳細
- [CLAUDE.md](CLAUDE.md) - AI開発ガイド

## 📜 ライセンス

このプロジェクトはMITライセンスの下で公開されています。

## 🆕 更新履歴

- **v3.0** (2025-10-16) - プロジェクト構造整理、hils_simulationに統合
- **v2.2** (2025-10-13) - コマンドパッケージ化、データフロー改善
- **v2.1** - データ収集機能追加
- **v2.0** - 1DOF HILS実装完了
- **v1.0** - 初期プロトタイプ
