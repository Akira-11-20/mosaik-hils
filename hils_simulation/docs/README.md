# HILS Simulation - 1DOF Configuration

模擬Hardware-in-the-Loop Simulation（HILS）システムの1自由度実装。

## 📁 ディレクトリ構成

```
hils_simulation/
├── simulators/          # シミュレーターモジュール
│   ├── controller_simulator.py   # PD制御器
│   ├── plant_simulator.py        # 推力測定器（スラストスタンド）
│   ├── env_simulator.py          # 環境シミュレーター（1DOF運動）
│   ├── bridge_simulator.py       # 通信ブリッジ（遅延・ジッター）
│   └── data_collector.py         # データ収集器（HDF5）
├── main_hils.py         # メインシナリオ
├── analyze_data.py      # データ解析・可視化スクリプト
├── results/             # シミュレーション結果（自動生成）
├── DATA_COLLECTION.md   # データ収集機能の詳細
└── README.md            # このファイル
```

## 🎯 システム構成

```
Controller → Bridge(cmd) → Plant → Bridge(sense) → Env → Controller (time-shifted)
```

### データフロー詳細

1. **制御指令経路（cmd）**:
   - Controller が推力指令を生成
   - Bridge(cmd) で遅延・ジッター・パケットロスを模擬
   - Plant（推力測定器）が指令を受信

2. **測定経路（sense）**:
   - Plant が測定した推力を出力
   - Bridge(sense) で遅延・ジッター・パケットロスを模擬
   - Env（環境シミュレーター）が推力を受け取り、運動方程式を積分

3. **状態フィードバック経路**:
   - Env が位置・速度を出力
   - Controller が状態を受信（**time-shifted接続**で循環依存を解決）

## 🚀 実行方法

### 1. 依存関係の確認

```bash
# プロジェクトルートで
uv sync
```

### 2. シミュレーション実行

```bash
cd hils_simulation
uv run python main_hils.py
```

### 3. 結果の確認

実行後、`results/YYYYMMDD-HHMMSS/` ディレクトリに以下のファイルが生成されます：

- `hils_data.h5` - 全シミュレーションデータ（HDF5形式）
- `*_dataflowGraph_*.png` - データフローグラフ
- `*_executionGraph.png` - 実行順序グラフ
- `*_executiontime.png` - 各シミュレーターの実行時間

### 4. データ解析

```bash
# 統計情報とグラフを生成
uv run python analyze_data.py results/20251013-183849/hils_data.h5 --save-plots
```

詳細は [DATA_COLLECTION.md](DATA_COLLECTION.md) を参照してください。

## ⚙️ パラメータ設定

[main_hils.py](main_hils.py) の冒頭で以下のパラメータを変更できます：

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

## 📊 各シミュレーターの詳細

### 1. ControllerSimulator（制御器）

- **モデル**: PDController
- **制御則**: `F = Kp * error - Kd * velocity`
- **ステップサイズ**: 10ms（制御周期）
- **入力**: position, velocity
- **出力**: command（JSON/dict: `{thrust, duration}`）, error

### 2. PlantSimulator（推力測定器）

- **モデル**: ThrustStand
- **機能**: 推力指令を受け取り、理想的な推力を出力
- **ステップサイズ**: 1ms
- **入力**: command（JSON/dict: `{thrust, duration}`）
- **出力**: measured_thrust, status

### 3. EnvSimulator（環境シミュレーター）

- **モデル**: Spacecraft1DOF
- **運動方程式**: `F = ma`, オイラー法で積分
- **ステップサイズ**: 1ms
- **入力**: force
- **出力**: position, velocity, acceleration

### 4. BridgeSimulator（通信ブリッジ）

- **モデル**: CommBridge
- **機能**: 遅延・ジッター・パケットロスを模擬
- **ステップサイズ**: 1ms（高頻度実行）
- **入力**: input（任意のデータ）
- **出力**: delayed_output, stats

### 5. DataCollectorSimulator（データ収集器）

- **モデル**: Collector
- **機能**: 全シミュレーションデータをHDF5形式で記録
- **ステップサイズ**: 1ms
- **入力**: 全シミュレーターからの全属性（動的）
- **出力**: HDF5ファイル（`hils_data.h5`）

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

## 🛠️ 今後の拡張

- [ ] 補償機能の実装（先行送出、補間、Nowcasting）
- [ ] 6DOF版への拡張（姿勢制御）
- [x] データ収集とプロット機能（HDF5形式）
- [x] データ解析スクリプト（`analyze_data.py`）
- [ ] リアルタイムモニタリングダッシュボード
- [ ] 実機制御器との統合
- [ ] HDF5データ圧縮オプション

## 📚 参考

- 詳細な設計書: `../docs/hils_delay_compensation_plan.md`
- Mosaikガイド: `../docs/mosaik_beginner_guide.md`
