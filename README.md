# Mosaik HILS (Hardware-in-the-Loop Simulation)

## 概要 (Overview)

このプロジェクトは、MosaikフレームワークをベースとしたHILS（Hardware-in-the-Loop Simulation）システムです。数値シミュレーション、ハードウェアインターフェース、データ収集、およびWebベースの可視化を統合したコシミュレーション環境を提供します。

This project implements a HILS (Hardware-in-the-Loop Simulation) system based on the Mosaik framework. It provides a co-simulation environment that integrates numerical simulation, hardware interfaces, data collection, and web-based visualization.

## システム構成 (System Architecture)

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Numerical      │    │  Hardware       │    │  Data           │
│  Simulator      │◄──►│  Simulator      │◄──►│  Collector      │
│                 │    │                 │    │                 │
│ - 正弦波生成      │    │ - センサー読取り │    │ - JSON保存      │
│ - 数学的モデル   │    │ - アクチュエータ │    │ - リアルタイム  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │   WebVis        │
                    │  Visualization  │
                    │                 │
                    │ - リアルタイム  │
                    │   グラフ表示    │
                    │ - ポート8002    │
                    └─────────────────┘
```

## ファイル構成 (File Structure)

```
mosaik-hils/
├── main.py                 # メインシナリオファイル (Main scenario file)
├── numerical_simulator.py  # 数値シミュレーター (Numerical simulator)
├── hardware_simulator.py   # ハードウェアシミュレーター (Hardware simulator)
├── data_collector.py       # データ収集器 (Data collector)
├── pyproject.toml          # プロジェクト設定 (Project configuration)
├── uv.lock                 # 依存関係ロック (Dependency lock file)
└── README.md               # このファイル (This file)
```

## 各コンポーネントの説明 (Component Description)

### 1. メインシナリオ (`main.py`)
- 全シミュレーターの統合管理
- エンティティ間の接続設定
- WebVis可視化の設定
- シミュレーション実行制御

### 2. 数値シミュレーター (`numerical_simulator.py`)
- **機能**: 正弦波生成による数学的モデル
- **入力**: ハードウェアからのフィードバック（オプション）
- **出力**: `sin(time * step_size * 0.1)` による正弦波
- **用途**: 制御信号やセンサー信号のシミュレーション

### 3. ハードウェアシミュレーター (`hardware_simulator.py`)
- **機能**: 物理デバイスとのインターフェースをシミュレート
- **センサー**: 0.5V～1.5Vのランダム値生成
- **アクチュエータ**: 数値シミュレーターからのコマンド受信
- **接続**: シリアル接続のシミュレーション

### 4. データコレクター (`data_collector.py`)
- **機能**: 全シミュレーターからのデータ収集
- **保存形式**: HDF5形式（`simulation_data.h5`）
- **表示**: リアルタイムコンソール出力
- **データ構造**: 時刻 + 属性_ソースID の形式

## 実行方法 (How to Run)

### 必要条件 (Prerequisites)
- Python 3.9以上
- uv（Pythonパッケージマネージャー）

### インストール (Installation)
```bash
# リポジトリをクローン
git clone <repository-url>
cd mosaik-hils

# 依存関係をインストール
uv sync
```

### 実行 (Execution)
```bash
# シミュレーション実行
uv run python main.py
```

### 可視化アクセス (Visualization Access)
シミュレーション実行後、以下のURLにアクセス:
```
http://localhost:8002
```

## 出力ファイル (Output Files)

### `simulation_data.h5`
収集された時系列データは HDF5 ファイルとして保存され、`/steps` グループ配下に
`time`, `output_NumSim_0`, `sensor_value_HW_0`, `actuator_command_HW_0` といった列名で
データセットが生成されます。Python からの読み取り例:
```python
import h5py

with h5py.File("simulation_data.h5", "r") as f:
    time = f["steps/time"][:]
    numerical = f["steps/output_NumSim_0"][:]
    sensor = f["steps/sensor_value_HW_0"][:]
```

### `execution_time.png` (オプション)
matplotlibが利用可能な場合、実行時間のプロットが生成されます。

## 設定パラメータ (Configuration Parameters)

### シミュレーション設定
- **実行時間**: 300ステップ
- **リアルタイムファクター**: 0.5（実時間より高速）
- **ステップサイズ**: 1秒

### カスタマイズ可能な値
- 数値モデル初期値: `initial_value=1.0`
- 数値モデルステップサイズ: `step_size=0.5`
- ハードウェアデバイスID: `device_id="sensor_01"`
- 接続タイプ: `connection_type="serial"`

## 開発・拡張 (Development & Extension)

### 新しいシミュレーターの追加
1. `mosaik_api.Simulator`を継承したクラスを作成
2. `meta`辞書でモデル仕様を定義
3. `init`, `create`, `step`, `get_data`メソッドを実装
4. `main.py`の`sim_config`に追加

### 実ハードウェアとの接続
`hardware_simulator.py`の以下のメソッドを実装:
- `_read_sensor_data()`: 実際のセンサーからデータ読み取り
- `_send_actuator_command()`: 実際のアクチュエーターに制御信号送信

## トラブルシューティング (Troubleshooting)

### よくある問題

**ポート8002が使用中**
```bash
# ポート使用状況確認
lsof -i :8002
# プロセス終了
kill -9 <PID>
```

**依存関係エラー**
```bash
# 依存関係を再インストール
uv sync --reinstall
```

**WebVis表示されない**
- ブラウザで `http://localhost:8002` にアクセス
- ファイアウォール設定を確認
- シミュレーション実行中かどうか確認

## ライセンス (License)

このプロジェクトはMITライセンスの下で公開されています。

## 参考資料 (References)

- [Mosaik Documentation](https://mosaik.readthedocs.io/)
- [Mosaik API Reference](https://mosaik-api.readthedocs.io/)
- [HILS概要](https://en.wikipedia.org/wiki/Hardware-in-the-loop_simulation)
