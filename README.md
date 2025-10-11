# Mosaik HILS (Hardware-in-the-Loop Simulation)

## 概要 (Overview)

このプロジェクトは、MosaikフレームワークをベースとしたHILS（Hardware-in-the-Loop Simulation）システムです。数値シミュレーション、ハードウェアインターフェース、**通信遅延モデリング**、データ収集、および**カスタマイズ可能なWebベース可視化**を統合したコシミュレーション環境を提供します。

This project implements a HILS (Hardware-in-the-Loop Simulation) system based on the Mosaik framework. It provides a co-simulation environment that integrates numerical simulation, hardware interfaces, **communication delay modeling**, data collection, and **customizable web-based visualization**.

## ✨ 主な機能 (Key Features)

- 🔢 **数値シミュレーション** - 正弦波信号生成
- ⚙️ **ハードウェアシミュレーション** - センサー・アクチュエーターI/O
- 🔄 **通信遅延モデリング** - ジッター・パケットロス対応
- 📊 **リアルタイムデータ収集** - HDF5形式での保存
- 🌐 **カスタマイズ可能WebVis** - 独自統計表示対応

## 📁 プロジェクト構造 (Project Structure)

```
mosaik-hils/
├── src/                      # ソースコード
│   ├── simulators/           # Mosaikシミュレーター群
│   │   ├── numerical_simulator.py
│   │   ├── hardware_simulator.py
│   │   ├── delay_simulator.py     # 🆕 遅延モデリング
│   │   └── data_collector.py
│   └── webvis/               # WebVis関連
│       ├── customize_webvis.py    # 🆕 WebVisカスタマイズ
│       └── custom_webvis.py
├── scripts/                  # 実行スクリプト
│   ├── run_simulation.py
│   └── setup_webvis.py
├── docs/                     # ドキュメント
│   ├── PROJECT_STRUCTURE.md
│   └── WEBVIS_CUSTOMIZATION.md
├── logs/                     # シミュレーション結果
└── main.py                   # メインファイル
```

詳細な構造については [`docs/PROJECT_STRUCTURE.md`](docs/PROJECT_STRUCTURE.md) を参照してください。

## 🚀 クイックスタート (Quick Start)

### 1. 環境構築
```bash
# 依存関係のインストール
uv sync

# WebVisカスタマイズの初期設定
python scripts/setup_webvis.py --apply
```

### 2. シミュレーション実行
```bash
# 基本実行
uv run python main.py

# スクリプト経由（オプション指定可能）
python scripts/run_simulation.py --steps 100 --delay 5.0 --jitter 2.0
```

### 3. 結果確認
- **WebVis**: http://localhost:8002 (カスタム統計表示付き)
- **ログファイル**: `logs/YYYYMMDD-HHMMSS/simulation_data.h5`

## 🔧 システム構成 (System Architecture)

### データフロー
```
NumericalSim → DelayNode → HardwareSim
     ↓            ↓           ↓
     DataCollector ← ← ← ← ← ← ←
            ↓
       HDF5 File + WebVis
```

### 主要コンポーネント

1. **NumericalSimulator** (`src/simulators/numerical_simulator.py`)
   - 正弦波信号の生成
   - パラメータ調整可能な数値モデル

2. **DelaySimulator** (`src/simulators/delay_simulator.py`) 🆕
   - 通信遅延のモデリング
   - ガウシアンジッター対応
   - パケットロス・順序制御

3. **HardwareSimulator** (`src/simulators/hardware_simulator.py`)
   - センサー読取りシミュレーション
   - アクチュエーター制御シミュレーション

4. **DataCollector** (`src/simulators/data_collector.py`)
   - HDF5形式でのデータ保存
   - リアルタイムコンソール出力

5. **CustomWebVis** (`src/webvis/`) 🆕
   - WebVisのカスタマイズ機能
   - 遅延統計の表示
   - uv sync後の永続化対応

## 🎛️ 実行オプション (Execution Options)

### 基本実行
```bash
# 通常実行
uv run python main.py

# WebVisなし
SKIP_MOSAIK_WEBVIS=1 uv run python main.py
```

### 高度な実行オプション
```bash
# スクリプト経由
python scripts/run_simulation.py \
  --steps 200 \           # シミュレーションステップ数
  --rt-factor 0.1 \       # リアルタイムファクター
  --delay 5.0 \           # 基本遅延（ステップ）
  --jitter 1.5 \          # ジッター標準偏差
  --packet-loss 0.005 \   # パケットロス率
  --no-webvis             # WebVis無効化
```

## 📊 データ分析 (Data Analysis)

### HDF5データ構造
```python
import h5py

with h5py.File('logs/YYYYMMDD-HHMMSS/simulation_data.h5', 'r') as f:
    steps = f['steps']
    print(f"Time: {steps['time'][:]}")
    print(f"Numerical Output: {steps['output_NumericalSim-0.NumSim_0'][:]}")
    print(f"Delayed Output: {steps['delayed_output_DelaySim-0.DelayNode_0'][:]}")
    print(f"Hardware Sensor: {steps['sensor_value_HardwareSim-0.HW_0'][:]}")
```

### 統計情報
遅延ノードの詳細統計は、WebVis右上パネルまたはHDF5の`stats`フィールドで確認できます。

## 🎨 WebVisカスタマイズ (WebVis Customization)

### 自動カスタマイズ
```bash
# main.py実行時に自動適用
uv run python main.py

# 手動適用
python scripts/setup_webvis.py --apply

# 状態確認
python scripts/setup_webvis.py --check

# 元に戻す
python scripts/setup_webvis.py --restore
```

### カスタマイズ内容
- タイトル変更: "mosaik-web (HILS Custom)"
- 遅延統計パネル: 右上に詳細統計表示
- uv sync対応: パッケージ更新後も自動再適用

詳細は [`docs/WEBVIS_CUSTOMIZATION.md`](docs/WEBVIS_CUSTOMIZATION.md) を参照してください。

## 🔬 開発ガイド (Development Guide)

### 新しいシミュレーター追加
1. `src/simulators/` にファイル作成
2. `mosaik_api.Simulator` を継承
3. `main.py` の `sim_config` に追加

### WebVisカスタマイズ拡張
1. `src/webvis/customize_webvis.py` を編集
2. HTML/CSS/JavaScript追加

### 詳細情報
- 📄 [`docs/PROJECT_STRUCTURE.md`](docs/PROJECT_STRUCTURE.md) - 詳細な構造説明
- 🤖 [`CLAUDE.md`](CLAUDE.md) - AI開発ガイド

## 📝 使用技術 (Technologies)

- **Python 3.9+** - プログラミング言語
- **Mosaik 3.5+** - コシミュレーションフレームワーク
- **HDF5** - データ保存形式
- **uv** - 依存関係管理
- **D3.js** - WebVis可視化
- **WebSocket** - リアルタイム通信

## 📈 パフォーマンス

- **リアルタイムファクター**: 0.5（2倍速実行）
- **遅延精度**: ステップレベル（設定可能）
- **データレート**: 毎秒数百ポイント
- **メモリ使用量**: 通常100MB以下

## 🤝 コントリビューション

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📜 ライセンス

このプロジェクトはMITライセンスの下で公開されています。

## 🆕 更新履歴

- **v2.0** - 通信遅延モデリング機能追加
- **v2.1** - WebVisカスタマイズ機能追加
- **v2.2** - プロジェクト構造整理とスクリプト化