# HILS Simulation - Quick Start Guide

## 概要

HILS Simulationは、シナリオベースのアーキテクチャを採用し、複数のシミュレーションモードを統一されたインターフェースで実行できるフレームワークです。

## クイックスタート

### 基本的な使い方

```bash
# ヘルプを表示
uv run python main.py --help

# HILSシナリオを実行（デフォルト）
uv run python main.py

# または明示的に
uv run python main.py hils

# RTシナリオを実行（通信遅延なし）
uv run python main.py rt

# 逆補償シナリオを実行
uv run python main.py inverse_comp

# Pure Pythonシナリオを実行（Mosaikなし）
uv run python main.py pure_python
```

### ショートカット（1文字）

シナリオ名は1文字のショートカットでも実行できます：

```bash
uv run python main.py h   # HILS
uv run python main.py r   # RT
uv run python main.py i   # Inverse Compensation
uv run python main.py p   # Pure Python

# フルネームでも実行可能
uv run python main.py hils
uv run python main.py rt
uv run python main.py inverse_comp
uv run python main.py pure_python
```

### パラメータのカスタマイズ

`.env`ファイルを編集してパラメータを変更できます:

```bash
# .env
SIMULATION_TIME=5.0          # シミュレーション時間 [s]
TIME_RESOLUTION=0.0001       # 時間分解能 [s]
CMD_DELAY=20                 # コマンド遅延 [ms]
SENSE_DELAY=30               # センシング遅延 [ms]
KP=15.0                      # 比例ゲイン
KD=5.0                       # 微分ゲイン
TARGET_POSITION=5.0          # 目標位置 [m]
ENABLE_INVERSE_COMP=True     # 逆補償を有効化
```

## 利用可能なシナリオ

### 1. HILS (Hardware-in-the-Loop Simulation)
- **用途**: 通信遅延を含む完全なHILS構成
- **特徴**: cmd/sense両経路で遅延・ジッタ・パケットロスを模擬
- **出力**: `results/YYYYMMDD-HHMMSS/`

### 2. RT (Real-Time)
- **用途**: 通信遅延なしのベースライン比較
- **特徴**: 直接接続、理想的な制御ループ
- **出力**: `results_rt/YYYYMMDD-HHMMSS/`

### 3. InverseComp (Inverse Compensation)
- **用途**: 逆補償を用いた遅延補償の評価
- **特徴**: コマンド経路に逆補償器を挿入
- **出力**: `results/YYYYMMDD-HHMMSS_inverse_comp/`

### 4. PurePython
- **用途**: Mosaikフレームワークのオーバーヘッド評価
- **特徴**: 純粋なPython実装、最小限のオーバーヘッド
- **出力**: `results_pure/YYYYMMDD-HHMMSS/`

## ファイル構成

```
hils_simulation/
├── config/                       # 設定管理
│   ├── parameters.py            # パラメータ管理
│   └── sim_config.py            # シミュレーター設定
├── scenarios/                    # シナリオ実装
│   ├── base_scenario.py         # 基底クラス
│   ├── hils_scenario.py         # HILS
│   ├── rt_scenario.py           # RT
│   ├── inverse_comp_scenario.py # 逆補償
│   └── pure_python_scenario.py  # Pure Python
├── main.py                       # エントリーポイント
├── archive/                      # v1実装のアーカイブ
│   ├── main_hils.py             # (旧) HILS
│   ├── main_hils_rt.py          # (旧) RT
│   ├── main_hils_with_inverse_comp.py  # (旧) 逆補償
│   └── main_pure_python.py      # (旧) Pure Python
├── V2_ARCHITECTURE.md            # アーキテクチャ詳細
└── V2_README.md                  # このファイル
```

## v1からの移行

v1の`main_*.py`ファイルは`archive/`ディレクトリに移動されました。参照用として保持されていますが、今後の開発は現在のシナリオベースのアーキテクチャで行うことを推奨します。

v1ファイルを使用する場合:

```bash
# v1（アーカイブ版）
cd archive
uv run python main_hils.py
uv run python main_hils_rt.py
uv run python main_hils_with_inverse_comp.py
uv run python main_pure_python.py
```

## プログラムからの利用

```python
from config.parameters import SimulationParameters
from scenarios import HILSScenario

# パラメータのカスタマイズ
params = SimulationParameters.from_env()
params.simulation_time = 10.0
params.control.kp = 20.0

# シナリオの実行
scenario = HILSScenario(params)
scenario.run()
```

## 結果の分析

```bash
# データ分析スクリプトを実行
uv run python analyze_data.py results/YYYYMMDD-HHMMSS/hils_data.h5 --save-plots
```

## トラブルシューティング

### モジュールが見つからないエラー

```bash
# hils_simulation/ディレクトリから実行してください
cd hils_simulation
uv run python main.py hils
```

### パラメータが反映されない

`.env`ファイルを編集後、シミュレーションを再実行してください。環境変数は起動時に読み込まれます。

## さらに詳しく

- **アーキテクチャ詳細**: [V2_ARCHITECTURE.md](V2_ARCHITECTURE.md)
- **実装サマリー**: [docs/SUMMARY.md](docs/SUMMARY.md)
- **データ収集**: [docs/DATA_COLLECTION.md](docs/DATA_COLLECTION.md)

## 次のステップ

1. 各シナリオを実行して結果を比較
2. パラメータを変更して遅延の影響を評価
3. 新しいシナリオの追加（詳細は[V2_ARCHITECTURE.md](V2_ARCHITECTURE.md)参照）

---

**Note**: フィードバックやバグレポートは大歓迎です。
