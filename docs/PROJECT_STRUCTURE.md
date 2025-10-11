# Project Structure

このドキュメントでは、Mosaik HILSプロジェクトのディレクトリ構造とファイル配置について説明します。

## 📁 ディレクトリ構造

```
mosaik-hils/
├── src/                          # ソースコード
│   ├── __init__.py
│   ├── simulators/               # Mosaikシミュレーター群
│   │   ├── __init__.py
│   │   ├── numerical_simulator.py    # 数値シミュレーター
│   │   ├── hardware_simulator.py     # ハードウェアシミュレーター
│   │   ├── delay_simulator.py        # 遅延シミュレーター
│   │   └── data_collector.py         # データ収集器
│   └── webvis/                   # WebVis関連
│       ├── __init__.py
│       ├── customize_webvis.py       # WebVisカスタマイズ
│       ├── custom_webvis.py          # カスタムWebVisシミュレーター
│       └── local_webvis.py           # ローカルWebVis（非推奨）
├── scripts/                      # 実行スクリプト
│   ├── setup_webvis.py              # WebVis設定スクリプト
│   └── run_simulation.py            # シミュレーション実行スクリプト
├── docs/                         # ドキュメント
│   ├── PROJECT_STRUCTURE.md         # このファイル
│   └── WEBVIS_CUSTOMIZATION.md      # WebVisカスタマイズガイド
├── logs/                         # シミュレーション実行ログ
│   └── YYYYMMDD-HHMMSS/             # タイムスタンプ付きログディレクトリ
├── mosaik_web_local/             # ローカルWebVisコピー（実験用）
├── main.py                       # メインシミュレーションファイル
├── CLAUDE.md                     # AI開発ガイド
├── README.md                     # プロジェクト概要
├── pyproject.toml               # Python依存関係管理
└── uv.lock                      # 依存関係ロックファイル
```

## 📄 ファイルの役割

### メインファイル
- **`main.py`** - シミュレーションのメインエントリーポイント
  - 全シミュレーターの設定と起動
  - WebVisカスタマイズの自動適用
  - エンティティ接続とデータフロー定義

### Simulatorsパッケージ (`src/simulators/`)
- **`numerical_simulator.py`** - 正弦波信号生成
- **`hardware_simulator.py`** - ハードウェアI/Oシミュレーション
- **`delay_simulator.py`** - 通信遅延・ジッター・パケットロス
- **`data_collector.py`** - HDF5データ保存とリアルタイム表示

### WebVisパッケージ (`src/webvis/`)
- **`customize_webvis.py`** - WebVisカスタマイズの核心機能
- **`custom_webvis.py`** - 独自WebVisシミュレーター実装
- **`local_webvis.py`** - 実験的ローカル版（非推奨）

### Scriptsディレクトリ (`scripts/`)
- **`setup_webvis.py`** - WebVisセットアップ・メンテナンス
- **`run_simulation.py`** - 高度なシミュレーション実行オプション

### Docsディレクトリ (`docs/`)
- **`PROJECT_STRUCTURE.md`** - このファイル
- **`WEBVIS_CUSTOMIZATION.md`** - WebVisカスタマイズ詳細ガイド

## 🚀 使用方法

### 基本実行
```bash
# 通常のシミュレーション実行
uv run python main.py

# WebVisなしで実行
SKIP_MOSAIK_WEBVIS=1 uv run python main.py
```

### スクリプト経由での実行
```bash
# 高度なオプション付き実行
python scripts/run_simulation.py --steps 100 --rt-factor 0.1

# WebVisカスタマイズ管理
python scripts/setup_webvis.py --apply
python scripts/setup_webvis.py --check
python scripts/setup_webvis.py --restore
```

### 開発者向け実行
```bash
# パッケージ更新後のWebVisカスタマイズ再適用
uv sync
python scripts/setup_webvis.py --apply

# 特定シミュレーター単体テスト
uv run python -m src.simulators.delay_simulator
```

## 🔧 開発ガイドライン

### 新しいシミュレーター追加
1. `src/simulators/` に新しい `.py` ファイルを作成
2. `mosaik_api.Simulator` を継承したクラスを実装
3. `main.py` の `sim_config` に設定を追加
4. 必要に応じて接続設定を追加

### WebVisカスタマイズ拡張
1. `src/webvis/customize_webvis.py` の `_apply_customizations()` を編集
2. HTMLの変更は文字列置換で実装
3. 新機能はJavaScript/CSSで追加

### ドキュメント更新
- `docs/` ディレクトリに新しいMarkdownファイルを追加
- `README.md` の目次を更新
- `CLAUDE.md` に開発ノートを追記

## 🏗️ アーキテクチャ原則

### パッケージ分離
- **Simulators**: Mosaikシミュレーターロジック
- **WebVis**: UI/可視化関連機能
- **Scripts**: 実行・メンテナンス機能
- **Docs**: ドキュメント

### 依存関係管理
- **uv**: Python依存関係管理
- **src**: パッケージとしてのインポート
- **相対インポート**: パッケージ内部での利用

### データフロー
```
NumericalSim → DelaySim → HardwareSim
     ↓           ↓          ↓
     DataCollector ← ← ← ← ←
            ↓
         HDF5 File
```

この構造により、機能別の明確な分離と保守性の向上を実現しています。