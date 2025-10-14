# HDF5データの時刻ベース変換ツール

このディレクトリには、HILS シミュレーションのHDF5データをステップ数ベースから時刻ベースに変換するツールが含まれています。

## スクリプト一覧

### 1. convert_to_time_based.py
HDF5データをステップ数ベースから時刻ベースに変換するメインスクリプト。

### 2. plot_time_based_data.py
変換後の時刻ベースHDF5データを可視化するスクリプト。

### 3. read_time_based_data.py
時刻ベースHDF5データを読み込むためのユーティリティスクリプト。

## 使用方法

### 基本的な変換

```bash
# 単純な変換(出力ファイル名は自動生成: 元のファイル名_time_based.h5)
uv run python convert_to_time_based.py hils_simulation/results/20251013-215104/hils_data.h5

# 出力ファイル名を指定
uv run python convert_to_time_based.py input.h5 -o output.h5
```

### 変換後のデータ構造

変換後のHDF5ファイルは以下の構造を持ちます:

```
/
├── metadata/                          # メタデータグループ
│   ├── @description                   # データの説明
│   ├── @original_file                 # 元のファイルパス
│   ├── @time_column                   # 時刻カラム名
│   ├── @num_samples                   # サンプル数
│   ├── @time_start                    # 開始時刻[s]
│   ├── @time_end                      # 終了時刻[s]
│   └── @sampling_period               # サンプリング周期[s]
│
├── data/                              # データグループ
│   ├── time_s                         # 時刻データ[s] (インデックス)
│   ├── position_EnvSim-0...           # 各種シミュレーションデータ
│   ├── velocity_EnvSim-0...
│   └── ...
│
└── statistics/                        # 統計情報グループ
    ├── position_EnvSim-0.../
    │   ├── @mean                      # 平均値
    │   ├── @std                       # 標準偏差
    │   ├── @min                       # 最小値
    │   ├── @max                       # 最大値
    │   ├── @count                     # 有効データ数
    │   └── @nan_count                 # NaN数
    └── ...
```

### データの可視化

#### 利用可能なカラムのリスト表示

```bash
uv run python plot_time_based_data.py hils_data_time_based.h5 --list
```

#### 自動選択されたカラムをプロット

```bash
# 主要なカラムを自動選択してプロット(画面表示)
uv run python plot_time_based_data.py hils_data_time_based.h5
```

#### 特定のカラムをプロット

```bash
# 完全一致または部分一致でカラムを指定
uv run python plot_time_based_data.py hils_data_time_based.h5 --columns position velocity acceleration

# パターンマッチング(Spacecraftを含むすべてのカラム)
uv run python plot_time_based_data.py hils_data_time_based.h5 --columns Spacecraft
```

#### 画像ファイルとして保存

```bash
# PNGファイルとして保存
uv run python plot_time_based_data.py hils_data_time_based.h5 --columns position velocity --output plot.png
```

## Pythonからのデータ読み込み例

変換後のHDF5ファイルは通常のh5pyで読み込めます:

```python
import h5py
import numpy as np
import matplotlib.pyplot as plt

# ファイルを開く
with h5py.File('hils_data_time_based.h5', 'r') as f:
    # メタデータを取得
    metadata = f['metadata']
    print(f"サンプリング周期: {metadata.attrs['sampling_period']}s")

    # データを取得
    data = f['data']
    time = data['time_s'][:]
    position = data['position_EnvSim-0.Spacecraft1DOF_0'][:]
    velocity = data['velocity_EnvSim-0.Spacecraft1DOF_0'][:]

    # プロット(時刻が横軸)
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(time, position)
    plt.ylabel('Position [m]')
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(time, velocity)
    plt.xlabel('Time [s]')
    plt.ylabel('Velocity [m/s]')
    plt.grid(True)

    plt.tight_layout()
    plt.show()
```

## pandasとの連携

pandasでDataFrameとして読み込む場合:

```python
import h5py
import pandas as pd

with h5py.File('hils_data_time_based.h5', 'r') as f:
    data = f['data']

    # すべてのカラムをDataFrameに変換
    df_dict = {}
    for column_name in data.keys():
        dataset = data[column_name]
        # 数値データのみ取得
        if dataset.dtype != object:
            df_dict[column_name] = dataset[:]

    df = pd.DataFrame(df_dict)

    # time_sをインデックスに設定
    df.set_index('time_s', inplace=True)

    print(df.head())
    print(f"\nデータ形状: {df.shape}")

    # CSVとして保存
    df.to_csv('hils_data.csv')
```

## 統計情報の活用

```python
import h5py

with h5py.File('hils_data_time_based.h5', 'r') as f:
    stats = f['statistics']

    # 各カラムの統計情報を表示
    for column_name in sorted(stats.keys()):
        col_stats = stats[column_name]
        print(f"\n{column_name}:")
        print(f"  平均: {col_stats.attrs['mean']:.6f}")
        print(f"  標準偏差: {col_stats.attrs['std']:.6f}")
        print(f"  範囲: [{col_stats.attrs['min']:.6f}, {col_stats.attrs['max']:.6f}]")
        print(f"  データ数: {col_stats.attrs['count']}")
```

## トラブルシューティング

### エラー: "入力ファイルが見つかりません"
- ファイルパスが正しいか確認してください
- 相対パスまたは絶対パスで指定してください

### エラー: "'time_s'カラムが見つかりません"
- 元のHDF5ファイルにtime_sカラムが存在することを確認してください
- data_collector.pyがtime_sを記録していることを確認してください

### 日本語フォントの警告
- matplotlibの日本語フォント警告は表示には影響しません
- 完全に解決したい場合は、日本語フォント(例: IPAフォント)をインストールしてください

## 変換の利点

1. **時刻ベースの解析が容易**: 横軸が時刻になるため、物理的な意味が直感的
2. **メタデータの保存**: サンプリング周期や時刻範囲などの情報が自動保存
3. **統計情報の自動計算**: 各カラムの平均、標準偏差などが事前計算済み
4. **圧縮による容量削減**: gzip圧縮によりファイルサイズを削減
5. **外部ツールとの連携**: pandas、matplotlibなど標準的なツールで簡単に読み込み可能

## ライセンス

このプロジェクトと同じライセンスが適用されます。
