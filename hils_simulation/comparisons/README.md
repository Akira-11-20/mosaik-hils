# Comparison Scripts

このディレクトリには、異なるシミュレーション方式の結果を比較するためのスクリプトが含まれています。

## スクリプト一覧

### 1. compare_all.py
3つのシミュレーション方式（HILS、RT、Pure Python）の制御性能を比較・可視化します。

**使用方法:**
```bash
cd hils_simulation/comparisons
uv run python compare_all.py \
  ../results/YYYYMMDD-HHMMSS/hils_data.h5 \
  ../results_rt/YYYYMMDD-HHMMSS/hils_data.h5 \
  ../results_pure/YYYYMMDD-HHMMSS/hils_data.h5
```

**比較対象:**
- HILS: 通信遅延あり（Mosaikベース）
- RT: 通信遅延なし（Mosaikベース）
- Pure Python: Mosaikなしの理想的なシミュレーション

**出力:**
- 位置、速度、制御入力、制御誤差の3-way比較プロット
- 制御性能指標の統計比較（RMS誤差、最大誤差、整定時間など）
- 結果は `comparison_results/` に保存

---

### 2. compare_hils_rt.py
HILSシステム（通信遅延あり）とRTシステム（通信遅延なし）の制御性能を比較します。

**使用方法:**
```bash
cd hils_simulation/comparisons
uv run python compare_hils_rt.py \
  ../results/YYYYMMDD-HHMMSS/hils_data.h5 \
  ../results_rt/YYYYMMDD-HHMMSS/hils_data.h5
```

**比較対象:**
- HILS: 通信遅延あり
- RT: 通信遅延なし

**出力:**
- 位置、速度、制御入力、制御誤差の比較プロット
- 制御性能指標の差分比較
- 結果は `comparison_results/` に保存

---

### 3. compare_pure_rt.py
Pure Python（Mosaikなし）とRT（Mosaikベース）を比較し、Mosaikフレームワークのオーバーヘッドを評価します。

**使用方法:**
```bash
cd hils_simulation/comparisons
uv run python compare_pure_rt.py \
  ../results_pure/YYYYMMDD-HHMMSS/hils_data.h5 \
  ../results_rt/YYYYMMDD-HHMMSS/hils_data.h5
```

**比較対象:**
- Pure Python: Mosaikなしの素のPythonシミュレーション
- RT (Mosaik): Mosaikベースだが通信遅延なし

**評価項目:**
- Mosaikフレームワークによる性能への影響
- 制御性能の違い（RMS誤差、オーバーシュート、整定時間）
- 両方とも通信遅延なし、制御周期10msで条件を揃えて評価

**出力:**
- 位置、速度、制御入力、制御誤差の比較プロット
- Mosaikオーバーヘッドの定量評価
- 結果は `comparison_results/` に保存

---

### 4. compare_positions.py
2つのHDF5ファイルから位置データを読み込んで比較プロットを作成します。

**使用方法:**
```bash
cd hils_simulation/comparisons
uv run python compare_positions.py \
  ../results/YYYYMMDD-HHMMSS/hils_data.h5 \
  ../results/YYYYMMDD-HHMMSS/hils_data.h5 \
  --output comparison_results
```

**出力:**
- 位置データの重ね合わせプロット
- 統計情報（最終位置、最大位置差分など）
- デフォルトは `comparison_results/` に保存

---

## 出力ディレクトリのカスタマイズ

すべてのスクリプトは `--output` オプションで出力先を変更できます：

```bash
uv run python compare_all.py \
  ../results/xxx/hils_data.h5 \
  ../results_rt/xxx/hils_data.h5 \
  ../results_pure/xxx/hils_data.h5 \
  --output /path/to/custom/output
```

## 出力ファイル

各スクリプトは以下のファイルを生成します：

- **PNG画像**: 比較プロット（300dpi）
- **JSONファイル**: 性能指標の数値データ

### compare_all.py の出力例
```
comparison_results/
├── comparison_all.png
└── comparison_all_metrics.json
```

### compare_hils_rt.py の出力例
```
comparison_results/
├── comparison_hils_rt.png
└── comparison_metrics.json
```

### compare_pure_rt.py の出力例
```
comparison_results/
├── comparison_pure_rt.png
└── comparison_pure_rt_metrics.json
```

### compare_positions.py の出力例
```
comparison_results/
└── position_comparison.png
```

---

## 注意事項

1. **データファイルのパス**:
   - `comparisons/` ディレクトリから実行する場合、親ディレクトリのデータにアクセスするため `../` を使用してください
   - 例: `../results/20251017-123456/hils_data.h5`

2. **HDF5データ形式**:
   - すべてのスクリプトは `/data` グループ内のデータセットを読み込みます
   - 古い形式（トップレベルのデータセット）にも対応しています

3. **必要な依存関係**:
   - h5py
   - matplotlib
   - numpy
   - すべて `uv run python` で自動的にインストールされます

4. **実行場所**:
   - `comparisons/` ディレクトリ内から実行することを推奨
   - 他の場所から実行する場合は、パスを適切に調整してください

---

## 性能指標の説明

各スクリプトは以下の制御性能指標を計算します：

- **RMS Error**: 制御誤差の二乗平均平方根
- **Max Error**: 最大制御誤差
- **Overshoot**: 目標位置を超えた量
- **Settling Time**: 誤差が5%以内に収まる時刻
- **Final Error**: 最後の10%区間の平均誤差

---

## トラブルシューティング

### エラー: "HDF5 file not found"
- ファイルパスが正しいか確認してください
- `comparisons/` ディレクトリから実行している場合は `../` を使用してください

### エラー: "No key found with suffix"
- HDF5ファイルの形式が異なる可能性があります
- `h5dump` コマンドでファイル構造を確認してください

### グラフが表示されない
- スクリプトはPNG画像として保存します（画面表示はしません）
- 出力ディレクトリを確認してください
