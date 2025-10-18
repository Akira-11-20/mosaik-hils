# Archive - v1 Implementation & Legacy Tools

このディレクトリには、v2実装への移行前の元のファイルと、現在使用されていない古いツールが保存されています。

## アーカイブされたファイル

### メインスクリプト (v1)
- **main_hils.py** - HILS基本版（通信遅延あり）
- **main_hils_rt.py** - RT版（通信遅延なし、Mosaik使用）
- **main_hils_with_inverse_comp.py** - HILS + 逆補償版
- **main_pure_python.py** - Pure Python版（Mosaikなし）

### 比較ツール（旧版）
- **comparisons/** - 個別の比較スクリプト群
  - `compare_all.py` - 3-wayシナリオ比較
  - `compare_hils_rt.py` - HILS vs RT比較
  - `compare_pure_rt.py` - Pure Python vs RT比較
  - `compare_positions.py` - 位置データ比較
- **comparison_results/** - 比較結果の出力先

**Note**: 現在は[visualize_results.py](../visualize_results.py)で統合的な可視化が可能なため、これらの個別スクリプトは使用されていません。

## v2への移行理由

これらのファイルは以下の理由でアーカイブされました：

1. **コードの重複**: 4つのファイルで同じロジックが繰り返されていた
   - パラメータ読み込み（`get_env_float`など）
   - シミュレーション設定保存（`save_simulation_config`）
   - グラフ生成ロジック
   - エンティティ作成・接続の基本パターン

2. **保守性の問題**: 新しいパラメータを追加する際、4ファイル全てを修正する必要があった

3. **拡張性の問題**: 新しいシナリオを追加する際、既存ファイルをコピー＆ペーストする必要があった

## v2での対応

v2では以下のような構造に再編成されました：

```
hils_simulation/
├── config/
│   ├── parameters.py          # パラメータ管理を一元化
│   └── sim_config.py          # シミュレーター設定を一元化
├── scenarios/
│   ├── base_scenario.py       # 共通ロジックを基底クラスに
│   ├── hils_scenario.py       # HILSシナリオ（差分のみ）
│   ├── rt_scenario.py         # RTシナリオ（差分のみ）
│   ├── inverse_comp_scenario.py  # 逆補償シナリオ（差分のみ）
│   └── pure_python_scenario.py   # Pure Pythonシナリオ（差分のみ）
└── main_v2.py                 # 統一エントリーポイント
```

## v1ファイルの使い方（参考用）

これらのファイルは参照用として保持されていますが、直接実行することも可能です：

```bash
# アーカイブディレクトリから実行
cd archive
uv run python main_hils.py
uv run python main_hils_rt.py
uv run python main_hils_with_inverse_comp.py
uv run python main_pure_python.py
```

ただし、今後の開発・メンテナンスはv2をベースに行うことを推奨します。

## v2の使い方

```bash
# v2では統一されたインターフェースで実行
uv run python main.py hils          # main_hils.py と同等
uv run python main.py rt            # main_hils_rt.py と同等
uv run python main.py inverse_comp  # main_hils_with_inverse_comp.py と同等
uv run python main.py pure_python   # main_pure_python.py と同等
```

詳細は [README_scenarios.md](../README_scenarios.md) および [V2_ARCHITECTURE.md](../V2_ARCHITECTURE.md) を参照してください。

## 比較ツールの代替

旧版の比較スクリプトの代わりに、現在は以下を使用してください：

```bash
# 統合的な可視化ツール（推奨）
cd hils_simulation
uv run python visualize_results.py results/YYYYMMDD-HHMMSS/hils_data.h5

# 複数シナリオの比較
uv run python visualize_results.py \
  results/YYYYMMDD-HHMMSS/hils_data.h5 \
  results_rt/YYYYMMDD-HHMMSS/hils_data.h5 \
  results_pure/YYYYMMDD-HHMMSS/hils_data.h5
```

`visualize_results.py`は、旧版の比較スクリプトの機能を統合し、より使いやすいインターフェースで提供しています。

---

**アーカイブ日時**: 2025-10-18
**理由**:
- メインスクリプト: v2実装への移行
- 比較ツール: visualize_results.pyへの統合
