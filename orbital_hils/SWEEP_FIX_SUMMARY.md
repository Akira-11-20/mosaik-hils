# Parameter Sweep Fix Summary

## 問題の診断

ユーザーが報告した「軌道が正しく伝播していない」問題を調査した結果、以下の原因が特定されました：

### 1. Controller Type未指定の問題

**問題**:
- `.env`ファイルで`CONTROLLER_TYPE=hohmann`が設定されていた
- しかし、`run_parameter_sweep.py`が常に`OrbitalScenario`をハードコードで使用していた
- このため、Hohmann遷移制御が実行されず、自由軌道運動のみが行われていた

**結果**:
- 軌道高度が一定(408 km ± 0.014 km)のまま維持された
- 推力がほぼゼロ(0.004-0.033 N)でノイズのみ
- 軌道は正しく伝播しているが、制御が実行されていなかった

### 2. パラメータ設定の問題

**問題**: `.env`ファイルの63行目
```bash
HOHMANN_INITIAL_ALTITUDE_KM=ALTITUDE_KM   # ❌ 文字列として解釈される
```

**修正後**:
```bash
HOHMANN_INITIAL_ALTITUDE_KM=408.0         # ✅ 数値として正しく設定
```

## 実施した修正

### 1. `run_parameter_sweep.py`の更新

**変更内容**:
```python
# Before: 常にOrbitalScenarioを使用
scenario = OrbitalScenario(config=CONFIG_ISS)

# After: CONTROLLER_TYPEに基づいてシナリオを選択
controller_type = os.environ.get("CONTROLLER_TYPE", "zero")
if controller_type == "hohmann":
    scenario = HohmannScenario(config=CONFIG_ISS)
else:
    scenario = OrbitalScenario(config=CONFIG_ISS)
```

**効果**:
- 環境変数`CONTROLLER_TYPE`に基づいて適切なシナリオクラスを自動選択
- Hohmann遷移、PD制御、自由軌道運動を柔軟に切り替え可能

### 2. `.env`ファイルの修正

**変更内容**:
```bash
# Before
HOHMANN_INITIAL_ALTITUDE_KM=ALTITUDE_KM

# After
HOHMANN_INITIAL_ALTITUDE_KM=408.0
```

**効果**:
- Hohmann遷移の初期高度が正しく数値として読み込まれる

### 3. 新規スイープスクリプトの追加

**ファイル**: `scripts/sweeps/examples/sweep_hohmann_inverse_comp.py`

**目的**:
- Hohmann遷移におけるInverse Compensationの効果を検証
- Plant time constantとON/OFFの組み合わせをテスト

**スイープ設定**:
```python
sweep_params = {
    "CONTROLLER_TYPE": ["hohmann"],       # Hohmann遷移を明示的に指定
    "PLANT_TIME_CONSTANT": [10.0, 50.0],  # Plant lag
    "INVERSE_COMPENSATION": [True, False], # Compensation ON/OFF
    "INVERSE_COMPENSATION_GAIN": [10.0],   # Fixed gain
    "SIMULATION_TIME": [3000.0],           # ~50分のシミュレーション
    "HOHMANN_INITIAL_ALTITUDE_KM": [408.0],
    "HOHMANN_TARGET_ALTITUDE_KM": [500.0],
    "HOHMANN_START_TIME": [100.0],
}
```

### 4. ドキュメントの更新

**ファイル**: `scripts/sweeps/README.md`

**追加内容**:
- Controller Typeの選択に関する説明
- 新しいHohmannスイープ例の追加
- 重要な注意事項セクションの追加

## 使用方法

### 1. Dry-runで設定確認

```bash
cd orbital_hils
uv run python scripts/sweeps/examples/sweep_hohmann_inverse_comp.py --dry-run
```

### 2. スイープ実行

```bash
uv run python scripts/sweeps/examples/sweep_hohmann_inverse_comp.py
```

### 3. 結果の比較可視化

スイープ完了後、自動的に比較可視化が生成されます：
- `altitude_thrust_comparison.png`
- `3d_trajectory_comparison.png`
- `trajectory_interactive.html`
- `phase_comparison.png`

## 期待される動作

修正後のシステムでは：

1. **Hohmann遷移が正しく実行される**
   - t=100sで第一バーン開始
   - 高度が408kmから500kmへ上昇
   - 目標高度到達後、第二バーン実行

2. **Inverse Compensationの効果が可視化される**
   - Plant lagによる遅れが補償される
   - ON/OFF比較で効果を定量評価可能

3. **スイープ結果が自動的に整理される**
   - タイムスタンプ付きディレクトリに保存
   - サマリーファイル生成
   - 比較プロット自動生成

## トラブルシューティング

### Q: まだ軌道が変化しない

A: 以下を確認してください：
- `.env`で`CONTROLLER_TYPE=hohmann`が設定されているか
- スイープパラメータに`"CONTROLLER_TYPE": ["hohmann"]`が含まれているか
- `HOHMANN_TARGET_ALTITUDE_KM`が初期高度と異なる値か

### Q: Inverse Compensationが効いていない

A: 以下を確認してください：
- `INVERSE_COMPENSATION=True`が設定されているか
- `INVERSE_COMPENSATION_GAIN`が適切な値か(推奨: 10.0以上)
- `PLANT_TIME_CONSTANT`がゼロでないか

## 次のステップ

1. 短時間テスト実行で動作確認
2. 本番パラメータでフルスイープ実行
3. 結果分析とInverse Compensation効果の評価
