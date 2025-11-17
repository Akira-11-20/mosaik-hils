# PD Inverse Compensation Verification

シンプルなPD制御システムに対してInverse Compensationを適用し、その効果を検証するMosaikシミュレーション。

## 概要

このプロジェクトは、PD制御の応答に対してInverse Compensatorがどのように影響するかを検証するために作成されました。

### システム構成

**3つのシンプルなノード:**

1. **Plant (SimplePlant)**: 目標値（デフォルト: 10m）を出力
2. **PD Controller**: Plantの目標値に向かって位置を制御
3. **Inverse Compensator** (オプション): PD制御の現在位置に補償をかける

### アーキテクチャ

**シナリオ1: 補償なし (No Compensation)**
```
Plant (target=10) → PD Controller → (内部で位置更新)
```

**シナリオ2: 補償あり (With Compensation)**
```
Plant (target=10) → PD Controller → Inverse Compensator → PD (feedback)
                                           ↓
                                    予測位置をフィードバック
```

## Inverse Compensation の計算式

HILS/Orbital HILSと同じ式を使用：

```
y_comp[k] = gain * y[k] - (gain - 1) * y[k-1]
```

これは以下のように書き換えられます：

```
y_comp[k] = y[k] + (gain - 1) * (y[k] - y[k-1])
```

**パラメータの効果:**
- `gain > 1.0`: 積極的な予測（より先を予測）
- `gain = 1.0`: 補償なし（pass-through）
- `gain < 1.0`: ダンピング効果

## 使用方法

### インストール

親ディレクトリ（mosaik-hils）で依存関係をインストール：

```bash
cd /path/to/mosaik-hils
uv sync
```

### 実行

```bash
cd pd_inverse_verification

# デフォルトパラメータで実行 (comp_gain=1.0)
uv run python main.py

# 補償ゲインを指定して実行
uv run python main.py 1.5   # comp_gain=1.5

# ヘルプを表示
uv run python main.py --help
```

### パラメータ

`main.py` 内でパラメータを調整可能：

```python
params = {
    'sim_time': 5.0,            # シミュレーション時間 [s]
    'time_resolution': 0.001,   # Mosaikタイム分解能
    'step_size': 0.01,          # シミュレータステップサイズ [s]
    'kp': 2.0,                  # 比例ゲイン
    'kd': 0.5,                  # 微分ゲイン
    'target': 10.0,             # 目標位置 [m]
    'initial_position': 0.0,    # 初期位置 [m]
    'comp_gain': 1.0,           # 補償ゲイン
}
```

## 出力

### 結果ディレクトリ

```
results/
├── YYYYMMDD-HHMMSS_no_comp/      # 補償なしの結果
│   ├── hils_data.h5               # HDF5データ
│   └── params.json                # パラメータ
├── YYYYMMDD-HHMMSS_with_comp/    # 補償ありの結果
│   ├── hils_data.h5
│   └── params.json
└── comparison.png                 # 比較プロット
```

### 比較プロット

以下の4つのプロットが自動生成されます：

1. **Position Tracking**: 目標値と実際の位置の追従性能
2. **Tracking Error**: 追従誤差の時間変化
3. **Control Output**: PD制御の出力（速度指令）
4. **Velocity**: 速度とInverse Compensatorの出力

### 性能指標

シミュレーション終了後、以下の指標が表示されます：

- **RMSE** (Root Mean Square Error): 追従誤差の二乗平均平方根
- **Settling Time**: 2%整定時間
- **RMSE Improvement**: 補償による改善率

## ファイル構成

```
pd_inverse_verification/
├── simulators/
│   ├── __init__.py
│   ├── plant_simulator.py         # Plant (目標値出力)
│   ├── pd_controller.py           # PD制御器
│   └── inverse_compensator.py     # Inverse補償器
├── config/                         # (未使用)
├── results/                        # 結果保存ディレクトリ
├── main.py                         # メインシミュレーション
└── README.md                       # このファイル
```

## シミュレータの詳細

### Plant Simulator

- **モデル**: `SimplePlant`
- **機能**: 定数目標値を出力
- **パラメータ**: `target` (目標値)
- **出力**: `target_output` (目標値)

### PD Controller Simulator

- **モデル**: `PDController`
- **機能**: PD制御則で目標値に追従
  - 制御則: `u = Kp * e + Kd * de/dt`
  - 内部で位置を積分して更新
- **パラメータ**: `kp`, `kd`, `dt`, `initial_position`
- **入力**: `target_position` (目標位置)
- **出力**: `position`, `control_output`, `velocity`, `error`

### Inverse Compensator Simulator

- **モデル**: `InverseCompensator`
- **機能**: 一つ前の値を使って補償計算
  - 式: `y_comp[k] = gain * y[k] - (gain - 1) * y[k-1]`
- **パラメータ**: `comp_gain`, `dt`
- **入力**: `position` (現在位置)
- **出力**: `output` (補償後の位置)

## 検証内容

このシミュレーションでは以下を検証します：

1. **PD制御の基本動作**: 目標値への追従性能
2. **Inverse Compensationの効果**: 補償ゲインによる応答の変化
3. **ゲインの影響**: 異なる`comp_gain`値での挙動比較

## 注意事項

- Plantには**遅れや遅延は一切含まれません**（シンプルなpass-through）
- 1次遅れ（first-order lag）も含まれません
- Inverse Compensatorは、PD制御の**現在位置**に対してのみ作用します
- このシミュレーションは、時間遅延の補償ではなく、**制御応答の補償**を検証するものです

## 比較: HILSプロジェクトとの違い

| 項目 | pd_inverse_verification | hils_simulation/orbital_hils |
|------|-------------------------|------------------------------|
| 目的 | PD制御応答の補償検証 | 通信遅延の補償 |
| Plant | 定数出力のみ | 1次遅れ + 時間遅延 |
| 制御対象 | PD制御の位置 | 宇宙機の推力/軌道 |
| 補償対象 | 制御ループの応答 | 通信経路の遅延 |
| 補償式 | 同じ（HILS/Orbitalと同一） | 同じ |

## ライセンス

親プロジェクト（mosaik-hils）に準拠
