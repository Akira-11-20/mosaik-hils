# Maximum Correntropy Kalman Filter (MCKF) 実装

## 📚 概要

このディレクトリには、**Maximum Correntropy Kalman Filter (MCKF)** の完全な実装が含まれています。MCKFは、通信遅延・パケット損失・非ガウスノイズに対してロバストなカルマンフィルタの拡張版です。

### 🎯 MCKFが解決する3つの問題

1. **ランダム通信遅延**: 0〜K ステップの可変遅延
2. **パケット損失**: 観測データが届かない場合の補完
3. **非ガウスノイズ**: 外れ値を含むセンサノイズへの頑健性

---

## 🏗️ ファイル構成

```
delay_estimation/
├── estimators/
│   ├── mckf.py              # MCKF本体の実装 ⭐
│   └── kalman_filter.py     # 標準KF (比較用)
├── test_mckf.py             # MCKFのテスト・デモスクリプト ⭐
├── MCKF.md                  # MCKF理論の詳細説明
└── MCKF_README.md           # このファイル
```

---

## 🚀 クイックスタート

### 実行方法

```bash
# プロジェクトルートから
cd delay_estimation

# MCKFのデモシミュレーションを実行
uv run python test_mckf.py
```

### 実行結果

1. **コンソール出力**: RMSE、反復回数などの統計情報
2. **プロット**:
   - 姿勢角の推定精度
   - 角速度の推定精度
   - 推定誤差の時系列
   - 通信遅延とMCKF反復回数
3. **保存先**: `results/mckf_test_YYYYMMDD_HHMMSS/`

---

## 💡 アルゴリズムの仕組み

### MCKFの3段階構造

```
┌─────────────────────────────────────────────────────────┐
│ Stage 1: 遅延・損失のモデル化 (Bernoulli分布)           │
│   - 複数の遅延した観測を統合                             │
│   - パケット損失時は予測値で補完                         │
│   - 「遅延なし」の等価モデルに変換                       │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ Stage 2: ノイズの無相関化 (Decorrelation)               │
│   - プロセスノイズと観測ノイズの相関を除去               │
│   - ラグランジュ乗数法で最適なパラメータを計算           │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ Stage 3: 最大コレンロピー更新 (IRLS)                    │
│   - ガウスカーネルで外れ値を抑制                         │
│   - 不動点反復法で状態を更新                             │
│   - 2〜5回の反復で収束                                   │
└─────────────────────────────────────────────────────────┘
```

### コレンロピー基準 vs. 最小二乗基準

| 基準 | 式 | 外れ値への頑健性 |
|-----|---|---------------|
| **最小二乗 (MSE)** | `Σ e²` | ❌ 弱い (誤差の2乗) |
| **コレンロピー (MCC)** | `Σ exp(-e²/2η²)` | ✅ 強い (指数的抑制) |

#### ガウスカーネルの効果

```python
重み = exp(-残差² / (2*η²))

残差が小さい → 重み ≈ 1 (信頼できるデータ)
残差が大きい → 重み ≈ 0 (外れ値として無視)
```

---

## 📖 使用例

### 基本的な使い方

```python
from estimators.mckf import MaximumCorrentropyKalmanFilter
import numpy as np

# システム行列の定義 (例: 2次積分系)
A = np.array([[1, 0.1], [0, 1]])   # 状態遷移
B = np.array([[0.005], [0.1]])     # 制御入力
C = np.array([[1, 0]])             # 観測 (位置のみ)

# ノイズ共分散
Q = np.diag([0.001, 0.01])         # プロセスノイズ
R = np.array([[0.1]])              # 観測ノイズ

# 初期状態
x0 = np.array([0.0, 0.0])
P0 = np.eye(2)

# MCKFの作成
mckf = MaximumCorrentropyKalmanFilter(
    A, B, C, Q, R, x0, P0,
    max_delay=5,            # 最大5ステップ遅延を考慮
    kernel_bandwidth=2.0,   # η=2.0 (小→外れ値抑制強)
    max_iterations=10,
    convergence_threshold=1e-4
)

# フィルタリングループ
for k in range(num_steps):
    # 制御入力
    u = np.array([control_input])

    # 観測取得 (遅延あり、Noneならパケット損失)
    measurement = get_delayed_measurement(k)

    # MCKF更新
    x_est, P_est, info = mckf.step(measurement, k, u)

    print(f"推定状態: {x_est}")
    print(f"反復回数: {info['num_iterations']}")
```

### パラメータチューニング

| パラメータ | 推奨値 | 効果 |
|-----------|-------|------|
| `kernel_bandwidth` (η) | 1.0〜5.0 | 小→外れ値抑制強、大→標準KFに近い |
| `max_iterations` | 5〜10 | 通常2〜5回で収束 |
| `convergence_threshold` | 1e-4〜1e-3 | 小→高精度、大→高速 |
| `max_delay` | ネットワーク依存 | 通信遅延の最大値を設定 |

---

## 📊 デモシミュレーション

### シナリオ

- **システム**: 1自由度宇宙機の姿勢制御
  - 状態: [姿勢角 θ, 角速度 ω]
  - 制御入力: トルク u
  - 観測: 姿勢角 θ のみ

- **通信環境**:
  - ランダム遅延: 0〜5 ステップ
  - パケット損失: 5%
  - 観測ノイズ: 外れ値10%含む

### 期待される結果

MCKFは標準KFと比較して:
- ✅ **外れ値の影響を抑制**: 推定誤差が小さい
- ✅ **遅延に対して頑健**: 遅延が変動しても安定
- ✅ **パケット損失に対応**: 欠損データを自動補完

---

## 🔬 理論背景

### コレンロピー (Correntropy)

2つの確率変数の類似度を測る新しい尺度:

```
V(X, Y) = E[κ(X - Y)]
```

ここで `κ` はガウスカーネル:
```
κ(e) = exp(-e² / 2η²)
```

### 最大コレンロピー基準 (MCC)

状態推定を最適化:
```
x̂ = argmax Σ κ(観測 - 予測)
```

これにより、外れ値を自動的に抑制。

### 不動点反復法 (Fixed-Point Iteration)

MCCは非線形最適化問題なので、反復的に解く:

```python
x̂ ← x̂_pred + K_tilde * (y - C*x̂_pred)
```

重み `K_tilde` は残差に依存するため、収束するまで更新。

---

## 🧪 実験・カスタマイズ

### シミュレーションパラメータの変更

[test_mckf.py](test_mckf.py) の `run_comparison_simulation()` で設定:

```python
results = run_comparison_simulation(
    total_time=20.0,           # シミュレーション時間 [s]
    dt=0.1,                    # サンプリング時間 [s]
    max_delay=5,               # 最大遅延ステップ
    measurement_noise_std=0.1, # 観測ノイズ標準偏差
    outlier_prob=0.1,          # 外れ値発生確率 (10%)
    seed=42                    # 乱数シード
)
```

### 異なる動力学系への適用

1. `create_spacecraft_system()` を編集して、独自の状態空間モデルを定義
2. 制御入力 `generate_control_input()` をカスタマイズ
3. ノイズ統計 `Q`, `R` を調整

---

## 📈 性能評価指標

シミュレーション後に表示される指標:

- **RMSE** (Root Mean Square Error): 推定精度
- **平均反復回数**: MCKF計算コスト
- **遅延統計**: 実際の通信遅延分布

---

## 🔍 デバッグ・解析

### 統計情報の取得

```python
stats = mckf.get_statistics()

print(f"各ステップの反復回数: {stats['iterations']}")
print(f"コレンロピー重み: {stats['correntropy_weights']}")
```

### 遅延確率の学習

MCKFは観測された遅延履歴から確率を更新できます:

```python
observed_delays = [0, 1, 0, 2, 3, 0, 1, ...]  # 実測値
mckf.update_delay_probabilities(observed_delays)
```

---

## 📚 参考文献

1. **Maximum correntropy Kalman filter**
   Badong Chen, et al., Automatica, 2017

2. **Kalman filtering based on maximum correntropy criterion in presence of non-Gaussian noise**
   Reza Izanloo, et al., 2016

3. **Convergence of a Fixed-Point Algorithm under Maximum Correntropy Criterion**
   IEEE Signal Processing Letters, 2015

---

## 💻 実装の特徴

### コード品質

- ✅ **詳細な注釈**: 各関数・ステップに日本語コメント
- ✅ **型ヒント**: すべての関数にtype hints
- ✅ **エラーハンドリング**: 数値的不安定性への対応
- ✅ **モジュール化**: 各機能が独立したメソッド

### 数値安定性

- Cholesky分解によるホワイトニング
- Joseph形式の共分散更新
- 条件数チェックとフォールバック処理

---

## 🎓 学習のヒント

### MCKFを理解するステップ

1. **通常のカルマンフィルタを理解**
   → [estimators/kalman_filter.py](estimators/kalman_filter.py) を読む

2. **遅延モデルを理解**
   → [MCKF.md](MCKF.md) のStage 1を読む

3. **コレンロピーの効果を実感**
   → `test_mckf.py` を実行して、プロットを観察

4. **実装を読む**
   → [estimators/mckf.py](estimators/mckf.py) のコメントを追う

---

## ❓ FAQ

### Q1: MCKFはいつ使うべき?

**A**: 以下の条件が1つでも当てはまる場合:
- 通信遅延が可変・ランダム
- パケット損失が発生する
- センサノイズに外れ値が含まれる

### Q2: kernel_bandwidth をどう選ぶ?

**A**:
- 外れ値が多い → `η = 1.0〜2.0` (小さめ)
- 外れ値が少ない → `η = 3.0〜5.0` (大きめ)
- 試行錯誤で最適値を探す

### Q3: 計算コストは?

**A**:
- 標準KF: O(n³)
- MCKF: O(n³) × 反復回数 (通常2〜5回)
- 実用上は問題なし (n=2〜10程度)

### Q4: リアルタイム実装は可能?

**A**:
- ✅ 可能。Pythonでも100Hz以上で動作
- 最大反復回数を制限すれば高速化可能
- C++実装でさらに高速化可能

---

## 🛠️ トラブルシューティング

### Cholesky分解エラー

**原因**: 共分散行列が正定値でない
**対策**:
- ノイズ共分散 Q, R を大きくする
- 初期共分散 P0 を大きくする
- フォールバック処理で標準KFを使用 (実装済み)

### 収束しない

**原因**: kernel_bandwidth が小さすぎる
**対策**: η を大きくする (2.0 → 5.0)

### 推定精度が悪い

**原因**:
- 遅延モデルが不適切
- ノイズ統計が不正確

**対策**:
- `update_delay_probabilities()` で学習
- Q, R をチューニング

---

## 🚀 今後の拡張

- [ ] 非線形システムへの拡張 (MCKF-EKF)
- [ ] アダプティブカーネル帯域幅
- [ ] GPU高速化
- [ ] リアルタイムMCP実装

---

## 📞 Contact

質問・バグ報告は GitHub Issues へ

---

**Happy Filtering! 🎉**
