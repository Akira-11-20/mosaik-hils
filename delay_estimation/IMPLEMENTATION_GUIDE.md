# MCKF実装ガイド - どこを見ればいい？

## 🎯 実装の全体像

```
delay_estimation/
├── estimators/
│   ├── mckf_simple.py          ⭐ 簡易版MCKF（成功・2.5x改善）
│   └── mckf.py                 ⭐ 完全版MCKF（遅延対応・2.3x改善）
├── test_mckf_simple.py         ⭐ 簡易版テスト（動作確認済み）
├── test_mckf.py                ⭐ 完全版テスト（動作確認済み）
└── results/
    ├── mckf_simple_20251103_204018/  ⭐ 簡易版成功結果
    └── mckf_test_20251104_120014/    ⭐ 完全版成功結果
```

---

## 📖 読む順番（推奨）

### 1️⃣ まず理論を理解

**[MCKF.md](MCKF.md)** - MCKFの理論詳細
- 3段階構造の説明
- 数式の意味
- アルゴリズムの流れ

**所要時間**: 10分

---

### 2️⃣ 簡易版の実装を読む（動作確認済み✅）

**[estimators/mckf_simple.py](estimators/mckf_simple.py)** - 250行

#### 重要な関数とその場所:

| 関数 | 行番号 | 内容 |
|-----|-------|------|
| `__init__` | 32-73 | 初期化、パラメータ設定 |
| `predict` | 75-95 | 予測ステップ（標準KFと同じ） |
| `_gaussian_kernel` | 97-110 | ガウスカーネル重み計算 ⭐ |
| `update_mckf` | 112-228 | **MCKF更新（核心部分）** ⭐⭐⭐ |
| `step` | 230-254 | 完全なフィルタステップ |

#### 📍 特に重要な部分:

**① ガウスカーネル（97-110行）**
```python
def _gaussian_kernel(self, residual: np.ndarray) -> np.ndarray:
    """
    重み = exp(-residual² / (2*η²))
    小さい残差 → 重み ≈ 1 (信頼)
    大きい残差 → 重み ≈ 0 (外れ値)
    """
    return np.exp(-residual**2 / (2 * self.eta**2))
```

**② Information Form更新（171-202行）** ⭐⭐⭐
```python
# 逆共分散を計算
P_tilde_inv = L_P_inv.T @ T_x @ L_P_inv
R_tilde_inv = L_R_inv.T @ T_y @ L_R_inv

# Information formのカルマンゲイン
K_tilde = np.linalg.inv(
    self.C.T @ R_tilde_inv @ self.C + P_tilde_inv
) @ self.C.T @ R_tilde_inv
```

**これが成功の鍵！** MATLABコードから学んだInformation Formの正しい使い方。

**③ 不動点反復（128-223行）**
```python
for iteration in range(self.max_iter):
    # 残差計算 → 重み計算 → ゲイン更新 → 状態更新
    # 収束するまで繰り返し（通常3回程度）
```

---

### 3️⃣ テストスクリプトで使い方を学ぶ

**[test_mckf_simple.py](test_mckf_simple.py)** - 300行

#### 重要な関数:

| 関数 | 行番号 | 内容 |
|-----|-------|------|
| `create_spacecraft_system` | 23-43 | システムモデル作成（A, B, C行列） |
| `add_non_gaussian_noise` | 46-71 | 外れ値を含むノイズ生成 |
| `run_comparison` | 74-209 | KFとMCKFの比較実験 ⭐ |
| `plot_results` | 212-282 | 結果可視化 |

#### 📍 実験の核心部分（74-209行）:

```python
# MCKFの作成
mckf = SimpleMCKF(
    A, B, C, Q, R, x0, P0,
    kernel_bandwidth=2.0,     # η=2.0（小さいほど外れ値抑制）
    max_iterations=10
)

# 各ステップでの更新
for k in range(num_steps):
    # 外れ値を含むノイズ付加
    y_meas = add_non_gaussian_noise(y_true, std, outlier_prob=0.1)

    # MCKF更新
    x_mckf, P_mckf, info = mckf.step(y_meas, u)
```

---

### 4️⃣ 結果を確認

**[results/mckf_simple_20251103_204018/](results/mckf_simple_20251103_204018/)**

最新の成功結果:
- **Standard KF RMSE**: 0.1918 rad
- **Simple MCKF RMSE**: 0.0768 rad ← **2.5倍改善！**

**プロット**: [mckf_simple_comparison.png](https://github.com/Akira-11-20/mosaik-hils/blob/main/delay_estimation/results/mckf_simple_20251103_204018/mckf_simple_comparison.png)

---

## 🔧 完全版MCKF（遅延対応版）

**[estimators/mckf.py](estimators/mckf.py)** - 600行 ⭐ 動作確認済み

### 実装状態:

| 機能 | 実装状況 | 精度 |
|-----|---------|------|
| 遅延モデリング | ✅ 完了（式12厳密版） | 2.3x改善 |
| デコリレーション | ✅ 完了 | - |
| MCKF更新 | ✅ 完了（Information Form） | - |

### 重要な実装箇所:

**📍 遅延ノイズ共分散の厳密計算（[mckf.py:234-248行](estimators/mckf.py#L234-L248)）**

論文式(12)の厳密実装（2025-11-04追加）:
```python
# R̄ = Σ_{t=0}^{k} C*A^t*Q*(A^t)^T*C^T
R_bar = self.R.copy()
for t in range(int(delay)):
    A_t = np.linalg.matrix_power(self.A, t)
    R_bar += self.C @ A_t @ self.Q @ A_t.T @ self.C.T

# O = Σ_{t=0}^{k-1} A^t * Q * (A^{k-t-1})^T * C^T
O = np.zeros((self.n, self.p))
for t in range(int(delay)):
    A_t = np.linalg.matrix_power(self.A, t)
    A_k_minus_t_minus_1 = np.linalg.matrix_power(self.A, int(delay) - t - 1)
    O += A_t @ self.Q @ A_k_minus_t_minus_1.T @ C_bar.T
```

**📍 Information Form更新（[mckf.py:385-410行](estimators/mckf.py#L385-L410)）**

```python
P_tilde_inv = L_P_inv.T @ T_x @ L_P_inv  # 逆共分散
R_tilde_inv = L_R_inv.T @ T_y @ L_R_inv
K_tilde = np.linalg.inv(
    C_bar.T @ R_tilde_inv @ C_bar + P_tilde_inv
) @ C_bar.T @ R_tilde_inv
```

---

## 📚 ドキュメント一覧

| ファイル | 内容 | いつ読む？ |
|---------|------|-----------|
| **[MCKF.md](MCKF.md)** | 理論・数式詳細 | 最初に |
| **[MCKF_README.md](MCKF_README.md)** | 使い方・FAQ | 実装する前 |
| **[README_MCKF_implementation.md](README_MCKF_implementation.md)** | 実装状況 | 開発中に |
| **このファイル** | どこを見ればいい？ | 最初に！ |

---

## 🚀 実際に動かす

### 簡易版MCKF（動作確認済み・2.5x改善）

```bash
cd delay_estimation
uv run python test_mckf_simple.py
```

**実行時間**: 約10秒
**出力**: `results/mckf_simple_YYYYMMDD_HHMMSS/`
**性能**: Standard KF 0.19 rad → MCKF **0.08 rad** (2.5x改善)

### 完全版MCKF（動作確認済み・2.3x改善）

```bash
cd delay_estimation
uv run python test_mckf.py
```

**実行時間**: 約20秒
**出力**: `results/mckf_test_YYYYMMDD_HHMMSS/`
**性能**: Standard KF 1.28 rad → MCKF **0.55 rad** (2.3x改善)

---

## 🔍 デバッグ・カスタマイズ

### パラメータを変えたい

**[test_mckf_simple.py:297行](test_mckf_simple.py#L297)**

```python
results = run_comparison(
    total_time=20.0,              # シミュレーション時間
    dt=0.1,                       # サンプリング時間
    measurement_noise_std=0.1,    # 観測ノイズ
    outlier_prob=0.1,             # 外れ値確率（10%）
    kernel_bandwidth=2.0,         # η（小→外れ値抑制強）
    seed=42
)
```

### 中間値をデバッグ出力したい

**[mckf_simple.py:168行](estimators/mckf_simple.py#L168)** あたりに追加:

```python
# ④ ガウスカーネル重み
weight_pred = self._gaussian_kernel(white_pred_residual)
weight_obs = self._gaussian_kernel(white_innovation)

# デバッグ出力
print(f"Iteration {iteration}: weights_obs = {weight_obs}")
```

---

## 💡 よくある質問

### Q1: なぜSimpleMCKFは成功したのに、Full MCKFは失敗？

**A**: SimpleMCKFは**Information Form**を正しく実装したため。Full MCKFはまだ古い実装のまま。

### Q2: Information Formとは？

**A**: 共分散の**逆行列**を使う方法。数値的に安定で、重み付きKFに適している。

### Q3: kernel_bandwidth (η) はどう選ぶ？

**A**:
- 大きい (η=5.0) → ガウスフィルタに近い、外れ値抑制弱い
- 小さい (η=1.0) → 外れ値抑制強い、過剰に保守的になる可能性
- **推奨**: η=2.0～3.0

### Q4: 反復回数が多い（10回）のは問題？

**A**: 収束していない可能性。原因:
- ηが小さすぎる
- 初期共分散P0が大きすぎる
- 数値的に不安定

---

## 🎓 学習パス

### 初心者向け

1. [MCKF_README.md](MCKF_README.md) を読む
2. [test_mckf_simple.py](test_mckf_simple.py) を実行
3. プロットを観察
4. パラメータを変えて再実行

### 中級者向け

1. [MCKF.md](MCKF.md) で理論を学ぶ
2. [mckf_simple.py](estimators/mckf_simple.py) のコードを読む
3. デバッグ出力を追加して動作確認
4. カスタムシステムで実験

### 上級者向け

1. [mckf.py](estimators/mckf.py) の遅延処理を理解
2. Information Formを完全版に適用
3. 論文とコードを詳細比較
4. 新しい応用を開発

---

## 📞 トラブルシューティング

### エラー: `LinAlgError: Matrix is not positive definite`

**原因**: 共分散行列が正定値でない
**解決**:
- Q, R を大きくする
- P0 を調整
- `P += np.eye(n) * 1e-8` を追加

### MCKF精度が悪い

**チェック項目**:
1. Information Formを使っているか？
2. ηは適切か？（2.0～3.0）
3. 外れ値は実際に発生しているか？

---

**Happy Filtering! 🎉**

*Last Updated: 2025-11-03 20:40 (成功版)*
