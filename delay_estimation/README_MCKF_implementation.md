# MCKF実装の状況

## 📁 実装ファイル

### ✅ 完成

1. **[estimators/mckf.py](estimators/mckf.py)** - 完全版MCKF
   - k-step遅延・パケット損失対応
   - Bernoulli分布モデル
   - デコリレーション処理
   - 最大コレンロピー更新
   - 状態: 実装完了、要検証

2. **[estimators/mckf_simple.py](estimators/mckf_simple.py)** - 簡易版MCKF
   - 遅延処理なし
   - コレンロピー基準のみ
   - 状態: 実装完了、精度調整中

3. **[test_mckf.py](test_mckf.py)** - 完全版MCKFテスト
   - 宇宙機姿勢制御シナリオ
   - 遅延・損失・外れ値を含む環境
   - 状態: 実装完了、精度要改善

4. **[test_mckf_simple.py](test_mckf_simple.py)** - 簡易版MCKFテスト
   - 遅延なし、外れ値のみ
   - KFとの比較
   - 状態: 実装完了、デバッグ中

### 📚 ドキュメント

1. **[MCKF.md](MCKF.md)** - 理論詳細
   - 完全版MCKFの数式
   - 3段階構造の説明
   - 実装設計書

2. **[MCKF_README.md](MCKF_README.md)** - 使い方ガイド
   - クイックスタート
   - パラメータ説明
   - FAQ

3. **[Kalman Filters for Bernoulli Systems.pdf](../Kalman Filters for Bernoulli Systems.pdf)** - 参考論文
   - 2024年の最新論文
   - k-step遅延モデル
   - MCKF with遅延の詳細

## 🎯 実装の核心部分

### コレンロピー基準 (Maximum Correntropy Criterion)

```python
# ガウスカーネル
weight = exp(-residual² / (2*η²))

# 重み付き共分散 (論文 式28-29)
P_tilde = L_P^{-T} * T_x * L_P^{-1}
R_tilde = L_R^{-T} * T_y * L_R^{-1}

# 重み付きカルマンゲイン (論文 式27)
K_tilde = P_tilde * C^T * (C*P_tilde*C^T + R_tilde)^{-1}
```

### 不動点反復 (Fixed-Point Iteration)

```python
for iteration in range(max_iter):
    # ① 残差計算
    e = 観測 - 予測

    # ② ガウスカーネル重み
    weights = exp(-e²/(2*η²))

    # ③ 重み付き更新
    K_tilde = f(weights)
    x = x_pred + K_tilde * innovation

    # ④ 収束判定
    if converged:
        break
```

## 🐛 現在の問題点

### 問題1: MCKFの精度が標準KFより悪い

**症状**:
- SimpleMCKF RMSE: 3.84 rad
- Standard KF RMSE: 0.19 rad

**可能な原因**:
1. 重み行列の計算が不正確
2. Cholesky分解の順序問題 (upper vs lower)
3. 白色化の実装ミス
4. 初期共分散が不適切

### 問題2: 完全版MCKFが発散

**症状**:
- 遅延処理を含むと推定値が発散
- 初期ステップで大きくずれる

**可能な原因**:
1. 遅延観測の統合ロジックのバグ
2. `B^{-i}` の計算が不安定
3. デコリレーションのラグランジュ乗数計算エラー

## 🔬 検証項目

### ✅ 確認済み

- [ ] 論文の式(28-29)の実装
- [x] ガウスカーネルの実装
- [x] 不動点反復の収束ロジック
- [x] Cholesky分解

### ❌ 未確認

- [ ] 重み行列 `T_x, T_y` の正しい適用
- [ ] 白色化処理の正確性
- [ ] Joseph形式の共分散更新
- [ ] 遅延モデルの`C_bar`計算

## 📊 実験結果

### Experiment 1: Simple MCKF (No Delay)

```
Conditions:
- Time: 20s, dt=0.1s
- Outlier: 10% (10x noise)
- Kernel bandwidth η=2.0

Results:
- KF RMSE:    0.1918 rad ✓
- MCKF RMSE:  3.8415 rad ✗  (worse!)
- MCKF Iterations: 7.83
```

### Experiment 2: Full MCKF (With Delay)

```
Conditions:
- Time: 20s, dt=0.1s
- Max delay: 5 steps
- Outlier: 10%
- Kernel bandwidth η=3.0

Results:
- KF RMSE:    1.2833 rad
- MCKF RMSE:  5.0292 rad ✗ (diverged)
- MCKF Iterations: 9.43
```

## 🔍 デバッグ戦略

### Step 1: 最小限のMCKF検証

論文の式(26-30)のみを実装した最小限バージョンで動作確認

### Step 2: 段階的な機能追加

1. 基本MCKF (遅延なし、単純ガウスノイズ) → 動作確認
2. 外れ値追加 → ロバスト性確認
3. 遅延追加 → 完全版

### Step 3: 論文との数値比較

- 論文 Table 3-6 の結果を再現
- 同じパラメータで実験

## 📖 参考文献

### 主要論文

1. **Liu et al. (2024)** - "Modified Kalman and Maximum Correntropy Kalman Filters..."
   - 本実装のベース
   - k-step遅延モデル
   - MCKF with Bernoulli

2. **Chen et al. (2017)** - "Maximum correntropy Kalman filter"
   - コレンロピー基準の原論文
   - 固定点反復法

### 実装参考

- GitHub: [Ramune6110/Maximum-Correntropy-Kalman-Filter](https://github.com/Ramune6110/Maximum-Correntropy-Kalman-Filter)
- MATLAB実装

## 💡 次のステップ

1. **デバッグ優先**
   - [ ] SimpleMCKFの精度改善
   - [ ] 重み行列の計算を再確認
   - [ ] 論文のMATLABコードと比較

2. **検証**
   - [ ] 論文のシミュレーション結果を再現
   - [ ] 単純な1Dシステムでテスト

3. **ドキュメント化**
   - [x] 実装状況のまとめ
   - [ ] 各関数の詳細説明
   - [ ] トラブルシューティングガイド

## 📝 メモ

- Cholesky分解は `lower=True` (下三角) がnumpyのデフォルト
- 論文の`T^{-1}`は重み行列の**逆行列**
- 白色化は共分散を単位行列に変換する操作
- 不動点反復は通常2-5回で収束

---

**Last Updated**: 2025-11-03
**Status**: 🚧 Implementation完了, デバッグ中
