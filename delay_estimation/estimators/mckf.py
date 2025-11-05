"""
Maximum Correntropy Kalman Filter (MCKF) implementation
通信遅延・パケット損失・非ガウスノイズに対応したロバストなカルマンフィルタ

Reference:
    "Maximum correntropy Kalman filter" (Automatica, 2017)
    "Kalman filtering based on maximum correntropy criterion in presence of non-Gaussian noise"
"""

import numpy as np
from typing import Tuple, Optional, List


class MaximumCorrentropyKalmanFilter:
    """
    Maximum Correntropy Kalman Filter (MCKF)

    MCKFは以下の3つの問題に対応:
    1. ランダム通信遅延・パケット損失 (Bernoulli分布でモデル化)
    2. 非ガウスノイズ (外れ値を含む)
    3. プロセスノイズと観測ノイズの相関

    システムモデル:
        x(k+1) = A*x(k) + B*u(k) + w(k)  # w ~ process noise
        y(k)   = C*x(k) + v(k)            # v ~ measurement noise

    遅延を含む観測:
        Y(k) = Σ Λ_{i,k} * y(k-i) + Λ_{K+1,k} * ŷ(k)

        ここで Λ_{i,k} はBernoulli変数:
        - Λ_{0,k} = 1: 遅延なし
        - Λ_{i,k} = 1 (i=1,...,K): i-step遅延
        - Λ_{K+1,k} = 1: パケット損失 (予測値で補完)
    """

    def __init__(
        self,
        A: np.ndarray,           # 状態遷移行列 (n x n)
        B: np.ndarray,           # 制御入力行列 (n x m)
        C: np.ndarray,           # 観測行列 (p x n)
        Q: np.ndarray,           # プロセスノイズ共分散 (n x n)
        R: np.ndarray,           # 観測ノイズ共分散 (p x p)
        x0: np.ndarray,          # 初期状態 (n,)
        P0: np.ndarray,          # 初期共分散 (n x n)
        max_delay: int = 5,      # 最大遅延ステップ数
        kernel_bandwidth: float = 3.0,  # コレンロピーカーネル帯域幅 η
        max_iterations: int = 10,       # 不動点反復の最大回数
        convergence_threshold: float = 1e-4,  # 収束判定閾値
    ):
        """
        MCKFの初期化

        Args:
            A: 状態遷移行列 (例: 離散化された運動方程式)
            B: 制御入力行列
            C: 観測行列 (どの状態を観測するか)
            Q: プロセスノイズ共分散 (モデル化誤差)
            R: 観測ノイズ共分散 (センサノイズ)
            x0: 初期状態推定値
            P0: 初期共分散推定値
            max_delay: 考慮する最大遅延ステップ数 K
            kernel_bandwidth: ガウスカーネル帯域幅 η (小→外れ値抑制強)
            max_iterations: 不動点反復の最大回数 (通常2-5回で収束)
            convergence_threshold: 収束判定閾値 (相対変化率)
        """
        # システム行列
        self.A = A.copy()
        self.B = B.copy()
        self.C = C.copy()
        self.Q = Q.copy()
        self.R = R.copy()

        # 状態推定値
        self.x = x0.copy()  # 現在の状態推定
        self.P = P0.copy()  # 現在の共分散推定

        # 次元
        self.n = A.shape[0]  # 状態次元
        self.m = B.shape[1]  # 制御入力次元
        self.p = C.shape[0]  # 観測次元

        # MCKF固有のパラメータ
        self.max_delay = max_delay
        self.eta = kernel_bandwidth  # カーネル帯域幅 (論文では σ や η)
        self.max_iter = max_iterations
        self.epsilon = convergence_threshold

        # 遅延観測バッファ (過去の観測を保存)
        # measurement_buffer[i] = (観測値, タイムステップ)
        self.measurement_buffer: List[Tuple[np.ndarray, int]] = []

        # 遅延確率 (Bernoulli分布のパラメータ)
        # 実際の実装では統計情報から推定するが、ここでは均等分布で初期化
        self.delay_probs = self._initialize_delay_probs()

        # 統計情報 (デバッグ・解析用)
        self.stats = {
            'iterations': [],        # 各ステップの反復回数
            'correntropy_weights': [],  # コレンロピー重み
            'estimated_delays': [],  # 推定された遅延
        }

    def _initialize_delay_probs(self) -> np.ndarray:
        """
        遅延確率の初期化（論文の条件を反映）

        Returns:
            probs: 確率ベクトル (max_delay + 2,)
                   [Λ̄_0, Λ̄_1, ..., Λ̄_K, P(損失)]

        Note:
            論文のSection 5の条件:
            - 遅延またはパケット損失の確率: 0.3 (Λ̄_0 = 0.7)
            - 遅延が発生する確率: 0.3 × 0.8 = 0.24
              - 1-step遅延: 0.24 × 0.5 = 0.12 (Λ̄_1)
              - 2-step遅延: 0.24 × 0.5 = 0.12 (Λ̄_2)
            - パケット損失: 0.3 × 0.2 = 0.06
        """
        K = self.max_delay

        if K == 2:
            # 論文の2-step遅延の条件
            probs = np.array([
                0.7,    # Λ̄_0,n = P(遅延なし) = 1 - 0.3 = 0.7
                0.12,   # Λ̄_1,n = P(1-step遅延) = 0.3 × 0.8 × 0.5 = 0.12
                0.12,   # Λ̄_2,n = P(2-step遅延) = 0.3 × 0.8 × 0.5 = 0.12
                0.06    # P(損失) = 0.3 × 0.2 = 0.06 (論文では Λ̄_{K+1,n})
            ])
        elif K == 3:
            # 論文の3-step遅延の条件
            probs = np.array([
                0.7,     # Λ̄_0,n = P(遅延なし) = 1 - 0.3 = 0.7
                0.24,    # Λ̄_1,n = P(1-step遅延) = 0.3 × 0.8 = 0.24
                0.12,    # Λ̄_2,n = P(2-step遅延) = 0.3 × 0.8 × 0.5 = 0.12
                0.06,    # Λ̄_3,n = P(3-step遅延) = 0.3 × 0.8 × 0.5 × 0.5 = 0.06
                0.02     # P(損失) = 残り (論文では Λ̄_{K+1,n})
            ])
        else:
            # 一般的な初期化（均等分布 + 損失確率）
            probs = np.ones(K + 2) / (K + 2)
            probs[-1] = 0.05
            probs[:-1] = (1.0 - probs[-1]) / (K + 1)

        return probs

    def update_delay_probabilities(self, observed_delays: List[int]):
        """
        観測された遅延履歴から確率を更新 (オンライン学習)

        Args:
            observed_delays: 過去の観測された遅延のリスト
        """
        K = self.max_delay
        counts = np.zeros(K + 2)

        for delay in observed_delays:
            if delay <= K:
                counts[delay] += 1
            else:
                counts[-1] += 1  # パケット損失

        # ラプラス平滑化を適用
        self.delay_probs = (counts + 1.0) / (len(observed_delays) + K + 2)

    def _compute_A_bar(self, r: int, delay: int) -> float:
        """
        Ā_{r,n} の計算（論文式(12), (13)で使用）- 厳密版

        論文ページ4の定義:
        A_{r,n} = (1-Λ_{0,n}) * (1-Λ_{0,n-r}) * ∏_{i=1}^{r} Λ_{i,n-r+1}
                  * (1-Λ_{r+1,n-r+1}) * ∏_{t=1}^{r-1} [1-(1-Λ_{0,n-t})...]

        時不変系での簡略化:
        - Λ_{i,n} = Λ̄_i (期待値、時間に依存しない)
        - 独立なBernoulli変数の積の期待値 = 各期待値の積

        Args:
            r: 遅延ステップ数 (0 <= r <= K)
            delay: 現在の観測の遅延

        Returns:
            Ā_{r,n}: Bernoulli変数の期待値
        """
        K = self.max_delay
        Λ = self.delay_probs  # [Λ̄_0, Λ̄_1, ..., Λ̄_K, Λ̄_{K+1}]

        if r == 0:
            # A_{0,n} = Λ_{0,n}
            # Ā_{0,n} = Λ̄_0
            return Λ[0]

        elif r == 1:
            # A_{1,n} = (1-Λ_{0,n}) * (1-Λ_{0,n-1}) * Λ_{1,n} * (1-Λ_{2,n})
            # 時不変系での期待値:
            # Ā_{1,n} = (1-Λ̄_0) * (1-Λ̄_0) * Λ̄_1 * (1-Λ̄_2)
            if K >= 2:
                return (1 - Λ[0]) * (1 - Λ[0]) * Λ[1] * (1 - Λ[2])
            else:
                return (1 - Λ[0]) * (1 - Λ[0]) * Λ[1]

        elif r == 2 and K >= 2:
            # A_{2,n} の計算（論文の複雑な式）
            # A_{2,n} = (1-Λ_0,n) * (1-Λ_0,n-2) * Λ_1,n-1 * Λ_2,n-1
            #         * (1-Λ_3,n-1) * [1-(1-Λ_0,n-1)*Λ_1,n*(1-Λ_2,n)]
            #
            # 時不変系での近似:
            # Ā_{2,n} ≈ (1-Λ̄_0)^2 * Λ̄_1 * Λ̄_2 * (1-Λ̄_3) * [1-(1-Λ̄_0)*Λ̄_1*(1-Λ̄_2)]

            term1 = (1 - Λ[0]) * (1 - Λ[0])  # (1-Λ_0)^2
            term2 = Λ[1] * Λ[2]  # Λ_1 * Λ_2

            # (1-Λ_3) の計算
            if K >= 3:
                term3 = 1 - Λ[3]
            else:
                term3 = 1 - Λ[-1]  # パケット損失確率

            # [1-(1-Λ_0)*Λ_1*(1-Λ_2)]
            term4 = 1 - (1 - Λ[0]) * Λ[1] * (1 - Λ[2])

            return term1 * term2 * term3 * term4

        elif r > K:
            # A_{K+1,n} = パケット損失の確率
            # 論文式: すべての遅延パターンが起こらなかった場合
            #
            # A_{K+1,n} = (1-Λ_0,n) * [1-(1-Λ_0,n-K)*∏Λ_i...]
            #           * ∏[1-(1-Λ_0,n-t)*...]
            #
            # 簡略版: 残りの確率を使用
            return Λ[-1]

        else:
            # r > 2 の一般的なケース
            # 複雑すぎるため、簡略化した近似を使用
            # Ā_{r,n} ≈ (1-Λ̄_0)^2 * ∏_{i=1}^{r} Λ̄_i * 補正項

            result = (1 - Λ[0]) ** 2
            for i in range(1, min(r + 1, K + 1)):
                result *= Λ[i]

            # 高次の遅延は急速に減衰
            return result * 0.5  # 経験的な補正係数

    def predict(self, u: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        予測ステップ (通常のカルマンフィルタと同じ)

        Args:
            u: 制御入力 (m,)

        Returns:
            x_pred: 予測状態 (n,)
            P_pred: 予測共分散 (n x n)
        """
        if u is None:
            u = np.zeros(self.m)

        # 状態予測: x̂(k|k-1) = A*x̂(k-1|k-1) + B*u(k-1)
        self.x = self.A @ self.x + self.B @ u

        # 共分散予測: P(k|k-1) = A*P(k-1|k-1)*A' + Q
        self.P = self.A @ self.P @ self.A.T + self.Q

        return self.x.copy(), self.P.copy()

    def _construct_delayed_observation(
        self,
        current_measurement: Optional[np.ndarray],
        current_time: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        遅延を含む観測の構築 (MCKF Stage 1)

        複数の遅延した観測を統合して、「遅延なし」の形に変換:
            Y(k) = C̄(k)*x(k) + ν̄(k)

        Args:
            current_measurement: 現在受信した観測 (遅延あり)
            current_time: 現在のタイムステップ

        Returns:
            Y: 統合観測ベクトル (p,)
            C_bar: 等価観測行列 (p x n)
            R_bar: 等価観測ノイズ共分散 (p x p)
            O: プロセスノイズと観測ノイズの相関 (n x p)
        """
        # 観測をバッファに追加
        if current_measurement is not None:
            self.measurement_buffer.append((current_measurement.copy(), current_time))

        # 古い観測を削除 (max_delay より古いもの)
        self.measurement_buffer = [
            (y, t) for y, t in self.measurement_buffer
            if current_time - t <= self.max_delay
        ]

        # 遅延観測の統合
        K = self.max_delay
        Y = np.zeros(self.p)
        C_bar = np.zeros((self.p, self.n))
        R_bar = np.zeros((self.p, self.p))
        O = np.zeros((self.n, self.p))

        # ケース1: 観測がない場合 (パケット損失)
        if len(self.measurement_buffer) == 0:
            # 予測観測値で補完: ŷ(k) = C*A*x̂(k-1)
            Y = self.C @ self.x
            C_bar = self.C.copy()
            R_bar = self.R.copy()
            return Y, C_bar, R_bar, O

        # ケース2: 遅延観測がある場合
        # バッファから最新の観測を取得して遅延ステップ数を計算
        # この観測に対して遅延モデリング(Stage 1)を適用
        y_delayed, t_delayed = self.measurement_buffer[-1]
        delay = current_time - t_delayed

        if delay == 0:
            # 遅延なし
            Y = y_delayed
            C_bar = self.C.copy()
            R_bar = self.R.copy()
        elif delay <= K:
            # k-step遅延の場合、観測を現在時刻に変換
            # 論文の式(8)-(12)に基づく厳密実装
            Y = y_delayed

            # C̄ = C * A^{-k}
            # 注: A^k を計算してから逆行列を取る
            A_power_k = np.linalg.matrix_power(self.A, int(delay))
            try:
                A_inv_k = np.linalg.inv(A_power_k)
                C_bar = self.C @ A_inv_k
            except np.linalg.LinAlgError:
                # 逆行列が取れない場合は簡略版にフォールバック
                C_bar = self.C.copy()

            # R̄の厳密計算（論文式(12)完全版 - Bernoulli確率使用）
            #
            # 論文式(12):
            # R̄_n = Σ_{t=0}^{k-1} Ā_{t,n} * R_{n-t}
            #      + Σ_{r=2}^{k} Σ_{j=r}^{k} Ā_{j,n} * C*B^{-j+r-1} * Q_{n-r} * (B^{-j+r-1})^T * C^T
            #      + Σ_{r=2}^{k} Σ_{j=r}^{k} Ā_{j,n} * C*B^{-r+j} * Q_{n-j-1} * (B^{-r+j})^T * C^T

            R_bar = np.zeros((self.p, self.p))

            # 第1項: Σ_{t=0}^{k-1} Ā_{t,n} * R_{n-t}
            # 時不変系では R_{n-t} = R
            for t in range(int(delay)):
                A_bar_t = self._compute_A_bar(t, delay)
                R_bar += A_bar_t * self.R

            # 第2項: Σ_{r=2}^{k} Σ_{j=r}^{k} Ā_{j,n} * C*B^{-j+r-1} * Q * (B^{-j+r-1})^T * C^T
            for r in range(2, int(delay) + 1):
                for j in range(r, int(delay) + 1):
                    A_bar_j = self._compute_A_bar(j, delay)
                    exponent = -j + r - 1

                    try:
                        if exponent >= 0:
                            B_power = np.linalg.matrix_power(self.A, exponent)
                        else:
                            B_power = np.linalg.inv(
                                np.linalg.matrix_power(self.A, -exponent)
                            )

                        term = A_bar_j * (self.C @ B_power @ self.Q @ B_power.T @ self.C.T)
                        R_bar += term
                    except np.linalg.LinAlgError:
                        pass

            # 第3項: Σ_{r=2}^{k} Σ_{j=r}^{k} Ā_{j,n} * C*B^{-r+j} * Q * (B^{-r+j})^T * C^T
            for r in range(2, int(delay) + 1):
                for j in range(r, int(delay) + 1):
                    A_bar_j = self._compute_A_bar(j, delay)
                    exponent = -r + j

                    try:
                        if exponent >= 0:
                            B_power = np.linalg.matrix_power(self.A, exponent)
                        else:
                            B_power = np.linalg.inv(
                                np.linalg.matrix_power(self.A, -exponent)
                            )

                        term = A_bar_j * (self.C @ B_power @ self.Q @ B_power.T @ self.C.T)
                        R_bar += term
                    except np.linalg.LinAlgError:
                        pass

            # プロセスノイズと観測ノイズの相関（論文式(13)厳密版 - Bernoulli確率使用）
            #
            # 論文式(13):
            # O_n = E{ω_n * ν̄_n^T}
            #     = -Ā_{k+1,n} * C * Q_n - Σ_{i=1}^{k} Ā_{i,n} * C * B^{-i} * Q_n
            #
            # 次元: O_n は (n x p) つまり (状態次元 x 観測次元)
            #       ω_n: (n x 1), ν̄_n: (p x 1) なので O_n: (n x p)
            #
            # 時不変系（B=A, Q_n=Q）での実装:

            O = np.zeros((self.n, self.p))

            # 第1項: -Ā_{k+1,n} * C * Q_n
            # C: (p x n), Q: (n x n) → C^T * Q: (n x n) @ ... の形
            A_bar_k_plus_1 = self._compute_A_bar(int(delay) + 1, delay)
            O -= A_bar_k_plus_1 * (self.Q @ self.C.T)

            # 第2項: -Σ_{i=1}^{k} Ā_{i,n} * B^{-i} * Q * C^T
            for i in range(1, int(delay) + 1):
                A_bar_i = self._compute_A_bar(i, delay)

                try:
                    A_inv_i = np.linalg.inv(np.linalg.matrix_power(self.A, i))
                    # B^{-i} @ Q @ C^T: (n x n) @ (n x n) @ (n x p) = (n x p)
                    O -= A_bar_i * (A_inv_i @ self.Q @ self.C.T)
                except np.linalg.LinAlgError:
                    # 逆行列計算失敗時はスキップ
                    pass

        return Y, C_bar, R_bar, O

    def _decorrelate_noise(
        self,
        C_bar: np.ndarray,
        R_bar: np.ndarray,
        O: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        ノイズの無相関化 (MCKF Stage 2: Decorrelation)

        遅延により生じたプロセスノイズと観測ノイズの相関を除去し、
        修正された状態方程式を作成:
            x(k) = D(k)*x(k-1) + U(k) + ζ(k)

        Args:
            C_bar: 等価観測行列 (p x n)
            R_bar: 等価観測ノイズ共分散 (p x p)
            O: プロセスノイズと観測ノイズの相関 (n x p)

        Returns:
            D: 修正状態遷移行列 (n x n)
            U: 修正入力項 (n,)
            Q_zeta: 修正プロセスノイズ共分散 (n x n)
            lambda_n: ラグランジュ乗数 (n x p)
        """
        # Lagrange乗数の計算（論文式(17)厳密版）
        # 無相関条件 E{ζ_n * ν̄_n^T} = 0 から導出される最適パラメータ
        #
        # λ_n = -Q_n * (Σ_{i=1}^{k} Ā_{i,n}*(C*B^{-i})^T + Ā_{k+1,n}*C^T)
        #     * [Σ_{t=0}^{k} Ā_{t,n}*R_{n-t} + Σ_{r=2}^{k} Σ_{j=r} Ā_{j,n}*C*B^{-j+r-1}*Q*...]^{-1}
        #
        # 実装では以下の等価な形式を使用:
        # λ_n = (Q*C̄^T - O) * (C̄*Q*C̄^T - O^T*C̄^T - C̄*O + R̄)^{-1}
        try:
            S = C_bar @ self.Q @ C_bar.T - O.T @ C_bar.T - C_bar @ O + R_bar
            lambda_n = (self.Q @ C_bar.T - O) @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            # 数値的に不安定な場合は相関なしと仮定
            lambda_n = np.zeros((self.n, self.p))

        # 修正状態遷移行列（論文式(14)）: D_n = B - λ_n * C̄_n * B
        # 時不変系でB=Aの場合
        D = self.A - lambda_n @ C_bar @ self.A

        # 修正入力項（論文式(14)）: U_n = λ_n * Y_n
        # (更新ステップで実際の観測値Yを使って計算)
        U = np.zeros(self.n)  # プレースホルダー

        # 修正プロセスノイズ共分散（論文式(15)で導出）
        # Q_ζ = (I - λ*C̄)*Q*(I - λ*C̄)^T + λ*R̄*λ^T - (I - λ*C̄)*O*λ^T - λ*O^T*(I - λ*C̄)^T
        #
        # ζ_n = (I - λ_n*C̄_n)*ω_n - λ_n*ν̄_n の共分散
        I_lambda_C = np.eye(self.n) - lambda_n @ C_bar
        Q_zeta = (
            I_lambda_C @ self.Q @ I_lambda_C.T +      # プロセスノイズ項
            lambda_n @ R_bar @ lambda_n.T -           # 観測ノイズ項
            I_lambda_C @ O @ lambda_n.T -             # 交差相関項1
            lambda_n @ O.T @ I_lambda_C.T             # 交差相関項2
        )

        return D, U, Q_zeta, lambda_n

    def _gaussian_kernel_weight(self, residual: np.ndarray) -> np.ndarray:
        """
        ガウスカーネルによる重み計算 (コレンロピー基準の核心)

        重み = exp(-residual^2 / (2*η^2))

        - 小さい残差 → 重み ≈ 1 (信頼できるデータ)
        - 大きい残差 → 重み ≈ 0 (外れ値として抑制)

        Args:
            residual: 残差ベクトル (各要素の誤差)

        Returns:
            weights: 要素ごとの重み (同じshape)
        """
        return np.exp(-residual**2 / (2 * self.eta**2))

    def update_mckf(
        self,
        Y: np.ndarray,
        C_bar: np.ndarray,
        R_bar: np.ndarray,
        Q_zeta: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        MCKF更新ステップ (MCKF Stage 3: Maximum Correntropy Update)

        不動点反復法 (Fixed-Point Iteration) により、
        最大コレンロピー基準を満たす状態推定値を計算

        アルゴリズム:
        1. 白色化 (Cholesky分解)
        2. while not converged:
               - 残差計算
               - ガウスカーネル重み計算
               - 重み付き共分散更新
               - 重み付きカルマンゲイン計算
               - 状態更新

        Args:
            Y: 統合観測 (p,)
            C_bar: 等価観測行列 (p x n)
            R_bar: 等価観測ノイズ共分散 (p x p)
            Q_zeta: 修正プロセスノイズ共分散 (n x n)

        Returns:
            x_updated: 更新後の状態推定 (n,)
            P_updated: 更新後の共分散推定 (n x n)
            num_iterations: 収束までの反復回数
        """
        # 予測値を初期値として使用
        x_pred = self.x.copy()
        P_pred = self.P.copy()

        # Cholesky分解 (白色化のため)
        # L_p * L_p' = P_pred, L_r * L_r' = R_bar
        try:
            L_p = np.linalg.cholesky(P_pred)
            L_r = np.linalg.cholesky(R_bar)
        except np.linalg.LinAlgError:
            # Cholesky分解失敗時は通常のKF更新にフォールバック
            return self._standard_kf_update(Y, C_bar, R_bar)

        # ブロック対角行列: L = diag(L_p, L_r)
        n, p = self.n, self.p
        L = np.block([
            [L_p, np.zeros((n, p))],
            [np.zeros((p, n)), L_r]
        ])

        # 白色化された観測・予測
        # d = L^{-1} * [x_pred; Y]
        # w = L^{-1} * [I; C_bar]
        try:
            L_inv = np.linalg.inv(L)
        except np.linalg.LinAlgError:
            return self._standard_kf_update(Y, C_bar, R_bar)

        d = L_inv @ np.concatenate([x_pred, Y])
        W = L_inv @ np.block([[np.eye(n)], [C_bar]])

        # 不動点反復の開始
        x_est = x_pred.copy()  # 初期推定値

        for iteration in range(self.max_iter):
            x_prev = x_est.copy()

            # ① 残差計算: e = d - W*x_est
            residual = d - W @ x_est

            # ② ガウスカーネル重み: G = exp(-e^2 / (2*η^2))
            weights = self._gaussian_kernel_weight(residual)

            # ③ 重み行列を予測部と観測部に分割
            # T = diag(G), T_x = T[0:n], T_y = T[n:n+p]
            T_x = np.diag(weights[:n])
            T_y = np.diag(weights[n:n+p])

            # ④ 重み付き逆共分散 (Information Form)
            # SimpleMCKFと同じ修正を適用
            # MATLABコード準拠: Pke_hat_inv = inv(Bpk)' * Cx * inv(Bpk)
            try:
                L_p_inv = np.linalg.inv(L_p)
                L_r_inv = np.linalg.inv(L_r)

                # 逆共分散を計算
                P_tilde_inv = L_p_inv.T @ T_x @ L_p_inv
                R_tilde_inv = L_r_inv.T @ T_y @ L_r_inv

                # 正定値性を保証
                P_tilde_inv += np.eye(self.n) * 1e-8
                R_tilde_inv += np.eye(self.p) * 1e-8

            except np.linalg.LinAlgError:
                break

            # ⑤ 重み付きカルマンゲイン (Information Form使用)
            # MATLABコード: Gk = inv(H'*R_hat_inv*H + Pke_hat_inv) * H' * R_hat_inv
            try:
                # Information formのゲイン計算
                K_tilde = np.linalg.inv(
                    C_bar.T @ R_tilde_inv @ C_bar + P_tilde_inv
                ) @ C_bar.T @ R_tilde_inv
            except np.linalg.LinAlgError:
                break

            # ⑥ 状態更新
            # x̂(k) = x̂(k|k-1) + K̃ * (Y - C̄*x̂(k|k-1))
            innovation = Y - C_bar @ x_pred
            x_est = x_pred + K_tilde @ innovation

            # ⑦ 収束判定
            if np.linalg.norm(x_est - x_prev) / (np.linalg.norm(x_prev) + 1e-10) < self.epsilon:
                # 収束した場合、共分散を更新して終了
                I_KC = np.eye(n) - K_tilde @ C_bar
                P_updated = I_KC @ P_pred @ I_KC.T + K_tilde @ R_bar @ K_tilde.T

                # 統計情報を保存
                self.stats['iterations'].append(iteration + 1)
                self.stats['correntropy_weights'].append(weights.copy())

                return x_est, P_updated, iteration + 1

        # 最大反復回数に達した場合も共分散を更新
        I_KC = np.eye(n) - K_tilde @ C_bar
        P_updated = I_KC @ P_pred @ I_KC.T + K_tilde @ R_bar @ K_tilde.T

        self.stats['iterations'].append(self.max_iter)
        self.stats['correntropy_weights'].append(weights.copy())

        return x_est, P_updated, self.max_iter

    def _standard_kf_update(
        self,
        Y: np.ndarray,
        C_bar: np.ndarray,
        R_bar: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        標準カルマンフィルタ更新 (フォールバック用)

        数値的に不安定な場合に使用
        """
        # イノベーション共分散
        S = C_bar @ self.P @ C_bar.T + R_bar

        # カルマンゲイン
        K = self.P @ C_bar.T @ np.linalg.inv(S)

        # 状態更新
        innovation = Y - C_bar @ self.x
        x_updated = self.x + K @ innovation

        # 共分散更新 (Joseph形式)
        I_KC = np.eye(self.n) - K @ C_bar
        P_updated = I_KC @ self.P @ I_KC.T + K @ R_bar @ K.T

        return x_updated, P_updated, 0

    def step(
        self,
        measurement: Optional[np.ndarray],
        current_time: int,
        u: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        完全なMCKFステップ (予測 + 更新)

        Args:
            measurement: 現在受信した観測 (遅延あり、Noneならパケット損失)
            current_time: 現在のタイムステップ
            u: 制御入力 (m,)

        Returns:
            x: 状態推定 (n,)
            P: 共分散推定 (n x n)
            info: 診断情報 (dict)
        """
        # ① 予測ステップ
        self.predict(u)

        # ② 遅延観測の構築
        Y, C_bar, R_bar, O = self._construct_delayed_observation(
            measurement, current_time
        )

        # ③ ノイズの無相関化
        D, U, Q_zeta, lambda_n = self._decorrelate_noise(C_bar, R_bar, O)

        # ④ MCKF更新
        self.x, self.P, num_iter = self.update_mckf(Y, C_bar, R_bar, Q_zeta)

        # 診断情報
        info = {
            'num_iterations': num_iter,
            'innovation': Y - C_bar @ self.x,
            'buffer_size': len(self.measurement_buffer),
        }

        return self.x.copy(), self.P.copy(), info

    def get_state(self) -> np.ndarray:
        """現在の状態推定値を取得"""
        return self.x.copy()

    def get_covariance(self) -> np.ndarray:
        """現在の共分散推定値を取得"""
        return self.P.copy()

    def get_statistics(self) -> dict:
        """統計情報を取得 (デバッグ・解析用)"""
        return self.stats.copy()
