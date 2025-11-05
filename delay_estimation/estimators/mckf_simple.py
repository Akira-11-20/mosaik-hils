"""
Simplified Maximum Correntropy Kalman Filter (MCKF)
遅延なしの簡易版 - コレンロピー基準のロバスト性を実証

この実装は遅延処理を省略し、MCKFの核心である
「最大コレンロピー基準による外れ値抑制」に焦点を当てています。
"""

import numpy as np
from typing import Tuple, Optional


class SimpleMCKF:
    """
    簡易版MCKF - 遅延なし、コレンロピー基準のみ

    通常のカルマンフィルタ + ガウスカーネルによる重み付けで、
    外れ値を含む非ガウスノイズに対してロバストな推定を実現。

    アルゴリズム:
    1. 予測ステップ (通常のKFと同じ)
    2. 不動点反復による更新:
       - 残差を計算
       - ガウスカーネルで重み計算
       - 重み付き共分散でカルマンゲイン更新
       - 状態更新
       - 収束まで繰り返し
    """

    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        C: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        x0: np.ndarray,
        P0: np.ndarray,
        kernel_bandwidth: float = 2.0,
        max_iterations: int = 10,
        convergence_threshold: float = 1e-4,
    ):
        """
        Args:
            A: 状態遷移行列 (n x n)
            B: 制御入力行列 (n x m)
            C: 観測行列 (p x n)
            Q: プロセスノイズ共分散 (n x n)
            R: 観測ノイズ共分散 (p x p)
            x0: 初期状態 (n,)
            P0: 初期共分散 (n x n)
            kernel_bandwidth: カーネル帯域幅 η (小→外れ値抑制強)
            max_iterations: 不動点反復の最大回数
            convergence_threshold: 収束判定閾値
        """
        self.A = A
        self.B = B
        self.C = C
        self.Q = Q
        self.R = R

        self.x = x0.copy()
        self.P = P0.copy()

        self.n = A.shape[0]  # 状態次元
        self.m = B.shape[1]  # 制御入力次元
        self.p = C.shape[0]  # 観測次元

        # MCKF固有のパラメータ
        self.eta = kernel_bandwidth  # ガウスカーネル帯域幅
        self.max_iter = max_iterations
        self.epsilon = convergence_threshold

        # 統計情報
        self.num_iterations_history = []

    def predict(self, u: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        予測ステップ (通常のKFと同じ)

        Args:
            u: 制御入力 (m,)

        Returns:
            x_pred: 予測状態
            P_pred: 予測共分散
        """
        if u is None:
            u = np.zeros(self.m)

        # 状態予測
        self.x = self.A @ self.x + self.B @ u

        # 共分散予測
        self.P = self.A @ self.P @ self.A.T + self.Q

        return self.x.copy(), self.P.copy()

    def _gaussian_kernel(self, residual: np.ndarray) -> np.ndarray:
        """
        ガウスカーネル重みの計算

        weight = exp(-residual² / (2*η²))

        Args:
            residual: 残差ベクトル

        Returns:
            weights: 重みベクトル
        """
        return np.exp(-residual**2 / (2 * self.eta**2))

    def update_mckf(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        MCKF更新ステップ (不動点反復)

        最大コレンロピー基準で状態を更新:
        - 小さい残差 → 重み大 → 観測を信用
        - 大きい残差 → 重み小 → 観測を無視 (外れ値)

        Args:
            y: 観測 (p,)

        Returns:
            x_updated: 更新後の状態
            P_updated: 更新後の共分散
            num_iterations: 反復回数
        """
        # 予測値を保存
        x_pred = self.x.copy()
        P_pred = self.P.copy()

        # Cholesky分解 (白色化のため)
        try:
            L_P = np.linalg.cholesky(P_pred)
            L_R = np.linalg.cholesky(self.R)
        except np.linalg.LinAlgError:
            # 分解失敗時は通常のKFにフォールバック
            return self._standard_update(y)

        # 不動点反復の初期値
        x_est = x_pred.copy()

        for iteration in range(self.max_iter):
            x_prev = x_est.copy()

            # ① イノベーション (観測残差)
            innovation = y - self.C @ x_est

            # ② 予測残差
            pred_residual = x_est - x_pred

            # ③ 白色化
            # 共分散を単位行列に変換して、各要素を同じスケールで扱う
            try:
                L_P_inv = np.linalg.inv(L_P)
                L_R_inv = np.linalg.inv(L_R)

                # 白色化された残差
                white_pred_residual = L_P_inv @ pred_residual
                white_innovation = L_R_inv @ innovation

            except np.linalg.LinAlgError:
                break

            # ④ ガウスカーネル重み
            # 予測部と観測部それぞれに重み付け
            weight_pred = self._gaussian_kernel(white_pred_residual)
            weight_obs = self._gaussian_kernel(white_innovation)

            # ⑤ 重み付き逆共分散 (Information Form)
            # MATLABコードに準拠:
            # Pke_hat_inv = inv(Bpk)' * Cx * inv(Bpk)
            # R_hat_inv = inv(Brk)' * Cy * inv(Brk)
            # これは逆共分散 (Information matrix) を計算している

            T_x = np.diag(weight_pred)
            T_y = np.diag(weight_obs)

            try:
                # 逆共分散を計算
                # P̃⁻^{-1} = L_P^{-T} * T_x * L_P^{-1}
                # R̃^{-1} = L_R^{-T} * T_y * L_R^{-1}
                P_tilde_inv = L_P_inv.T @ T_x @ L_P_inv
                R_tilde_inv = L_R_inv.T @ T_y @ L_R_inv

                # 正定値性を保証
                P_tilde_inv += np.eye(self.n) * 1e-8
                R_tilde_inv += np.eye(self.p) * 1e-8

            except:
                break

            # ⑥ 重み付きカルマンゲイン (Information Formを使用)
            # MATLABコード: Gk = inv(H'*R_hat_inv*H + Pke_hat_inv) * H' * R_hat_inv
            try:
                # Information formのゲイン計算
                K_tilde = np.linalg.inv(
                    self.C.T @ R_tilde_inv @ self.C + P_tilde_inv
                ) @ self.C.T @ R_tilde_inv
            except np.linalg.LinAlgError:
                break

            # ⑦ 状態更新
            x_est = x_pred + K_tilde @ (y - self.C @ x_pred)

            # ⑧ 収束判定
            relative_change = np.linalg.norm(x_est - x_prev) / (np.linalg.norm(x_prev) + 1e-10)
            if relative_change < self.epsilon:
                # 収束したので共分散を更新して終了
                I_KC = np.eye(self.n) - K_tilde @ self.C
                P_updated = I_KC @ P_pred @ I_KC.T + K_tilde @ self.R @ K_tilde.T

                self.num_iterations_history.append(iteration + 1)
                return x_est, P_updated, iteration + 1

        # 最大反復回数到達
        I_KC = np.eye(self.n) - K_tilde @ self.C
        P_updated = I_KC @ P_pred @ I_KC.T + K_tilde @ self.R @ K_tilde.T

        self.num_iterations_history.append(self.max_iter)
        return x_est, P_updated, self.max_iter

    def _standard_update(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
        """標準カルマンフィルタ更新 (フォールバック)"""
        # イノベーション共分散
        S = self.C @ self.P @ self.C.T + self.R

        # カルマンゲイン
        K = self.P @ self.C.T @ np.linalg.inv(S)

        # 状態更新
        innovation = y - self.C @ self.x
        x_updated = self.x + K @ innovation

        # 共分散更新 (Joseph形式)
        I_KC = np.eye(self.n) - K @ self.C
        P_updated = I_KC @ self.P @ I_KC.T + K @ self.R @ K.T

        return x_updated, P_updated, 0

    def step(
        self, y: np.ndarray, u: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        完全なフィルタステップ (予測 + MCKF更新)

        Args:
            y: 観測 (p,)
            u: 制御入力 (m,)

        Returns:
            x: 状態推定
            P: 共分散推定
            info: 診断情報
        """
        # 予測
        self.predict(u)

        # MCKF更新
        self.x, self.P, num_iter = self.update_mckf(y)

        info = {
            'num_iterations': num_iter,
            'innovation': y - self.C @ self.x,
        }

        return self.x.copy(), self.P.copy(), info

    def get_state(self) -> np.ndarray:
        """現在の状態推定値"""
        return self.x.copy()

    def get_covariance(self) -> np.ndarray:
        """現在の共分散推定値"""
        return self.P.copy()

    def get_avg_iterations(self) -> float:
        """平均反復回数"""
        if len(self.num_iterations_history) == 0:
            return 0.0
        return np.mean(self.num_iterations_history)
