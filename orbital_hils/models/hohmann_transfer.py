"""
HohmannTransferModel - ホーマン遷移軌道計算モデル

lamberthubライブラリを使用して、任意の高度へのホーマン遷移を計算。

主な機能:
- 2回のインパルス推力計算（初期バーン、最終バーン）
- 遷移時間の計算
- ΔV要求量の計算
- 推力スケジューリング
"""

import numpy as np
from lamberthub import izzo2015


class HohmannTransferModel:
    """
    ホーマン遷移軌道計算モデル

    2つの円軌道間の最適な2インパルス遷移を計算。
    """

    def __init__(
        self,
        mu: float,
        initial_altitude: float,
        target_altitude: float,
        radius_body: float = 6378137.0,
        spacecraft_mass: float = 500.0,
        max_thrust: float = 1.0,
    ):
        """
        初期化

        Args:
            mu: 重力定数 [m³/s²]
            initial_altitude: 初期軌道高度 [m]
            target_altitude: 目標軌道高度 [m]
            radius_body: 天体半径 [m] (デフォルト: 地球半径)
            spacecraft_mass: 衛星質量 [kg]
            max_thrust: 最大推力 [N]
        """
        self.mu = mu
        self.r1 = radius_body + initial_altitude  # 初期軌道半径
        self.r2 = radius_body + target_altitude  # 目標軌道半径
        self.radius_body = radius_body
        self.mass = spacecraft_mass
        self.max_thrust = max_thrust

        # ホーマン遷移パラメータを計算
        self._calculate_hohmann_parameters()

        # 遷移状態
        self.transfer_active = False
        self.burn_phase = None  # "first_burn", "coast", "second_burn", "completed"
        self.transfer_start_time = None
        self.first_burn_time = None
        self.second_burn_time = None

        print("[HohmannTransfer] Initialized")
        print(f"  Initial orbit radius: {self.r1 / 1e3:.2f} km")
        print(f"  Target orbit radius: {self.r2 / 1e3:.2f} km")
        print(f"  Total ΔV: {self.total_delta_v:.2f} m/s")
        print(f"  Transfer time: {self.transfer_time / 60:.2f} min")

    def _calculate_hohmann_parameters(self):
        """ホーマン遷移パラメータを計算"""
        # 遷移軌道の半長軸
        self.a_transfer = (self.r1 + self.r2) / 2

        # 第1回目の速度変化 (初期円軌道 → 遷移楕円軌道)
        v_circular_1 = np.sqrt(self.mu / self.r1)
        v_transfer_periapsis = np.sqrt(self.mu * (2 / self.r1 - 1 / self.a_transfer))
        self.delta_v1 = v_transfer_periapsis - v_circular_1

        # 第2回目の速度変化 (遷移楕円軌道 → 目標円軌道)
        v_circular_2 = np.sqrt(self.mu / self.r2)
        v_transfer_apoapsis = np.sqrt(self.mu * (2 / self.r2 - 1 / self.a_transfer))
        self.delta_v2 = v_circular_2 - v_transfer_apoapsis

        # 総ΔV
        self.total_delta_v = abs(self.delta_v1) + abs(self.delta_v2)

        # 遷移時間 (半周期)
        self.transfer_time = np.pi * np.sqrt(self.a_transfer**3 / self.mu)

        # バーン時間（有限推力を仮定）
        # ΔV = (F/m) * t_burn → t_burn = ΔV * m / F
        max_accel = self.max_thrust / self.mass
        self.burn1_duration = abs(self.delta_v1) / max_accel
        self.burn2_duration = abs(self.delta_v2) / max_accel

        print(f"  ΔV1 (first burn): {self.delta_v1:.2f} m/s")
        print(f"  ΔV2 (second burn): {self.delta_v2:.2f} m/s")
        print(f"  Burn1 duration: {self.burn1_duration:.2f} s")
        print(f"  Burn2 duration: {self.burn2_duration:.2f} s")

    def start_transfer(self, current_time: float):
        """
        ホーマン遷移を開始

        Args:
            current_time: 現在時刻 [s]
        """
        self.transfer_active = True
        self.burn_phase = "first_burn"
        self.transfer_start_time = current_time
        self.first_burn_time = current_time
        self.second_burn_time = current_time + self.transfer_time

        print(f"\n[HohmannTransfer] Transfer started at t={current_time:.2f}s")
        print(f"  First burn: {current_time:.2f}s - {current_time + self.burn1_duration:.2f}s")
        print(f"  Coast phase: {current_time + self.burn1_duration:.2f}s - {self.second_burn_time:.2f}s")
        print(f"  Second burn: {self.second_burn_time:.2f}s - {self.second_burn_time + self.burn2_duration:.2f}s")

    def calculate_thrust(
        self,
        current_time: float,
        position: np.ndarray,
        velocity: np.ndarray,
    ) -> np.ndarray:
        """
        現在時刻での推力ベクトルを計算

        Args:
            current_time: 現在時刻 [s]
            position: 現在位置 [x, y, z] [m]
            velocity: 現在速度 [vx, vy, vz] [m/s]

        Returns:
            thrust: 推力ベクトル [Fx, Fy, Fz] [N]
        """
        if not self.transfer_active:
            return np.zeros(3)

        # 第1回バーン（初期軌道から遷移軌道へ）
        if self.burn_phase == "first_burn":
            if current_time < self.first_burn_time + self.burn1_duration:
                # 速度方向に推力
                v_norm = np.linalg.norm(velocity)
                if v_norm > 1e-6:
                    thrust_direction = velocity / v_norm
                    # ΔV1の符号に応じて推力方向を決定
                    if self.delta_v1 > 0:
                        thrust = self.max_thrust * thrust_direction
                    else:
                        thrust = -self.max_thrust * thrust_direction
                    return thrust
                else:
                    return np.zeros(3)
            else:
                # 第1回バーン終了、コースト開始
                self.burn_phase = "coast"
                print(f"[HohmannTransfer] First burn completed at t={current_time:.2f}s")
                return np.zeros(3)

        # コースト（惰性飛行）
        if self.burn_phase == "coast":
            if current_time >= self.second_burn_time:
                # 第2回バーン開始
                self.burn_phase = "second_burn"
                print(f"[HohmannTransfer] Second burn started at t={current_time:.2f}s")
            return np.zeros(3)

        # 第2回バーン（遷移軌道から目標軌道へ）
        if self.burn_phase == "second_burn":
            if current_time < self.second_burn_time + self.burn2_duration:
                # 速度方向に推力
                v_norm = np.linalg.norm(velocity)
                if v_norm > 1e-6:
                    thrust_direction = velocity / v_norm
                    # ΔV2の符号に応じて推力方向を決定
                    if self.delta_v2 > 0:
                        thrust = self.max_thrust * thrust_direction
                    else:
                        thrust = -self.max_thrust * thrust_direction
                    return thrust
                else:
                    return np.zeros(3)
            else:
                # 第2回バーン終了、遷移完了
                self.burn_phase = "completed"
                self.transfer_active = False
                print(f"[HohmannTransfer] Transfer completed at t={current_time:.2f}s")
                return np.zeros(3)

        # 遷移完了
        return np.zeros(3)

    def get_status(self) -> dict:
        """
        現在の遷移状態を取得

        Returns:
            status: 状態情報の辞書
        """
        return {
            "transfer_active": self.transfer_active,
            "burn_phase": self.burn_phase,
            "delta_v1": self.delta_v1,
            "delta_v2": self.delta_v2,
            "total_delta_v": self.total_delta_v,
            "transfer_time": self.transfer_time,
            "burn1_duration": self.burn1_duration,
            "burn2_duration": self.burn2_duration,
        }


class LambertTransferModel:
    """
    Lambert問題ベースの軌道遷移モデル

    任意の位置から任意の位置への遷移を計算。
    lamberthubライブラリを使用して解を求める。
    """

    def __init__(
        self,
        mu: float,
        spacecraft_mass: float = 500.0,
        max_thrust: float = 1.0,
    ):
        """
        初期化

        Args:
            mu: 重力定数 [m³/s²]
            spacecraft_mass: 衛星質量 [kg]
            max_thrust: 最大推力 [N]
        """
        self.mu = mu
        self.mass = spacecraft_mass
        self.max_thrust = max_thrust

        # 遷移状態
        self.transfer_active = False
        self.initial_velocity = None
        self.final_velocity = None
        self.delta_v1 = None
        self.delta_v2 = None
        self.transfer_time = None

        print("[LambertTransfer] Initialized")
        print(f"  Spacecraft mass: {self.mass} kg")
        print(f"  Max thrust: {self.max_thrust} N")

    def solve_lambert(
        self,
        r1: np.ndarray,
        r2: np.ndarray,
        tof: float,
        v1_current: np.ndarray = None,
        prograde: bool = True,
    ):
        """
        Lambert問題を解いて遷移軌道を計算

        Args:
            r1: 初期位置ベクトル [m]
            r2: 目標位置ベクトル [m]
            tof: 飛行時間 (Time of Flight) [s]
            v1_current: 現在の速度ベクトル [m/s] (ΔV計算用、Noneなら無視)
            prograde: 順行軌道かどうか (True: 順行, False: 逆行)

        Returns:
            (v1, v2): 初期速度、最終速度ベクトル [m/s]
        """
        print(f"\n[LambertTransfer] Solving Lambert problem...")
        print(f"  r1: {r1 / 1e3} km")
        print(f"  r2: {r2 / 1e3} km")
        print(f"  Time of flight: {tof:.2f} s ({tof / 60:.2f} min)")

        # lamberthubで解く
        # izzo2015は高速かつ安定したアルゴリズム
        M = 0  # 周回数（0 = 直接遷移）

        try:
            v1, v2 = izzo2015(self.mu, r1, r2, tof, M=M, prograde=prograde)

            print(f"  v1 (initial velocity): {v1} m/s")
            print(f"  v2 (final velocity): {v2} m/s")

            # ΔV計算
            if v1_current is not None:
                delta_v1 = np.linalg.norm(v1 - v1_current)
                print(f"  ΔV1 (initial burn): {delta_v1:.2f} m/s")

            self.initial_velocity = v1
            self.final_velocity = v2
            self.transfer_time = tof

            return v1, v2

        except Exception as e:
            print(f"  ⚠️  Lambert solver failed: {e}")
            return None, None

    def start_transfer(
        self,
        current_time: float,
        r1: np.ndarray,
        r2: np.ndarray,
        tof: float,
        v1_current: np.ndarray,
    ):
        """
        軌道遷移を開始

        Args:
            current_time: 現在時刻 [s]
            r1: 初期位置 [m]
            r2: 目標位置 [m]
            tof: 飛行時間 [s]
            v1_current: 現在の速度 [m/s]
        """
        v1, v2 = self.solve_lambert(r1, r2, tof, v1_current)

        if v1 is None or v2 is None:
            print("[LambertTransfer] Transfer start failed (Lambert solver error)")
            return False

        # ΔVを計算
        self.delta_v1 = v1 - v1_current
        # NOTE: delta_v2は目標位置に到達時に必要な速度変化
        # （目標軌道の速度に合わせる場合）

        self.transfer_active = True
        self.transfer_start_time = current_time

        print(f"\n[LambertTransfer] Transfer started at t={current_time:.2f}s")
        print(f"  ΔV1: {np.linalg.norm(self.delta_v1):.2f} m/s")

        return True


if __name__ == "__main__":
    # テスト: 400km → 600km ホーマン遷移
    print("=" * 60)
    print("Hohmann Transfer Test: 400km → 600km")
    print("=" * 60)

    # 地球の重力定数
    MU_EARTH = 3.986004418e14
    RADIUS_EARTH = 6378137.0

    # ホーマン遷移モデル
    hohmann = HohmannTransferModel(
        mu=MU_EARTH,
        initial_altitude=400e3,  # 400km
        target_altitude=600e3,  # 600km
        radius_body=RADIUS_EARTH,
        spacecraft_mass=500.0,
        max_thrust=1.0,
    )

    print("\n" + "=" * 60)
    print("Lambert Transfer Test")
    print("=" * 60)

    lambert = LambertTransferModel(
        mu=MU_EARTH,
        spacecraft_mass=500.0,
        max_thrust=1.0,
    )

    # 初期位置・速度（400km円軌道）
    r1 = np.array([RADIUS_EARTH + 400e3, 0.0, 0.0])
    v1 = np.array([0.0, np.sqrt(MU_EARTH / r1[0]), 0.0])

    # 目標位置（600km円軌道、90度先）
    r2_radius = RADIUS_EARTH + 600e3
    r2 = np.array([0.0, r2_radius, 0.0])

    # 飛行時間（ホーマン遷移の半周期）
    tof = hohmann.transfer_time

    # Lambert問題を解く
    v1_lambert, v2_lambert = lambert.solve_lambert(r1, r2, tof, v1)

    if v1_lambert is not None:
        print(f"\n✅ Lambert solution found:")
        print(f"  Required ΔV1: {np.linalg.norm(v1_lambert - v1):.2f} m/s")
        print(f"  Hohmann ΔV1: {abs(hohmann.delta_v1):.2f} m/s")
