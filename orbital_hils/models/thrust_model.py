"""
ThrustModel - 軌道制御用推力計算モデル

実装済みの制御アルゴリズム:
- PD制御 (PDThrustModel)
- ホーマン遷移 (HohmannThrustModel)
- Lambert問題ベース遷移 (LambertThrustModel)

将来的な実装予定:
- LQR (Linear Quadratic Regulator)
- MPC (Model Predictive Control)
"""

import numpy as np


class ThrustModel:
    """
    軌道制御用推力計算モデル（基底クラス）

    デフォルトではゼロ推力を返す（自由軌道運動）。
    制御を行う場合は、このクラスを継承して実装する。
    """

    def __init__(self, target_position=None, control_gain=1.0):
        """
        初期化

        Args:
            target_position: 目標位置 [x, y, z] [m]
            control_gain: 制御ゲイン
        """
        self.target = target_position if target_position is not None else np.zeros(3)
        self.gain = control_gain
        self.current_time = 0.0

        print("[ThrustModel] Initialized (zero thrust - free orbit)")
        print(f"  Target position: {self.target} m")
        print(f"  Control gain: {self.gain}")

    def calculate_thrust(self, position, velocity, time=None):
        """
        推力ベクトルを計算

        Args:
            position: 現在位置 [x, y, z] [m]
            velocity: 現在速度 [vx, vy, vz] [m/s]
            time: 現在時刻 [s] (オプション)

        Returns:
            thrust: 推力ベクトル [Fx, Fy, Fz] [N]
        """
        if time is not None:
            self.current_time = time

        # デフォルトはゼロ推力（自由軌道運動）
        return np.zeros(3)

    def update_target(self, target_position):
        """
        目標位置を更新

        Args:
            target_position: 新しい目標位置 [x, y, z] [m]
        """
        self.target = np.array(target_position)
        print(f"[ThrustModel] Target updated: {self.target} m")

    def update_gain(self, control_gain):
        """
        制御ゲインを更新

        Args:
            control_gain: 新しい制御ゲイン
        """
        self.gain = control_gain
        print(f"[ThrustModel] Gain updated: {self.gain}")


class PDThrustModel(ThrustModel):
    """
    PD制御による推力計算

    F = Kp*(r_target - r) + Kd*(v_target - v)
    """

    def __init__(
        self,
        target_position=None,
        kp=1e-3,
        kd=1e-2,
        max_thrust=1.0,
        target_velocity=None,
    ):
        """
        初期化

        Args:
            target_position: 目標位置 [m]
            kp: 位置ゲイン
            kd: 速度ゲイン
            max_thrust: 最大推力 [N]
            target_velocity: 目標速度 [m/s] (Noneなら0)
        """
        super().__init__(target_position, control_gain=kp)
        self.kp = kp
        self.kd = kd
        self.max_thrust = max_thrust
        self.target_velocity = target_velocity if target_velocity is not None else np.zeros(3)

        print("[PDThrustModel] Initialized")
        print(f"  Kp: {self.kp}")
        print(f"  Kd: {self.kd}")
        print(f"  Max thrust: {self.max_thrust} N")

    def calculate_thrust(self, position, velocity, time=None):
        """
        PD制御による推力計算

        Args:
            position: 現在位置 [m]
            velocity: 現在速度 [m/s]
            time: 現在時刻 [s] (オプション)

        Returns:
            thrust: 推力ベクトル [N]
        """
        if time is not None:
            self.current_time = time

        # 誤差計算
        position_error = self.target - position
        velocity_error = self.target_velocity - velocity

        # PD制御則
        thrust = self.kp * position_error + self.kd * velocity_error

        # 推力制限
        thrust_magnitude = np.linalg.norm(thrust)
        if thrust_magnitude > self.max_thrust:
            thrust = thrust * (self.max_thrust / thrust_magnitude)

        return thrust


class HohmannThrustModel(ThrustModel):
    """
    ホーマン遷移制御モデル

    2つの円軌道間の最適な2インパルス遷移を実行。
    """

    def __init__(
        self,
        mu: float,
        initial_altitude: float,
        target_altitude: float,
        radius_body: float = 6378137.0,
        spacecraft_mass: float = 500.0,
        max_thrust: float = 1.0,
        start_time: float = 0.0,
    ):
        """
        初期化

        Args:
            mu: 重力定数 [m³/s²]
            initial_altitude: 初期軌道高度 [m]
            target_altitude: 目標軌道高度 [m]
            radius_body: 天体半径 [m]
            spacecraft_mass: 衛星質量 [kg]
            max_thrust: 最大推力 [N]
            start_time: 遷移開始時刻 [s]
        """
        super().__init__(control_gain=1.0)

        # HohmannTransferModelをインポート
        from models.hohmann_transfer import HohmannTransferModel

        self.hohmann = HohmannTransferModel(
            mu=mu,
            initial_altitude=initial_altitude,
            target_altitude=target_altitude,
            radius_body=radius_body,
            spacecraft_mass=spacecraft_mass,
            max_thrust=max_thrust,
        )

        self.start_time = start_time
        self.transfer_started = False

        print("[HohmannThrustModel] Initialized")
        print(f"  Transfer will start at t={start_time:.2f}s")

    def calculate_thrust(self, position, velocity, time=None):
        """
        ホーマン遷移の推力を計算

        Args:
            position: 現在位置 [m]
            velocity: 現在速度 [m/s]
            time: 現在時刻 [s]

        Returns:
            thrust: 推力ベクトル [N]
        """
        if time is not None:
            self.current_time = time

        # 遷移開始判定
        if not self.transfer_started and self.current_time >= self.start_time:
            self.hohmann.start_transfer(self.current_time)
            self.transfer_started = True

        # 推力計算
        return self.hohmann.calculate_thrust(self.current_time, position, velocity)

    def get_status(self):
        """遷移状態を取得"""
        return self.hohmann.get_status()


class LambertThrustModel(ThrustModel):
    """
    Lambert問題ベースの軌道遷移制御モデル

    任意の位置から任意の位置への遷移を実行。
    """

    def __init__(
        self,
        mu: float,
        target_position: np.ndarray,
        flight_time: float,
        spacecraft_mass: float = 500.0,
        max_thrust: float = 1.0,
        start_time: float = 0.0,
    ):
        """
        初期化

        Args:
            mu: 重力定数 [m³/s²]
            target_position: 目標位置 [m]
            flight_time: 飛行時間 [s]
            spacecraft_mass: 衛星質量 [kg]
            max_thrust: 最大推力 [N]
            start_time: 遷移開始時刻 [s]
        """
        super().__init__(target_position=target_position, control_gain=1.0)

        # LambertTransferModelをインポート
        from models.hohmann_transfer import LambertTransferModel

        self.lambert = LambertTransferModel(
            mu=mu,
            spacecraft_mass=spacecraft_mass,
            max_thrust=max_thrust,
        )

        self.flight_time = flight_time
        self.start_time = start_time
        self.transfer_started = False
        self.delta_v = None

        print("[LambertThrustModel] Initialized")
        print(f"  Target position: {target_position / 1e3} km")
        print(f"  Flight time: {flight_time:.2f} s")

    def calculate_thrust(self, position, velocity, time=None):
        """
        Lambert遷移の推力を計算

        Args:
            position: 現在位置 [m]
            velocity: 現在速度 [m/s]
            time: 現在時刻 [s]

        Returns:
            thrust: 推力ベクトル [N]
        """
        if time is not None:
            self.current_time = time

        # 遷移開始判定
        if not self.transfer_started and self.current_time >= self.start_time:
            success = self.lambert.start_transfer(
                self.current_time,
                position,
                self.target,
                self.flight_time,
                velocity,
            )

            if success:
                self.transfer_started = True
                self.delta_v = self.lambert.delta_v1

        # インパルス推力を有限推力で近似
        # TODO: より詳細な実装
        return np.zeros(3)


if __name__ == "__main__":
    # テスト
    print("=" * 60)
    print("ThrustModel Test")
    print("=" * 60)

    model = ThrustModel(target_position=[0, 0, 0], control_gain=1.0)

    # テスト位置・速度
    test_position = np.array([6778137.0, 0.0, 0.0])
    test_velocity = np.array([0.0, 7500.0, 0.0])

    thrust = model.calculate_thrust(test_position, test_velocity)
    print("\nTest thrust calculation:")
    print(f"  Position: {test_position} m")
    print(f"  Velocity: {test_velocity} m/s")
    print(f"  Thrust: {thrust} N")
