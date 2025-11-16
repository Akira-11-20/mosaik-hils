"""
ThrustModel - 軌道制御用推力計算モデル

実装済みの制御アルゴリズム:
- PD制御 (PDThrustModel)
- ホーマン遷移 (HohmannThrustModel)
- Lambert問題ベース遷移 (LambertThrustModel)
- HCW編隊飛行 (HCWThrustModel)

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


class HCWThrustModel(ThrustModel):
    """
    Hill-Clohessy-Wiltshire (HCW) 方程式ベースの編隊飛行制御

    相対運動方程式を使用して、Chief衛星に対するDeputy衛星の
    相対位置を制御する。

    HCW方程式（円軌道近似）:
        ẍ - 2nẏ - 3n²x = Fx/m
        ÿ + 2nẋ       = Fy/m
        z̈ + n²z       = Fz/m

    ここで:
        (x, y, z): LVLH座標系での相対位置
        n: 平均運動（軌道角速度）= √(μ/a³)
        F: 制御力
        m: 衛星質量
    """

    def __init__(
        self,
        target_relative_position=None,
        target_relative_velocity=None,
        chief_position=None,
        chief_velocity=None,
        mu=3.986004418e14,
        kp_x=0.01,
        kp_y=0.01,
        kp_z=0.01,
        kd_x=0.1,
        kd_y=0.1,
        kd_z=0.1,
        max_thrust=1.0,
    ):
        """
        初期化

        Args:
            target_relative_position: 目標相対位置 [x, y, z] [m] (LVLH座標系)
            target_relative_velocity: 目標相対速度 [vx, vy, vz] [m/s] (LVLH座標系)
            chief_position: Chief衛星の初期位置 [x, y, z] [m] (ECI座標系)
            chief_velocity: Chief衛星の初期速度 [vx, vy, vz] [m/s] (ECI座標系)
            mu: 重力定数 [m³/s²]
            kp_x, kp_y, kp_z: 位置ゲイン
            kd_x, kd_y, kd_z: 速度ゲイン
            max_thrust: 最大推力 [N]
        """
        super().__init__(target_position=None, control_gain=1.0)

        # 目標相対位置・速度（LVLH座標系）
        self.target_rel_pos = (
            np.array(target_relative_position) if target_relative_position is not None else np.zeros(3)
        )
        self.target_rel_vel = (
            np.array(target_relative_velocity) if target_relative_velocity is not None else np.zeros(3)
        )

        # Chief衛星の初期状態（参照軌道）
        self.chief_pos_0 = np.array(chief_position) if chief_position is not None else None
        self.chief_vel_0 = np.array(chief_velocity) if chief_velocity is not None else None

        # 重力定数
        self.mu = mu

        # PD制御ゲイン
        self.kp = np.array([kp_x, kp_y, kp_z])
        self.kd = np.array([kd_x, kd_y, kd_z])

        # 推力制限
        self.max_thrust = max_thrust

        print("[HCWThrustModel] Initialized")
        print(f"  Target relative position (LVLH): {self.target_rel_pos} m")
        print(f"  Target relative velocity (LVLH): {self.target_rel_vel} m/s")
        print(f"  Kp: {self.kp}")
        print(f"  Kd: {self.kd}")
        print(f"  Max thrust: {self.max_thrust} N")

    def calculate_thrust(self, deputy_position, deputy_velocity, chief_position=None, chief_velocity=None, time=None):
        """
        HCW方程式ベースのPD制御による推力計算

        Args:
            deputy_position: Deputy衛星の位置 [x, y, z] [m] (ECI座標系)
            deputy_velocity: Deputy衛星の速度 [vx, vy, vz] [m/s] (ECI座標系)
            chief_position: Chief衛星の位置 [x, y, z] [m] (ECI座標系、Noneなら初期値使用)
            chief_velocity: Chief衛星の速度 [vx, vy, vz] [m/s] (ECI座標系、Noneなら初期値使用)
            time: 現在時刻 [s] (オプション)

        Returns:
            thrust: 推力ベクトル [N] (ECI座標系)
        """
        if time is not None:
            self.current_time = time

        # Chief位置が指定されていない場合は初期値を使用
        if chief_position is None:
            chief_position = self.chief_pos_0
        if chief_velocity is None:
            chief_velocity = self.chief_vel_0

        if chief_position is None or chief_velocity is None:
            # Chiefデータがない場合はゼロ推力
            return np.zeros(3)

        # ECI座標系 → LVLH座標系への変換
        rel_pos_lvlh, rel_vel_lvlh = self._eci_to_lvlh(
            deputy_position, deputy_velocity, chief_position, chief_velocity
        )

        # 平均運動の計算（Chief衛星の軌道から）
        r_chief = np.linalg.norm(chief_position)
        n = np.sqrt(self.mu / r_chief**3)  # 平均運動 [rad/s]

        # 相対位置・速度の誤差（LVLH座標系）
        pos_error = rel_pos_lvlh - self.target_rel_pos
        vel_error = rel_vel_lvlh - self.target_rel_vel

        # HCW方程式ベースのPD制御（LVLH座標系）
        # F = -Kp*(rel_pos - target) - Kd*(rel_vel - target_vel) + HCW補償項

        # 基本PD制御
        thrust_lvlh = -self.kp * pos_error - self.kd * vel_error

        # HCW方程式の非線形項の補償（フィードフォワード）
        # ẍ - 2nẏ - 3n²x = 0 の右辺を補償
        thrust_lvlh[0] += 2 * n * rel_vel_lvlh[1] + 3 * n**2 * rel_pos_lvlh[0]
        # ÿ + 2nẋ = 0 の右辺を補償
        thrust_lvlh[1] += -2 * n * rel_vel_lvlh[0]
        # z̈ + n²z = 0 の右辺を補償
        thrust_lvlh[2] += n**2 * rel_pos_lvlh[2]

        # 推力制限（LVLH座標系で）
        thrust_magnitude = np.linalg.norm(thrust_lvlh)
        if thrust_magnitude > self.max_thrust:
            thrust_lvlh = thrust_lvlh * (self.max_thrust / thrust_magnitude)

        # LVLH座標系 → ECI座標系への変換
        thrust_eci = self._lvlh_to_eci(thrust_lvlh, chief_position, chief_velocity)

        return thrust_eci

    def _eci_to_lvlh(self, deputy_pos, deputy_vel, chief_pos, chief_vel):
        """
        ECI座標系からLVLH座標系への変換

        Args:
            deputy_pos: Deputy位置 (ECI)
            deputy_vel: Deputy速度 (ECI)
            chief_pos: Chief位置 (ECI)
            chief_vel: Chief速度 (ECI)

        Returns:
            rel_pos_lvlh: 相対位置 (LVLH)
            rel_vel_lvlh: 相対速度 (LVLH)
        """
        # ECI座標系での相対位置・速度
        rel_pos_eci = deputy_pos - chief_pos
        rel_vel_eci = deputy_vel - chief_vel

        # LVLH座標系の基底ベクトル
        # x軸: Chief位置の方向（Radial）
        x_hat = chief_pos / np.linalg.norm(chief_pos)

        # z軸: 角運動量ベクトルの方向（Cross-track）
        h = np.cross(chief_pos, chief_vel)
        z_hat = h / np.linalg.norm(h)

        # y軸: z × x（Along-track）
        y_hat = np.cross(z_hat, x_hat)

        # 回転行列（ECI → LVLH）
        R_eci_to_lvlh = np.array([x_hat, y_hat, z_hat])

        # 相対位置・速度をLVLH座標系に変換
        rel_pos_lvlh = R_eci_to_lvlh @ rel_pos_eci
        rel_vel_lvlh = R_eci_to_lvlh @ rel_vel_eci

        # LVLH座標系自体の回転を考慮（角速度 ω = h/r²）
        r_chief = np.linalg.norm(chief_pos)
        omega = np.linalg.norm(h) / (r_chief**2)
        omega_vec_lvlh = np.array([0, 0, omega])  # z軸周りの回転

        # 相対速度の補正: v_rel_lvlh = v_deputy_lvlh - v_chief_lvlh - ω × r_rel_lvlh
        rel_vel_lvlh -= np.cross(omega_vec_lvlh, rel_pos_lvlh)

        return rel_pos_lvlh, rel_vel_lvlh

    def _lvlh_to_eci(self, vec_lvlh, chief_pos, chief_vel):
        """
        LVLH座標系からECI座標系への変換

        Args:
            vec_lvlh: ベクトル (LVLH)
            chief_pos: Chief位置 (ECI)
            chief_vel: Chief速度 (ECI)

        Returns:
            vec_eci: ベクトル (ECI)
        """
        # LVLH座標系の基底ベクトル
        x_hat = chief_pos / np.linalg.norm(chief_pos)
        h = np.cross(chief_pos, chief_vel)
        z_hat = h / np.linalg.norm(h)
        y_hat = np.cross(z_hat, x_hat)

        # 回転行列（LVLH → ECI）
        R_lvlh_to_eci = np.array([x_hat, y_hat, z_hat]).T

        # ベクトルをECI座標系に変換
        vec_eci = R_lvlh_to_eci @ vec_lvlh

        return vec_eci
