"""
軌道力学シミュレーション用パラメータ設定

地球周回軌道の二体問題シミュレーションに必要な物理定数、
軌道要素、シミュレーション設定を定義。
"""

from dataclasses import dataclass

import numpy as np

# ========================================
# 物理定数
# ========================================


@dataclass
class CelestialBodyConstants:
    """中心天体の物理定数"""

    # 地球
    MU_EARTH: float = 3.986004418e14  # 重力定数 [m³/s²]
    RADIUS_EARTH: float = 6378137.0  # 赤道半径 [m]
    J2_EARTH: float = 1.08263e-3  # J2摂動項（将来の実装用）

    # 月（参考）
    MU_MOON: float = 4.9028e12  # 重力定数 [m³/s²]
    RADIUS_MOON: float = 1737400.0  # 半径 [m]


# ========================================
# 軌道パラメータ
# ========================================


@dataclass
class OrbitalParameters:
    """軌道シミュレーションパラメータ"""

    # 中心天体
    mu: float  # 重力定数 [m³/s²]
    radius_body: float  # 天体半径 [m]

    # 初期軌道要素（ケプラー要素）
    semi_major_axis: float  # 軌道長半径 [m]
    eccentricity: float  # 離心率 [-]
    inclination: float  # 軌道傾斜角 [deg]
    raan: float  # 昇交点赤経 [deg] (Right Ascension of Ascending Node)
    arg_periapsis: float  # 近地点引数 [deg]
    true_anomaly: float  # 真近点角 [deg]

    @property
    def altitude(self) -> float:
        """軌道高度 [m]（近地点での高度）"""
        periapsis = self.semi_major_axis * (1 - self.eccentricity)
        return periapsis - self.radius_body

    @property
    def orbital_period(self) -> float:
        """軌道周期 [s]"""
        return 2 * np.pi * np.sqrt(self.semi_major_axis**3 / self.mu)

    @property
    def mean_motion(self) -> float:
        """平均運動 [rad/s]"""
        return np.sqrt(self.mu / self.semi_major_axis**3)

    def to_cartesian(self) -> tuple:
        """
        ケプラー軌道要素から位置・速度ベクトルへ変換

        Returns:
            (position, velocity): 位置[m]と速度[m/s]のnumpy配列
        """
        # 角度をラジアンに変換
        i = np.radians(self.inclination)
        omega = np.radians(self.raan)
        w = np.radians(self.arg_periapsis)
        nu = np.radians(self.true_anomaly)

        # 軌道面座標系での位置・速度
        p = self.semi_major_axis * (1 - self.eccentricity**2)
        r_orbit = p / (1 + self.eccentricity * np.cos(nu))

        # 位置ベクトル（軌道面）
        r_pqw = np.array([r_orbit * np.cos(nu), r_orbit * np.sin(nu), 0.0])

        # 速度ベクトル（軌道面）
        v_pqw = np.sqrt(self.mu / p) * np.array([-np.sin(nu), self.eccentricity + np.cos(nu), 0.0])

        # 回転行列（PQW座標系 → ECI座標系）
        R = self._rotation_matrix_pqw_to_eci(omega, i, w)

        # ECI座標系へ変換
        position = R @ r_pqw
        velocity = R @ v_pqw

        return position, velocity

    @staticmethod
    def _rotation_matrix_pqw_to_eci(omega, i, w):
        """PQW座標系からECI座標系への回転行列"""
        cos_omega = np.cos(omega)
        sin_omega = np.sin(omega)
        cos_i = np.cos(i)
        sin_i = np.sin(i)
        cos_w = np.cos(w)
        sin_w = np.sin(w)

        R = np.array(
            [
                [
                    cos_omega * cos_w - sin_omega * sin_w * cos_i,
                    -cos_omega * sin_w - sin_omega * cos_w * cos_i,
                    sin_omega * sin_i,
                ],
                [
                    sin_omega * cos_w + cos_omega * sin_w * cos_i,
                    -sin_omega * sin_w + cos_omega * cos_w * cos_i,
                    -cos_omega * sin_i,
                ],
                [sin_w * sin_i, cos_w * sin_i, cos_i],
            ]
        )
        return R


# ========================================
# 衛星パラメータ
# ========================================


@dataclass
class SpacecraftParameters:
    """衛星物理パラメータ"""

    mass: float  # 質量 [kg]
    max_thrust: float  # 最大推力 [N]
    specific_impulse: float  # 比推力 [s] (将来の燃料消費計算用)

    @property
    def max_acceleration(self) -> float:
        """最大加速度 [m/s²]"""
        return self.max_thrust / self.mass


# ========================================
# シミュレーション設定
# ========================================


@dataclass
class OrbitalSimulationConfig:
    """軌道シミュレーション設定"""

    # 時間設定
    simulation_time: float  # 総シミュレーション時間 [s]
    time_resolution: float  # 時間解像度 [s]
    step_size: int  # ステップサイズ [時間単位]

    # 中心天体
    celestial_body: str  # "Earth", "Moon", etc.

    # 衛星
    spacecraft: SpacecraftParameters

    # 軌道
    orbit: OrbitalParameters

    @classmethod
    def create_leo_config(
        cls,
        altitude_km: float = 400.0,
        eccentricity: float = 0.0,
        inclination_deg: float = 51.6,
        simulation_time: float = 5700.0,  # デフォルト: 約1軌道周期
        time_resolution: float = 1.0,  # デフォルト: 1秒
    ):
        """
        低軌道（LEO）設定を作成

        Args:
            altitude_km: 軌道高度 [km]
            eccentricity: 離心率
            inclination_deg: 軌道傾斜角 [deg]
            simulation_time: シミュレーション時間 [s]
            time_resolution: 時間解像度 [s]

        Returns:
            OrbitalSimulationConfig
        """
        constants = CelestialBodyConstants()

        semi_major_axis = constants.RADIUS_EARTH + altitude_km * 1e3

        orbit = OrbitalParameters(
            mu=constants.MU_EARTH,
            radius_body=constants.RADIUS_EARTH,
            semi_major_axis=semi_major_axis,
            eccentricity=eccentricity,
            inclination=inclination_deg,
            raan=0.0,
            arg_periapsis=0.0,
            true_anomaly=0.0,
        )

        spacecraft = SpacecraftParameters(
            mass=500.0,  # 小型衛星相当
            max_thrust=1.0,  # 小推力スラスタ
            specific_impulse=200.0,
        )

        return cls(
            simulation_time=simulation_time,
            time_resolution=time_resolution,
            step_size=1,
            celestial_body="Earth",
            spacecraft=spacecraft,
            orbit=orbit,
        )

    @classmethod
    def create_iss_config(cls, simulation_time: float = 5500.0):
        """ISS相当の軌道設定"""
        return cls.create_leo_config(
            altitude_km=408.0,
            eccentricity=0.0,
            inclination_deg=51.64,
            simulation_time=simulation_time,
            time_resolution=1.0,
        )

    @classmethod
    def create_geo_config(cls, simulation_time: float = 86400.0):
        """
        静止軌道（GEO）設定を作成

        Args:
            simulation_time: シミュレーション時間 [s] (デフォルト: 1日)
        """
        constants = CelestialBodyConstants()

        # 静止軌道半径: (μ/n²)^(1/3), n = 2π/T, T = 86400s
        semi_major_axis = 42164000.0  # 約35786km高度

        orbit = OrbitalParameters(
            mu=constants.MU_EARTH,
            radius_body=constants.RADIUS_EARTH,
            semi_major_axis=semi_major_axis,
            eccentricity=0.0,
            inclination=0.0,
            raan=0.0,
            arg_periapsis=0.0,
            true_anomaly=0.0,
        )

        spacecraft = SpacecraftParameters(
            mass=2000.0,  # 通信衛星相当
            max_thrust=10.0,
            specific_impulse=300.0,
        )

        return OrbitalSimulationConfig(
            simulation_time=simulation_time,
            time_resolution=10.0,  # 10秒刻み
            step_size=1,
            celestial_body="Earth",
            spacecraft=spacecraft,
            orbit=orbit,
        )


# ========================================
# .env読み込み
# ========================================


def load_config_from_env(env_file: str = ".env") -> OrbitalSimulationConfig:
    """
    .envファイルから設定を読み込む

    Args:
        env_file: .envファイルのパス（デフォルト: ".env"）

    Returns:
        OrbitalSimulationConfig
    """
    try:
        import os
        from pathlib import Path

        from dotenv import load_dotenv

        # .envファイルを読み込み
        env_path = Path(__file__).parent.parent / env_file
        load_dotenv(env_path)

        # 各パラメータを読み込み
        simulation_time = float(os.getenv("SIMULATION_TIME", "5500.0"))
        time_resolution = float(os.getenv("TIME_RESOLUTION", "1.0"))

        # 軌道パラメータ
        altitude_km = float(os.getenv("ALTITUDE_KM", "408.0"))
        eccentricity = float(os.getenv("ECCENTRICITY", "0.0"))
        inclination_deg = float(os.getenv("INCLINATION_DEG", "51.64"))
        raan_deg = float(os.getenv("RAAN_DEG", "0.0"))
        arg_periapsis_deg = float(os.getenv("ARG_PERIAPSIS_DEG", "0.0"))
        true_anomaly_deg = float(os.getenv("TRUE_ANOMALY_DEG", "0.0"))

        # 衛星パラメータ
        spacecraft_mass = float(os.getenv("SPACECRAFT_MASS", "500.0"))
        max_thrust = float(os.getenv("MAX_THRUST", "1.0"))
        specific_impulse = float(os.getenv("SPECIFIC_IMPULSE", "200.0"))

        # 設定を作成
        constants = CelestialBodyConstants()
        semi_major_axis = constants.RADIUS_EARTH + altitude_km * 1e3

        orbit = OrbitalParameters(
            mu=constants.MU_EARTH,
            radius_body=constants.RADIUS_EARTH,
            semi_major_axis=semi_major_axis,
            eccentricity=eccentricity,
            inclination=inclination_deg,
            raan=raan_deg,
            arg_periapsis=arg_periapsis_deg,
            true_anomaly=true_anomaly_deg,
        )

        spacecraft = SpacecraftParameters(
            mass=spacecraft_mass, max_thrust=max_thrust, specific_impulse=specific_impulse
        )

        print(f"✅ Configuration loaded from {env_file}")
        return OrbitalSimulationConfig(
            simulation_time=simulation_time,
            time_resolution=time_resolution,
            step_size=1,
            celestial_body="Earth",
            spacecraft=spacecraft,
            orbit=orbit,
        )

    except ImportError:
        print("⚠️  python-dotenv not installed, using default ISS config")
        return CONFIG_ISS
    except Exception as e:
        print(f"⚠️  Error loading .env: {e}, using default ISS config")
        return CONFIG_ISS


def get_env_param(key: str, default: any, convert_type: type = str) -> any:
    """
    環境変数から単一のパラメータを取得

    Args:
        key: 環境変数のキー
        default: デフォルト値
        convert_type: 変換する型

    Returns:
        変換後の値
    """
    try:
        import os
        from pathlib import Path

        from dotenv import load_dotenv

        env_path = Path(__file__).parent.parent / ".env"
        if env_path.exists():
            load_dotenv(env_path)

        value = os.getenv(key)
        if value is None:
            return default

        if convert_type == bool:
            return value.lower() in ("true", "1", "yes", "on")
        else:
            return convert_type(value)

    except Exception:
        return default


# ========================================
# プリセット設定
# ========================================

# ISS軌道
CONFIG_ISS = OrbitalSimulationConfig.create_iss_config()

# 低軌道（400km、円軌道）
CONFIG_LEO_400 = OrbitalSimulationConfig.create_leo_config(altitude_km=400.0)

# 低軌道（600km、円軌道）
CONFIG_LEO_600 = OrbitalSimulationConfig.create_leo_config(altitude_km=600.0)

# 静止軌道
CONFIG_GEO = OrbitalSimulationConfig.create_geo_config()


if __name__ == "__main__":
    # 設定のテスト表示
    print("=" * 60)
    print("ISS Configuration")
    print("=" * 60)
    config = CONFIG_ISS
    orbit = config.orbit

    print(f"Altitude: {orbit.altitude / 1e3:.2f} km")
    print(f"Orbital Period: {orbit.orbital_period:.2f} s ({orbit.orbital_period / 60:.2f} min)")
    print(f"Eccentricity: {orbit.eccentricity}")
    print(f"Inclination: {orbit.inclination} deg")

    position, velocity = orbit.to_cartesian()
    print("\nInitial State (ECI frame):")
    print(f"  Position: {position / 1e3} km")
    print(f"  Velocity: {velocity} m/s")
    print(f"  |r|: {np.linalg.norm(position) / 1e3:.2f} km")
    print(f"  |v|: {np.linalg.norm(velocity):.2f} m/s")

    print("\nSpacecraft:")
    print(f"  Mass: {config.spacecraft.mass} kg")
    print(f"  Max Thrust: {config.spacecraft.max_thrust} N")
    print(f"  Max Accel: {config.spacecraft.max_acceleration * 1e3:.3f} mm/s²")
