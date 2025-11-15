"""
ホーマン遷移シナリオ（.env設定ベース）

.envファイルからパラメータを読み込み、ホーマン遷移シミュレーションを実行します。

使用方法:
    cd orbital_hils
    uv run python -m scenarios.hohmann_scenario
"""

import sys
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.orbital_parameters import (
    CelestialBodyConstants,
    OrbitalSimulationConfig,
    get_env_param,
)

from scenarios.orbital_scenario import OrbitalScenario


class HohmannScenario(OrbitalScenario):
    """
    ホーマン遷移シナリオ

    .envファイルから以下のパラメータを読み込みます：
    - CONTROLLER_TYPE: 制御タイプ (zero/pd/hohmann)
    - HOHMANN_INITIAL_ALTITUDE_KM: 初期軌道高度 [km]
    - HOHMANN_TARGET_ALTITUDE_KM: 目標軌道高度 [km]
    - HOHMANN_START_TIME: 遷移開始時刻 [s]
    - MAX_THRUST: 最大推力 [N]
    - SPACECRAFT_MASS: 衛星質量 [kg]
    - PLANT_TIME_CONSTANT: Plant時定数 [s]
    - PLANT_NOISE_STD: Plant計測ノイズ標準偏差
    """

    def __init__(self, config: OrbitalSimulationConfig = None):
        """
        初期化

        Args:
            config: シミュレーション設定（Noneなら.envから読み込み）
        """
        if config is None:
            config = self._load_config_from_env()

        super().__init__(config=config)

        # .envからホーマン遷移パラメータを読み込み
        self.controller_type = get_env_param("CONTROLLER_TYPE", "hohmann", str)
        self.hohmann_initial_altitude = get_env_param("HOHMANN_INITIAL_ALTITUDE_KM", 400.0, float) * 1e3
        self.hohmann_target_altitude = get_env_param("HOHMANN_TARGET_ALTITUDE_KM", 600.0, float) * 1e3
        self.hohmann_start_time = get_env_param("HOHMANN_START_TIME", 100.0, float)
        self.plant_time_constant = get_env_param("PLANT_TIME_CONSTANT", 10.0, float)
        self.plant_noise_std = get_env_param("PLANT_NOISE_STD", 0.01, float)

        print("\n[HohmannScenario] Configuration:")
        print(f"  Controller type: {self.controller_type}")
        print(f"  Initial altitude: {self.hohmann_initial_altitude / 1e3:.2f} km")
        print(f"  Target altitude: {self.hohmann_target_altitude / 1e3:.2f} km")
        print(f"  Transfer start time: {self.hohmann_start_time:.2f} s")
        print(f"  Max thrust: {self.config.spacecraft.max_thrust:.2f} N")
        print(f"  Spacecraft mass: {self.config.spacecraft.mass:.2f} kg")
        print(f"  Plant time constant: {self.plant_time_constant:.2f} s")
        print(f"  Plant noise std: {self.plant_noise_std:.4f}")

    def _load_config_from_env(self) -> OrbitalSimulationConfig:
        """
        .envからシミュレーション設定を読み込む

        Returns:
            OrbitalSimulationConfig
        """
        from config.orbital_parameters import load_config_from_env

        return load_config_from_env()

    def setup_entities(self):
        """エンティティのセットアップ（ホーマン遷移用）"""
        print("\n[HohmannScenario] Setting up entities...")

        # 初期状態の計算
        position, velocity = self.config.orbit.to_cartesian()

        # Controller（ホーマン遷移またはその他）
        controller_sim = self.world.start(
            "OrbitalControllerSim",
            time_resolution=self.config.time_resolution,
            step_size=self.config.step_size,
        )

        if self.controller_type == "hohmann":
            self.controller = controller_sim.OrbitalController(
                controller_type="hohmann",
                mu=self.config.orbit.mu,
                initial_altitude=self.hohmann_initial_altitude,
                target_altitude=self.hohmann_target_altitude,
                radius_body=self.config.orbit.radius_body,
                spacecraft_mass=self.config.spacecraft.mass,
                max_thrust=self.config.spacecraft.max_thrust,
                start_time=self.hohmann_start_time,
            )
            print("  ✅ Hohmann transfer controller created")
            print(f"     {self.hohmann_initial_altitude / 1e3:.0f}km → {self.hohmann_target_altitude / 1e3:.0f}km")
            print(f"     Start time: {self.hohmann_start_time:.0f}s")
        elif self.controller_type == "pd":
            # PD制御器
            target_position = [
                get_env_param("TARGET_POSITION_X", 0.0, float),
                get_env_param("TARGET_POSITION_Y", 0.0, float),
                get_env_param("TARGET_POSITION_Z", 0.0, float),
            ]
            control_gain = get_env_param("CONTROL_GAIN", 1.0, float)
            self.controller = controller_sim.OrbitalController(
                controller_type="pd",
                target_position=target_position,
                control_gain=control_gain,
                max_thrust=self.config.spacecraft.max_thrust,
            )
            print("  ✅ PD controller created")
        else:
            # ゼロ推力（自由軌道運動）
            self.controller = controller_sim.OrbitalController(
                controller_type="zero",
                max_thrust=self.config.spacecraft.max_thrust,
            )
            print("  ✅ Zero-thrust controller created (free orbit)")

        # Plant
        plant_sim = self.world.start(
            "OrbitalPlantSim",
            time_resolution=self.config.time_resolution,
            step_size=self.config.step_size,
        )
        self.plant = plant_sim.OrbitalThrustStand(
            time_constant=self.plant_time_constant,
            noise_std=self.plant_noise_std,
        )
        print(f"  ✅ Plant created (τ={self.plant_time_constant}s, σ={self.plant_noise_std})")

        # Environment
        env_sim = self.world.start(
            "OrbitalEnvSim",
            time_resolution=self.config.time_resolution,
            step_size=self.config.step_size,
        )
        self.spacecraft = env_sim.OrbitalSpacecraft(
            mass=self.config.spacecraft.mass,
            mu=self.config.orbit.mu,
            initial_position=position.tolist(),
            initial_velocity=velocity.tolist(),
            radius_earth=self.config.orbit.radius_body,
        )
        print("  ✅ Environment created")

        # Data Collector
        minimal_mode = get_env_param("MINIMAL_DATA_MODE", False, bool)
        collector_sim = self.world.start(
            "DataCollector",
            time_resolution=self.config.time_resolution,
            step_size=self.config.step_size,
        )
        self.collector = collector_sim.Collector(
            output_dir=str(self.output_dir),
            minimal_mode=minimal_mode,
        )
        print("  ✅ Data collector created")


def main():
    """メイン実行"""
    print("=" * 70)
    print("Hohmann Transfer Scenario (.env based)")
    print("=" * 70)

    # ホーマン遷移パラメータのプレビュー
    controller_type = get_env_param("CONTROLLER_TYPE", "hohmann", str)

    if controller_type == "hohmann":
        from models.hohmann_transfer import HohmannTransferModel

        initial_alt = get_env_param("HOHMANN_INITIAL_ALTITUDE_KM", 400.0, float) * 1e3
        target_alt = get_env_param("HOHMANN_TARGET_ALTITUDE_KM", 600.0, float) * 1e3
        max_thrust = get_env_param("MAX_THRUST", 10.0, float)
        mass = get_env_param("SPACECRAFT_MASS", 500.0, float)

        constants = CelestialBodyConstants()

        # ホーマン遷移パラメータ計算
        hohmann = HohmannTransferModel(
            mu=constants.MU_EARTH,
            initial_altitude=initial_alt,
            target_altitude=target_alt,
            radius_body=constants.RADIUS_EARTH,
            spacecraft_mass=mass,
            max_thrust=max_thrust,
        )

        status = hohmann.get_status()
        print("\n📊 Hohmann Transfer Parameters:")
        print(f"   ΔV1: {status['delta_v1']:+.2f} m/s")
        print(f"   ΔV2: {status['delta_v2']:+.2f} m/s")
        print(f"   Total ΔV: {status['total_delta_v']:.2f} m/s")
        print(f"   Transfer time: {status['transfer_time'] / 60:.2f} min")
        print(f"   Burn1 duration: {status['burn1_duration'] / 60:.2f} min")
        print(f"   Burn2 duration: {status['burn2_duration'] / 60:.2f} min")
        print(
            f"   Total maneuver time: {(status['burn1_duration'] + status['transfer_time'] + status['burn2_duration']) / 60:.2f} min"
        )

    # シナリオ実行
    print("\n▶️  Running simulation...")
    scenario = HohmannScenario()
    result_dir = scenario.run()

    print("\n✅ Simulation completed!")
    print(f"📁 Results: {result_dir}")
    print("\n💡 Results include auto-generated phase-colored plots (PNG & HTML)")
    print("=" * 70)


if __name__ == "__main__":
    main()
