"""
ホーマン遷移HILSシミュレーション実行スクリプト

400km円軌道から600km円軌道への遷移を
Mosaikを使った完全なHILSシミュレーションで実行。

実行方法:
    cd orbital_hils
    uv run python examples/run_hohmann_simulation.py
"""

import sys
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.orbital_parameters import (
    CelestialBodyConstants,
    OrbitalParameters,
    OrbitalSimulationConfig,
    SpacecraftParameters,
)
from scenarios.orbital_scenario import OrbitalScenario


def create_hohmann_transfer_config():
    """ホーマン遷移用のシミュレーション設定を作成"""
    constants = CelestialBodyConstants()

    # 初期軌道: 400km円軌道
    initial_altitude = 400e3  # 400km
    semi_major_axis = constants.RADIUS_EARTH + initial_altitude

    orbit = OrbitalParameters(
        mu=constants.MU_EARTH,
        radius_body=constants.RADIUS_EARTH,
        semi_major_axis=semi_major_axis,
        eccentricity=0.0,
        inclination=51.64,  # ISS相当
        raan=0.0,
        arg_periapsis=0.0,
        true_anomaly=0.0,
    )

    # 衛星パラメータ（より大きな推力を使用）
    spacecraft = SpacecraftParameters(
        mass=500.0,  # 500kg
        max_thrust=10.0,  # 10N（より大きな推力）
        specific_impulse=200.0,
    )

    # シミュレーション時間（遷移時間 + バーン時間 + マージン）
    # ホーマン遷移時間: 約2839秒（47分）
    # バーン時間: 約2777秒（46分）× 2回
    # 合計: 約8400秒（2.3時間）
    simulation_time = 10000.0  # 10000秒（約2.8時間）
    time_resolution = 1.0  # 1秒刻み

    config = OrbitalSimulationConfig(
        simulation_time=simulation_time,
        time_resolution=time_resolution,
        step_size=1,
        celestial_body="Earth",
        spacecraft=spacecraft,
        orbit=orbit,
    )

    return config


def main():
    """メイン実行関数"""
    print("=" * 70)
    print("Hohmann Transfer HILS Simulation")
    print("=" * 70)
    print("\n🚀 Mission: Transfer from 400km to 600km circular orbit")
    print("   Using Mosaik-based HILS simulation\n")

    # 設定を作成
    config = create_hohmann_transfer_config()

    print("📋 Simulation Configuration:")
    print("   Initial altitude: 400 km")
    print("   Target altitude: 600 km")
    print(f"   Spacecraft mass: {config.spacecraft.mass} kg")
    print(f"   Max thrust: {config.spacecraft.max_thrust} N")
    print(f"   Simulation time: {config.simulation_time / 60:.2f} min")
    print(f"   Time resolution: {config.time_resolution} s")

    # ホーマン遷移パラメータを計算（確認用）
    from models.hohmann_transfer import HohmannTransferModel

    hohmann = HohmannTransferModel(
        mu=config.orbit.mu,
        initial_altitude=400e3,
        target_altitude=600e3,
        radius_body=config.orbit.radius_body,
        spacecraft_mass=config.spacecraft.mass,
        max_thrust=config.spacecraft.max_thrust,
    )

    status = hohmann.get_status()
    print("\n📊 Hohmann Transfer Parameters:")
    print(f"   ΔV1: {status['delta_v1']:+.2f} m/s")
    print(f"   ΔV2: {status['delta_v2']:+.2f} m/s")
    print(f"   Total ΔV: {status['total_delta_v']:.2f} m/s")
    print(f"   Transfer time: {status['transfer_time'] / 60:.2f} min")
    print(f"   Burn1 duration: {status['burn1_duration'] / 60:.2f} min")
    print(f"   Burn2 duration: {status['burn2_duration'] / 60:.2f} min")

    # シミュレーション実行
    print("\n🔧 Creating scenario...")

    # カスタムシナリオを作成
    scenario = create_hohmann_scenario(config)

    print("\n▶️  Running simulation...")
    result_dir = scenario.run()

    print("\n✅ Simulation completed!")
    print(f"📁 Results saved to: {result_dir}")
    print("\n💡 To visualize results:")
    print("   cd orbital_hils")
    print(f"   uv run python scripts/analysis/visualize_orbital_results.py {result_dir}/hils_data.h5")


def create_hohmann_scenario(config):
    """
    ホーマン遷移用のカスタムシナリオを作成

    OrbitalScenarioを継承して、コントローラーにホーマン遷移モデルを使用。
    """

    class HohmannTransferScenario(OrbitalScenario):
        """ホーマン遷移シナリオ"""

        def setup_entities(self):
            """エンティティのセットアップ（カスタマイズ版）"""
            print("\n[HohmannScenario] Setting up entities...")

            # 初期状態の計算
            position, velocity = self.config.orbit.to_cartesian()

            # Controller（ホーマン遷移モード）
            controller_sim = self.world.start(
                "OrbitalControllerSim",
                time_resolution=self.config.time_resolution,
                step_size=self.config.step_size,
            )
            self.controller = controller_sim.OrbitalController(
                controller_type="hohmann",
                mu=self.config.orbit.mu,
                initial_altitude=400e3,  # 400km
                target_altitude=700e3,  # 600km
                radius_body=self.config.orbit.radius_body,
                spacecraft_mass=self.config.spacecraft.mass,
                max_thrust=self.config.spacecraft.max_thrust,
                start_time=100.0,  # 100秒後に遷移開始
            )

            # Plant
            plant_sim = self.world.start(
                "OrbitalPlantSim",
                time_resolution=self.config.time_resolution,
                step_size=self.config.step_size,
            )
            self.plant = plant_sim.OrbitalThrustStand(
                time_constant=10.0,  # 10秒の時定数
                noise_std=0.01,  # 1%のノイズ
            )

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

            # Data Collector
            collector_sim = self.world.start(
                "DataCollector",
                time_resolution=self.config.time_resolution,
                step_size=self.config.step_size,
            )
            self.collector = collector_sim.Collector(
                output_dir=str(self.output_dir),
                minimal_mode=False,  # 全データを記録
            )

            print("  ✅ All entities created (Hohmann transfer mode)")

    # シナリオを作成して返す
    return HohmannTransferScenario(config=config)


if __name__ == "__main__":
    main()
