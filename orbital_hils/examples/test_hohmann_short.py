"""
ホーマン遷移の短時間テスト

シミュレーション開始100秒後にホーマン遷移を開始し、
第1バーンの開始を確認する短時間シミュレーション。

実行方法:
    cd orbital_hils
    uv run python examples/test_hohmann_short.py
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


def create_short_hohmann_config():
    """短時間ホーマン遷移テスト用の設定"""
    constants = CelestialBodyConstants()

    # 初期軌道: 400km円軌道
    initial_altitude = 400e3
    semi_major_axis = constants.RADIUS_EARTH + initial_altitude

    orbit = OrbitalParameters(
        mu=constants.MU_EARTH,
        radius_body=constants.RADIUS_EARTH,
        semi_major_axis=semi_major_axis,
        eccentricity=0.0,
        inclination=51.64,
        raan=0.0,
        arg_periapsis=0.0,
        true_anomaly=0.0,
    )

    # より大きな推力で高速化（100N）
    spacecraft = SpacecraftParameters(
        mass=500.0,
        max_thrust=100.0,  # 100N推力
        specific_impulse=200.0,
    )

    # 短時間シミュレーション（500秒 = 8.3分）
    simulation_time = 500.0
    time_resolution = 1.0

    return OrbitalSimulationConfig(
        simulation_time=simulation_time,
        time_resolution=time_resolution,
        step_size=1,
        celestial_body="Earth",
        spacecraft=spacecraft,
        orbit=orbit,
    )


class ShortHohmannScenario(OrbitalScenario):
    """短時間ホーマン遷移テストシナリオ"""

    def setup_entities(self):
        """エンティティのセットアップ"""
        print("\n[ShortHohmannScenario] Setting up entities...")

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
            target_altitude=600e3,  # 600km
            radius_body=self.config.orbit.radius_body,
            spacecraft_mass=self.config.spacecraft.mass,
            max_thrust=self.config.spacecraft.max_thrust,
            start_time=100.0,  # 100秒後に遷移開始
        )

        # Plant（応答を速く）
        plant_sim = self.world.start(
            "OrbitalPlantSim",
            time_resolution=self.config.time_resolution,
            step_size=self.config.step_size,
        )
        self.plant = plant_sim.OrbitalThrustStand(
            time_constant=1.0,  # 1秒の時定数（速い応答）
            noise_std=0.001,  # ノイズ少なめ
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
            minimal_mode=False,
        )

        print("  ✅ All entities created (Short Hohmann test)")


def main():
    """メイン実行"""
    print("=" * 70)
    print("Short Hohmann Transfer Test")
    print("=" * 70)
    print("\n🎯 Goal: Verify Hohmann transfer initiation")
    print("   - Start transfer at t=100s")
    print("   - Run first burn for ~300s")
    print("   - Total simulation: 500s (8.3 min)\n")

    # 設定を作成
    config = create_short_hohmann_config()

    # ホーマン遷移パラメータを表示
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
    print("📊 Hohmann Parameters (100N thrust):")
    print(f"   ΔV1: {status['delta_v1']:+.2f} m/s")
    print(f"   ΔV2: {status['delta_v2']:+.2f} m/s")
    print(f"   Total ΔV: {status['total_delta_v']:.2f} m/s")
    print(f"   Transfer time: {status['transfer_time'] / 60:.2f} min")
    print(f"   Burn1 duration: {status['burn1_duration']:.2f} s ({status['burn1_duration'] / 60:.2f} min)")
    print(f"   Burn2 duration: {status['burn2_duration']:.2f} s")

    # シナリオ実行
    print("\n▶️  Running simulation...")
    scenario = ShortHohmannScenario(config=config)
    result_dir = scenario.run()

    print("\n✅ Test completed!")
    print(f"📁 Results: {result_dir}")
    print("\n💡 Expected timeline:")
    print("   t = 0-100s      : No thrust (free orbit)")
    print(f"   t = 100-{100 + status['burn1_duration']:.0f}s : First burn (should see thrust)")
    print("   t > 400s        : Still in first burn")
    print("\n📊 To visualize:")
    print(f"   uv run python scripts/analysis/visualize_orbital_results.py {result_dir}/hils_data.h5")


if __name__ == "__main__":
    main()
