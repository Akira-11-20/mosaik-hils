"""
Orbital Scenario - 軌道力学シミュレーション

地球周回軌道の二体問題をシミュレート。
制御入力なしで自由軌道運動を観察。
"""

from pathlib import Path
from typing import Optional

import mosaik

from config.orbital_parameters import OrbitalSimulationConfig


class OrbitalScenario:
    """
    軌道力学シミュレーションシナリオ

    構成:
        - OrbitalEnvSimulator: 軌道力学エンジン（二体問題）
        - DataCollector: データ収集・記録
    """

    def __init__(self, config: Optional[OrbitalSimulationConfig] = None):
        """
        シナリオの初期化

        Args:
            config: 軌道シミュレーション設定
        """
        from config.orbital_parameters import CONFIG_ISS

        self.config = config if config is not None else CONFIG_ISS
        self.world: Optional[mosaik.World] = None
        self.run_dir: Optional[Path] = None

        # エンティティの保存
        self.spacecraft = None
        self.collector = None

    @property
    def scenario_name(self) -> str:
        return "Orbital"

    @property
    def scenario_description(self) -> str:
        orbit = self.config.orbit
        return f"Two-body orbital dynamics - {orbit.altitude/1e3:.0f}km altitude, {orbit.eccentricity:.2f} eccentricity"

    @property
    def results_base_dir(self) -> str:
        return "results_orbital"

    def setup_output_directory(self, suffix: str = "") -> Path:
        """出力ディレクトリの作成"""
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        dir_name = f"{timestamp}{suffix}" if suffix else timestamp
        self.run_dir = Path(self.results_base_dir) / dir_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        return self.run_dir

    def save_configuration(self):
        """設定の保存"""
        if self.run_dir is None:
            raise RuntimeError("Output directory not set")

        import json

        config_dict = {
            "scenario": self.scenario_name,
            "description": self.scenario_description,
            "simulation": {
                "time": self.config.simulation_time,
                "time_resolution": self.config.time_resolution,
                "step_size": self.config.step_size,
            },
            "celestial_body": self.config.celestial_body,
            "orbit": {
                "semi_major_axis": self.config.orbit.semi_major_axis,
                "eccentricity": self.config.orbit.eccentricity,
                "inclination": self.config.orbit.inclination,
                "raan": self.config.orbit.raan,
                "arg_periapsis": self.config.orbit.arg_periapsis,
                "true_anomaly": self.config.orbit.true_anomaly,
                "altitude_km": self.config.orbit.altitude / 1e3,
                "period_min": self.config.orbit.orbital_period / 60,
            },
            "spacecraft": {
                "mass": self.config.spacecraft.mass,
                "max_thrust": self.config.spacecraft.max_thrust,
                "specific_impulse": self.config.spacecraft.specific_impulse,
            },
        }

        config_path = self.run_dir / "orbital_config.json"
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2)

        print(f"💾 Configuration saved: {config_path}")

    def print_header(self):
        """ヘッダー表示"""
        print("=" * 70)
        print(f"{self.scenario_name} Simulation - Two-Body Orbital Dynamics")
        print(f"{self.scenario_description}")
        print("=" * 70)

    def print_simulation_info(self):
        """シミュレーション情報の表示"""
        orbit = self.config.orbit
        sc = self.config.spacecraft

        print(f"\n🛰️  Orbital Parameters:")
        print(f"   Altitude: {orbit.altitude / 1e3:.2f} km")
        print(f"   Semi-major axis: {orbit.semi_major_axis / 1e3:.2f} km")
        print(f"   Eccentricity: {orbit.eccentricity:.4f}")
        print(f"   Inclination: {orbit.inclination:.2f}°")
        print(f"   Orbital period: {orbit.orbital_period / 60:.2f} min")

        print(f"\n🚀 Spacecraft:")
        print(f"   Mass: {sc.mass} kg")
        print(f"   Max thrust: {sc.max_thrust} N")

        print(f"\n⏱️  Simulation:")
        print(f"   Duration: {self.config.simulation_time} s ({self.config.simulation_time / 60:.2f} min)")
        print(f"   Time resolution: {self.config.time_resolution} s")
        total_steps = int(self.config.simulation_time / self.config.time_resolution)
        print(f"   Total steps: {total_steps}")

    def create_world(self) -> mosaik.World:
        """Mosaikワールドの作成"""
        sim_config = {
            "OrbitalEnvSim": {
                "python": "simulators.orbital_env_simulator:OrbitalEnvSimulator",
            },
            "DataCollector": {
                "python": "simulators.data_collector:DataCollectorSimulator",
            },
        }

        self.world = mosaik.World(
            sim_config,
            time_resolution=self.config.time_resolution,
            cache=False,
        )
        return self.world

    def setup_entities(self):
        """エンティティの作成"""
        # 初期状態の計算
        position, velocity = self.config.orbit.to_cartesian()

        # 軌道環境シミュレータ
        orbital_sim = self.world.start(
            "OrbitalEnvSim",
            time_resolution=self.config.time_resolution,
            step_size=self.config.step_size,
        )

        self.spacecraft = orbital_sim.OrbitalSpacecraft(
            mass=self.config.spacecraft.mass,
            mu=self.config.orbit.mu,
            initial_position=position.tolist(),
            initial_velocity=velocity.tolist(),
            radius_earth=self.config.orbit.radius_body,
        )

        # データコレクター
        collector_sim = self.world.start(
            "DataCollector",
            step_size=self.config.step_size,
        )

        self.collector = collector_sim.Collector(output_dir=str(self.run_dir))

        print(f"   ✅ Spacecraft entity created")
        print(f"   ✅ Data collector created: {self.run_dir}")

    def connect_entities(self):
        """エンティティの接続"""
        # 現在は制御入力なし（自由軌道運動）
        # 将来的には制御器を追加可能
        print(f"   ℹ️  Free orbital motion (no control input)")

    def setup_data_collection(self):
        """データ収集の設定"""
        # 全ての軌道状態を記録
        attrs = [
            "position_x",
            "position_y",
            "position_z",
            "position_norm",
            "velocity_x",
            "velocity_y",
            "velocity_z",
            "velocity_norm",
            "acceleration_x",
            "acceleration_y",
            "acceleration_z",
            "altitude",
            "semi_major_axis",
            "eccentricity",
            "specific_energy",
        ]

        self.world.connect(
            self.spacecraft,
            self.collector,
            *attrs,
        )
        print(f"   ✅ Data collection configured")

    def generate_plots(self):
        """プロット生成"""
        if self.run_dir is None:
            return

        print(f"\n📊 Generating plots...")
        try:
            from scripts.analysis.visualize_orbital_results import plot_orbital_simulation

            # DataCollectorが生成するHDF5ファイル名を検索
            h5_files = list(self.run_dir.glob("*.h5"))
            if not h5_files:
                print(f"   ⚠️  No HDF5 data file found in {self.run_dir}")
                return

            h5_path = h5_files[0]
            plot_orbital_simulation(str(h5_path), output_dir=str(self.run_dir))
            print(f"   ✅ Plots saved to {self.run_dir}/")
        except ImportError:
            print(f"   ℹ️  Visualization script not found (will create later)")
        except Exception as e:
            print(f"   ⚠️  Plot generation failed: {e}")

    def run(self):
        """シミュレーションの実行"""
        # ヘッダー表示
        self.print_header()

        # 出力ディレクトリ設定
        self.setup_output_directory()
        print(f"📁 Output directory: {self.run_dir}")

        # 設定保存
        self.save_configuration()

        # シミュレーション情報表示
        self.print_simulation_info()

        # ワールド作成
        print(f"\n🌍 Creating Mosaik World...")
        self.create_world()

        # エンティティ作成
        print(f"\n📦 Creating entities...")
        self.setup_entities()

        # エンティティ接続
        print(f"\n🔗 Connecting data flows...")
        self.connect_entities()

        # データ収集設定
        print(f"\n📊 Setting up data collection...")
        self.setup_data_collection()

        # シミュレーション実行
        total_steps = int(self.config.simulation_time / self.config.time_resolution)
        print(f"\n▶️  Running simulation for {self.config.simulation_time}s ({total_steps} steps)...")
        print("=" * 70)

        self.world.run(until=total_steps)

        print("=" * 70)
        print("✅ Simulation completed successfully!")

        # プロット生成
        self.generate_plots()

        # フッター
        print("\n" + "=" * 70)
        print(f"{self.scenario_name} Simulation Finished")
        print(f"Results saved to: {self.run_dir}")
        print("=" * 70)


if __name__ == "__main__":
    # デフォルトISS軌道でシミュレーション
    scenario = OrbitalScenario()
    scenario.run()
