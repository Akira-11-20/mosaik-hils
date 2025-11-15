"""
OrbitalScenario - 軌道HILS シナリオ

6自由度軌道力学シミュレーションの制御ループを実装。

データフロー:
    OrbitalEnv → OrbitalController → OrbitalPlant → OrbitalEnv
    (全コンポーネント → DataCollector)
"""

import sys
from datetime import datetime
from pathlib import Path

import mosaik

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.orbital_parameters import CONFIG_ISS, get_env_param


class OrbitalScenario:
    """
    軌道HILS シナリオ

    制御ループ:
        1. OrbitalEnv: 軌道力学エンジン（RK4積分）
        2. OrbitalController: 制御器（推力指令計算）
        3. OrbitalPlant: 推力計測デバイス（1次遅れ + ノイズ）

    データ収集:
        全コンポーネントのデータをHDF5形式で記録
    """

    def __init__(self, config=None):
        """
        初期化

        Args:
            config: OrbitalSimulationConfig（デフォルト: CONFIG_ISS）
        """
        self.config = config if config is not None else CONFIG_ISS
        self.world = None
        self.controller = None
        self.plant = None
        self.spacecraft = None
        self.collector = None
        self.inverse_compensator = None

        # 結果ディレクトリ
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.output_dir = Path(__file__).parent.parent / "results_orbital" / timestamp
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"[OrbitalScenario] Output directory: {self.output_dir}")

    def create_world(self):
        """Mosaikワールドの作成"""
        print("\n[OrbitalScenario] Creating Mosaik world...")

        # シミュレーター設定
        sim_config = {
            "OrbitalControllerSim": {
                "python": "simulators.controller_simulator:OrbitalControllerSimulator",
            },
            "OrbitalPlantSim": {
                "python": "simulators.plant_simulator:OrbitalPlantSimulator",
            },
            "OrbitalEnvSim": {
                "python": "simulators.env_simulator:OrbitalEnvSimulator",
            },
            "InverseCompensatorSim": {
                "python": "simulators.inverse_compensator_simulator:InverseCompensatorSimulator",
            },
            "DataCollector": {
                "python": "simulators.data_collector:DataCollectorSimulator",
            },
        }

        # .envから設定を読み込み
        debug_mode = get_env_param("MOSAIK_DEBUG_MODE", True, bool)
        show_dataflow = get_env_param("SHOW_DATAFLOW", True, bool)

        # ワールド作成
        self.world = mosaik.World(
            sim_config,
            time_resolution=self.config.time_resolution,
            debug=debug_mode,  # .envから読み込み
        )

        print(f"  Time resolution: {self.config.time_resolution} s")
        print(f"  Simulation time: {self.config.simulation_time} s")
        print(f"  Debug mode: {'ON' if debug_mode else 'OFF'}")

        # データフロー表示フラグを保存
        self.show_dataflow = show_dataflow

    def setup_entities(self):
        """エンティティのセットアップ"""
        print("\n[OrbitalScenario] Setting up entities...")

        # 初期状態の計算
        position, velocity = self.config.orbit.to_cartesian()

        # .envからパラメータを読み込み
        target_pos_x = get_env_param("TARGET_POSITION_X", 0.0, float)
        target_pos_y = get_env_param("TARGET_POSITION_Y", 0.0, float)
        target_pos_z = get_env_param("TARGET_POSITION_Z", 0.0, float)
        control_gain = get_env_param("CONTROL_GAIN", 1.0, float)
        plant_time_constant = get_env_param("PLANT_TIME_CONSTANT", 10.0, float)
        plant_noise_std = get_env_param("PLANT_NOISE_STD", 0.01, float)
        minimal_data_mode = get_env_param("MINIMAL_DATA_MODE", False, bool)
        self.use_inverse_compensation = get_env_param("INVERSE_COMPENSATION", False, bool)
        inverse_compensation_gain = get_env_param("INVERSE_COMPENSATION_GAIN", 1.0, float)

        # Controller
        controller_sim = self.world.start(
            "OrbitalControllerSim",
            time_resolution=self.config.time_resolution,
            step_size=self.config.step_size,
        )
        self.controller = controller_sim.OrbitalController(
            target_position=[target_pos_x, target_pos_y, target_pos_z],
            control_gain=control_gain,
        )

        # Plant
        plant_sim = self.world.start(
            "OrbitalPlantSim",
            time_resolution=self.config.time_resolution,
            step_size=self.config.step_size,
        )
        self.plant = plant_sim.OrbitalThrustStand(
            time_constant=plant_time_constant,
            noise_std=plant_noise_std,
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
        
        if self.use_inverse_compensation:
            # Inverse Compensator
            inverse_comp_sim = self.world.start(
                "InverseCompensatorSim",
                time_resolution=self.config.time_resolution,
                step_size=self.config.step_size,
            )
            self.inverse_compensator = inverse_comp_sim.InverseCompensator(
                gain=inverse_compensation_gain,
            )
            print(f"  ✅ Inverse Compensator created (Gain={inverse_compensation_gain})")

        # Data Collector
        collector_sim = self.world.start(
            "DataCollector",
            time_resolution=self.config.time_resolution,
            step_size=self.config.step_size,
        )
        self.collector = collector_sim.Collector(output_dir=str(self.output_dir), minimal_mode=minimal_data_mode)

        print("  ✅ All entities created")

    def connect_entities(self):
        """エンティティの接続（データフロー定義）"""
        print("\n[OrbitalScenario] Connecting entities...")

        # データフロー表示（.envから制御）
        if self.show_dataflow:
            print("\n  📊 Data Flow:")
            print("  ┌─────────────────────────────────────────────────────────┐")
            print("  │                   Control Loop                          │")
            print("  └─────────────────────────────────────────────────────────┘")
            print("  [1] OrbitalEnv → OrbitalController")
            print("      └─ position_x/y/z, velocity_x/y/z (same-step)")

        # フィードバック: Env → Controller (same-step for immediate response)
        self.world.connect(
            self.spacecraft,
            self.controller,
            ("position_x", "position_x"),
            ("position_y", "position_y"),
            ("position_z", "position_z"),
            ("velocity_x", "velocity_x"),
            ("velocity_y", "velocity_y"),
            ("velocity_z", "velocity_z"),
        )

        # 指令: Controller → Plant
        if self.show_dataflow:
            print("  [2] OrbitalController → OrbitalPlant")
            print("      └─ thrust_command_x/y/z (same-step)")

        self.world.connect(
            self.controller,
            self.plant,
            ("thrust_command_x", "command_x"),
            ("thrust_command_y", "command_y"),
            ("thrust_command_z", "command_z"),
        )
        
        if self.use_inverse_compensation:
            if self.show_dataflow:
                print("  [2.5] InverseCompensator → OrbitalPlant")
                print("      └─ compensated_command_x/y/z (same-step)")
            
            self.world.connect(
                self.plant,
                self.inverse_compensator,
                ("measured_force_x", "input_force_x"),
                ("measured_force_y", "input_force_y"),
                ("measured_force_z", "input_force_z"),
            )
            
            self.world.connect(
                self.inverse_compensator,
                self.spacecraft,
                ("compensated_force_x", "force_x"),
                ("compensated_force_y", "force_y"),
                ("compensated_force_z", "force_z"),
                time_shifted=True,
                initial_data={
                    "compensated_force_x": 0.0,
                    "compensated_force_y": 0.0,
                    "compensated_force_z": 0.0,
                },
            )
        else:
            
        # 計測: Plant → Env (time-shifted to break cycle)
            if self.show_dataflow:
                print("  [3] OrbitalPlant → OrbitalEnv")
                print("      └─ measured_force_x/y/z (time-shifted, breaks cycle)")

            self.world.connect(
                self.plant,
                self.spacecraft,
                ("measured_force_x", "force_x"),
                ("measured_force_y", "force_y"),
                ("measured_force_z", "force_z"),
                time_shifted=True,
                initial_data={
                    "measured_force_x": 0.0,
                    "measured_force_y": 0.0,
                    "measured_force_z": 0.0,
                },
            )

        print("\n  ✅ Control loop connected")
        print("  ℹ️  Loop: Env → Controller → Plant → [time-shift] → Env")

    def setup_data_collection(self):
        """データ収集の設定"""
        print("\n[OrbitalScenario] Setting up data collection...")

        # Controller data
        self.world.connect(
            self.controller,
            self.collector,
            "thrust_command_x",
            "thrust_command_y",
            "thrust_command_z",
        )

        # Plant data
        self.world.connect(
            self.plant,
            self.collector,
            "measured_force_x",
            "measured_force_y",
            "measured_force_z",
            "norm_measured_force",
            "alpha",
        )

        # Environment data
        self.world.connect(
            self.spacecraft,
            self.collector,
            "position_x",
            "position_y",
            "position_z",
            "position_norm",
            "velocity_x",
            "velocity_y",
            "velocity_z",
            "velocity_norm",
            "force_x",
            "force_y",
            "force_z",
            "norm_force",
            "acceleration_x",
            "acceleration_y",
            "acceleration_z",
            "altitude",
            "semi_major_axis",
            "eccentricity",
            "specific_energy",
        )
        
        if self.use_inverse_compensation:
            # Inverse Compensator data
            self.world.connect(
                self.inverse_compensator,
                self.collector,
                "input_force_x",
                "input_force_y",
                "input_force_z",
                "input_norm_force",
                "compensated_force_x",
                "compensated_force_y",
                "compensated_force_z",
                "compensated_norm_force",
                "gain",
            )

        print("  ✅ Data collection configured")

    def run(self):
        """シミュレーションの実行"""
        print("\n" + "=" * 70)
        print("Orbital HILS Simulation")
        print("=" * 70)
        print(f"Orbital altitude: {self.config.orbit.altitude / 1e3:.2f} km")
        print(f"Orbital period: {self.config.orbit.orbital_period / 60:.2f} min")
        print(f"Simulation time: {self.config.simulation_time / 60:.2f} min")
        print("=" * 70)

        # ワールド作成
        self.create_world()

        # エンティティのセットアップ
        self.setup_entities()

        # 接続
        self.connect_entities()

        # データ収集
        self.setup_data_collection()

        # シミュレーション実行
        print("\n[OrbitalScenario] Running simulation...")
        print(f"  Duration: {self.config.simulation_time} s")

        self.world.run(until=self.config.simulation_time)

        print("\n[OrbitalScenario] ✅ Simulation completed")
        print(f"[OrbitalScenario] 📁 Results: {self.output_dir}")

        # 実行グラフの保存
        self._save_execution_graph()

        # 自動可視化
        self._auto_visualize()

        return self.output_dir

    def _save_execution_graph(self):
        """実行グラフを保存"""
        try:
            print("\n[OrbitalScenario] 📊 Generating execution graph...")

            # 共通ユーティリティを使用してカスタムグラフを生成
            try:
                import sys

                # プロジェクトルートをパスに追加
                project_root = Path(__file__).parent.parent.parent
                sys.path.insert(0, str(project_root))

                from common_utils import (
                    plot_dataflow_graph_custom,
                    plot_execution_graph_with_data_only,
                )

                # データフローグラフ（ノード接続図）
                plot_dataflow_graph_custom(
                    self.world,
                    folder=str(self.output_dir),
                    show_plot=False,
                    dpi=600,
                    format="png",
                    exclude_nodes=["DataCollector-0"],  # DataCollectorを除外
                )
                print("  ✅ Custom dataflow graph saved (dataflowGraph_custom.png)")

                # 実行グラフ（データやり取りタイミング）
                plot_execution_graph_with_data_only(
                    self.world,
                    title="Orbital HILS Execution Graph",
                    folder=str(self.output_dir),
                    show_plot=False,
                    save_plot=True,
                )
                print("  ✅ Custom execution graph saved (data-only view)")
            except ImportError as e:
                print(f"  ⚠️  Custom graph failed: {e}")
            except Exception as e:
                print(f"  ⚠️  Custom graph error: {e}")

            # Mosaikの標準DOTファイルも保存
            import glob
            import shutil
            import subprocess

            dot_file = self.output_dir / "execution_graph.dot"
            dot_files = glob.glob(str(Path.cwd() / "execution_graph*.dot"))

            if dot_files:
                latest_dot = max(dot_files, key=lambda p: Path(p).stat().st_mtime)
                shutil.copy(latest_dot, dot_file)

                # PNGに変換（graphviz利用可能な場合）
                png_file = self.output_dir / "execution_graph_full.png"
                result = subprocess.run(
                    ["dot", "-Tpng", str(dot_file), "-o", str(png_file)],
                    capture_output=True,
                    text=True,
                )

                if result.returncode == 0:
                    print(f"  ✅ Full execution graph saved: {png_file.name}")
                else:
                    print(f"  📄 DOT file saved: {dot_file.name} (install graphviz for PNG)")

        except Exception as e:
            print(f"  ⚠️  Could not save execution graph: {e}")

    def _auto_visualize(self):
        """シミュレーション結果の自動可視化"""
        # .envからAUTO_VISUALIZEフラグを読み込み
        auto_visualize = get_env_param("AUTO_VISUALIZE", True, bool)

        if not auto_visualize:
            print("\n[OrbitalScenario] ⏭️  Auto-visualization disabled (set AUTO_VISUALIZE=true in .env to enable)")
            return

        print("\n[OrbitalScenario] 📊 Auto-generating visualizations...")

        h5_file = self.output_dir / "hils_data.h5"

        if not h5_file.exists():
            print("  ⚠️  HDF5 file not found, skipping visualization")
            return

        try:
            import subprocess

            # 静的プロット生成
            print("  📈 Generating static plots...")
            result = subprocess.run(
                [
                    "python",
                    "scripts/analysis/visualize_orbital_results.py",
                    str(h5_file),
                ],
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                print("  ✅ Static plots generated")
            else:
                print(f"  ⚠️  Static plots failed: {result.stderr}")

            # インタラクティブプロット生成
            print("  🌐 Generating interactive plots...")
            result = subprocess.run(
                [
                    "python",
                    "scripts/analysis/visualize_orbital_interactive.py",
                    str(h5_file),
                ],
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                print("  ✅ Interactive plots generated")
            else:
                print(f"  ⚠️  Interactive plots failed: {result.stderr}")

        except Exception as e:
            print(f"  ⚠️  Auto-visualization failed: {e}")


if __name__ == "__main__":
    # テスト実行
    scenario = OrbitalScenario(config=CONFIG_ISS)
    scenario.run()
