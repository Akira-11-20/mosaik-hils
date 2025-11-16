"""
FormationFlyingScenario - Formation Flying シナリオ

2機の衛星による編隊飛行シミュレーション:
  - Chaser (追跡機): 制御あり（PD制御でターゲットを追尾）
  - Target (目標機): 制御なし（自由軌道運動）

データフロー:
    Chaser: OrbitalEnv → OrbitalController → OrbitalPlant → OrbitalEnv
    Target: OrbitalEnv (free orbital motion)
    (全コンポーネント → DataCollector)
"""

import sys
from datetime import datetime
from pathlib import Path

import mosaik

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.orbital_parameters import CONFIG_ISS, get_env_param


class FormationFlyingScenario:
    """
    Formation Flying シナリオ

    2機の衛星による編隊飛行:
        1. Chaser (追跡機): 制御ループあり
           - OrbitalEnv: 軌道力学エンジン（RK4積分）
           - OrbitalController: 制御器（推力指令計算）
           - OrbitalPlant: 推力計測デバイス（1次遅れ + ノイズ）

        2. Target (目標機): 自由軌道運動
           - OrbitalEnv: 軌道力学エンジン（制御なし）

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

        # Chaser (追跡機) のエンティティ
        self.chaser_controller = None
        self.chaser_plant = None
        self.chaser_spacecraft = None

        # Target (目標機) のエンティティ
        self.target_spacecraft = None

        # データコレクター
        self.collector = None
        self.inverse_compensator = None

        # 結果ディレクトリ
        import os

        output_dir_override = os.environ.get("OUTPUT_DIR_OVERRIDE")
        if output_dir_override:
            self.output_dir = Path(output_dir_override)
        else:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            self.output_dir = Path(__file__).parent.parent / "results_orbital" / timestamp

        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"[FormationFlyingScenario] Output directory: {self.output_dir}")

    def create_world(self):
        """Mosaikワールドの作成"""
        print("\n[FormationFlyingScenario] Creating Mosaik world...")

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
            debug=debug_mode,
        )

        print(f"  Time resolution: {self.config.time_resolution} s")
        print(f"  Simulation time: {self.config.simulation_time} s")
        print(f"  Debug mode: {'ON' if debug_mode else 'OFF'}")

        # データフロー表示フラグを保存
        self.show_dataflow = show_dataflow

    def setup_entities(self):
        """エンティティのセットアップ"""
        print("\n[FormationFlyingScenario] Setting up entities...")

        # 初期状態の計算（基準軌道）
        position_base, velocity_base = self.config.orbit.to_cartesian()

        # .envからパラメータを読み込み
        # Chaser の目標はTarget衛星の位置
        formation_offset_x = get_env_param("FORMATION_OFFSET_X", 100.0, float)
        formation_offset_y = get_env_param("FORMATION_OFFSET_Y", 0.0, float)
        formation_offset_z = get_env_param("FORMATION_OFFSET_Z", 0.0, float)

        control_gain = get_env_param("CONTROL_GAIN", 1.0, float)
        max_thrust = get_env_param("MAX_THRUST", 1.0, float)
        plant_time_constant = get_env_param("PLANT_TIME_CONSTANT", 10.0, float)
        plant_noise_std = get_env_param("PLANT_NOISE_STD", 0.01, float)
        minimal_data_mode = get_env_param("MINIMAL_DATA_MODE", False, bool)
        self.use_inverse_compensation = get_env_param("INVERSE_COMPENSATION", False, bool)
        inverse_compensation_gain = get_env_param("INVERSE_COMPENSATION_GAIN", 1.0, float)

        print("\n[FormationFlyingScenario] 🚀 Creating Chaser (controlled spacecraft)...")

        # Chaser Controller
        controller_sim = self.world.start(
            "OrbitalControllerSim",
            time_resolution=self.config.time_resolution,
            step_size=self.config.step_size,
        )
        # HCW制御: LVLH座標系での相対位置を目標に
        # 目標相対位置（LVLH）= [0, 0, 0] （Targetと同じ位置）
        controller_type = get_env_param("FORMATION_CONTROLLER_TYPE", "hcw", str)

        if controller_type == "hcw":
            # HCW編隊飛行制御
            self.chaser_controller = controller_sim.OrbitalController(
                target_position=[0.0, 0.0, 0.0],  # 目標相対位置（LVLH）
                target_velocity=[0.0, 0.0, 0.0],  # 目標相対速度（LVLH）
                control_gain=control_gain,
                max_thrust=max_thrust,
                mu=self.config.orbit.mu,
                controller_type="hcw",
            )
            print(f"  Controller: HCW Formation Flying")
            print(f"  Target relative position (LVLH): [0, 0, 0] m")
            print(f"  Max thrust: {max_thrust} N")
        else:
            # 従来のPD制御
            self.chaser_controller = controller_sim.OrbitalController(
                target_position=position_base.tolist(),  # Targetの初期位置
                target_velocity=velocity_base.tolist(),  # Targetの初期速度
                control_gain=control_gain,
                max_thrust=max_thrust,
                controller_type="pd",
            )
            print(f"  Controller: PD Control")
            print(f"  Target position: {position_base / 1e3} km")
            print(f"  Target velocity: {velocity_base} m/s")
            print(f"  Max thrust: {max_thrust} N")

        # Chaser Plant
        plant_sim = self.world.start(
            "OrbitalPlantSim",
            time_resolution=self.config.time_resolution,
            step_size=self.config.step_size,
        )
        self.chaser_plant = plant_sim.OrbitalThrustStand(
            time_constant=plant_time_constant,
            noise_std=plant_noise_std,
        )

        # Chaser Environment（初期位置にオフセットを追加）
        chaser_position = position_base + [formation_offset_x, formation_offset_y, formation_offset_z]
        chaser_velocity = velocity_base.copy()  # 速度は同じ

        env_sim_chaser = self.world.start(
            "OrbitalEnvSim",
            time_resolution=self.config.time_resolution,
            step_size=self.config.step_size,
        )
        self.chaser_spacecraft = env_sim_chaser.OrbitalSpacecraft(
            mass=self.config.spacecraft.mass,
            mu=self.config.orbit.mu,
            initial_position=chaser_position.tolist(),
            initial_velocity=chaser_velocity.tolist(),
            radius_earth=self.config.orbit.radius_body,
        )

        print("\n[FormationFlyingScenario] 🎯 Creating Target (free-flying spacecraft)...")

        # Target Environment（基準軌道）
        env_sim_target = self.world.start(
            "OrbitalEnvSim",
            time_resolution=self.config.time_resolution,
            step_size=self.config.step_size,
        )
        self.target_spacecraft = env_sim_target.OrbitalSpacecraft(
            mass=self.config.spacecraft.mass,
            mu=self.config.orbit.mu,
            initial_position=position_base.tolist(),
            initial_velocity=velocity_base.tolist(),
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
        self.collector = collector_sim.Collector(
            output_dir=str(self.output_dir),
            minimal_mode=minimal_data_mode
        )

        print("  ✅ All entities created")
        print(f"\n  📊 Formation Configuration:")
        print(f"    Chaser initial offset: [{formation_offset_x}, {formation_offset_y}, {formation_offset_z}] m")
        print(f"    Control gain: {control_gain}")
        print(f"    Plant time constant: {plant_time_constant} s")

    def connect_entities(self):
        """エンティティの接続（データフロー定義）"""
        print("\n[FormationFlyingScenario] Connecting entities...")

        # データフロー表示（.envから制御）
        controller_type = get_env_param("FORMATION_CONTROLLER_TYPE", "hcw", str)

        if self.show_dataflow:
            print("\n  📊 Data Flow:")
            print("  ┌─────────────────────────────────────────────────────────┐")
            print("  │                 Formation Flying Control                │")
            print("  └─────────────────────────────────────────────────────────┘")
            print("  [Chaser Loop]")
            print("  [1] Chaser Env → Chaser Controller")
            print("      └─ position_x/y/z, velocity_x/y/z (same-step)")
            if controller_type == "hcw":
                print("  [1b] Target Env → Chaser Controller (Chief reference)")
                print("      └─ chief_position_x/y/z, chief_velocity_x/y/z (same-step)")

        # === Chaser の制御ループ ===

        # フィードバック: Chaser Env → Controller (same-step)
        self.world.connect(
            self.chaser_spacecraft,
            self.chaser_controller,
            ("position_x", "position_x"),
            ("position_y", "position_y"),
            ("position_z", "position_z"),
            ("velocity_x", "velocity_x"),
            ("velocity_y", "velocity_y"),
            ("velocity_z", "velocity_z"),
        )

        # HCW制御の場合: Target → Controller (Chief参照)
        if controller_type == "hcw":
            self.world.connect(
                self.target_spacecraft,
                self.chaser_controller,
                ("position_x", "chief_position_x"),
                ("position_y", "chief_position_y"),
                ("position_z", "chief_position_z"),
                ("velocity_x", "chief_velocity_x"),
                ("velocity_y", "chief_velocity_y"),
                ("velocity_z", "chief_velocity_z"),
            )

        if self.show_dataflow:
            print("  [2] Chaser Controller → Chaser Plant")
            print("      └─ thrust_command_x/y/z (same-step)")

        # 指令: Controller → Plant
        self.world.connect(
            self.chaser_controller,
            self.chaser_plant,
            ("thrust_command_x", "command_x"),
            ("thrust_command_y", "command_y"),
            ("thrust_command_z", "command_z"),
        )

        if self.use_inverse_compensation:
            if self.show_dataflow:
                print("  [3] Chaser Plant → InverseCompensator → Chaser Env")
                print("      └─ compensated_force_x/y/z (time-shifted)")

            self.world.connect(
                self.chaser_plant,
                self.inverse_compensator,
                ("measured_force_x", "input_force_x"),
                ("measured_force_y", "input_force_y"),
                ("measured_force_z", "input_force_z"),
            )

            self.world.connect(
                self.inverse_compensator,
                self.chaser_spacecraft,
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
            if self.show_dataflow:
                print("  [3] Chaser Plant → Chaser Env")
                print("      └─ measured_force_x/y/z (time-shifted, breaks cycle)")

            # 計測: Plant → Env (time-shifted to break cycle)
            self.world.connect(
                self.chaser_plant,
                self.chaser_spacecraft,
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

        if self.show_dataflow:
            print("\n  [Target]")
            print("  [4] Target Env (free orbital motion)")
            print("      └─ No control inputs")

        print("\n  ✅ Control loop connected")
        print("  ℹ️  Chaser Loop: Env → Controller → Plant → [time-shift] → Env")
        print("  ℹ️  Target: Free orbital motion (no control)")

    def setup_data_collection(self):
        """データ収集の設定"""
        print("\n[FormationFlyingScenario] Setting up data collection...")

        # === Chaser のデータ収集 ===

        # Controller data
        self.world.connect(
            self.chaser_controller,
            self.collector,
            "thrust_command_x",
            "thrust_command_y",
            "thrust_command_z",
        )

        # Plant data
        self.world.connect(
            self.chaser_plant,
            self.collector,
            "measured_force_x",
            "measured_force_y",
            "measured_force_z",
            "norm_measured_force",
            "alpha",
        )

        # Chaser Environment data
        self.world.connect(
            self.chaser_spacecraft,
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

        # === Target のデータ収集 ===

        # Target Environment data
        self.world.connect(
            self.target_spacecraft,
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

        print("  ✅ Data collection configured")

    def run(self):
        """シミュレーションの実行"""
        print("\n" + "=" * 70)
        print("Formation Flying HILS Simulation")
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
        print("\n[FormationFlyingScenario] Running simulation...")
        print(f"  Duration: {self.config.simulation_time} s")

        self.world.run(until=self.config.simulation_time)

        print("\n[FormationFlyingScenario] ✅ Simulation completed")
        print(f"[FormationFlyingScenario] 📁 Results: {self.output_dir}")

        # 実行グラフの保存
        self._save_execution_graph()

        # 自動可視化
        self._auto_visualize()

        return self.output_dir

    def _save_execution_graph(self):
        """実行グラフを保存"""
        try:
            print("\n[FormationFlyingScenario] 📊 Generating execution graph...")

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
                    title="Formation Flying HILS Execution Graph",
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
            print("\n[FormationFlyingScenario] ⏭️  Auto-visualization disabled")
            return

        print("\n[FormationFlyingScenario] 📊 Auto-generating visualizations...")

        h5_file = self.output_dir / "hils_data.h5"

        if not h5_file.exists():
            print("  ⚠️  HDF5 file not found, skipping visualization")
            return

        try:
            import subprocess

            # Formation flyingの専用可視化スクリプトを呼ぶ
            print("  📈 Generating formation flying plots...")
            result = subprocess.run(
                [
                    "python",
                    "scripts/analysis/visualize_formation_flying.py",
                    str(h5_file),
                ],
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                print("  ✅ Formation flying plots generated")
            else:
                # スクリプトがなければ通常の軌道可視化を実行
                print("  ⚠️  Formation script not found, using standard orbital visualization")
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

        except Exception as e:
            print(f"  ⚠️  Auto-visualization failed: {e}")


if __name__ == "__main__":
    # テスト実行
    scenario = FormationFlyingScenario(config=CONFIG_ISS)
    scenario.run()
