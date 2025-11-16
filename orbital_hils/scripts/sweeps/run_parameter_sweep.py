"""
Orbital HILS Parameter Sweep Script

汎用的なパラメータスイープツール。.envファイルの任意のパラメータを
リストで指定して、全ての組み合わせでシミュレーションを実行します。

使用例:
    # 単一パラメータのスイープ
    python run_parameter_sweep.py

    # カスタム設定
    sweep_params = {
        "PLANT_TIME_CONSTANT": [5.0, 10.0, 20.0],
        "INVERSE_COMPENSATION_GAIN": [1.0, 2.0, 3.0],
    }
"""

import os
import sys
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Dict, List

from config.orbital_parameters import load_config_from_env
from scenarios.hohmann_scenario import HohmannScenario
from scenarios.orbital_scenario import OrbitalScenario

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class ParameterSweepConfig:
    """パラメータスイープ設定"""

    def __init__(
        self,
        sweep_params: Dict[str, List[Any]],
        base_env_file: str = ".env",
        output_base_dir: str = "results_sweep",
        description: str = "Parameter Sweep",
    ):
        """
        初期化

        Args:
            sweep_params: スイープするパラメータと値のリスト
                例: {"PLANT_TIME_CONSTANT": [5.0, 10.0, 20.0]}
            base_env_file: ベースとなる.envファイルのパス
            output_base_dir: 結果出力のベースディレクトリ
            description: スイープの説明
        """
        self.sweep_params = sweep_params
        self.base_env_file = base_env_file
        self.output_base_dir = output_base_dir
        self.description = description

        # ベース.envファイルの読み込み
        self.base_env = self._load_env_file(base_env_file)

        # スイープ設定の生成
        self.configs = self._generate_configs()

    def _load_env_file(self, env_file: str) -> Dict[str, str]:
        """
        .envファイルを読み込み

        Args:
            env_file: .envファイルのパス

        Returns:
            環境変数の辞書
        """
        env_path = project_root / env_file
        env_dict = {}

        if not env_path.exists():
            print(f"⚠️  Warning: {env_file} not found, using empty base")
            return env_dict

        with open(env_path) as f:
            for line in f:
                line = line.strip()
                # コメントと空行をスキップ
                if not line or line.startswith("#"):
                    continue
                # KEY=VALUE形式をパース
                if "=" in line:
                    key, value = line.split("=", 1)
                    env_dict[key.strip()] = value.strip()

        return env_dict

    def _generate_configs(self) -> List[Dict[str, Any]]:
        """
        全てのパラメータ組み合わせの設定を生成

        Returns:
            設定の辞書のリスト
        """
        # スイープするパラメータ名と値のリストを取得
        param_names = list(self.sweep_params.keys())
        param_values = list(self.sweep_params.values())

        # 全ての組み合わせを生成
        configs = []
        for values in product(*param_values):
            config = dict(zip(param_names, values))
            configs.append(config)

        return configs

    def get_env_for_config(self, config: Dict[str, Any]) -> Dict[str, str]:
        """
        指定された設定用の環境変数辞書を生成

        Args:
            config: パラメータ設定

        Returns:
            環境変数の辞書
        """
        # ベース環境変数をコピー
        env = self.base_env.copy()

        # スイープパラメータで上書き
        for key, value in config.items():
            env[key] = str(value)

        return env

    def get_config_label(self, config: Dict[str, Any]) -> str:
        """
        設定のラベルを生成

        Args:
            config: パラメータ設定

        Returns:
            ラベル文字列
        """
        parts = []
        for key, value in config.items():
            # キー名を短縮（例: PLANT_TIME_CONSTANT -> tau）
            short_key = self._shorten_key(key)
            parts.append(f"{short_key}={value}")

        return "_".join(parts)

    def _shorten_key(self, key: str) -> str:
        """
        パラメータ名を短縮

        Args:
            key: パラメータ名

        Returns:
            短縮名
        """
        # よく使うパラメータの短縮名マッピング
        short_names = {
            "PLANT_TIME_CONSTANT": "tau",
            "PLANT_NOISE_STD": "noise",
            "INVERSE_COMPENSATION": "inv_comp",
            "INVERSE_COMPENSATION_GAIN": "gain",
            "CONTROL_GAIN": "Kp",
            "SIMULATION_TIME": "T",
            "TIME_RESOLUTION": "dt",
            "SPACECRAFT_MASS": "mass",
            "ALTITUDE_KM": "alt",
        }

        return short_names.get(key, key.lower())


def run_sweep(sweep_config: ParameterSweepConfig, dry_run: bool = False):
    """
    パラメータスイープを実行

    Args:
        sweep_config: スイープ設定
        dry_run: True の場合、実行せずに設定を表示するのみ
    """
    # 出力ディレクトリの作成
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    sweep_dir = project_root / sweep_config.output_base_dir / f"{timestamp}_sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"{sweep_config.description}")
    print("=" * 70)
    print(f"Total configurations: {len(sweep_config.configs)}")
    print(f"Output directory: {sweep_dir}")
    print()

    # 設定の表示
    print("Sweep parameters:")
    for key, values in sweep_config.sweep_params.items():
        print(f"  {key}: {values}")
    print()

    if dry_run:
        print("🔍 Dry run mode - showing configurations without execution:\n")
        for i, config in enumerate(sweep_config.configs, 1):
            label = sweep_config.get_config_label(config)
            print(f"{i}. {label}")
            for key, value in config.items():
                print(f"     {key} = {value}")
            print()
        return

    # スイープ実行
    results = []
    for i, config in enumerate(sweep_config.configs, 1):
        label = sweep_config.get_config_label(config)
        print(f"\n{'=' * 70}")
        print(f"Running {i}/{len(sweep_config.configs)}: {label}")
        print(f"{'=' * 70}")

        # 結果ディレクトリの設定
        result_subdir = sweep_dir / f"{i:03d}_{label}"
        os.environ["OUTPUT_DIR_OVERRIDE"] = str(result_subdir)

        try:
            # シミュレーション実行
            # 1. スイープパラメータを環境変数に設定
            #    (シナリオがget_env_param()で読み込むパラメータ用)
            for key, value in config.items():
                os.environ[key] = str(value)

            # 2. ベース設定を.envから読み込み（環境変数が優先される）
            orbital_config = load_config_from_env()

            # 3. OrbitalSimulationConfigの属性を直接上書き
            for key, value in config.items():
                # Spacecraft parameters
                if key == "SPACECRAFT_MASS":
                    orbital_config.spacecraft.mass = float(value)
                elif key == "MAX_THRUST":
                    orbital_config.spacecraft.max_thrust = float(value)
                elif key == "SPECIFIC_IMPULSE":
                    orbital_config.spacecraft.specific_impulse = float(value)

                # Orbital parameters
                elif key == "ALTITUDE_KM":
                    # Recalculate semi-major axis
                    altitude_m = float(value) * 1000.0
                    orbital_config.orbit.semi_major_axis = orbital_config.orbit.radius_body + altitude_m
                elif key == "ECCENTRICITY":
                    orbital_config.orbit.eccentricity = float(value)
                elif key == "INCLINATION_DEG":
                    orbital_config.orbit.inclination = float(value)
                elif key == "RAAN_DEG":
                    orbital_config.orbit.raan = float(value)
                elif key == "ARG_PERIAPSIS_DEG":
                    orbital_config.orbit.arg_periapsis = float(value)
                elif key == "TRUE_ANOMALY_DEG":
                    orbital_config.orbit.true_anomaly = float(value)

                # Simulation parameters
                elif key == "SIMULATION_TIME":
                    orbital_config.simulation_time = float(value)
                elif key == "TIME_RESOLUTION":
                    orbital_config.time_resolution = float(value)

            # 3. CONTROLLER_TYPEに基づいてシナリオを選択
            controller_type = config.get("CONTROLLER_TYPE", os.environ.get("CONTROLLER_TYPE", "zero"))
            if controller_type == "hohmann":
                scenario = HohmannScenario(config=orbital_config)
            else:
                scenario = OrbitalScenario(config=orbital_config)
            result_dir = scenario.run()

            # 結果を記録
            results.append(
                {
                    "index": i,
                    "label": label,
                    "config": config,
                    "result_dir": result_dir,
                    "status": "success",
                }
            )

            print(f"✅ Completed: {label}")

        except Exception as e:
            print(f"❌ Failed: {label}")
            print(f"   Error: {e}")

            results.append(
                {
                    "index": i,
                    "label": label,
                    "config": config,
                    "result_dir": None,
                    "status": "failed",
                    "error": str(e),
                }
            )

    # サマリーの出力
    print("\n" + "=" * 70)
    print("Sweep Summary")
    print("=" * 70)

    success_count = sum(1 for r in results if r["status"] == "success")
    failed_count = len(results) - success_count

    print(f"Total simulations: {len(results)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {failed_count}")
    print()

    if failed_count > 0:
        print("Failed simulations:")
        for r in results:
            if r["status"] == "failed":
                print(f"  {r['label']}: {r.get('error', 'Unknown error')}")
        print()

    # 結果の保存
    summary_file = sweep_dir / "sweep_summary.txt"
    with open(summary_file, "w") as f:
        f.write(f"{sweep_config.description}\n")
        f.write(f"{'=' * 70}\n\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Total configurations: {len(results)}\n")
        f.write(f"Successful: {success_count}\n")
        f.write(f"Failed: {failed_count}\n\n")

        f.write("Sweep parameters:\n")
        for key, values in sweep_config.sweep_params.items():
            f.write(f"  {key}: {values}\n")
        f.write("\n")

        f.write("Results:\n")
        for r in results:
            f.write(f"\n{r['index']}. {r['label']}\n")
            f.write(f"   Status: {r['status']}\n")
            if r["status"] == "success":
                f.write(f"   Directory: {r['result_dir']}\n")
            else:
                f.write(f"   Error: {r.get('error', 'Unknown')}\n")

    print(f"📁 Summary saved to: {summary_file}")
    print(f"📁 All results in: {sweep_dir}")

    # 自動可視化（成功したシミュレーションが2つ以上ある場合）
    if success_count >= 2:
        print("\n" + "=" * 70)
        print("Generating comparison visualizations...")
        print("=" * 70)

        try:
            import subprocess

            # 比較可視化スクリプトを実行
            result = subprocess.run(
                [
                    "python",
                    str(project_root / "scripts/analysis/compare_sweep_results.py"),
                    str(sweep_dir),
                    "--with-phases",
                ],
                capture_output=True,
                text=True,
                cwd=str(project_root),
            )

            if result.returncode == 0:
                print("✅ Comparison visualizations generated")
            else:
                print(f"⚠️  Visualization failed: {result.stderr}")

        except Exception as e:
            print(f"⚠️  Could not generate visualizations: {e}")


# ============================================================================
# 使用例
# ============================================================================

if __name__ == "__main__":
    import sys

    # Example 1: Plant time constant sweep
    sweep_params_example1 = {
        "PLANT_TIME_CONSTANT": [5.0, 10.0, 20.0, 50.0],
        "PLANT_NOISE_STD": [0.0, 0.01, 0.05],
    }

    # Example 2: Inverse compensation sweep
    sweep_params_example2 = {
        "INVERSE_COMPENSATION": [True, False],
        "INVERSE_COMPENSATION_GAIN": [100.0],
        "PLANT_TIME_CONSTANT": [100.0],
        "CONTROLLER_TYPE": ["hohmann"],
    }

    # Example 3: Controller gain sweep
    sweep_params_example3 = {
        "CONTROL_GAIN": [0.001, 0.01, 0.1, 1.0],
        "SIMULATION_TIME": [100.0],  # Short simulation for quick test
    }

    # 使用するスイープパラメータを選択
    # ここを変更してスイープ内容をカスタマイズ
    sweep_params = sweep_params_example2

    # スイープ設定の作成
    config = ParameterSweepConfig(
        sweep_params=sweep_params,
        base_env_file=".env",
        output_base_dir="results_sweep",
        description="Orbital HILS Parameter Sweep",
    )

    # コマンドライン引数をチェック
    dry_run = "--dry-run" in sys.argv

    if dry_run:
        print("🔍 Dry run mode - showing configuration without execution\n")

    # 実行
    run_sweep(config, dry_run=dry_run)
