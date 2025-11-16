"""
Sweep Results Comparison Visualization

パラメータスイープ結果を比較可視化するスクリプト。
複数のシミュレーション結果を同時にプロットして比較。

使用例:
    # スイープディレクトリを指定
    python compare_sweep_results.py /path/to/results_sweep/20251116-010424_sweep

    # 特定の結果のみを比較
    python compare_sweep_results.py /path/to/sweep --indices 1 2 3

    # フェーズ可視化も生成
    python compare_sweep_results.py /path/to/sweep --with-phases
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import h5py
import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def load_orbital_data(h5_path: Path) -> Dict:
    """
    HDF5ファイルから軌道データを読み込む

    Args:
        h5_path: HDF5ファイルのパス

    Returns:
        dict: 時系列データ
    """
    data = {}

    with h5py.File(h5_path, "r") as f:
        # 時間データ
        data["time"] = f["time"]["time_s"][:]

        # 衛星データグループを検索
        spacecraft_group_name = [k for k in f.keys() if "OrbitalSpacecraft" in k][0]
        sc_group = f[spacecraft_group_name]

        # データ読み込み
        data["position_x"] = sc_group["position_x"][:]
        data["position_y"] = sc_group["position_y"][:]
        data["position_z"] = sc_group["position_z"][:]
        data["altitude"] = sc_group["altitude"][:]
        data["velocity_norm"] = sc_group["velocity_norm"][:]
        data["semi_major_axis"] = sc_group["semi_major_axis"][:]
        data["eccentricity"] = sc_group["eccentricity"][:]

        # 推力データ（EnvSimへの入力 = 実際に宇宙機に作用した力）
        # Note: norm_forceはEnvSimに入力された力（InvCompがあればその出力、なければPlantの出力）
        data["norm_force"] = sc_group["norm_force"][:]

        # Plant出力も読み込み（比較用）
        try:
            plant_group_name = [k for k in f.keys() if "OrbitalThrustStand" in k][0]
            plant_group = f[plant_group_name]
            data["norm_measured_force"] = plant_group["norm_measured_force"][:]
        except (IndexError, KeyError):
            data["norm_measured_force"] = None

        # Inverse compensator データ（存在する場合）
        try:
            inv_comp_name = [k for k in f.keys() if "InverseCompensator" in k][0]
            inv_comp_group = f[inv_comp_name]
            data["compensated_norm_force"] = inv_comp_group["compensated_norm_force"][:]
        except (IndexError, KeyError):
            data["compensated_norm_force"] = None

    return data


def find_sweep_results(sweep_dir: Path) -> List[Dict]:
    """
    スイープディレクトリから結果ファイルを検索

    Args:
        sweep_dir: スイープディレクトリのパス

    Returns:
        List[Dict]: 結果情報のリスト
    """
    results = []

    # サブディレクトリをスキャン
    for subdir in sorted(sweep_dir.iterdir()):
        if not subdir.is_dir():
            continue

        # HDF5ファイルを探す
        h5_files = list(subdir.glob("*.h5"))
        if not h5_files:
            continue

        h5_file = h5_files[0]  # 最初のHDF5ファイルを使用

        # ディレクトリ名から設定を取得
        dir_name = subdir.name
        # 例: "001_tau=10.0_gain=2.0" -> "tau=10.0, gain=2.0"
        if "_" in dir_name:
            parts = dir_name.split("_", 1)
            if len(parts) == 2:
                index = int(parts[0])
                # パラメータ部分を整形（_ を ", " に置換）
                param_str = parts[1]
                # 各パラメータを ", " で区切る
                label = param_str.replace("_", ", ").replace("=", "=")
            else:
                index = 0
                label = dir_name
        else:
            index = 0
            label = dir_name

        results.append(
            {
                "index": index,
                "label": label,
                "path": h5_file,
                "dir": subdir,
            }
        )

    return results


def plot_altitude_thrust_comparison(results: List[Dict], output_path: Path):
    """
    高度と推力の比較プロット

    Args:
        results: 結果情報のリスト
        output_path: 出力ファイルパス
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    for result in results:
        data = load_orbital_data(result["path"])
        time_min = data["time"] / 60.0
        alt_km = data["altitude"] / 1e3

        # 高度プロット
        ax1.plot(time_min, alt_km, label=result["label"], alpha=0.7, linewidth=1.5)

        # 推力プロット（EnvSimに入力された力 = 宇宙機に実際に作用した力）
        ax2.plot(time_min, data["norm_force"], label=result["label"], alpha=0.7, linewidth=1.5)

    # 高度グラフ設定
    ax1.set_xlabel("Time [min]")
    ax1.set_ylabel("Altitude [km]")
    ax1.set_title("Altitude Comparison")
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    # 推力グラフ設定
    ax2.set_xlabel("Time [min]")
    ax2.set_ylabel("Thrust [N]")
    ax2.set_title("Thrust Comparison")
    ax2.grid(True, alpha=0.3)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✅ Saved: {output_path}")


def plot_3d_trajectory_comparison(results: List[Dict], output_path: Path):
    """
    3D軌道比較プロット

    Args:
        results: 結果情報のリスト
        output_path: 出力ファイルパス
    """
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")

    # 地球を描画
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    R_earth = 6371  # km
    x_earth = R_earth * np.outer(np.cos(u), np.sin(v))
    y_earth = R_earth * np.outer(np.sin(u), np.sin(v))
    z_earth = R_earth * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x_earth, y_earth, z_earth, color="lightblue", alpha=0.3)

    # 軌道プロット
    for i, result in enumerate(results):
        data = load_orbital_data(result["path"])
        pos_km = {
            "x": data["position_x"] / 1e3,
            "y": data["position_y"] / 1e3,
            "z": data["position_z"] / 1e3,
        }

        ax.plot(pos_km["x"], pos_km["y"], pos_km["z"], label=result["label"], alpha=0.7, linewidth=1.5)

        # 開始点
        ax.scatter(pos_km["x"][0], pos_km["y"][0], pos_km["z"][0], s=100, marker="o", alpha=0.8)

    ax.set_xlabel("X [km]")
    ax.set_ylabel("Y [km]")
    ax.set_zlabel("Z [km]")
    ax.set_title("3D Orbital Trajectory Comparison")
    ax.legend(bbox_to_anchor=(1.15, 1), loc="upper left")

    # アスペクト比を等しくする
    max_range = np.array(
        [
            max(abs(data["position_x"])) / 1e3,
            max(abs(data["position_y"])) / 1e3,
            max(abs(data["position_z"])) / 1e3,
        ]
    ).max()
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✅ Saved: {output_path}")


def generate_interactive_html(results: List[Dict], output_path: Path):
    """
    インタラクティブHTML軌道比較

    Args:
        results: 結果情報のリスト
        output_path: 出力ファイルパス
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("⚠️  plotly not installed. Skipping interactive HTML generation.")
        print("   Install with: uv add plotly")
        return

    fig = go.Figure()

    # 地球を描画
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    R_earth = 6371  # km
    x_earth = R_earth * np.outer(np.cos(u), np.sin(v))
    y_earth = R_earth * np.outer(np.sin(u), np.sin(v))
    z_earth = R_earth * np.outer(np.ones(np.size(u)), np.cos(v))

    fig.add_trace(
        go.Surface(
            x=x_earth,
            y=y_earth,
            z=z_earth,
            colorscale="Blues",
            showscale=False,
            opacity=0.3,
            name="Earth",
        )
    )

    # 軌道プロット
    for result in results:
        data = load_orbital_data(result["path"])
        pos_km = {
            "x": data["position_x"] / 1e3,
            "y": data["position_y"] / 1e3,
            "z": data["position_z"] / 1e3,
        }

        fig.add_trace(
            go.Scatter3d(
                x=pos_km["x"],
                y=pos_km["y"],
                z=pos_km["z"],
                mode="lines",
                name=result["label"],
                line=dict(width=3),
            )
        )

        # 開始点
        fig.add_trace(
            go.Scatter3d(
                x=[pos_km["x"][0]],
                y=[pos_km["y"][0]],
                z=[pos_km["z"][0]],
                mode="markers",
                name=f"{result['label']} (start)",
                marker=dict(size=8),
                showlegend=False,
            )
        )

    fig.update_layout(
        title="3D Orbital Trajectory Comparison (Interactive)",
        scene=dict(
            xaxis_title="X [km]",
            yaxis_title="Y [km]",
            zaxis_title="Z [km]",
            aspectmode="cube",
        ),
        height=800,
    )

    fig.write_html(output_path)
    print(f"✅ Saved: {output_path}")


def plot_phase_comparison(results: List[Dict], output_path: Path):
    """
    フェーズ比較プロット（高度-速度、高度-軌道要素）

    Args:
        results: 結果情報のリスト
        output_path: 出力ファイルパス
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    for result in results:
        data = load_orbital_data(result["path"])
        alt_km = data["altitude"] / 1e3
        vel_km_s = data["velocity_norm"] / 1e3
        sma_km = data["semi_major_axis"] / 1e3

        # 高度 vs 速度
        axes[0, 0].plot(alt_km, vel_km_s, label=result["label"], alpha=0.7, linewidth=1.5)

        # 高度 vs 軌道長半径
        axes[0, 1].plot(alt_km, sma_km, label=result["label"], alpha=0.7, linewidth=1.5)

        # 高度 vs 離心率
        axes[1, 0].plot(alt_km, data["eccentricity"], label=result["label"], alpha=0.7, linewidth=1.5)

        # 時間 vs 高度
        time_min = data["time"] / 60.0
        axes[1, 1].plot(time_min, alt_km, label=result["label"], alpha=0.7, linewidth=1.5)

    # グラフ設定
    axes[0, 0].set_xlabel("Altitude [km]")
    axes[0, 0].set_ylabel("Velocity [km/s]")
    axes[0, 0].set_title("Altitude vs Velocity")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    axes[0, 1].set_xlabel("Altitude [km]")
    axes[0, 1].set_ylabel("Semi-major Axis [km]")
    axes[0, 1].set_title("Altitude vs Semi-major Axis")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    axes[1, 0].set_xlabel("Altitude [km]")
    axes[1, 0].set_ylabel("Eccentricity")
    axes[1, 0].set_title("Altitude vs Eccentricity")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    axes[1, 1].set_xlabel("Time [min]")
    axes[1, 1].set_ylabel("Altitude [km]")
    axes[1, 1].set_title("Time vs Altitude")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✅ Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare sweep simulation results")
    parser.add_argument("sweep_dir", type=str, help="Sweep directory path")
    parser.add_argument("--indices", nargs="+", type=int, help="Specific indices to compare")
    parser.add_argument("--with-phases", action="store_true", help="Generate phase plots")
    parser.add_argument("--output-dir", type=str, help="Output directory (default: sweep_dir/comparison)")

    args = parser.parse_args()

    # スイープディレクトリ
    sweep_dir = Path(args.sweep_dir)
    if not sweep_dir.exists():
        print(f"❌ Error: Sweep directory not found: {sweep_dir}")
        return

    # 結果を検索
    print(f"\n🔍 Scanning sweep directory: {sweep_dir}")
    results = find_sweep_results(sweep_dir)

    if not results:
        print("❌ No results found in sweep directory")
        return

    print(f"📊 Found {len(results)} simulation results")

    # インデックスフィルタリング
    if args.indices:
        results = [r for r in results if r["index"] in args.indices]
        print(f"   Filtering to {len(results)} results based on indices: {args.indices}")

    if not results:
        print("❌ No results match the specified indices")
        return

    # 出力ディレクトリ
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = sweep_dir / "comparison"

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output directory: {output_dir}\n")

    # 結果リストを表示
    print("Results to compare:")
    for r in results:
        print(f"  [{r['index']}] {r['label']}")
    print()

    # 可視化生成
    print("Generating visualizations...")

    # 1. 高度と推力の比較
    plot_altitude_thrust_comparison(results, output_dir / "altitude_thrust_comparison.png")

    # 2. 3D軌道比較
    plot_3d_trajectory_comparison(results, output_dir / "3d_trajectory_comparison.png")

    # 3. インタラクティブHTML
    generate_interactive_html(results, output_dir / "trajectory_interactive.html")

    # 4. フェーズプロット（オプション）
    if args.with_phases:
        plot_phase_comparison(results, output_dir / "phase_comparison.png")

    print(f"\n✅ Comparison complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
