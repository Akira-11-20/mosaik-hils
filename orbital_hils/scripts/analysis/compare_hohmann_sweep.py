"""
Hohmann Transfer Sweep Comparison Visualization

Hohmann transfer のパラメータスイープ結果を比較可視化するスクリプト。
高度変化、推力使用量、ベースラインとの差分を含む。

使用例:
    # スイープディレクトリを指定
    python compare_hohmann_sweep.py /path/to/results_sweep/20251116-154228_sweep

    # 特定の結果のみを比較
    python compare_hohmann_sweep.py /path/to/sweep --indices 1 2 3
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


def load_hohmann_data(h5_path: Path) -> Dict:
    """
    HDF5ファイルからHohmann transferデータを読み込む

    Args:
        h5_path: HDF5ファイルのパス

    Returns:
        dict: 時系列データ
    """
    data = {}

    with h5py.File(h5_path, "r") as f:
        # 時間データ
        data["time"] = f["time"]["time_s"][:]

        # 宇宙機データグループを検索（1機のみ）
        env_groups = [k for k in f.keys() if "OrbitalEnvSim" in k]
        if not env_groups:
            raise ValueError("No OrbitalEnvSim group found")

        spacecraft_group = f[env_groups[0]]

        # 軌道要素
        data["altitude"] = spacecraft_group["altitude"][:]
        data["semi_major_axis"] = spacecraft_group["semi_major_axis"][:]
        data["eccentricity"] = spacecraft_group["eccentricity"][:]
        data["specific_energy"] = spacecraft_group["specific_energy"][:]

        # 位置・速度
        data["position_x"] = spacecraft_group["position_x"][:]
        data["position_y"] = spacecraft_group["position_y"][:]
        data["position_z"] = spacecraft_group["position_z"][:]
        data["position_norm"] = spacecraft_group["position_norm"][:]

        data["velocity_x"] = spacecraft_group["velocity_x"][:]
        data["velocity_y"] = spacecraft_group["velocity_y"][:]
        data["velocity_z"] = spacecraft_group["velocity_z"][:]
        data["velocity_norm"] = spacecraft_group["velocity_norm"][:]

        # 推力データ（環境への入力force）
        try:
            data["norm_force"] = spacecraft_group["norm_force"][:]
            data["force_x"] = spacecraft_group["force_x"][:]
            data["force_y"] = spacecraft_group["force_y"][:]
            data["force_z"] = spacecraft_group["force_z"][:]
        except KeyError:
            data["norm_force"] = None

        # Controller データ（存在する場合）
        try:
            ctrl_group_name = [k for k in f.keys() if "OrbitalControllerSim" in k][0]
            ctrl_group = f[ctrl_group_name]
            thrust_x = ctrl_group["thrust_command_x"][:]
            thrust_y = ctrl_group["thrust_command_y"][:]
            thrust_z = ctrl_group["thrust_command_z"][:]
            data["norm_thrust_command"] = np.sqrt(thrust_x**2 + thrust_y**2 + thrust_z**2)
        except (IndexError, KeyError):
            data["norm_thrust_command"] = None

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

        # ラベル生成（ディレクトリ名から）
        label = subdir.name.split("_", 1)[1] if "_" in subdir.name else subdir.name

        results.append({"path": h5_files[0], "label": label, "dir": subdir})

    return results


def plot_altitude_thrust_comparison(results: List[Dict], output_path: Path):
    """
    高度と推力の比較プロット

    Args:
        results: 結果情報のリスト
        output_path: 出力ファイルパス
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    for result in results:
        data = load_hohmann_data(result["path"])
        time_min = data["time"] / 60.0

        # ベースラインの場合は太い線で強調（被さっても見やすいように）
        is_baseline = "baseline" in result["label"].lower()
        linewidth = 4 if is_baseline else 2
        linestyle = "-" if is_baseline else "-"
        alpha = 1.0 if is_baseline else 0.7

        # 高度プロット
        ax1.plot(
            time_min,
            data["altitude"] / 1e3,  # Convert to km
            label=result["label"],
            alpha=alpha,
            linewidth=linewidth,
            linestyle=linestyle,
        )

        # 推力プロット（Env入力force）
        if data["norm_force"] is not None:
            ax2.plot(
                time_min,
                data["norm_force"],
                label=result["label"],
                alpha=alpha,
                linewidth=linewidth,
                linestyle=linestyle,
            )
        elif data["norm_thrust_command"] is not None:
            ax2.plot(
                time_min,
                data["norm_thrust_command"],
                label=result["label"],
                alpha=alpha,
                linewidth=linewidth,
                linestyle=linestyle,
            )

    # 高度グラフ設定
    ax1.set_xlabel("Time [min]", fontsize=12)
    ax1.set_ylabel("Altitude [km]", fontsize=12)
    ax1.set_title("Hohmann Transfer: Altitude Comparison", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)

    # 推力グラフ設定
    ax2.set_xlabel("Time [min]", fontsize=12)
    ax2.set_ylabel("Thrust [N]", fontsize=12)
    ax2.set_title("Hohmann Transfer: Thrust Magnitude", fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✅ Saved: {output_path}")


def plot_orbital_elements_comparison(results: List[Dict], output_path: Path):
    """
    軌道要素の比較プロット

    Args:
        results: 結果情報のリスト
        output_path: 出力ファイルパス
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    for result in results:
        data = load_hohmann_data(result["path"])
        time_min = data["time"] / 60.0

        # ベースラインの場合は太い線で強調（被さっても見やすいように）
        is_baseline = "baseline" in result["label"].lower()
        linewidth = 4 if is_baseline else 2
        alpha = 1.0 if is_baseline else 0.7

        # Semi-major axis
        axes[0].plot(time_min, data["semi_major_axis"] / 1e3, label=result["label"], alpha=alpha, linewidth=linewidth)

        # Eccentricity
        axes[1].plot(time_min, data["eccentricity"], label=result["label"], alpha=alpha, linewidth=linewidth)

        # Specific energy
        axes[2].plot(
            time_min,
            data["specific_energy"] / 1e6,  # MJ/kg
            label=result["label"],
            alpha=alpha,
            linewidth=linewidth,
        )

    # グラフ設定
    axes[0].set_ylabel("Semi-major Axis [km]", fontsize=12)
    axes[0].set_title("Hohmann Transfer: Orbital Elements Comparison", fontsize=14, fontweight="bold")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)

    axes[1].set_ylabel("Eccentricity [-]", fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)

    axes[2].set_ylabel("Specific Energy [MJ/kg]", fontsize=12)
    axes[2].set_xlabel("Time [min]", fontsize=12)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✅ Saved: {output_path}")


def plot_baseline_difference(results: List[Dict], output_dir: Path):
    """
    ベースラインとの差分プロット（高度と推力）

    Args:
        results: 結果情報のリスト
        output_dir: 出力ディレクトリ
    """
    # ベースラインを探す
    baseline_result = None
    other_results = []

    for result in results:
        if "baseline" in result["label"].lower():
            baseline_result = result
        else:
            other_results.append(result)

    if baseline_result is None:
        print("⚠️  No baseline found, skipping difference plots")
        return

    if not other_results:
        print("⚠️  No non-baseline results found, skipping difference plots")
        return

    # ベースラインデータを読み込み
    baseline_data = load_hohmann_data(baseline_result["path"])
    baseline_time = baseline_data["time"]

    # 差分プロット1: 高度と推力の差分
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    for result in other_results:
        data = load_hohmann_data(result["path"])
        time_min = data["time"] / 60.0

        # 時間軸が一致していることを確認
        if len(data["time"]) == len(baseline_time) and np.allclose(data["time"], baseline_time):
            # 高度の差分
            altitude_diff = (data["altitude"] - baseline_data["altitude"]) / 1e3  # km
            ax1.plot(time_min, altitude_diff, label=result["label"], alpha=0.8, linewidth=2)

            # 推力の差分（Env入力force）
            if data["norm_force"] is not None and baseline_data["norm_force"] is not None:
                thrust_diff = data["norm_force"] - baseline_data["norm_force"]
                ax2.plot(time_min, thrust_diff, label=result["label"], alpha=0.8, linewidth=2)
            elif data["norm_thrust_command"] is not None and baseline_data["norm_thrust_command"] is not None:
                thrust_diff = data["norm_thrust_command"] - baseline_data["norm_thrust_command"]
                ax2.plot(time_min, thrust_diff, label=result["label"], alpha=0.8, linewidth=2)
        else:
            print(f"⚠️  Time mismatch for {result['label']}, skipping")

    # ゼロラインを追加
    ax1.axhline(y=0, color="black", linestyle="--", linewidth=1, alpha=0.5, label="Baseline")
    ax2.axhline(y=0, color="black", linestyle="--", linewidth=1, alpha=0.5, label="Baseline")

    # グラフ設定
    ax1.set_ylabel("Δ Altitude [km]", fontsize=12)
    ax1.set_title(
        "Difference from Baseline: Altitude",
        fontsize=14,
        fontweight="bold",
    )
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)

    ax2.set_xlabel("Time [min]", fontsize=12)
    ax2.set_ylabel("Δ Thrust [N]", fontsize=12)
    ax2.set_title(
        "Difference from Baseline: Thrust Magnitude",
        fontsize=14,
        fontweight="bold",
    )
    ax2.grid(True, alpha=0.3)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)

    plt.tight_layout()
    output_path = output_dir / "hohmann_baseline_difference.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✅ Saved: {output_path}")

    # 差分プロット2: 軌道要素の差分
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    for result in other_results:
        data = load_hohmann_data(result["path"])
        time_min = data["time"] / 60.0

        if len(data["time"]) == len(baseline_time) and np.allclose(data["time"], baseline_time):
            # Semi-major axis の差分
            sma_diff = (data["semi_major_axis"] - baseline_data["semi_major_axis"]) / 1e3  # km
            axes[0].plot(time_min, sma_diff, label=result["label"], alpha=0.8, linewidth=2)

            # Eccentricity の差分
            ecc_diff = data["eccentricity"] - baseline_data["eccentricity"]
            axes[1].plot(time_min, ecc_diff, label=result["label"], alpha=0.8, linewidth=2)

            # Specific energy の差分
            energy_diff = (data["specific_energy"] - baseline_data["specific_energy"]) / 1e6  # MJ/kg
            axes[2].plot(time_min, energy_diff, label=result["label"], alpha=0.8, linewidth=2)

    # ゼロラインを追加
    for ax in axes:
        ax.axhline(y=0, color="black", linestyle="--", linewidth=1, alpha=0.5, label="Baseline")

    # グラフ設定
    axes[0].set_ylabel("Δ Semi-major Axis [km]", fontsize=12)
    axes[0].set_title(
        "Difference from Baseline: Orbital Elements",
        fontsize=14,
        fontweight="bold",
    )
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)

    axes[1].set_ylabel("Δ Eccentricity [-]", fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)

    axes[2].set_ylabel("Δ Specific Energy [MJ/kg]", fontsize=12)
    axes[2].set_xlabel("Time [min]", fontsize=12)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)

    plt.tight_layout()
    output_path = output_dir / "hohmann_baseline_orbital_difference.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✅ Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Hohmann Transfer Sweep Comparison Visualization")
    parser.add_argument("sweep_dir", type=str, help="スイープディレクトリのパス")
    parser.add_argument(
        "--indices",
        type=int,
        nargs="+",
        help="比較する結果のインデックス（指定しない場合は全て）",
    )
    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir)
    if not sweep_dir.exists():
        print(f"❌ Error: Directory not found: {sweep_dir}")
        return

    print(f"🔍 Scanning Hohmann transfer sweep directory: {sweep_dir}")

    # 結果を検索
    results = find_sweep_results(sweep_dir)
    if not results:
        print("❌ No results found")
        return

    print(f"📊 Found {len(results)} simulation results")

    # インデックス指定がある場合はフィルタ
    if args.indices:
        results = [r for i, r in enumerate(results, 1) if i in args.indices]
        print(f"Filtering to {len(results)} results based on indices")

    # 出力ディレクトリ作成
    output_dir = sweep_dir / "comparison"
    output_dir.mkdir(exist_ok=True)
    print(f"📁 Output directory: {output_dir}")

    print("\nResults to compare:")
    for i, result in enumerate(results, 1):
        print(f"  [{i}] {result['label']}")

    if len(results) < 2:
        print("\n⚠️  Need at least 2 results for comparison")
        return

    print("\nGenerating Hohmann transfer comparison visualizations...")

    # 1. 高度と推力の比較
    plot_altitude_thrust_comparison(results, output_dir / "hohmann_altitude_thrust_comparison.png")

    # 2. 軌道要素の比較
    plot_orbital_elements_comparison(results, output_dir / "hohmann_orbital_elements_comparison.png")

    # 3. ベースラインとの差分（NEW!）
    print()
    plot_baseline_difference(results, output_dir)

    print(f"\n✅ Hohmann transfer comparison complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
