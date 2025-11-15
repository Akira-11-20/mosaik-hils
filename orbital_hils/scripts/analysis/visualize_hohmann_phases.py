"""
ホーマン遷移の各フェーズを色分けして可視化

各フェーズを異なる色で表示：
- 遷移前（青）
- 第1バーン（赤）
- コーストフェーズ（緑）
- 第2バーン（オレンジ）
- 遷移後（紫）

使用方法:
    uv run python scripts/analysis/visualize_hohmann_phases.py <HDF5_FILE>
"""

import argparse
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def detect_phases(thrust_magnitude, time, threshold=1.0):
    """
    推力データから各フェーズを検出

    Args:
        thrust_magnitude: 推力の大きさ [N]
        time: 時刻 [s]
        threshold: 推力ありと判定する閾値 [N]

    Returns:
        phases: フェーズのリスト [(start_idx, end_idx, phase_name, color), ...]
    """
    burning = thrust_magnitude > threshold
    phases = []

    # 第1バーンの検出
    first_burn_start = None
    first_burn_end = None
    for i in range(len(burning) - 1):
        if not burning[i] and burning[i + 1] and first_burn_start is None:
            first_burn_start = i + 1
        if burning[i] and not burning[i + 1] and first_burn_start is not None and first_burn_end is None:
            first_burn_end = i
            break

    # 第2バーンの検出
    second_burn_start = None
    second_burn_end = None
    if first_burn_end is not None:
        for i in range(first_burn_end + 1, len(burning) - 1):
            if not burning[i] and burning[i + 1] and second_burn_start is None:
                second_burn_start = i + 1
            if burning[i] and not burning[i + 1] and second_burn_start is not None:
                second_burn_end = i
                break

    # フェーズの定義
    if first_burn_start is not None:
        # Pre-transfer
        phases.append((0, first_burn_start, "Pre-transfer", "#4A90E2"))  # 青

        # First burn
        if first_burn_end is not None:
            phases.append((first_burn_start, first_burn_end, "First Burn", "#E74C3C"))  # 赤

            # Coast phase
            if second_burn_start is not None:
                phases.append((first_burn_end, second_burn_start, "Coast Phase", "#2ECC71"))  # 緑

                # Second burn
                if second_burn_end is not None:
                    phases.append((second_burn_start, second_burn_end, "Second Burn", "#F39C12"))  # オレンジ

                    # Post-transfer
                    phases.append((second_burn_end, len(time), "Post-transfer", "#9B59B6"))  # 紫
                else:
                    # Still in second burn
                    phases.append((second_burn_start, len(time), "Second Burn (ongoing)", "#F39C12"))
            else:
                # Still in coast
                phases.append((first_burn_end, len(time), "Coast Phase (ongoing)", "#2ECC71"))
        else:
            # Still in first burn
            phases.append((first_burn_start, len(time), "First Burn (ongoing)", "#E74C3C"))
    else:
        # No burn detected
        phases.append((0, len(time), "Free Orbit", "#95A5A6"))  # グレー

    return phases


def plot_3d_trajectory_with_phases(h5_file, output_dir=None, dpi=150):
    """
    3D軌道を各フェーズで色分けして表示

    Args:
        h5_file: HDF5ファイルパス
        output_dir: 出力ディレクトリ（Noneなら入力ファイルと同じ）
        dpi: 解像度
    """
    with h5py.File(h5_file, "r") as f:
        time = f["time"]["time_s"][:]

        # 軌道データ
        env = f["OrbitalEnvSim-0_OrbitalSpacecraft_0"]
        position_x = env["position_x"][:]
        position_y = env["position_y"][:]
        position_z = env["position_z"][:]

        # 推力データ
        ctrl = f["OrbitalControllerSim-0_OrbitalController_0"]
        thrust_x = ctrl["thrust_command_x"][:]
        thrust_y = ctrl["thrust_command_y"][:]
        thrust_z = ctrl["thrust_command_z"][:]
        thrust_mag = np.sqrt(thrust_x**2 + thrust_y**2 + thrust_z**2)

    # フェーズの検出
    phases = detect_phases(thrust_mag, time)

    # プロット
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection="3d")

    # 各フェーズを色分けしてプロット（地球より先に描画して手前に表示）
    for start_idx, end_idx, phase_name, color in phases:
        ax.plot(
            position_x[start_idx:end_idx] / 1e3,
            position_y[start_idx:end_idx] / 1e3,
            position_z[start_idx:end_idx] / 1e3,
            color=color,
            linewidth=3,
            label=phase_name,
            alpha=0.9,
            zorder=10,  # 手前に描画
        )

        # 開始点にマーカー
        if start_idx < len(position_x):
            ax.scatter(
                position_x[start_idx] / 1e3,
                position_y[start_idx] / 1e3,
                position_z[start_idx] / 1e3,
                color=color,
                s=150,
                marker="o",
                edgecolors="black",
                linewidth=2,
                zorder=15,
                alpha=1.0,
            )

    # 地球を描画（軌道より後ろに配置）
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    radius_earth = 6378137.0
    x_earth = radius_earth * np.outer(np.cos(u), np.sin(v))
    y_earth = radius_earth * np.outer(np.sin(u), np.sin(v))
    z_earth = radius_earth * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x_earth / 1e3, y_earth / 1e3, z_earth / 1e3, color="lightblue", alpha=0.2, zorder=1)

    ax.set_xlabel("X [km]", fontsize=12)
    ax.set_ylabel("Y [km]", fontsize=12)
    ax.set_zlabel("Z [km]", fontsize=12)
    ax.set_title("Hohmann Transfer Trajectory (Phase-Colored)", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)

    # 軸を等スケールに
    max_range = (
        np.array(
            [
                position_x.max() - position_x.min(),
                position_y.max() - position_y.min(),
                position_z.max() - position_z.min(),
            ]
        ).max()
        / 2e3
    )

    mid_x = (position_x.max() + position_x.min()) / 2e3
    mid_y = (position_y.max() + position_y.min()) / 2e3
    mid_z = (position_z.max() + position_z.min()) / 2e3

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.tight_layout()

    # 保存
    if output_dir is None:
        output_dir = Path(h5_file).parent
    output_file = Path(output_dir) / "orbital_3d_trajectory_phases.png"
    plt.savefig(output_file, dpi=dpi, bbox_inches="tight")
    print(f"✅ Saved: {output_file}")
    plt.close()


def plot_altitude_with_phases(h5_file, output_dir=None, dpi=150):
    """
    高度の時系列を各フェーズで色分けして表示

    Args:
        h5_file: HDF5ファイルパス
        output_dir: 出力ディレクトリ
        dpi: 解像度
    """
    with h5py.File(h5_file, "r") as f:
        time = f["time"]["time_s"][:]

        # 軌道データ
        env = f["OrbitalEnvSim-0_OrbitalSpacecraft_0"]
        altitude = env["altitude"][:]

        # 推力データ
        ctrl = f["OrbitalControllerSim-0_OrbitalController_0"]
        thrust_x = ctrl["thrust_command_x"][:]
        thrust_y = ctrl["thrust_command_y"][:]
        thrust_z = ctrl["thrust_command_z"][:]
        thrust_mag = np.sqrt(thrust_x**2 + thrust_y**2 + thrust_z**2)

    # フェーズの検出
    phases = detect_phases(thrust_mag, time)

    # プロット
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # 高度プロット
    for start_idx, end_idx, phase_name, color in phases:
        ax1.plot(
            time[start_idx:end_idx] / 60,
            altitude[start_idx:end_idx] / 1e3,
            color=color,
            linewidth=2.5,
            label=phase_name,
            alpha=0.8,
        )

    ax1.set_ylabel("Altitude [km]", fontsize=12)
    ax1.set_title("Hohmann Transfer - Altitude Profile", fontsize=14, fontweight="bold")
    ax1.legend(loc="best", fontsize=10)
    ax1.grid(True, alpha=0.3)

    # 推力プロット
    for start_idx, end_idx, phase_name, color in phases:
        ax2.plot(
            time[start_idx:end_idx] / 60,
            thrust_mag[start_idx:end_idx],
            color=color,
            linewidth=2.5,
            label=phase_name,
            alpha=0.8,
        )

    ax2.set_xlabel("Time [min]", fontsize=12)
    ax2.set_ylabel("Thrust [N]", fontsize=12)
    ax2.set_title("Thrust Profile", fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存
    if output_dir is None:
        output_dir = Path(h5_file).parent
    output_file = Path(output_dir) / "altitude_thrust_phases.png"
    plt.savefig(output_file, dpi=dpi, bbox_inches="tight")
    print(f"✅ Saved: {output_file}")
    plt.close()


def plot_orbital_elements_with_phases(h5_file, output_dir=None, dpi=150):
    """
    軌道要素を各フェーズで色分けして表示

    Args:
        h5_file: HDF5ファイルパス
        output_dir: 出力ディレクトリ
        dpi: 解像度
    """
    with h5py.File(h5_file, "r") as f:
        time = f["time"]["time_s"][:]

        # 軌道データ
        env = f["OrbitalEnvSim-0_OrbitalSpacecraft_0"]
        semi_major_axis = env["semi_major_axis"][:]
        eccentricity = env["eccentricity"][:]

        # 推力データ
        ctrl = f["OrbitalControllerSim-0_OrbitalController_0"]
        thrust_x = ctrl["thrust_command_x"][:]
        thrust_y = ctrl["thrust_command_y"][:]
        thrust_z = ctrl["thrust_command_z"][:]
        thrust_mag = np.sqrt(thrust_x**2 + thrust_y**2 + thrust_z**2)

    # フェーズの検出
    phases = detect_phases(thrust_mag, time)

    # プロット
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # 軌道長半径
    for start_idx, end_idx, phase_name, color in phases:
        ax1.plot(
            time[start_idx:end_idx] / 60,
            semi_major_axis[start_idx:end_idx] / 1e3,
            color=color,
            linewidth=2.5,
            label=phase_name,
            alpha=0.8,
        )

    ax1.set_ylabel("Semi-major Axis [km]", fontsize=12)
    ax1.set_title("Orbital Elements Evolution", fontsize=14, fontweight="bold")
    ax1.legend(loc="best", fontsize=10)
    ax1.grid(True, alpha=0.3)

    # 離心率
    for start_idx, end_idx, phase_name, color in phases:
        ax2.plot(
            time[start_idx:end_idx] / 60,
            eccentricity[start_idx:end_idx],
            color=color,
            linewidth=2.5,
            label=phase_name,
            alpha=0.8,
        )

    ax2.set_xlabel("Time [min]", fontsize=12)
    ax2.set_ylabel("Eccentricity [-]", fontsize=12)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存
    if output_dir is None:
        output_dir = Path(h5_file).parent
    output_file = Path(output_dir) / "orbital_elements_phases.png"
    plt.savefig(output_file, dpi=dpi, bbox_inches="tight")
    print(f"✅ Saved: {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize Hohmann transfer with phase coloring")
    parser.add_argument("h5_file", help="HDF5 file path")
    parser.add_argument("--output-dir", help="Output directory (default: same as input file)")
    parser.add_argument("--dpi", type=int, default=150, help="DPI for output images (default: 150)")

    args = parser.parse_args()

    h5_file = Path(args.h5_file)
    if not h5_file.exists():
        print(f"❌ Error: File not found: {h5_file}")
        sys.exit(1)

    output_dir = args.output_dir if args.output_dir else h5_file.parent

    print("=" * 70)
    print("Hohmann Transfer Phase Visualization")
    print("=" * 70)
    print(f"Input: {h5_file}")
    print(f"Output: {output_dir}")
    print()

    print("📊 Generating visualizations...")

    # 各プロットを生成
    plot_3d_trajectory_with_phases(h5_file, output_dir, args.dpi)
    plot_altitude_with_phases(h5_file, output_dir, args.dpi)
    plot_orbital_elements_with_phases(h5_file, output_dir, args.dpi)

    print()
    print("✅ All visualizations completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
