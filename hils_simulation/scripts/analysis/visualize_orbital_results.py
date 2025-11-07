"""
軌道シミュレーション結果の可視化

HDF5ファイルから軌道データを読み込み、以下をプロット:
- 3D軌道
- 位置・速度の時間変化
- 軌道要素の時間変化
"""

import argparse
import h5py
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_orbital_data(h5_path: str):
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
        data["position_norm"] = sc_group["position_norm"][:]
        data["velocity_x"] = sc_group["velocity_x"][:]
        data["velocity_y"] = sc_group["velocity_y"][:]
        data["velocity_z"] = sc_group["velocity_z"][:]
        data["velocity_norm"] = sc_group["velocity_norm"][:]
        data["acceleration_x"] = sc_group["acceleration_x"][:]
        data["acceleration_y"] = sc_group["acceleration_y"][:]
        data["acceleration_z"] = sc_group["acceleration_z"][:]
        data["altitude"] = sc_group["altitude"][:]
        data["semi_major_axis"] = sc_group["semi_major_axis"][:]
        data["eccentricity"] = sc_group["eccentricity"][:]
        data["specific_energy"] = sc_group["specific_energy"][:]

    return data


def plot_orbital_simulation(h5_path: str, output_dir: str = None):
    """
    軌道シミュレーション結果をプロット

    Args:
        h5_path: HDF5ファイルのパス
        output_dir: 出力ディレクトリ（Noneの場合はHDF5ファイルと同じ場所）
    """
    # データ読み込み
    data = load_orbital_data(h5_path)

    # 出力ディレクトリ設定
    if output_dir is None:
        output_dir = Path(h5_path).parent
    else:
        output_dir = Path(output_dir)

    # 単位変換
    time_min = data["time"] / 60.0  # 秒 → 分
    pos_km = {
        "x": data["position_x"] / 1e3,
        "y": data["position_y"] / 1e3,
        "z": data["position_z"] / 1e3,
        "norm": data["position_norm"] / 1e3,
    }
    vel_km_s = {
        "x": data["velocity_x"] / 1e3,
        "y": data["velocity_y"] / 1e3,
        "z": data["velocity_z"] / 1e3,
        "norm": data["velocity_norm"] / 1e3,
    }
    altitude_km = data["altitude"] / 1e3
    sma_km = data["semi_major_axis"] / 1e3

    # プロット1: 3D軌道
    fig1 = plt.figure(figsize=(10, 10))
    ax1 = fig1.add_subplot(111, projection="3d")

    # 地球（球体）
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    earth_radius = 6378.137  # km
    x_earth = earth_radius * np.outer(np.cos(u), np.sin(v))
    y_earth = earth_radius * np.outer(np.sin(u), np.sin(v))
    z_earth = earth_radius * np.outer(np.ones(np.size(u)), np.cos(v))
    ax1.plot_surface(x_earth, y_earth, z_earth, color="blue", alpha=0.3)

    # 軌道
    ax1.plot(pos_km["x"], pos_km["y"], pos_km["z"], "r-", linewidth=2, label="Orbit")
    ax1.plot([pos_km["x"][0]], [pos_km["y"][0]], [pos_km["z"][0]], "go", markersize=10, label="Start")
    ax1.plot([pos_km["x"][-1]], [pos_km["y"][-1]], [pos_km["z"][-1]], "ro", markersize=10, label="End")

    ax1.set_xlabel("X [km]")
    ax1.set_ylabel("Y [km]")
    ax1.set_zlabel("Z [km]")
    ax1.set_title("3D Orbital Trajectory")
    ax1.legend()
    ax1.set_box_aspect([1, 1, 1])

    # 軸範囲を等しく設定
    max_range = np.max(pos_km["norm"]) * 1.1
    ax1.set_xlim([-max_range, max_range])
    ax1.set_ylim([-max_range, max_range])
    ax1.set_zlim([-max_range, max_range])

    fig1.tight_layout()
    fig1.savefig(output_dir / "orbital_3d_trajectory.png", dpi=300)
    print(f"   Saved: orbital_3d_trajectory.png")
    plt.close(fig1)

    # プロット2: 位置の時間変化
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))

    # 位置成分
    axes2[0, 0].plot(time_min, pos_km["x"], label="X")
    axes2[0, 0].plot(time_min, pos_km["y"], label="Y")
    axes2[0, 0].plot(time_min, pos_km["z"], label="Z")
    axes2[0, 0].set_xlabel("Time [min]")
    axes2[0, 0].set_ylabel("Position [km]")
    axes2[0, 0].set_title("Position Components")
    axes2[0, 0].legend()
    axes2[0, 0].grid(True)

    # 動径距離
    axes2[0, 1].plot(time_min, pos_km["norm"], "b-")
    axes2[0, 1].set_xlabel("Time [min]")
    axes2[0, 1].set_ylabel("Radial Distance [km]")
    axes2[0, 1].set_title("Orbital Radius |r|")
    axes2[0, 1].grid(True)

    # 高度
    axes2[1, 0].plot(time_min, altitude_km, "g-")
    axes2[1, 0].set_xlabel("Time [min]")
    axes2[1, 0].set_ylabel("Altitude [km]")
    axes2[1, 0].set_title("Altitude (from Earth surface)")
    axes2[1, 0].grid(True)

    # XY平面
    axes2[1, 1].plot(pos_km["x"], pos_km["y"], "r-")
    axes2[1, 1].plot(0, 0, "bo", markersize=10, label="Earth")
    axes2[1, 1].set_xlabel("X [km]")
    axes2[1, 1].set_ylabel("Y [km]")
    axes2[1, 1].set_title("Orbit (XY plane)")
    axes2[1, 1].legend()
    axes2[1, 1].grid(True)
    axes2[1, 1].axis("equal")

    fig2.tight_layout()
    fig2.savefig(output_dir / "orbital_position.png", dpi=300)
    print(f"   Saved: orbital_position.png")
    plt.close(fig2)

    # プロット3: 速度の時間変化
    fig3, axes3 = plt.subplots(2, 2, figsize=(14, 10))

    # 速度成分
    axes3[0, 0].plot(time_min, vel_km_s["x"], label="Vx")
    axes3[0, 0].plot(time_min, vel_km_s["y"], label="Vy")
    axes3[0, 0].plot(time_min, vel_km_s["z"], label="Vz")
    axes3[0, 0].set_xlabel("Time [min]")
    axes3[0, 0].set_ylabel("Velocity [km/s]")
    axes3[0, 0].set_title("Velocity Components")
    axes3[0, 0].legend()
    axes3[0, 0].grid(True)

    # 速度ノルム
    axes3[0, 1].plot(time_min, vel_km_s["norm"], "b-")
    axes3[0, 1].set_xlabel("Time [min]")
    axes3[0, 1].set_ylabel("Speed [km/s]")
    axes3[0, 1].set_title("Orbital Speed |v|")
    axes3[0, 1].grid(True)

    # 比エネルギー
    axes3[1, 0].plot(time_min, data["specific_energy"] / 1e6, "m-")
    axes3[1, 0].set_xlabel("Time [min]")
    axes3[1, 0].set_ylabel("Specific Energy [MJ/kg]")
    axes3[1, 0].set_title("Specific Orbital Energy")
    axes3[1, 0].grid(True)

    # 速度ベクトル（XY平面）
    skip = max(1, len(time_min) // 20)  # 矢印を間引く
    axes3[1, 1].plot(pos_km["x"], pos_km["y"], "r-", alpha=0.3)
    axes3[1, 1].quiver(
        pos_km["x"][::skip],
        pos_km["y"][::skip],
        vel_km_s["x"][::skip],
        vel_km_s["y"][::skip],
        scale=50,
        width=0.003,
    )
    axes3[1, 1].plot(0, 0, "bo", markersize=10, label="Earth")
    axes3[1, 1].set_xlabel("X [km]")
    axes3[1, 1].set_ylabel("Y [km]")
    axes3[1, 1].set_title("Velocity Vectors (XY plane)")
    axes3[1, 1].legend()
    axes3[1, 1].grid(True)
    axes3[1, 1].axis("equal")

    fig3.tight_layout()
    fig3.savefig(output_dir / "orbital_velocity.png", dpi=300)
    print(f"   Saved: orbital_velocity.png")
    plt.close(fig3)

    # プロット4: 軌道要素
    fig4, axes4 = plt.subplots(2, 2, figsize=(14, 10))

    # 軌道長半径
    axes4[0, 0].plot(time_min, sma_km, "b-")
    axes4[0, 0].set_xlabel("Time [min]")
    axes4[0, 0].set_ylabel("Semi-major Axis [km]")
    axes4[0, 0].set_title("Orbital Semi-major Axis")
    axes4[0, 0].grid(True)

    # 離心率
    axes4[0, 1].plot(time_min, data["eccentricity"], "g-")
    axes4[0, 1].set_xlabel("Time [min]")
    axes4[0, 1].set_ylabel("Eccentricity [-]")
    axes4[0, 1].set_title("Orbital Eccentricity")
    axes4[0, 1].grid(True)

    # 高度統計
    axes4[1, 0].hist(altitude_km, bins=50, edgecolor="black")
    axes4[1, 0].set_xlabel("Altitude [km]")
    axes4[1, 0].set_ylabel("Frequency")
    axes4[1, 0].set_title("Altitude Distribution")
    axes4[1, 0].grid(True)

    # 軌道要素の偏差（理想値からのずれ）
    sma_mean = np.mean(sma_km)
    ecc_mean = np.mean(data["eccentricity"])
    axes4[1, 1].plot(time_min, (sma_km - sma_mean) * 1e3, label="SMA error [m]")
    axes4[1, 1].plot(time_min, (data["eccentricity"] - ecc_mean) * 1e6, label="Ecc error [×10⁻⁶]")
    axes4[1, 1].set_xlabel("Time [min]")
    axes4[1, 1].set_ylabel("Error")
    axes4[1, 1].set_title("Orbital Element Errors (numerical drift)")
    axes4[1, 1].legend()
    axes4[1, 1].grid(True)

    fig4.tight_layout()
    fig4.savefig(output_dir / "orbital_elements.png", dpi=300)
    print(f"   Saved: orbital_elements.png")
    plt.close(fig4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize orbital simulation results")
    parser.add_argument("h5_file", type=str, help="Path to HDF5 data file")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for plots")

    args = parser.parse_args()

    plot_orbital_simulation(args.h5_file, args.output_dir)
    print("✅ Visualization completed!")
