"""
Formation Flying 可視化スクリプト

2機の衛星（Chaser と Target）の軌道と相対位置を可視化。

使用法:
    python visualize_formation_flying.py <hdf5_file>
"""

import argparse
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def load_formation_data(h5_file):
    """
    HDF5ファイルから編隊飛行データを読み込み

    Args:
        h5_file: HDF5ファイルパス

    Returns:
        data: データ辞書
    """
    with h5py.File(h5_file, "r") as f:
        # 時間データ
        time_s = f["time"]["time_s"][:]

        # Chaser のデータ（制御あり）
        # EnvSimのインスタンスを特定（通常は0がChaser、1がTarget）
        chaser_group = None
        target_group = None

        for key in f.keys():
            if key.startswith("OrbitalEnvSim"):
                if chaser_group is None:
                    chaser_group = f[key]
                elif target_group is None:
                    target_group = f[key]

        if chaser_group is None or target_group is None:
            raise ValueError("Could not find both Chaser and Target spacecraft data")

        # Chaser データ
        chaser_pos_x = chaser_group["position_x"][:]
        chaser_pos_y = chaser_group["position_y"][:]
        chaser_pos_z = chaser_group["position_z"][:]
        chaser_vel_x = chaser_group["velocity_x"][:]
        chaser_vel_y = chaser_group["velocity_y"][:]
        chaser_vel_z = chaser_group["velocity_z"][:]
        chaser_altitude = chaser_group["altitude"][:]

        # Target データ
        target_pos_x = target_group["position_x"][:]
        target_pos_y = target_group["position_y"][:]
        target_pos_z = target_group["position_z"][:]
        target_vel_x = target_group["velocity_x"][:]
        target_vel_y = target_group["velocity_y"][:]
        target_vel_z = target_group["velocity_z"][:]
        target_altitude = target_group["altitude"][:]

        # Controller データ（Chaserのみ）
        controller_group = None
        for key in f.keys():
            if key.startswith("OrbitalControllerSim"):
                controller_group = f[key]
                break

        if controller_group is not None:
            thrust_cmd_x = controller_group["thrust_command_x"][:]
            thrust_cmd_y = controller_group["thrust_command_y"][:]
            thrust_cmd_z = controller_group["thrust_command_z"][:]
        else:
            # No controller data
            thrust_cmd_x = np.zeros_like(time_s)
            thrust_cmd_y = np.zeros_like(time_s)
            thrust_cmd_z = np.zeros_like(time_s)

        # Plant データ（Chaserのみ）
        plant_group = None
        for key in f.keys():
            if key.startswith("OrbitalPlantSim"):
                plant_group = f[key]
                break

        if plant_group is not None:
            measured_force_x = plant_group["measured_force_x"][:]
            measured_force_y = plant_group["measured_force_y"][:]
            measured_force_z = plant_group["measured_force_z"][:]
        else:
            measured_force_x = np.zeros_like(time_s)
            measured_force_y = np.zeros_like(time_s)
            measured_force_z = np.zeros_like(time_s)

    # 相対位置・速度の計算
    rel_pos_x = chaser_pos_x - target_pos_x
    rel_pos_y = chaser_pos_y - target_pos_y
    rel_pos_z = chaser_pos_z - target_pos_z
    rel_distance = np.sqrt(rel_pos_x**2 + rel_pos_y**2 + rel_pos_z**2)

    rel_vel_x = chaser_vel_x - target_vel_x
    rel_vel_y = chaser_vel_y - target_vel_y
    rel_vel_z = chaser_vel_z - target_vel_z
    rel_velocity = np.sqrt(rel_vel_x**2 + rel_vel_y**2 + rel_vel_z**2)

    # 推力ノルム
    thrust_norm = np.sqrt(thrust_cmd_x**2 + thrust_cmd_y**2 + thrust_cmd_z**2)
    force_norm = np.sqrt(measured_force_x**2 + measured_force_y**2 + measured_force_z**2)

    return {
        "time_s": time_s,
        # Chaser
        "chaser_pos_x": chaser_pos_x,
        "chaser_pos_y": chaser_pos_y,
        "chaser_pos_z": chaser_pos_z,
        "chaser_vel_x": chaser_vel_x,
        "chaser_vel_y": chaser_vel_y,
        "chaser_vel_z": chaser_vel_z,
        "chaser_altitude": chaser_altitude,
        # Target
        "target_pos_x": target_pos_x,
        "target_pos_y": target_pos_y,
        "target_pos_z": target_pos_z,
        "target_vel_x": target_vel_x,
        "target_vel_y": target_vel_y,
        "target_vel_z": target_vel_z,
        "target_altitude": target_altitude,
        # 相対
        "rel_pos_x": rel_pos_x,
        "rel_pos_y": rel_pos_y,
        "rel_pos_z": rel_pos_z,
        "rel_distance": rel_distance,
        "rel_vel_x": rel_vel_x,
        "rel_vel_y": rel_vel_y,
        "rel_vel_z": rel_vel_z,
        "rel_velocity": rel_velocity,
        # 制御
        "thrust_cmd_x": thrust_cmd_x,
        "thrust_cmd_y": thrust_cmd_y,
        "thrust_cmd_z": thrust_cmd_z,
        "thrust_norm": thrust_norm,
        "measured_force_x": measured_force_x,
        "measured_force_y": measured_force_y,
        "measured_force_z": measured_force_z,
        "force_norm": force_norm,
    }


def plot_formation_flying(data, output_dir):
    """
    Formation Flying の可視化

    Args:
        data: データ辞書
        output_dir: 出力ディレクトリ
    """
    time_s = data["time_s"]
    time_min = time_s / 60.0

    # ========================================
    # Figure 1: 3D軌道図（Chaser と Target）
    # ========================================
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    # 地球の描画（球）
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    R_earth = 6378137.0  # 地球半径 [m]
    x_earth = R_earth * np.outer(np.cos(u), np.sin(v))
    y_earth = R_earth * np.outer(np.sin(u), np.sin(v))
    z_earth = R_earth * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x_earth, y_earth, z_earth, color="blue", alpha=0.3, label="Earth")

    # Chaser 軌道
    ax.plot(
        data["chaser_pos_x"],
        data["chaser_pos_y"],
        data["chaser_pos_z"],
        "r-",
        linewidth=2,
        label="Chaser (controlled)",
    )

    # Target 軌道
    ax.plot(
        data["target_pos_x"],
        data["target_pos_y"],
        data["target_pos_z"],
        "b--",
        linewidth=2,
        label="Target (free)",
    )

    # 初期位置マーカー
    ax.scatter(
        data["chaser_pos_x"][0],
        data["chaser_pos_y"][0],
        data["chaser_pos_z"][0],
        color="red",
        s=100,
        marker="o",
        label="Chaser start",
    )
    ax.scatter(
        data["target_pos_x"][0],
        data["target_pos_y"][0],
        data["target_pos_z"][0],
        color="blue",
        s=100,
        marker="s",
        label="Target start",
    )

    # 最終位置マーカー
    ax.scatter(
        data["chaser_pos_x"][-1],
        data["chaser_pos_y"][-1],
        data["chaser_pos_z"][-1],
        color="darkred",
        s=150,
        marker="^",
        label="Chaser end",
    )
    ax.scatter(
        data["target_pos_x"][-1],
        data["target_pos_y"][-1],
        data["target_pos_z"][-1],
        color="darkblue",
        s=150,
        marker="v",
        label="Target end",
    )

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title("Formation Flying: 3D Orbits", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # 軸の範囲を調整
    max_range = max(
        np.max(np.abs(data["chaser_pos_x"])),
        np.max(np.abs(data["chaser_pos_y"])),
        np.max(np.abs(data["chaser_pos_z"])),
    )
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range, max_range)

    plt.tight_layout()
    fig.savefig(output_dir / "formation_3d_orbits.png", dpi=300, bbox_inches="tight")
    print("  ✅ Saved: formation_3d_orbits.png")
    plt.close(fig)

    # ========================================
    # Figure 2: 相対位置（時系列）
    # ========================================
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    # Relative X
    axes[0].plot(time_min, data["rel_pos_x"], "r-", linewidth=1.5)
    axes[0].set_ylabel("Relative X [m]", fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title("Formation Flying: Relative Position", fontsize=14, fontweight="bold")

    # Relative Y
    axes[1].plot(time_min, data["rel_pos_y"], "g-", linewidth=1.5)
    axes[1].set_ylabel("Relative Y [m]", fontsize=12)
    axes[1].grid(True, alpha=0.3)

    # Relative Z
    axes[2].plot(time_min, data["rel_pos_z"], "b-", linewidth=1.5)
    axes[2].set_ylabel("Relative Z [m]", fontsize=12)
    axes[2].grid(True, alpha=0.3)

    # Relative Distance (norm)
    axes[3].plot(time_min, data["rel_distance"], "k-", linewidth=2)
    axes[3].set_ylabel("Distance [m]", fontsize=12)
    axes[3].set_xlabel("Time [min]", fontsize=12)
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / "formation_relative_position.png", dpi=300, bbox_inches="tight")
    print("  ✅ Saved: formation_relative_position.png")
    plt.close(fig)

    # ========================================
    # Figure 3: 相対速度（時系列）
    # ========================================
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    # Relative Vx
    axes[0].plot(time_min, data["rel_vel_x"], "r-", linewidth=1.5)
    axes[0].set_ylabel("Relative Vx [m/s]", fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title("Formation Flying: Relative Velocity", fontsize=14, fontweight="bold")

    # Relative Vy
    axes[1].plot(time_min, data["rel_vel_y"], "g-", linewidth=1.5)
    axes[1].set_ylabel("Relative Vy [m/s]", fontsize=12)
    axes[1].grid(True, alpha=0.3)

    # Relative Vz
    axes[2].plot(time_min, data["rel_vel_z"], "b-", linewidth=1.5)
    axes[2].set_ylabel("Relative Vz [m/s]", fontsize=12)
    axes[2].grid(True, alpha=0.3)

    # Relative Velocity (norm)
    axes[3].plot(time_min, data["rel_velocity"], "k-", linewidth=2)
    axes[3].set_ylabel("Velocity [m/s]", fontsize=12)
    axes[3].set_xlabel("Time [min]", fontsize=12)
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / "formation_relative_velocity.png", dpi=300, bbox_inches="tight")
    print("  ✅ Saved: formation_relative_velocity.png")
    plt.close(fig)

    # ========================================
    # Figure 4: 制御入力（推力）
    # ========================================
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    # Thrust Command X
    axes[0].plot(time_min, data["thrust_cmd_x"], "r-", linewidth=1.5, label="Command")
    axes[0].plot(time_min, data["measured_force_x"], "b--", linewidth=1.5, alpha=0.7, label="Measured")
    axes[0].set_ylabel("Thrust X [N]", fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right")
    axes[0].set_title("Chaser Thrust Command & Measured Force", fontsize=14, fontweight="bold")

    # Thrust Command Y
    axes[1].plot(time_min, data["thrust_cmd_y"], "g-", linewidth=1.5, label="Command")
    axes[1].plot(time_min, data["measured_force_y"], "b--", linewidth=1.5, alpha=0.7, label="Measured")
    axes[1].set_ylabel("Thrust Y [N]", fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper right")

    # Thrust Command Z
    axes[2].plot(time_min, data["thrust_cmd_z"], "b-", linewidth=1.5, label="Command")
    axes[2].plot(time_min, data["measured_force_z"], "b--", linewidth=1.5, alpha=0.7, label="Measured")
    axes[2].set_ylabel("Thrust Z [N]", fontsize=12)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="upper right")

    # Thrust Norm
    axes[3].plot(time_min, data["thrust_norm"], "k-", linewidth=2, label="Command Norm")
    axes[3].plot(time_min, data["force_norm"], "r--", linewidth=2, alpha=0.7, label="Measured Norm")
    axes[3].set_ylabel("Thrust Norm [N]", fontsize=12)
    axes[3].set_xlabel("Time [min]", fontsize=12)
    axes[3].grid(True, alpha=0.3)
    axes[3].legend(loc="upper right")

    plt.tight_layout()
    fig.savefig(output_dir / "formation_thrust.png", dpi=300, bbox_inches="tight")
    print("  ✅ Saved: formation_thrust.png")
    plt.close(fig)

    # ========================================
    # Figure 5: 高度比較
    # ========================================
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))

    ax.plot(time_min, data["chaser_altitude"] / 1e3, "r-", linewidth=2, label="Chaser")
    ax.plot(time_min, data["target_altitude"] / 1e3, "b--", linewidth=2, label="Target")
    ax.set_ylabel("Altitude [km]", fontsize=12)
    ax.set_xlabel("Time [min]", fontsize=12)
    ax.set_title("Altitude Comparison", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=12)

    plt.tight_layout()
    fig.savefig(output_dir / "formation_altitude.png", dpi=300, bbox_inches="tight")
    print("  ✅ Saved: formation_altitude.png")
    plt.close(fig)

    # ========================================
    # Figure 6: 相対位置（3Dプロット - 局所座標系）
    # ========================================
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    # 相対位置の3Dプロット
    ax.plot(
        data["rel_pos_x"],
        data["rel_pos_y"],
        data["rel_pos_z"],
        "purple",
        linewidth=2,
        label="Relative trajectory",
    )

    # 初期位置
    ax.scatter(
        data["rel_pos_x"][0],
        data["rel_pos_y"][0],
        data["rel_pos_z"][0],
        color="green",
        s=200,
        marker="o",
        label="Start",
    )

    # 最終位置
    ax.scatter(
        data["rel_pos_x"][-1],
        data["rel_pos_y"][-1],
        data["rel_pos_z"][-1],
        color="red",
        s=200,
        marker="^",
        label="End",
    )

    # 目標位置（原点）
    ax.scatter(0, 0, 0, color="blue", s=300, marker="*", label="Target (origin)", zorder=10)

    ax.set_xlabel("Relative X [m]")
    ax.set_ylabel("Relative Y [m]")
    ax.set_zlabel("Relative Z [m]")
    ax.set_title("Relative Position (Chaser w.r.t. Target)", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / "formation_relative_3d.png", dpi=300, bbox_inches="tight")
    print("  ✅ Saved: formation_relative_3d.png")
    plt.close(fig)

    # ========================================
    # Figure 7: 相対位置（インタラクティブ3D - Plotly）
    # ========================================
    # カラーマップ：時間経過に応じて色を変える
    time_normalized = np.linspace(0, 1, len(data["rel_pos_x"]))

    # メイントラジェクトリ
    trace_trajectory = go.Scatter3d(
        x=data["rel_pos_x"],
        y=data["rel_pos_y"],
        z=data["rel_pos_z"],
        mode="lines",
        line=dict(
            color=time_normalized,
            colorscale="Viridis",
            width=4,
            colorbar=dict(title="Time (normalized)", x=1.1),
        ),
        name="Relative trajectory",
        hovertemplate="X: %{x:.2f} m<br>Y: %{y:.2f} m<br>Z: %{z:.2f} m<extra></extra>",
    )

    # 初期位置マーカー
    trace_start = go.Scatter3d(
        x=[data["rel_pos_x"][0]],
        y=[data["rel_pos_y"][0]],
        z=[data["rel_pos_z"][0]],
        mode="markers",
        marker=dict(size=10, color="green", symbol="circle"),
        name="Start",
        hovertemplate="Start<br>X: %{x:.2f} m<br>Y: %{y:.2f} m<br>Z: %{z:.2f} m<extra></extra>",
    )

    # 最終位置マーカー
    trace_end = go.Scatter3d(
        x=[data["rel_pos_x"][-1]],
        y=[data["rel_pos_y"][-1]],
        z=[data["rel_pos_z"][-1]],
        mode="markers",
        marker=dict(size=12, color="red", symbol="diamond"),
        name="End",
        hovertemplate="End<br>X: %{x:.2f} m<br>Y: %{y:.2f} m<br>Z: %{z:.2f} m<extra></extra>",
    )

    # 目標位置（原点）
    trace_target = go.Scatter3d(
        x=[0],
        y=[0],
        z=[0],
        mode="markers",
        marker=dict(size=15, color="blue", symbol="cross"),
        name="Target (origin)",
        hovertemplate="Target<br>X: 0 m<br>Y: 0 m<br>Z: 0 m<extra></extra>",
    )

    # レイアウト設定
    layout = go.Layout(
        title=dict(
            text="Formation Flying: Relative Position (Interactive 3D)",
            font=dict(size=18, color="black"),
            x=0.5,
            xanchor="center",
        ),
        scene=dict(
            xaxis=dict(title="Relative X [m]", backgroundcolor="white", gridcolor="lightgray"),
            yaxis=dict(title="Relative Y [m]", backgroundcolor="white", gridcolor="lightgray"),
            zaxis=dict(title="Relative Z [m]", backgroundcolor="white", gridcolor="lightgray"),
            aspectmode="cube",
        ),
        showlegend=True,
        legend=dict(x=0.7, y=0.95, bgcolor="rgba(255,255,255,0.8)"),
        hovermode="closest",
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    # 図の作成
    fig_plotly = go.Figure(data=[trace_trajectory, trace_start, trace_end, trace_target], layout=layout)

    # HTML保存
    html_path = output_dir / "formation_relative_3d_interactive.html"
    fig_plotly.write_html(html_path)
    print("  ✅ Saved: formation_relative_3d_interactive.html")


def main():
    parser = argparse.ArgumentParser(description="Formation Flying データ可視化")
    parser.add_argument("h5_file", type=str, help="HDF5ファイルパス")
    args = parser.parse_args()

    h5_file = Path(args.h5_file)
    output_dir = h5_file.parent

    print(f"\n{'=' * 70}")
    print("Formation Flying Visualization")
    print(f"{'=' * 70}")
    print(f"Input: {h5_file}")
    print(f"Output: {output_dir}")
    print(f"{'=' * 70}\n")

    # データ読み込み
    print("📊 Loading formation flying data...")
    data = load_formation_data(h5_file)
    print(f"  ✅ Data loaded ({len(data['time_s'])} time steps)")

    # 可視化
    print("\n📈 Generating plots...")
    plot_formation_flying(data, output_dir)

    print(f"\n{'=' * 70}")
    print("✅ Visualization completed")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
