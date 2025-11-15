"""
ホーマン遷移の各フェーズを色分けしてインタラクティブ可視化 (Plotly版)

各フェーズを異なる色で表示：
- 遷移前（青）
- 第1バーン（赤）
- コーストフェーズ（緑）
- 第2バーン（オレンジ）
- 遷移後（紫）

使用方法:
    uv run python scripts/analysis/visualize_hohmann_phases_interactive.py <HDF5_FILE>
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


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


def create_earth_sphere(radius_km=6378.137, resolution=30):
    """
    地球の球体メッシュを作成

    Args:
        radius_km: 地球半径 [km]
        resolution: 球体の解像度

    Returns:
        dict: x, y, z座標
    """
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    x = radius_km * np.outer(np.cos(u), np.sin(v))
    y = radius_km * np.outer(np.sin(u), np.sin(v))
    z = radius_km * np.outer(np.ones(np.size(u)), np.cos(v))
    return {"x": x, "y": y, "z": z}


def plot_3d_trajectory_with_phases(h5_file, output_dir=None):
    """
    3D軌道を各フェーズで色分けしてインタラクティブ表示

    Args:
        h5_file: HDF5ファイルパス
        output_dir: 出力ディレクトリ（Noneなら入力ファイルと同じ）
    """
    with h5py.File(h5_file, "r") as f:
        time = f["time"]["time_s"][:]

        # 軌道データ
        env = f["OrbitalEnvSim-0_OrbitalSpacecraft_0"]
        position_x = env["position_x"][:]
        position_y = env["position_y"][:]
        position_z = env["position_z"][:]
        altitude = env["altitude"][:]

        # 推力データ
        ctrl = f["OrbitalControllerSim-0_OrbitalController_0"]
        thrust_x = ctrl["thrust_command_x"][:]
        thrust_y = ctrl["thrust_command_y"][:]
        thrust_z = ctrl["thrust_command_z"][:]
        thrust_mag = np.sqrt(thrust_x**2 + thrust_y**2 + thrust_z**2)

    # フェーズの検出
    phases = detect_phases(thrust_mag, time)

    # 地球の球体
    earth = create_earth_sphere()

    # 3Dプロット
    fig = go.Figure()

    # 地球を描画（透明度高め）
    fig.add_trace(
        go.Surface(
            x=earth["x"],
            y=earth["y"],
            z=earth["z"],
            colorscale=[[0, "lightblue"], [1, "lightblue"]],
            showscale=False,
            name="Earth",
            opacity=0.3,
            hoverinfo="name",
        )
    )

    # 各フェーズを色分けしてプロット
    for start_idx, end_idx, phase_name, color in phases:
        fig.add_trace(
            go.Scatter3d(
                x=position_x[start_idx:end_idx] / 1e3,
                y=position_y[start_idx:end_idx] / 1e3,
                z=position_z[start_idx:end_idx] / 1e3,
                mode="lines",
                line=dict(color=color, width=4),
                name=phase_name,
                hovertemplate="<b>"
                + phase_name
                + "</b><br>"
                + "Time: %{text:.2f} min<br>"
                + "X: %{x:.2f} km<br>"
                + "Y: %{y:.2f} km<br>"
                + "Z: %{z:.2f} km<br>"
                + "Alt: %{customdata:.2f} km<br>"
                + "<extra></extra>",
                text=time[start_idx:end_idx] / 60.0,
                customdata=altitude[start_idx:end_idx] / 1e3,
            )
        )

        # 開始点にマーカー
        if start_idx < len(position_x):
            fig.add_trace(
                go.Scatter3d(
                    x=[position_x[start_idx] / 1e3],
                    y=[position_y[start_idx] / 1e3],
                    z=[position_z[start_idx] / 1e3],
                    mode="markers",
                    marker=dict(size=6, color=color, line=dict(color="black", width=2)),
                    name=f"{phase_name} start",
                    showlegend=False,
                    hovertemplate=f"<b>{phase_name} Start</b><br>"
                    + f"Time: {time[start_idx] / 60.0:.2f} min<br>"
                    + "X: %{x:.2f} km<br>"
                    + "Y: %{y:.2f} km<br>"
                    + "Z: %{z:.2f} km<br>"
                    + "<extra></extra>",
                )
            )

    # レイアウト設定
    max_range = np.max(np.sqrt(position_x**2 + position_y**2 + position_z**2)) / 1e3 * 1.1
    fig.update_layout(
        title="Hohmann Transfer Trajectory - Phase Colored (Interactive)",
        scene=dict(
            xaxis=dict(title="X [km]", range=[-max_range, max_range]),
            yaxis=dict(title="Y [km]", range=[-max_range, max_range]),
            zaxis=dict(title="Z [km]", range=[-max_range, max_range]),
            aspectmode="cube",
        ),
        width=1200,
        height=900,
        showlegend=True,
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.8)"),
    )

    # 保存
    if output_dir is None:
        output_dir = Path(h5_file).parent
    output_file = Path(output_dir) / "orbital_3d_trajectory_phases_interactive.html"
    fig.write_html(str(output_file))
    print(f"✅ Saved: {output_file}")


def plot_altitude_thrust_with_phases(h5_file, output_dir=None):
    """
    高度と推力の時系列を各フェーズで色分けしてインタラクティブ表示

    Args:
        h5_file: HDF5ファイルパス
        output_dir: 出力ディレクトリ
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

    # サブプロット作成
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("Altitude Profile", "Thrust Profile"),
        vertical_spacing=0.12,
        shared_xaxes=True,
    )

    # 高度プロット
    for start_idx, end_idx, phase_name, color in phases:
        fig.add_trace(
            go.Scatter(
                x=time[start_idx:end_idx] / 60.0,
                y=altitude[start_idx:end_idx] / 1e3,
                mode="lines",
                line=dict(color=color, width=3),
                name=phase_name,
                hovertemplate="<b>"
                + phase_name
                + "</b><br>"
                + "Time: %{x:.2f} min<br>"
                + "Altitude: %{y:.2f} km<br>"
                + "<extra></extra>",
            ),
            row=1,
            col=1,
        )

    # 推力プロット
    for start_idx, end_idx, phase_name, color in phases:
        fig.add_trace(
            go.Scatter(
                x=time[start_idx:end_idx] / 60.0,
                y=thrust_mag[start_idx:end_idx],
                mode="lines",
                line=dict(color=color, width=3),
                name=phase_name,
                showlegend=False,
                hovertemplate="<b>"
                + phase_name
                + "</b><br>"
                + "Time: %{x:.2f} min<br>"
                + "Thrust: %{y:.2f} N<br>"
                + "<extra></extra>",
            ),
            row=2,
            col=1,
        )

    # 軸ラベル
    fig.update_xaxes(title_text="Time [min]", row=2, col=1)
    fig.update_yaxes(title_text="Altitude [km]", row=1, col=1)
    fig.update_yaxes(title_text="Thrust [N]", row=2, col=1)

    # レイアウト
    fig.update_layout(
        title="Hohmann Transfer - Altitude & Thrust (Interactive)",
        height=900,
        showlegend=True,
        hovermode="x unified",
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.8)"),
    )

    # 保存
    if output_dir is None:
        output_dir = Path(h5_file).parent
    output_file = Path(output_dir) / "altitude_thrust_phases_interactive.html"
    fig.write_html(str(output_file))
    print(f"✅ Saved: {output_file}")


def plot_orbital_elements_with_phases(h5_file, output_dir=None):
    """
    軌道要素を各フェーズで色分けしてインタラクティブ表示

    Args:
        h5_file: HDF5ファイルパス
        output_dir: 出力ディレクトリ
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

    # サブプロット作成
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("Semi-major Axis", "Eccentricity"),
        vertical_spacing=0.12,
        shared_xaxes=True,
    )

    # 軌道長半径
    for start_idx, end_idx, phase_name, color in phases:
        fig.add_trace(
            go.Scatter(
                x=time[start_idx:end_idx] / 60.0,
                y=semi_major_axis[start_idx:end_idx] / 1e3,
                mode="lines",
                line=dict(color=color, width=3),
                name=phase_name,
                hovertemplate="<b>"
                + phase_name
                + "</b><br>"
                + "Time: %{x:.2f} min<br>"
                + "SMA: %{y:.2f} km<br>"
                + "<extra></extra>",
            ),
            row=1,
            col=1,
        )

    # 離心率
    for start_idx, end_idx, phase_name, color in phases:
        fig.add_trace(
            go.Scatter(
                x=time[start_idx:end_idx] / 60.0,
                y=eccentricity[start_idx:end_idx],
                mode="lines",
                line=dict(color=color, width=3),
                name=phase_name,
                showlegend=False,
                hovertemplate="<b>"
                + phase_name
                + "</b><br>"
                + "Time: %{x:.2f} min<br>"
                + "Eccentricity: %{y:.6f}<br>"
                + "<extra></extra>",
            ),
            row=2,
            col=1,
        )

    # 軸ラベル
    fig.update_xaxes(title_text="Time [min]", row=2, col=1)
    fig.update_yaxes(title_text="Semi-major Axis [km]", row=1, col=1)
    fig.update_yaxes(title_text="Eccentricity [-]", row=2, col=1)

    # レイアウト
    fig.update_layout(
        title="Orbital Elements Evolution (Interactive)",
        height=900,
        showlegend=True,
        hovermode="x unified",
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.8)"),
    )

    # 保存
    if output_dir is None:
        output_dir = Path(h5_file).parent
    output_file = Path(output_dir) / "orbital_elements_phases_interactive.html"
    fig.write_html(str(output_file))
    print(f"✅ Saved: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Visualize Hohmann transfer with phase coloring (Interactive)")
    parser.add_argument("h5_file", help="HDF5 file path")
    parser.add_argument("--output-dir", help="Output directory (default: same as input file)")

    args = parser.parse_args()

    h5_file = Path(args.h5_file)
    if not h5_file.exists():
        print(f"❌ Error: File not found: {h5_file}")
        sys.exit(1)

    output_dir = args.output_dir if args.output_dir else h5_file.parent

    print("=" * 70)
    print("Hohmann Transfer Phase Visualization (Interactive)")
    print("=" * 70)
    print(f"Input: {h5_file}")
    print(f"Output: {output_dir}")
    print()

    print("📊 Generating interactive visualizations...")

    # 各プロットを生成
    plot_3d_trajectory_with_phases(h5_file, output_dir)
    plot_altitude_thrust_with_phases(h5_file, output_dir)
    plot_orbital_elements_with_phases(h5_file, output_dir)

    print()
    print("✅ All interactive visualizations completed!")
    print(f"📁 Open the HTML files in your browser to view interactive plots")
    print("=" * 70)


if __name__ == "__main__":
    main()
