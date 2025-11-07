"""
軌道シミュレーション結果のインタラクティブ可視化 (Plotly版)

HDF5ファイルから軌道データを読み込み、インタラクティブなHTMLプロットを生成:
- 3D軌道（回転・ズーム可能）
- アニメーション機能付き軌道再生
- ホバーでデータ詳細表示
- 時系列グラフ（インタラクティブ）
"""

import argparse
import h5py
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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


def create_earth_sphere(radius_km=6378.137, resolution=50):
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


def plot_3d_trajectory_interactive(data, output_dir=None):
    """
    3D軌道のインタラクティブプロット

    Args:
        data: 軌道データ
        output_dir: 出力ディレクトリ
    """
    # 単位変換
    pos_km = {
        "x": data["position_x"] / 1e3,
        "y": data["position_y"] / 1e3,
        "z": data["position_z"] / 1e3,
    }
    time_min = data["time"] / 60.0
    altitude_km = data["altitude"] / 1e3

    # 地球の球体
    earth = create_earth_sphere()

    # 3Dプロット
    fig = go.Figure()

    # 地球
    fig.add_trace(
        go.Surface(
            x=earth["x"],
            y=earth["y"],
            z=earth["z"],
            colorscale=[[0, "lightblue"], [1, "darkblue"]],
            showscale=False,
            name="Earth",
            opacity=0.6,
            hoverinfo="name",
        )
    )

    # 軌道（全体）
    fig.add_trace(
        go.Scatter3d(
            x=pos_km["x"],
            y=pos_km["y"],
            z=pos_km["z"],
            mode="lines",
            line=dict(color="red", width=3),
            name="Orbit",
            hovertemplate="<b>Orbit</b><br>"
            + "Time: %{text:.2f} min<br>"
            + "X: %{x:.2f} km<br>"
            + "Y: %{y:.2f} km<br>"
            + "Z: %{z:.2f} km<br>"
            + "Alt: %{customdata:.2f} km<br>"
            + "<extra></extra>",
            text=time_min,
            customdata=altitude_km,
        )
    )

    # スタート地点
    fig.add_trace(
        go.Scatter3d(
            x=[pos_km["x"][0]],
            y=[pos_km["y"][0]],
            z=[pos_km["z"][0]],
            mode="markers",
            marker=dict(size=8, color="green", symbol="circle"),
            name="Start",
            hovertemplate="<b>Start</b><br>"
            + "X: %{x:.2f} km<br>"
            + "Y: %{y:.2f} km<br>"
            + "Z: %{z:.2f} km<br>"
            + "<extra></extra>",
        )
    )

    # エンド地点
    fig.add_trace(
        go.Scatter3d(
            x=[pos_km["x"][-1]],
            y=[pos_km["y"][-1]],
            z=[pos_km["z"][-1]],
            mode="markers",
            marker=dict(size=8, color="orange", symbol="circle"),
            name="End",
            hovertemplate="<b>End</b><br>"
            + "X: %{x:.2f} km<br>"
            + "Y: %{y:.2f} km<br>"
            + "Z: %{z:.2f} km<br>"
            + "<extra></extra>",
        )
    )

    # レイアウト設定
    max_range = np.max(data["position_norm"] / 1e3) * 1.1
    fig.update_layout(
        title="3D Orbital Trajectory (Interactive)",
        scene=dict(
            xaxis=dict(title="X [km]", range=[-max_range, max_range]),
            yaxis=dict(title="Y [km]", range=[-max_range, max_range]),
            zaxis=dict(title="Z [km]", range=[-max_range, max_range]),
            aspectmode="cube",
        ),
        width=1000,
        height=800,
        showlegend=True,
    )

    # 保存
    if output_dir:
        output_path = Path(output_dir) / "orbital_3d_interactive.html"
        fig.write_html(str(output_path))
        print(f"   Saved: orbital_3d_interactive.html")

    return fig


def plot_3d_trajectory_animated(data, output_dir=None):
    """
    アニメーション付き3D軌道プロット

    Args:
        data: 軌道データ
        output_dir: 出力ディレクトリ
    """
    # 単位変換
    pos_km = {
        "x": data["position_x"] / 1e3,
        "y": data["position_y"] / 1e3,
        "z": data["position_z"] / 1e3,
    }
    time_min = data["time"] / 60.0

    # データを間引く（アニメーション用）
    skip = max(1, len(data["time"]) // 200)  # 最大200フレーム
    indices = np.arange(0, len(data["time"]), skip)

    # 地球の球体
    earth = create_earth_sphere()

    # フレームの作成
    frames = []
    for i, idx in enumerate(indices):
        frame_data = [
            # 地球（固定）
            go.Surface(
                x=earth["x"],
                y=earth["y"],
                z=earth["z"],
                colorscale=[[0, "lightblue"], [1, "darkblue"]],
                showscale=False,
                opacity=0.6,
                hoverinfo="skip",
            ),
            # 軌道トレイル
            go.Scatter3d(
                x=pos_km["x"][: idx + 1],
                y=pos_km["y"][: idx + 1],
                z=pos_km["z"][: idx + 1],
                mode="lines",
                line=dict(color="red", width=2),
                name="Orbit Trail",
                hoverinfo="skip",
            ),
            # 現在位置
            go.Scatter3d(
                x=[pos_km["x"][idx]],
                y=[pos_km["y"][idx]],
                z=[pos_km["z"][idx]],
                mode="markers",
                marker=dict(size=10, color="yellow", symbol="diamond"),
                name=f"Spacecraft (t={time_min[idx]:.1f}min)",
                hovertemplate=f"<b>Time: {time_min[idx]:.2f} min</b><br>"
                + "X: %{x:.2f} km<br>"
                + "Y: %{y:.2f} km<br>"
                + "Z: %{z:.2f} km<br>"
                + "<extra></extra>",
            ),
        ]
        frames.append(go.Frame(data=frame_data, name=str(i)))

    # 初期フレーム
    fig = go.Figure(data=frames[0].data, frames=frames)

    # アニメーション設定
    max_range = np.max(data["position_norm"] / 1e3) * 1.1
    fig.update_layout(
        title="3D Orbital Animation",
        scene=dict(
            xaxis=dict(title="X [km]", range=[-max_range, max_range]),
            yaxis=dict(title="Y [km]", range=[-max_range, max_range]),
            zaxis=dict(title="Z [km]", range=[-max_range, max_range]),
            aspectmode="cube",
        ),
        width=1000,
        height=800,
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            {
                                "frame": {"duration": 50, "redraw": True},
                                "fromcurrent": True,
                                "mode": "immediate",
                            },
                        ],
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                            },
                        ],
                    ),
                ],
                x=0.1,
                y=1.15,
            )
        ],
        sliders=[
            dict(
                active=0,
                steps=[
                    dict(
                        args=[
                            [str(i)],
                            {
                                "frame": {"duration": 0, "redraw": True},
                                "mode": "immediate",
                            },
                        ],
                        label=f"{time_min[idx]:.1f}min",
                        method="animate",
                    )
                    for i, idx in enumerate(indices)
                ],
                x=0.1,
                len=0.9,
                y=0,
            )
        ],
    )

    # 保存
    if output_dir:
        output_path = Path(output_dir) / "orbital_3d_animation.html"
        fig.write_html(str(output_path))
        print(f"   Saved: orbital_3d_animation.html")

    return fig


def plot_timeseries_interactive(data, output_dir=None):
    """
    時系列データのインタラクティブプロット

    Args:
        data: 軌道データ
        output_dir: 出力ディレクトリ
    """
    # 単位変換
    time_min = data["time"] / 60.0
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

    # サブプロット作成
    fig = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=(
            "Position Components",
            "Orbital Radius",
            "Velocity Components",
            "Orbital Speed",
            "Altitude",
            "Orbital Elements",
        ),
        specs=[
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "scatter"}],
        ],
        vertical_spacing=0.1,
        horizontal_spacing=0.1,
    )

    # 位置成分
    fig.add_trace(
        go.Scatter(
            x=time_min,
            y=pos_km["x"],
            mode="lines",
            name="X",
            line=dict(color="red"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=time_min,
            y=pos_km["y"],
            mode="lines",
            name="Y",
            line=dict(color="green"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=time_min,
            y=pos_km["z"],
            mode="lines",
            name="Z",
            line=dict(color="blue"),
        ),
        row=1,
        col=1,
    )

    # 動径距離
    fig.add_trace(
        go.Scatter(x=time_min, y=pos_km["norm"], mode="lines", name="|r|", showlegend=False),
        row=1,
        col=2,
    )

    # 速度成分
    fig.add_trace(
        go.Scatter(
            x=time_min,
            y=vel_km_s["x"],
            mode="lines",
            name="Vx",
            line=dict(color="red", dash="dash"),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=time_min,
            y=vel_km_s["y"],
            mode="lines",
            name="Vy",
            line=dict(color="green", dash="dash"),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=time_min,
            y=vel_km_s["z"],
            mode="lines",
            name="Vz",
            line=dict(color="blue", dash="dash"),
        ),
        row=2,
        col=1,
    )

    # 速度ノルム
    fig.add_trace(
        go.Scatter(x=time_min, y=vel_km_s["norm"], mode="lines", name="|v|", showlegend=False),
        row=2,
        col=2,
    )

    # 高度
    fig.add_trace(
        go.Scatter(x=time_min, y=altitude_km, mode="lines", name="Altitude", showlegend=False),
        row=3,
        col=1,
    )

    # 軌道要素
    fig.add_trace(
        go.Scatter(
            x=time_min,
            y=sma_km,
            mode="lines",
            name="SMA",
            line=dict(color="purple"),
        ),
        row=3,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=time_min,
            y=data["eccentricity"] * 1000,  # スケール調整
            mode="lines",
            name="Ecc×1000",
            line=dict(color="orange"),
            yaxis="y2",
        ),
        row=3,
        col=2,
    )

    # 軸ラベル更新
    fig.update_xaxes(title_text="Time [min]", row=3, col=1)
    fig.update_xaxes(title_text="Time [min]", row=3, col=2)
    fig.update_yaxes(title_text="Position [km]", row=1, col=1)
    fig.update_yaxes(title_text="Radius [km]", row=1, col=2)
    fig.update_yaxes(title_text="Velocity [km/s]", row=2, col=1)
    fig.update_yaxes(title_text="Speed [km/s]", row=2, col=2)
    fig.update_yaxes(title_text="Altitude [km]", row=3, col=1)
    fig.update_yaxes(title_text="SMA [km]", row=3, col=2)

    # レイアウト
    fig.update_layout(
        title="Orbital Parameters Time Series (Interactive)",
        height=1200,
        showlegend=True,
        hovermode="x unified",
    )

    # 保存
    if output_dir:
        output_path = Path(output_dir) / "orbital_timeseries_interactive.html"
        fig.write_html(str(output_path))
        print(f"   Saved: orbital_timeseries_interactive.html")

    return fig


def plot_orbital_simulation_interactive(h5_path: str, output_dir: str = None):
    """
    軌道シミュレーション結果をインタラクティブにプロット

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

    print(f"📊 Generating interactive plots...")

    # 3D軌道（静止）
    plot_3d_trajectory_interactive(data, output_dir)

    # 3D軌道（アニメーション）
    plot_3d_trajectory_animated(data, output_dir)

    # 時系列グラフ
    plot_timeseries_interactive(data, output_dir)

    print(f"✅ All interactive plots saved to {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize orbital simulation results (Interactive Plotly version)"
    )
    parser.add_argument("h5_file", type=str, help="Path to HDF5 data file")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for plots")

    args = parser.parse_args()

    plot_orbital_simulation_interactive(args.h5_file, args.output_dir)
