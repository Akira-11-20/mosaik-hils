"""
Formation Flying Sweep Comparison Visualization

Formation flying のパラメータスイープ結果を比較可視化するスクリプト。
相対位置の3D可視化（PNG & インタラクティブHTML）を含む。

使用例:
    # スイープディレクトリを指定
    python compare_formation_sweep.py /path/to/results_sweep/20251116-154228_sweep

    # 特定の結果のみを比較
    python compare_formation_sweep.py /path/to/sweep --indices 1 2 3
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


def load_formation_data(h5_path: Path) -> Dict:
    """
    HDF5ファイルからFormation flyingデータを読み込む

    Args:
        h5_path: HDF5ファイルのパス

    Returns:
        dict: 時系列データ
    """
    data = {}

    with h5py.File(h5_path, "r") as f:
        # 時間データ
        data["time"] = f["time"]["time_s"][:]

        # Chaser と Target のデータグループを検索
        env_groups = [k for k in f.keys() if "OrbitalEnvSim" in k]
        if len(env_groups) < 2:
            raise ValueError(f"Expected 2 spacecraft, found {len(env_groups)}")

        # 通常、0がChaser、1がTarget
        chaser_group = f[env_groups[0]]
        target_group = f[env_groups[1]]

        # Chaser データ
        data["chaser_pos_x"] = chaser_group["position_x"][:]
        data["chaser_pos_y"] = chaser_group["position_y"][:]
        data["chaser_pos_z"] = chaser_group["position_z"][:]
        data["chaser_vel_x"] = chaser_group["velocity_x"][:]
        data["chaser_vel_y"] = chaser_group["velocity_y"][:]
        data["chaser_vel_z"] = chaser_group["velocity_z"][:]
        data["chaser_altitude"] = chaser_group["altitude"][:]

        # Target データ
        data["target_pos_x"] = target_group["position_x"][:]
        data["target_pos_y"] = target_group["position_y"][:]
        data["target_pos_z"] = target_group["position_z"][:]
        data["target_altitude"] = target_group["altitude"][:]

        # 相対位置・速度の計算
        data["rel_pos_x"] = data["chaser_pos_x"] - data["target_pos_x"]
        data["rel_pos_y"] = data["chaser_pos_y"] - data["target_pos_y"]
        data["rel_pos_z"] = data["chaser_pos_z"] - data["target_pos_z"]
        data["rel_distance"] = np.sqrt(data["rel_pos_x"] ** 2 + data["rel_pos_y"] ** 2 + data["rel_pos_z"] ** 2)

        # 推力データ（Chaser環境への入力force）
        try:
            # ChaserのEnvSimから推力を取得（measured forceではなくenv入力force）
            data["norm_force"] = chaser_group["norm_force"][:]
        except (IndexError, KeyError):
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
        if not subdir.is_dir() or subdir.name == "comparison":
            continue

        # HDF5ファイルを探す
        h5_files = list(subdir.glob("*.h5"))
        if not h5_files:
            continue

        h5_file = h5_files[0]  # 最初のHDF5ファイルを使用

        # ディレクトリ名から設定を取得
        dir_name = subdir.name
        if "_" in dir_name:
            parts = dir_name.split("_", 1)
            if len(parts) == 2:
                index = int(parts[0])
                label = parts[1].replace("_", ", ")
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


def plot_relative_distance_comparison(results: List[Dict], output_path: Path):
    """
    相対距離の比較プロット

    Args:
        results: 結果情報のリスト
        output_path: 出力ファイルパス
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    for result in results:
        data = load_formation_data(result["path"])
        time_min = data["time"] / 60.0

        # ベースラインの場合は太い線で強調（被さっても見やすいように）
        is_baseline = "baseline" in result["label"].lower()
        linewidth = 4 if is_baseline else 2
        linestyle = "-" if is_baseline else "-"
        alpha = 1.0 if is_baseline else 0.7

        # 相対距離プロット
        ax1.plot(
            time_min,
            data["rel_distance"],
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

    # 相対距離グラフ設定
    ax1.set_xlabel("Time [min]", fontsize=12)
    ax1.set_ylabel("Relative Distance [m]", fontsize=12)
    ax1.set_title("Formation Flying: Relative Distance Comparison", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)

    # 推力グラフ設定
    ax2.set_xlabel("Time [min]", fontsize=12)
    ax2.set_ylabel("Thrust [N]", fontsize=12)
    ax2.set_title("Thrust Magnitude Comparison", fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✅ Saved: {output_path}")


def plot_relative_position_comparison(results: List[Dict], output_path: Path):
    """
    相対位置（X, Y, Z）の比較プロット

    Args:
        results: 結果情報のリスト
        output_path: 出力ファイルパス
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    for result in results:
        data = load_formation_data(result["path"])
        time_min = data["time"] / 60.0

        # ベースラインの場合は太い線で強調（被さっても見やすいように）
        is_baseline = "baseline" in result["label"].lower()
        linewidth = 4 if is_baseline else 2
        alpha = 1.0 if is_baseline else 0.7

        # Relative X
        axes[0].plot(time_min, data["rel_pos_x"], label=result["label"], alpha=alpha, linewidth=linewidth)

        # Relative Y
        axes[1].plot(time_min, data["rel_pos_y"], label=result["label"], alpha=alpha, linewidth=linewidth)

        # Relative Z
        axes[2].plot(time_min, data["rel_pos_z"], label=result["label"], alpha=alpha, linewidth=linewidth)

    # グラフ設定
    axes[0].set_ylabel("Relative X [m]", fontsize=12)
    axes[0].set_title("Formation Flying: Relative Position Comparison", fontsize=14, fontweight="bold")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)

    axes[1].set_ylabel("Relative Y [m]", fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)

    axes[2].set_ylabel("Relative Z [m]", fontsize=12)
    axes[2].set_xlabel("Time [min]", fontsize=12)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✅ Saved: {output_path}")


def plot_relative_3d_comparison(results: List[Dict], output_path: Path):
    """
    相対位置3D比較プロット（PNG）

    Args:
        results: 結果情報のリスト
        output_path: 出力ファイルパス
    """
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection="3d")

    # カラーマップの準備
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    # 各シミュレーションの相対軌道をプロット
    for i, result in enumerate(results):
        data = load_formation_data(result["path"])

        # ベースラインの場合は太い線で強調（被さっても見やすいように）
        is_baseline = "baseline" in result["label"].lower()
        linewidth = 5 if is_baseline else 2
        alpha = 1.0 if is_baseline else 0.7

        # 相対位置の3Dプロット
        ax.plot(
            data["rel_pos_x"],
            data["rel_pos_y"],
            data["rel_pos_z"],
            label=result["label"],
            alpha=alpha,
            linewidth=linewidth,
            color=colors[i],
        )

        # 初期位置マーカー
        ax.scatter(
            data["rel_pos_x"][0],
            data["rel_pos_y"][0],
            data["rel_pos_z"][0],
            s=150,
            marker="o",
            color=colors[i],
            alpha=0.9,
            edgecolors="black",
            linewidths=1.5,
        )

        # 最終位置マーカー
        ax.scatter(
            data["rel_pos_x"][-1],
            data["rel_pos_y"][-1],
            data["rel_pos_z"][-1],
            s=200,
            marker="^",
            color=colors[i],
            alpha=0.9,
            edgecolors="black",
            linewidths=1.5,
        )

    # 目標位置（原点）を強調
    ax.scatter(
        0,
        0,
        0,
        color="gold",
        s=400,
        marker="*",
        label="Target (origin)",
        zorder=100,
        edgecolors="black",
        linewidths=2,
    )

    ax.set_xlabel("Relative X [m]", fontsize=12, fontweight="bold")
    ax.set_ylabel("Relative Y [m]", fontsize=12, fontweight="bold")
    ax.set_zlabel("Relative Z [m]", fontsize=12, fontweight="bold")
    ax.set_title(
        "Formation Flying: Relative Position 3D Comparison",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    ax.legend(bbox_to_anchor=(1.15, 1), loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)

    # 軸のアスペクト比を等しくする
    max_range = 0
    for result in results:
        data = load_formation_data(result["path"])
        max_range = max(
            max_range,
            np.max(np.abs(data["rel_pos_x"])),
            np.max(np.abs(data["rel_pos_y"])),
            np.max(np.abs(data["rel_pos_z"])),
        )

    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range, max_range)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✅ Saved: {output_path}")


def generate_relative_3d_interactive(results: List[Dict], output_path: Path):
    """
    相対位置3Dインタラクティブ比較（HTML）

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

    # カラーパレット
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    # 各シミュレーションの相対軌道をプロット
    for i, result in enumerate(results):
        data = load_formation_data(result["path"])
        color = colors[i % len(colors)]

        # ベースラインの場合は太い線で強調（被さっても見やすいように）
        is_baseline = "baseline" in result["label"].lower()
        line_width = 8 if is_baseline else 4

        # メイントラジェクトリ
        fig.add_trace(
            go.Scatter3d(
                x=data["rel_pos_x"],
                y=data["rel_pos_y"],
                z=data["rel_pos_z"],
                mode="lines",
                name=result["label"],
                line=dict(
                    color=color,
                    width=line_width,
                ),
                hovertemplate=f"{result['label']}<br>"
                + "X: %{x:.2f} m<br>Y: %{y:.2f} m<br>Z: %{z:.2f} m<extra></extra>",
            )
        )

        # 初期位置マーカー
        fig.add_trace(
            go.Scatter3d(
                x=[data["rel_pos_x"][0]],
                y=[data["rel_pos_y"][0]],
                z=[data["rel_pos_z"][0]],
                mode="markers",
                name=f"{result['label']} (start)",
                marker=dict(size=8, color=color, symbol="circle", line=dict(color="black", width=1)),
                hovertemplate=f"{result['label']} (start)<br>"
                + "X: %{x:.2f} m<br>Y: %{y:.2f} m<br>Z: %{z:.2f} m<extra></extra>",
                showlegend=False,
            )
        )

        # 最終位置マーカー
        fig.add_trace(
            go.Scatter3d(
                x=[data["rel_pos_x"][-1]],
                y=[data["rel_pos_y"][-1]],
                z=[data["rel_pos_z"][-1]],
                mode="markers",
                name=f"{result['label']} (end)",
                marker=dict(size=10, color=color, symbol="diamond", line=dict(color="black", width=1)),
                hovertemplate=f"{result['label']} (end)<br>"
                + "X: %{x:.2f} m<br>Y: %{y:.2f} m<br>Z: %{z:.2f} m<extra></extra>",
                showlegend=False,
            )
        )

    # 目標位置（原点）
    fig.add_trace(
        go.Scatter3d(
            x=[0],
            y=[0],
            z=[0],
            mode="markers",
            name="Target (origin)",
            marker=dict(size=15, color="gold", symbol="cross", line=dict(color="black", width=2)),
            hovertemplate="Target (origin)<br>X: 0 m<br>Y: 0 m<br>Z: 0 m<extra></extra>",
        )
    )

    # レイアウト設定
    fig.update_layout(
        title=dict(
            text="Formation Flying: Relative Position 3D Comparison (Interactive)",
            font=dict(size=20, color="black"),
            x=0.5,
            xanchor="center",
        ),
        scene=dict(
            xaxis=dict(
                title="Relative X [m]",
                backgroundcolor="white",
                gridcolor="lightgray",
                showbackground=True,
            ),
            yaxis=dict(
                title="Relative Y [m]",
                backgroundcolor="white",
                gridcolor="lightgray",
                showbackground=True,
            ),
            zaxis=dict(
                title="Relative Z [m]",
                backgroundcolor="white",
                gridcolor="lightgray",
                showbackground=True,
            ),
            aspectmode="cube",
        ),
        showlegend=True,
        legend=dict(x=0.7, y=0.95, bgcolor="rgba(255,255,255,0.9)", bordercolor="black", borderwidth=1),
        hovermode="closest",
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=900,
    )

    fig.write_html(output_path)
    print(f"✅ Saved: {output_path}")


def plot_baseline_difference(results: List[Dict], output_dir: Path):
    """
    ベースラインとの差分プロット（相対距離と推力）

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
    baseline_data = load_formation_data(baseline_result["path"])
    baseline_time = baseline_data["time"]

    # 差分プロット1: 相対距離の差分
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    for result in other_results:
        data = load_formation_data(result["path"])
        time_min = data["time"] / 60.0

        # 時間軸が一致していることを確認（必要に応じて補間）
        if len(data["time"]) == len(baseline_time) and np.allclose(data["time"], baseline_time):
            # 相対距離の差分
            distance_diff = data["rel_distance"] - baseline_data["rel_distance"]
            ax1.plot(time_min, distance_diff, label=result["label"], alpha=0.8, linewidth=2)

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
    ax1.set_ylabel("Δ Relative Distance [m]", fontsize=12)
    ax1.set_title(
        "Difference from Baseline: Relative Distance",
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
    output_path = output_dir / "formation_baseline_difference.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✅ Saved: {output_path}")

    # 差分プロット2: 相対位置（X, Y, Z）の差分
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    for result in other_results:
        data = load_formation_data(result["path"])
        time_min = data["time"] / 60.0

        if len(data["time"]) == len(baseline_time) and np.allclose(data["time"], baseline_time):
            # X, Y, Zの差分
            diff_x = data["rel_pos_x"] - baseline_data["rel_pos_x"]
            diff_y = data["rel_pos_y"] - baseline_data["rel_pos_y"]
            diff_z = data["rel_pos_z"] - baseline_data["rel_pos_z"]

            axes[0].plot(time_min, diff_x, label=result["label"], alpha=0.8, linewidth=2)
            axes[1].plot(time_min, diff_y, label=result["label"], alpha=0.8, linewidth=2)
            axes[2].plot(time_min, diff_z, label=result["label"], alpha=0.8, linewidth=2)

    # ゼロラインを追加
    for ax in axes:
        ax.axhline(y=0, color="black", linestyle="--", linewidth=1, alpha=0.5, label="Baseline")

    # グラフ設定
    axes[0].set_ylabel("Δ Relative X [m]", fontsize=12)
    axes[0].set_title(
        "Difference from Baseline: Relative Position",
        fontsize=14,
        fontweight="bold",
    )
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)

    axes[1].set_ylabel("Δ Relative Y [m]", fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)

    axes[2].set_ylabel("Δ Relative Z [m]", fontsize=12)
    axes[2].set_xlabel("Time [min]", fontsize=12)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)

    plt.tight_layout()
    output_path = output_dir / "formation_baseline_position_difference.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✅ Saved: {output_path}")


def plot_altitude_comparison(results: List[Dict], output_path: Path):
    """
    高度比較プロット（Chaser と Target）

    Args:
        results: 結果情報のリスト
        output_path: 出力ファイルパス
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))

    for i, result in enumerate(results):
        data = load_formation_data(result["path"])
        time_min = data["time"] / 60.0

        # Chaser altitude
        ax.plot(
            time_min,
            data["chaser_altitude"] / 1e3,
            label=f"{result['label']} (Chaser)",
            alpha=0.7,
            linewidth=2,
            linestyle="-",
        )

        # Target altitude (dashed)
        ax.plot(
            time_min,
            data["target_altitude"] / 1e3,
            label=f"{result['label']} (Target)",
            alpha=0.5,
            linewidth=1.5,
            linestyle="--",
        )

    ax.set_xlabel("Time [min]", fontsize=12)
    ax.set_ylabel("Altitude [km]", fontsize=12)
    ax.set_title("Formation Flying: Altitude Comparison", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✅ Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare formation flying sweep simulation results")
    parser.add_argument("sweep_dir", type=str, help="Sweep directory path")
    parser.add_argument("--indices", nargs="+", type=int, help="Specific indices to compare")
    parser.add_argument("--output-dir", type=str, help="Output directory (default: sweep_dir/comparison)")

    args = parser.parse_args()

    # スイープディレクトリ
    sweep_dir = Path(args.sweep_dir)
    if not sweep_dir.exists():
        print(f"❌ Error: Sweep directory not found: {sweep_dir}")
        return

    # 結果を検索
    print(f"\n🔍 Scanning formation flying sweep directory: {sweep_dir}")
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
    print("Generating formation flying comparison visualizations...\n")

    # 1. 相対距離と推力の比較
    plot_relative_distance_comparison(results, output_dir / "formation_distance_thrust_comparison.png")

    # 2. 相対位置（X, Y, Z）の比較
    plot_relative_position_comparison(results, output_dir / "formation_relative_position_comparison.png")

    # 3. 相対位置3D比較（PNG）
    plot_relative_3d_comparison(results, output_dir / "formation_relative_3d_comparison.png")

    # 4. 相対位置3Dインタラクティブ（HTML）
    generate_relative_3d_interactive(results, output_dir / "formation_relative_3d_interactive.html")

    # 5. 高度比較
    plot_altitude_comparison(results, output_dir / "formation_altitude_comparison.png")

    # 6. ベースラインとの差分（NEW!）
    print()
    plot_baseline_difference(results, output_dir)

    print(f"\n✅ Formation flying comparison complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
