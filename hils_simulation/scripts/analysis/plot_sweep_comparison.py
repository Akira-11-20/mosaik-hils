"""
Sweep Results Comparison Plotter

指定されたsweepディレクトリ配下の各シミュレーション結果について、
position/velocityの絶対値とbaselineからの差分を4つのプロットで縦に並べた図を作成する。

使用方法:
    cd /home/akira/mosaik-hils/hils_simulation
    uv run python scripts/analysis/plot_sweep_comparison.py results/20251111-183809_sweep
"""

import argparse
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


def load_hdf5_data(h5_path):
    """HDF5ファイルからデータを読み込む（階層構造対応）"""
    hdf5_data = {}
    with h5py.File(h5_path, "r") as f:
        # 旧形式（data/以下にフラット）の対応
        if "data" in f:
            for key in f["data"].keys():
                hdf5_data[key] = f["data"][key][:]
        else:
            # 新形式（ノードごとにグループ化）の対応
            def read_group(group, prefix=""):
                """再帰的にグループを読み込む"""
                for key in group.keys():
                    item = group[key]
                    if isinstance(item, h5py.Group):
                        # グループの場合、再帰的に読み込む
                        read_group(item, prefix=f"{key}_")
                    elif isinstance(item, h5py.Dataset):
                        # データセットの場合、フラット化したキー名で保存
                        parts = item.name.split("/")
                        if len(parts) >= 2:
                            # /group_name/attr_name -> attr_name_group_name
                            group_name = parts[1]
                            attr_name = parts[-1]
                            flat_key = f"{attr_name}_{group_name}" if group_name != "time" else attr_name
                        else:
                            flat_key = item.name.replace("/", "_")
                            if flat_key.startswith("_"):
                                flat_key = flat_key[1:]
                        hdf5_data[flat_key] = item[:]

            read_group(f)
    return hdf5_data


def find_key_by_prefix_and_suffix(key_data, prefix, suffix):
    """プレフィックスとサフィックスでデータセットキーを検索"""
    for k in key_data.keys():
        if k.startswith(prefix) and k.endswith(suffix):
            return k
    return None


def find_key_by_suffix(key_data, suffix):
    """キーの接尾辞でデータセットキーを検索"""
    for k in key_data.keys():
        if k.endswith(suffix):
            return k
    return None


def load_simulation_data(h5_file):
    """シミュレーションデータをロード（位置と速度のみ）"""
    data = load_hdf5_data(h5_file)

    # 時刻データ
    time = data.get("time_s", np.array([]))

    # 位置データキーの検索
    pos_key = find_key_by_prefix_and_suffix(data, "position_", "Spacecraft1DOF_0")
    if not pos_key:
        pos_key = find_key_by_suffix(data, "position_Spacecraft")

    if not pos_key:
        return None

    # 速度データキーの検索
    vel_key = pos_key.replace("position", "velocity")

    position = data.get(pos_key, np.array([]))
    velocity = data.get(vel_key, np.array([]))

    if len(time) == 0 or len(position) == 0 or len(velocity) == 0:
        return None

    return {
        "time": time,
        "position": position,
        "velocity": velocity,
    }


def load_simulation_config(sim_dir):
    """
    シミュレーションディレクトリから設定を読み込む

    Returns:
        dict: 設定データ、またはNone
    """
    config_file = sim_dir / "simulation_config.json"
    if not config_file.exists():
        return None

    try:
        with open(config_file) as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️  Warning: Failed to load config from {config_file}: {e}")
        return None


def create_unified_comparison_plot(baseline_data, sim_data_list, sim_names, sim_configs, output_path):
    """
    全シミュレーションをまとめた比較プロットを作成

    Args:
        baseline_data: baselineのシミュレーションデータ
        sim_data_list: 比較するシミュレーションデータのリスト
        sim_names: シミュレーション名のリスト
        sim_configs: シミュレーション設定のリスト
        output_path: 出力ファイルパス
    """
    # 4つのサブプロットを縦に並べる
    fig, axes = plt.subplots(4, 1, figsize=(12, 16))

    # 色パレット（compare_tau_sweep_results.py を参考に）
    colors = ["#e74c3c", "#f39c12", "#9b59b6", "#3498db", "#2ecc71"]

    # Baseline用の色とスタイル（黒色、実線、太線）
    baseline_color = "black"
    baseline_linestyle = "-"
    baseline_linewidth = 2

    # 時刻データを取得
    time_baseline = baseline_data["time"]
    pos_baseline = baseline_data["position"]
    vel_baseline = baseline_data["velocity"]

    # プロット1: Position（絶対値）
    axes[0].plot(
        time_baseline,
        pos_baseline,
        label="Baseline (RT)",
        color=baseline_color,
        linestyle=baseline_linestyle,
        lw=baseline_linewidth,
        alpha=0.9,
        zorder=100,
    )

    # プロット2: Position差分
    axes[1].axhline(y=0, color="k", linestyle="--", alpha=0.3, zorder=1)

    # プロット3: Velocity（絶対値）
    axes[2].plot(
        time_baseline,
        vel_baseline,
        label="Baseline (RT)",
        color=baseline_color,
        linestyle=baseline_linestyle,
        lw=baseline_linewidth,
        alpha=0.9,
        zorder=100,
    )

    # プロット4: Velocity差分
    axes[3].axhline(y=0, color="k", linestyle="--", alpha=0.3, zorder=1)

    # 各シミュレーションをプロット（順順：小さい遅延から大きい遅延へ）
    for idx, (sim_data, sim_name, sim_config) in enumerate(zip(sim_data_list, sim_names, sim_configs)):
        # 色パレットから色を取得
        color = colors[idx % len(colors)]
        linewidth = 1.5
        linestyle = "--"  # 破線

        time_sim = sim_data["time"]
        pos_sim = sim_data["position"]
        vel_sim = sim_data["velocity"]

        # 時刻を統一（線形補間）- baselineの時刻を基準にする
        pos_sim_interp = np.interp(time_baseline, time_sim, pos_sim)
        vel_sim_interp = np.interp(time_baseline, time_sim, vel_sim)

        # 差分を計算
        pos_diff = pos_sim_interp - pos_baseline
        vel_diff = vel_sim_interp - vel_baseline

        # 短い名前を作成（見やすくするため）
        short_name = sim_name
        if "cmd" in sim_name and "ms" in sim_name:
            # 例: 20251111-183844_cmd5ms_sense0ms_comp_tau100ms -> cmd5ms
            parts = sim_name.split("_")
            for part in parts:
                if "cmd" in part and "ms" in part:
                    short_name = part
                    break

        # Inverse compensationが有効な場合、alpha値を追加
        if sim_config:
            inv_comp = sim_config.get("inverse_compensation", {})
            if inv_comp.get("enabled", False):
                alpha = inv_comp.get("gain", None)
                if alpha is not None:
                    # cmd_delay_sをmsに変換
                    cmd_delay_ms = sim_config.get("communication", {}).get("cmd_delay_s", 0) * 1000
                    short_name = f"cmd{cmd_delay_ms:.0f}ms α={alpha}"

        # プロット1: Position（絶対値）
        axes[0].plot(
            time_sim,
            pos_sim,
            label=short_name,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
        )

        # プロット2: Position差分
        axes[1].plot(
            time_baseline,
            pos_diff,
            label=short_name,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
        )

        # プロット3: Velocity（絶対値）
        axes[2].plot(
            time_sim,
            vel_sim,
            label=short_name,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
        )

        # プロット4: Velocity差分
        axes[3].plot(
            time_baseline,
            vel_diff,
            label=short_name,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
        )

    # 軸設定
    axes[0].set_xlabel("Time [s]", fontsize=11)
    axes[0].set_ylabel("Position [m]", fontsize=11)
    axes[0].set_title("Position Trajectory Comparison", fontsize=12, fontweight="bold")
    axes[0].legend(fontsize=9, loc="best")
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Time [s]", fontsize=11)
    axes[1].set_ylabel("Position Deviation from RT [m]", fontsize=11)
    axes[1].set_title("Position Deviation from RT Baseline", fontsize=12, fontweight="bold")
    axes[1].legend(fontsize=9, loc="best")
    axes[1].grid(True, alpha=0.3)

    axes[2].set_xlabel("Time [s]", fontsize=11)
    axes[2].set_ylabel("Velocity [m/s]", fontsize=11)
    axes[2].set_title("Velocity Trajectory Comparison", fontsize=12, fontweight="bold")
    axes[2].legend(fontsize=9, loc="best")
    axes[2].grid(True, alpha=0.3)

    axes[3].set_xlabel("Time [s]", fontsize=11)
    axes[3].set_ylabel("Velocity Deviation from RT [m/s]", fontsize=11)
    axes[3].set_title("Velocity Deviation from RT Baseline", fontsize=12, fontweight="bold")
    axes[3].legend(fontsize=9, loc="best")
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✓ Saved unified comparison plot: {output_path}")

    # メモリ解放
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate comparison plots (absolute + difference) for sweep results")
    parser.add_argument(
        "sweep_dir",
        type=str,
        help="Path to sweep directory (e.g., results/20251111-183809_sweep)",
    )
    parser.add_argument(
        "--baseline-name",
        type=str,
        default="baseline",
        help="Substring to identify baseline directory (default: 'baseline')",
    )

    args = parser.parse_args()

    # sweepディレクトリのパス
    sweep_dir = Path(args.sweep_dir)
    if not sweep_dir.exists():
        print(f"❌ Error: Sweep directory not found: {sweep_dir}")
        return

    print(f"📂 Processing sweep directory: {sweep_dir}")

    # サブディレクトリを取得
    subdirs = [d for d in sweep_dir.iterdir() if d.is_dir()]
    if len(subdirs) == 0:
        print(f"❌ Error: No subdirectories found in {sweep_dir}")
        return

    print(f"   Found {len(subdirs)} subdirectories")

    # baselineを検索
    baseline_dir = None
    for subdir in subdirs:
        if args.baseline_name in subdir.name.lower():
            baseline_dir = subdir
            break

    if baseline_dir is None:
        print(f"❌ Error: Baseline directory not found (looking for '{args.baseline_name}' in name)")
        print("   Available directories:")
        for subdir in subdirs:
            print(f"     - {subdir.name}")
        return

    print(f"📊 Baseline: {baseline_dir.name}")

    # baselineデータのロード
    baseline_h5 = baseline_dir / "hils_data.h5"
    if not baseline_h5.exists():
        print(f"❌ Error: Baseline HDF5 file not found: {baseline_h5}")
        return

    baseline_data = load_simulation_data(baseline_h5)
    if baseline_data is None:
        print(f"❌ Error: Failed to load baseline data from {baseline_h5}")
        return

    print(f"   Loaded baseline data: {len(baseline_data['time'])} time steps")

    # 各シミュレーションのデータをロード
    sim_data_list = []
    sim_names = []
    sim_configs = []

    print("\n📊 Loading simulation data...")
    for subdir in sorted(subdirs):
        # baselineはスキップ
        if subdir == baseline_dir:
            continue

        h5_file = subdir / "hils_data.h5"
        if not h5_file.exists():
            print(f"⚠️  Skipping {subdir.name}: No hils_data.h5 found")
            continue

        # データのロード
        sim_data = load_simulation_data(h5_file)
        if sim_data is None:
            print(f"⚠️  Skipping {subdir.name}: Failed to load data")
            continue

        # 設定のロード
        sim_config = load_simulation_config(subdir)

        sim_data_list.append(sim_data)
        sim_names.append(subdir.name)
        sim_configs.append(sim_config)
        print(f"   ✓ Loaded: {subdir.name}")

    if len(sim_data_list) == 0:
        print("❌ Error: No valid simulation data found")
        return

    # 統一プロットを作成
    output_path = sweep_dir / "unified_comparison.png"
    print("\n📈 Creating unified comparison plot...")
    try:
        create_unified_comparison_plot(baseline_data, sim_data_list, sim_names, sim_configs, output_path)
    except Exception as e:
        print(f"❌ Error creating unified plot: {e}")
        import traceback

        traceback.print_exc()
        return

    print(f"\n✅ Done! Processed {len(sim_data_list)} simulations")
    print(f"   Unified comparison plot saved to: {output_path}")


if __name__ == "__main__":
    main()
