"""
Interactive HILS Simulation Results Viewer

marimoベースの対話的な結果可視化アプリケーション。
複数のシミュレーション結果を選択して比較できる。

使用方法:
    cd /home/akira/mosaik-hils
    marimo edit hils_simulation/visualize_results.py
"""

import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import json
    import os
    from pathlib import Path

    import h5py
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    return Path, h5py, json, np, os, plt


@app.cell
def _(Path, json, os):
    """結果ディレクトリのスキャン"""

    # カレントディレクトリを基準にパスを設定
    cwd = Path(os.getcwd())

    # hils_simulationディレクトリを探す
    if cwd.name == "hils_simulation":
        base_dir = cwd
    elif (cwd / "hils_simulation").exists():
        # プロジェクトルートから実行された場合
        base_dir = cwd / "hils_simulation"
    else:
        # それでも見つからない場合は親ディレクトリを確認
        base_dir = cwd.parent / "hils_simulation" if cwd.parent.name != cwd.name else cwd

    all_results = []

    # 3つのディレクトリをスキャン
    for dir_type, dir_name in [
        ("HILS", "results"),
        ("RT", "results_rt"),
        ("Pure", "results_pure"),
    ]:
        result_dir = base_dir / dir_name
        if not result_dir.exists():
            continue

        for subdir in sorted(result_dir.iterdir(), reverse=True):
            if not subdir.is_dir():
                continue

            h5_file = subdir / "hils_data.h5"
            if not h5_file.exists():
                continue

            # 設定ファイルの読み込み
            config_file = subdir / "simulation_config.json"
            config = {}
            if config_file.exists():
                with open(config_file, "r") as f:
                    config = json.load(f)

            # 遅延情報
            comm = config.get("communication", {})
            cmd_delay = comm.get("cmd_delay_s", 0) * 1000
            sense_delay = comm.get("sense_delay_s", 0) * 1000

            if cmd_delay == 0 and sense_delay == 0:
                delay_info = "No delay"
            else:
                delay_info = f"Cmd:{cmd_delay:.0f}ms, Sense:{sense_delay:.0f}ms"

            label = f"[{dir_type}] {subdir.name} ({delay_info})"

            all_results.append(
                {
                    "type": dir_type,
                    "name": subdir.name,
                    "h5_file": str(h5_file),
                    "config": config,
                    "label": label,
                }
            )
    return all_results, base_dir


@app.cell
def _(all_results, base_dir, mo):
    """ヘッダー表示"""
    if not all_results:
        title = mo.md(
            f"""
    ⚠️ **No results found**

    Base directory: `{base_dir}`

    Check these directories:
    - `{base_dir}/results/`
    - `{base_dir}/results_rt/`
    - `{base_dir}/results_pure/`
    """
        )
    else:
        result_list_md = "\n".join([f"- {r['label']}" for r in all_results[:5]])
        more_md = f"\n- ... and {len(all_results) - 5} more" if len(all_results) > 5 else ""

        title = mo.md(
            f"""
    # HILS Simulation Results Viewer

    Found **{len(all_results)}** simulation results

    _Base directory: `{base_dir}`_

    **Available results:**
    {result_list_md}{more_md}

    ---

    **Select up to 3 results to compare:**
    """
        )
    return (title,)


@app.cell
def _(title):
    """タイトル表示"""
    title
    return


@app.cell
def _(all_results, mo):
    """ドロップダウン作成"""
    if len(all_results) == 0:
        dd1 = None
        dd2 = None
        dd3 = None
        dd_ui = None
    else:
        # オプション: ラベル -> インデックス
        opts = {r["label"]: i for i, r in enumerate(all_results)}

        # 最初と2番目のラベルを取得
        labels = list(opts.keys())
        first_label = labels[0] if len(labels) > 0 else None
        second_label = labels[1] if len(labels) > 1 else "(None)"

        dd1 = mo.ui.dropdown(opts, value=first_label, label="📊 Result 1")

        dd2 = mo.ui.dropdown({**{"(None)": -1}, **opts}, value=second_label, label="📊 Result 2")

        dd3 = mo.ui.dropdown({**{"(None)": -1}, **opts}, value="(None)", label="📊 Result 3")

        dd_ui = mo.vstack([dd1, dd2, dd3])
    return dd1, dd2, dd3, dd_ui


@app.cell
def _(dd_ui):
    """ドロップダウン表示"""
    dd_ui
    return


@app.cell
def _(all_results, dd1, dd2, dd3):
    """選択された結果の取得"""
    selected_results = []
    if len(all_results) > 0 and dd1 is not None:
        for dd in [dd1, dd2, dd3]:
            if dd is not None and dd.value is not None:
                idx = dd.value
                if isinstance(idx, int) and idx >= 0 and idx < len(all_results):
                    selected_results.append(all_results[idx])
    return (selected_results,)


@app.cell
def _(h5py, mo, np, plt, selected_results):
    """プロット生成"""

    def generate_comparison_plot(results_list):
        """比較プロットを生成する関数"""

        def load_hdf5_data(h5_path):
            """HDF5ファイルからデータを読み込む"""
            hdf5_data = {}
            with h5py.File(h5_path, "r") as f:
                if "data" in f:
                    for key in f["data"].keys():
                        hdf5_data[key] = f["data"][key][:]
                else:
                    for key in f.keys():
                        if isinstance(f[key], h5py.Dataset):
                            hdf5_data[key] = f[key][:]
            return hdf5_data

        def find_key_by_suffix(key_data, suffix):
            """キーの接尾辞でデータセットキーを検索"""
            for k in key_data.keys():
                if k.endswith(suffix):
                    return k
            return None

        def find_key_by_prefix_and_suffix(key_data, prefix, suffix):
            """プレフィックスとサフィックスでデータセットキーを検索"""
            for k in key_data.keys():
                if k.startswith(prefix) and k.endswith(suffix):
                    return k
            return None

        if len(results_list) == 0:
            return mo.md("⚠️ Please select at least one result")

        # プロット作成
        fig, axes = plt.subplots(4, 1, figsize=(14, 16))
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
        styles = ["-", "--", ":"]

        for plot_idx, result_item in enumerate(results_list):
            try:
                result_data = load_hdf5_data(result_item["h5_file"])
                time_data = result_data.get("time_s", np.array([]))

                # データキーの検索（compare_all.py と同じロジック）

                # HILS/RT の場合: position_EnvSim-0.Spacecraft1DOF_0
                result_pos_key = find_key_by_prefix_and_suffix(
                    result_data, "position_", "Spacecraft1DOF_0"
                )

                # Pure Python の場合: position_Spacecraft
                if not result_pos_key:
                    result_pos_key = find_key_by_suffix(result_data, "position_Spacecraft")

                if not result_pos_key:
                    print(f"⚠️ Warning: Could not find position key for {result_item['name']}")
                    print(f"   Available keys: {list(result_data.keys())[:10]}")
                    continue

                # Velocity: position を velocity に置き換え
                result_vel_key = result_pos_key.replace("position", "velocity")

                # Thrust: command_..._thrust (compare_all.py と同じ)
                result_thrust_key = find_key_by_suffix(result_data, "_thrust")

                # Error: error_..._Controller...
                result_error_key = find_key_by_prefix_and_suffix(
                    result_data, "error_", "Controller_0"
                )
                if not result_error_key:
                    result_error_key = find_key_by_suffix(result_data, "error_Controller")

                # データ取得
                result_position = result_data.get(result_pos_key, np.array([]))
                result_velocity = result_data.get(result_vel_key, np.array([]))
                result_thrust = result_data.get(result_thrust_key, np.array([]))
                result_error = result_data.get(result_error_key, np.array([]))

                # デバッグ情報
                print(f"\n{result_item['name']}:")
                print(f"  pos_key: {result_pos_key} (len={len(result_position)})")
                print(f"  vel_key: {result_vel_key} (len={len(result_velocity)})")
                print(f"  thrust_key: {result_thrust_key} (len={len(result_thrust)})")
                print(f"  error_key: {result_error_key} (len={len(result_error)})")

                result_label = result_item["label"]
                result_color = colors[plot_idx % len(colors)]
                result_style = styles[plot_idx % len(styles)]

                # プロット
                if len(result_position) > 0:
                    axes[0].plot(
                        time_data,
                        result_position,
                        color=result_color,
                        ls=result_style,
                        label=result_label,
                        lw=1.5,
                        alpha=0.8,
                    )
                else:
                    print(f"  ⚠️ No position data")

                if len(result_velocity) > 0:
                    axes[1].plot(
                        time_data,
                        result_velocity,
                        color=result_color,
                        ls=result_style,
                        label=result_label,
                        lw=1.5,
                        alpha=0.8,
                    )
                else:
                    print(f"  ⚠️ No velocity data")

                if len(result_thrust) > 0:
                    axes[2].plot(
                        time_data,
                        result_thrust,
                        color=result_color,
                        ls=result_style,
                        label=result_label,
                        lw=1.5,
                        alpha=0.8,
                    )
                else:
                    print(f"  ⚠️ No thrust data")

                if len(result_error) > 0:
                    axes[3].plot(
                        time_data,
                        result_error,
                        color=result_color,
                        ls=result_style,
                        label=result_label,
                        lw=1.5,
                        alpha=0.8,
                    )
                else:
                    print(f"  ⚠️ No error data")

            except Exception as exc:
                import traceback

                print(f"\n❌ Error loading {result_item['name']}:")
                print(f"  {exc}")
                print(traceback.format_exc())

        # 軸設定
        plot_target = 5.0
        if len(results_list) > 0:
            plot_target = results_list[0]["config"].get("control", {}).get("target_position_m", 5.0)

        axes[0].axhline(plot_target, color="k", linestyle=":", label="Target", lw=1)
        axes[0].set_ylabel("Position [m]", fontsize=11)
        axes[0].set_title("Position Comparison", fontsize=13, fontweight="bold")
        axes[0].legend(fontsize=9, loc="best")
        axes[0].grid(True, alpha=0.3)

        axes[1].set_ylabel("Velocity [m/s]", fontsize=11)
        axes[1].set_title("Velocity Comparison", fontsize=13, fontweight="bold")
        axes[1].legend(fontsize=9, loc="best")
        axes[1].grid(True, alpha=0.3)

        axes[2].set_ylabel("Thrust [N]", fontsize=11)
        axes[2].set_title("Control Input Comparison", fontsize=13, fontweight="bold")
        axes[2].legend(fontsize=9, loc="best")
        axes[2].grid(True, alpha=0.3)

        axes[3].set_xlabel("Time [s]", fontsize=11)
        axes[3].set_ylabel("Position Error [m]", fontsize=11)
        axes[3].set_title("Control Error Comparison", fontsize=13, fontweight="bold")
        axes[3].axhline(0, color="k", linestyle=":", lw=1)
        axes[3].legend(fontsize=9, loc="best")
        axes[3].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    # 関数を呼び出してプロットを生成
    plot_fig = generate_comparison_plot(selected_results)
    return (plot_fig,)


@app.cell
def _(plot_fig):
    """プロット表示"""
    plot_fig
    return


if __name__ == "__main__":
    app.run()
