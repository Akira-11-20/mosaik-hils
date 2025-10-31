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
    import plotly.graph_objects as go
    return Path, go, h5py, json, np, os, pd, plt


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

            # 逆補償情報
            inv_comp = config.get("inverse_compensation", {})
            if inv_comp.get("enabled", False):
                alpha = inv_comp.get("gain", 0)
                delay_info = f"{delay_info}, α={alpha:.1f}"

            # Plant遅延情報
            plant = config.get("plant", {})
            plant_enabled = plant.get("enable_lag", True)
            plant_tau = plant.get("time_constant_s", 0) * 1000 if plant_enabled else None

            if plant_tau is not None and plant_tau > 0:
                delay_info = f"{delay_info}, Plant-τ:{plant_tau:.0f}ms"
            elif not plant_enabled:
                delay_info = f"{delay_info}, Plant:ideal"

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
    """ヘッダー表示とプロット数選択"""
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
        num_plots_selector = None
    else:
        result_list_md = "\n".join([f"- {r['label']}" for r in all_results[:5]])
        more_md = f"\n- ... and {len(all_results) - 5} more" if len(all_results) > 5 else ""

        # プロット数選択（数値入力）
        max_selectable = min(len(all_results), 10)  # 最大10個まで
        num_plots_selector = mo.ui.number(
            start=1,
            stop=max_selectable,
            step=1,
            value=min(4, max_selectable),
            label=f"Number of results to plot (1-{max_selectable})",
        )

        title = mo.md(
            f"""
    # HILS Simulation Results Viewer

    Found **{len(all_results)}** simulation results

    _Base directory: `{base_dir}`_

    **Available results:**
    {result_list_md}{more_md}

    ---

    **Select how many results to compare:**
    """
        )
    return num_plots_selector, title


@app.cell
def _(mo, num_plots_selector, title):
    """タイトル表示"""
    title_display = mo.vstack([title, num_plots_selector] if num_plots_selector else [title])
    title_display
    return


@app.cell
def _(all_results, mo, num_plots_selector):
    """ドロップダウン作成（動的）"""
    if len(all_results) == 0 or num_plots_selector is None:
        result_dropdowns_array = None
    else:
        # プロット数を取得
        num_plots = num_plots_selector.value if num_plots_selector.value else 4

        # オプション: ラベル -> インデックス
        opts = {r["label"]: i for i, r in enumerate(all_results)}
        opts_with_none = {**{"(None)": -1}, **opts}

        # 最新のラベル（降順でソートされている）
        labels = list(opts.keys())

        # ヘルパー関数でドロップダウンを作成
        def _create_dropdown(index):
            default_val = labels[index] if index < len(labels) else "(None)"
            if index == 0:
                return mo.ui.dropdown(opts, value=default_val, label=f"📊 Result {index+1}")
            else:
                return mo.ui.dropdown(opts_with_none, value=default_val, label=f"📊 Result {index+1} (optional)")

        # mo.ui.arrayでラップしてリアクティビティを確保
        dropdowns_list = [_create_dropdown(i) for i in range(num_plots)]
        result_dropdowns_array = mo.ui.array(dropdowns_list)
    return (result_dropdowns_array,)


@app.cell
def _(result_dropdowns_array):
    """ドロップダウン表示"""
    result_dropdowns_array
    return


@app.cell
def _(all_results, result_dropdowns_array):
    """選択された結果の取得"""
    # mo.ui.arrayの.valueを使ってリアクティビティを確保
    selected_results = []
    if len(all_results) > 0 and result_dropdowns_array is not None:
        # mo.ui.arrayの.valueは各要素の値のリストを返す
        dropdown_values = result_dropdowns_array.value

        # 有効な値のみフィルタリング
        selected_results = [
            all_results[val]
            for val in dropdown_values
            if val is not None
            and isinstance(val, int)
            and val >= 0
            and val < len(all_results)
        ]
    return (selected_results,)


@app.cell
def _(np):
    """メトリクス計算関数"""

    def calculate_detailed_metrics(
        time: np.ndarray,
        position: np.ndarray,
        velocity: np.ndarray,
        error: np.ndarray,
        thrust: np.ndarray,
        target_position: float,
    ):
        """
        詳細な制御性能指標を計算

        Returns:
            Dict with performance metrics
        """
        metrics = {}

        # 基本統計
        metrics["rms_error"] = np.sqrt(np.mean(error**2))
        metrics["max_error"] = np.max(np.abs(error))
        metrics["mean_abs_error"] = np.mean(np.abs(error))
        metrics["std_error"] = np.std(error)

        # オーバーシュート
        overshoot = np.max(position) - target_position
        metrics["overshoot"] = overshoot
        metrics["overshoot_percent"] = (overshoot / target_position) * 100 if target_position != 0 else 0

        # アンダーシュート
        undershoot = target_position - np.min(position)
        metrics["undershoot"] = undershoot
        metrics["undershoot_percent"] = (undershoot / target_position) * 100 if target_position != 0 else 0

        # 整定時間（誤差が5%以内に収まる時刻）
        settling_threshold = 0.05 * abs(target_position)
        settled_indices = np.where(np.abs(error) <= settling_threshold)[0]
        if len(settled_indices) > 0:
            for idx in settled_indices:
                if np.all(np.abs(error[idx:]) <= settling_threshold):
                    metrics["settling_time_5pct"] = time[idx]
                    break
            else:
                metrics["settling_time_5pct"] = None
        else:
            metrics["settling_time_5pct"] = None

        # 整定時間（2%基準）
        settling_threshold_2 = 0.02 * abs(target_position)
        settled_indices_2 = np.where(np.abs(error) <= settling_threshold_2)[0]
        if len(settled_indices_2) > 0:
            for idx in settled_indices_2:
                if np.all(np.abs(error[idx:]) <= settling_threshold_2):
                    metrics["settling_time_2pct"] = time[idx]
                    break
            else:
                metrics["settling_time_2pct"] = None
        else:
            metrics["settling_time_2pct"] = None

        # 最終誤差（最後の10%の平均）
        final_window = max(int(len(error) * 0.1), 1)
        metrics["final_error"] = np.mean(np.abs(error[-final_window:]))
        metrics["final_std"] = np.std(error[-final_window:])

        # 立ち上がり時間（目標の10%から90%に到達する時間）
        pos_10pct = 0.1 * target_position
        pos_90pct = 0.9 * target_position
        idx_10 = np.where(position >= pos_10pct)[0]
        idx_90 = np.where(position >= pos_90pct)[0]
        if len(idx_10) > 0 and len(idx_90) > 0:
            metrics["rise_time"] = time[idx_90[0]] - time[idx_10[0]]
        else:
            metrics["rise_time"] = None

        # ピーク時間
        peak_idx = np.argmax(position)
        metrics["peak_time"] = time[peak_idx]
        metrics["peak_value"] = position[peak_idx]

        # 制御入力の統計
        metrics["mean_thrust"] = np.mean(np.abs(thrust))
        metrics["max_thrust"] = np.max(np.abs(thrust))
        metrics["thrust_variation"] = np.std(thrust)

        # 制御入力の総変化量（Total Variation）
        if len(thrust) > 1:
            metrics["control_effort"] = np.sum(np.abs(np.diff(thrust)))
        else:
            metrics["control_effort"] = 0

        # 速度の統計
        metrics["max_velocity"] = np.max(np.abs(velocity))
        metrics["final_velocity"] = np.mean(np.abs(velocity[-final_window:]))

        # ISE (Integral of Squared Error)
        if len(time) > 1:
            dt = time[1] - time[0]
            metrics["ise"] = np.sum(error**2) * dt
        else:
            metrics["ise"] = 0

        # IAE (Integral of Absolute Error)
        if len(time) > 1:
            dt = time[1] - time[0]
            metrics["iae"] = np.sum(np.abs(error)) * dt
        else:
            metrics["iae"] = 0

        # ITAE (Integral of Time-weighted Absolute Error)
        if len(time) > 1:
            dt = time[1] - time[0]
            metrics["itae"] = np.sum(time * np.abs(error)) * dt
        else:
            metrics["itae"] = 0

        return metrics
    return (calculate_detailed_metrics,)


@app.cell
def _(h5py):
    """HDF5データ読み込み関数（共通）"""

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
                            # 例: BridgeSim-0_CommBridge_0/buffer_size -> buffer_size_BridgeSim-0_CommBridge_0
                            flat_key = f"{item.name.replace('/', '_')}"
                            if flat_key.startswith("_"):
                                flat_key = flat_key[1:]
                            # 階層を逆にしてフラット化: group_name/attr -> attr_group_name
                            parts = item.name.split("/")
                            if len(parts) >= 2:
                                # /group_name/attr_name -> attr_name_group_name
                                group_name = parts[1]
                                attr_name = parts[-1]
                                flat_key = f"{attr_name}_{group_name}" if group_name != "time" else attr_name
                            hdf5_data[flat_key] = item[:]

                read_group(f)
        return hdf5_data
    return (load_hdf5_data,)


@app.cell
def _(
    calculate_detailed_metrics,
    load_hdf5_data,
    mo,
    np,
    plt,
    selected_results,
):
    """プロット生成とメトリクス計算"""

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

    def generate_comparison_plot_and_metrics(results_list):
        """比較プロットとメトリクスを生成する関数"""

        if len(results_list) == 0:
            return mo.md("⚠️ Please select at least one result"), []

        # メトリクスを保存するリスト
        all_metrics = []

        # プロット作成（5つに増やす：Position, Velocity, Command Thrust, Measured Thrust, Error）
        fig, axes = plt.subplots(5, 1, figsize=(14, 20))
        # 10色に拡張（カラーマップから取得）
        colors = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
        ]
        # 線スタイルを拡張
        styles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2, 1, 2)), (0, (3, 5, 1, 5)), (0, (1, 1)), (0, (5, 1)), (0, (3, 1, 1, 1, 1, 1))]

        for plot_idx, result_item in enumerate(results_list):
            try:
                result_data = load_hdf5_data(result_item["h5_file"])
                time_data = result_data.get("time_s", np.array([])).copy()

                # データキーの検索（compare_all.py と同じロジック）

                # HILS/RT の場合: position_EnvSim-0.Spacecraft1DOF_0
                result_pos_key = find_key_by_prefix_and_suffix(result_data, "position_", "Spacecraft1DOF_0")

                # Pure Python の場合: position_Spacecraft
                if not result_pos_key:
                    result_pos_key = find_key_by_suffix(result_data, "position_Spacecraft")

                if not result_pos_key:
                    print(f"⚠️ Warning: Could not find position key for {result_item['name']}")
                    print(f"   Available keys: {list(result_data.keys())[:10]}")
                    del result_data  # データを削除
                    continue

                # Velocity: position を velocity に置き換え
                result_vel_key = result_pos_key.replace("position", "velocity")

                # Command Thrust: コントローラーからの指令
                # 優先順位: compensated_output_thrust (逆補償あり) > command_thrust (逆補償なし)
                result_cmd_thrust_key = None
                for k in result_data.keys():
                    if k.startswith("compensated_output") and k.endswith("_thrust"):
                        result_cmd_thrust_key = k
                        break
                if not result_cmd_thrust_key:
                    for k in result_data.keys():
                        if k.startswith("command_thrust"):
                            result_cmd_thrust_key = k
                            break

                # Actual Thrust: Plantで測定された実際のthrust（遅延後）
                result_actual_thrust_key = None
                for k in result_data.keys():
                    if k.startswith("actual_thrust") and "ThrustStand" in k:
                        result_actual_thrust_key = k
                        break
                if not result_actual_thrust_key:
                    for k in result_data.keys():
                        if k.startswith("actual_thrust"):
                            result_actual_thrust_key = k
                            break

                # Error: error_..._Controller...
                # Error key search
                result_error_key = None
                for k in result_data.keys():
                    if k.startswith("error_") and "Controller" in k:
                        result_error_key = k
                        break
                if not result_error_key:
                    for k in result_data.keys():
                        if "error" in k and "Controller" in k:
                            result_error_key = k
                            break

                # データ取得（コピーを作成して元のresult_dataを削除可能にする）
                result_position = result_data.get(result_pos_key, np.array([])).copy()
                result_velocity = result_data.get(result_vel_key, np.array([])).copy()
                result_cmd_thrust = result_data.get(result_cmd_thrust_key, np.array([])).copy()
                result_actual_thrust = result_data.get(result_actual_thrust_key, np.array([])).copy()
                result_error = result_data.get(result_error_key, np.array([])).copy()

                # 大きなresult_dataを即座に削除
                del result_data

                # デバッグ情報
                print(f"\n{result_item['name']}:")
                print(f"  pos_key: {result_pos_key} (len={len(result_position)})")
                print(f"  vel_key: {result_vel_key} (len={len(result_velocity)})")
                print(f"  cmd_thrust_key: {result_cmd_thrust_key} (len={len(result_cmd_thrust)})")
                print(f"  actual_thrust_key: {result_actual_thrust_key} (len={len(result_actual_thrust)})")
                print(f"  error_key: {result_error_key} (len={len(result_error)})")

                # 目標位置の取得
                target_position = result_item["config"].get("control", {}).get("target_position_m", 5.0)

                # メトリクス計算（command thrustを使用）
                if len(time_data) > 0 and len(result_position) > 0 and len(result_velocity) > 0 and len(result_error) > 0 and len(result_cmd_thrust) > 0:
                    metrics = calculate_detailed_metrics(
                        time_data,
                        result_position,
                        result_velocity,
                        result_error,
                        result_cmd_thrust,
                        target_position,
                    )
                    all_metrics.append(
                        {
                            "name": result_item["name"],
                            "label": result_item["label"],
                            "type": result_item["type"],
                            "metrics": metrics,
                        }
                    )

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

                # Command Thrust（コントローラー出力）
                if len(result_cmd_thrust) > 0:
                    axes[2].plot(
                        time_data,
                        result_cmd_thrust,
                        color=result_color,
                        ls=result_style,
                        label=result_label,
                        lw=1.5,
                        alpha=0.8,
                    )
                else:
                    print(f"  ⚠️ No command thrust data")

                # Actual Thrust（Plant入力、遅延後）
                if len(result_actual_thrust) > 0:
                    axes[3].plot(
                        time_data,
                        result_actual_thrust,
                        color=result_color,
                        ls=result_style,
                        label=result_label,
                        lw=1.5,
                        alpha=0.8,
                    )
                else:
                    print(f"  ⚠️ No actual thrust data")

                if len(result_error) > 0:
                    axes[4].plot(
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

        axes[2].set_ylabel("Command Thrust [N]", fontsize=11)
        axes[2].set_title("Command Thrust (Controller Output)", fontsize=13, fontweight="bold")
        axes[2].legend(fontsize=9, loc="best")
        axes[2].grid(True, alpha=0.3)

        axes[3].set_ylabel("Actual Thrust [N]", fontsize=11)
        axes[3].set_title("Actual Thrust (Plant Input, After Delay)", fontsize=13, fontweight="bold")
        axes[3].legend(fontsize=9, loc="best")
        axes[3].grid(True, alpha=0.3)

        axes[4].set_xlabel("Time [s]", fontsize=11)
        axes[4].set_ylabel("Position Error [m]", fontsize=11)
        axes[4].set_title("Control Error Comparison", fontsize=13, fontweight="bold")
        axes[4].axhline(0, color="k", linestyle=":", lw=1)
        axes[4].legend(fontsize=9, loc="best")
        axes[4].grid(True, alpha=0.3)

        plt.tight_layout()

        # Figureを閉じてメモリを解放
        # matplotlibはfigureを表示後も保持するため、明示的に閉じる
        # 注意: marimoはfigオブジェクトを表示する前にこれを実行するため、
        # returnする前に閉じてはいけない
        return fig, all_metrics

    # 関数を呼び出してプロットとメトリクスを生成
    if len(selected_results) > 0:
        plot_fig, computed_metrics = generate_comparison_plot_and_metrics(selected_results)
    else:
        plot_fig = None
        computed_metrics = []
    return computed_metrics, plot_fig


@app.cell
def _(mo, plot_fig):
    """プロット表示"""
    if plot_fig is not None:
        # matplotlibのfigureをそのまま表示
        plot_display = plot_fig
    else:
        plot_display = mo.md("_No results selected for plotting._")
    return (plot_display,)


@app.cell
def _(plot_display):
    """プロット描画"""
    plot_display
    return


@app.cell
def _(mo):
    """インタラクティブプロットセクションヘッダー"""
    interactive_header = mo.md(
        """
    ---

    ## 📈 Interactive Plot Explorer

    Select a plot type to view an interactive version with zoom, pan, and hover capabilities.
        """
    )
    return (interactive_header,)


@app.cell
def _(interactive_header):
    """インタラクティブヘッダー表示"""
    interactive_header
    return


@app.cell
def _(mo):
    """プロット選択ドロップダウン"""
    plot_selector = mo.ui.dropdown(
        {
            "Position": "position",
            "Velocity": "velocity",
            "Command Thrust (Controller Output)": "cmd_thrust",
            "Actual Thrust (Plant Input, After Delay)": "actual_thrust",
            "Position Error": "error",
        },
        value="Position",
        label="Select Plot Type",
    )
    return (plot_selector,)


@app.cell
def _(plot_selector):
    """プロット選択表示"""
    plot_selector
    return


@app.cell
def _(go, load_hdf5_data, mo, np, plot_selector, selected_results):
    """インタラクティブプロット生成"""

    def generate_interactive_plot(results_list, plot_type):
        """Plotlyでインタラクティブなプロットを生成"""

        if len(results_list) == 0:
            return mo.md("_No results selected for interactive plot._")

        fig = go.Figure()

        # 10色に拡張
        colors = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
        ]
        # 線スタイルを拡張
        styles = ["solid", "dash", "dot", "dashdot", "longdash", "longdashdot",
                  (5, (10, 3)), (0, (5, 5)), (0, (3, 1, 1, 1)), (0, (1, 1))]

        # データキー検索用の関数
        def find_key_by_suffix(key_data, suffix):
            for k in key_data.keys():
                if k.endswith(suffix):
                    return k
            return None

        def find_key_by_prefix_and_suffix(key_data, prefix, suffix):
            for k in key_data.keys():
                if k.startswith(prefix) and k.endswith(suffix):
                    return k
            return None

        for plot_idx, result_item in enumerate(results_list):
            try:
                result_data = load_hdf5_data(result_item["h5_file"])
                time_data = result_data.get("time_s", np.array([])).copy()

                # データキーの検索
                result_pos_key = find_key_by_prefix_and_suffix(result_data, "position_", "Spacecraft1DOF_0")
                if not result_pos_key:
                    result_pos_key = find_key_by_suffix(result_data, "position_Spacecraft")

                if not result_pos_key:
                    del result_data
                    continue

                result_vel_key = result_pos_key.replace("position", "velocity")

                # Command Thrust: コントローラーからの指令
                result_cmd_thrust_key = None
                for k in result_data.keys():
                    if k.startswith("compensated_output") and k.endswith("_thrust"):
                        result_cmd_thrust_key = k
                        break
                if not result_cmd_thrust_key:
                    for k in result_data.keys():
                        if k.startswith("command_thrust"):
                            result_cmd_thrust_key = k
                            break

                # Actual Thrust: Plantで測定された実際のthrust（遅延後）
                result_actual_thrust_key = None
                for k in result_data.keys():
                    if k.startswith("actual_thrust") and "ThrustStand" in k:
                        result_actual_thrust_key = k
                        break
                if not result_actual_thrust_key:
                    for k in result_data.keys():
                        if k.startswith("actual_thrust"):
                            result_actual_thrust_key = k
                            break

                # Error key search
                result_error_key = None
                for k in result_data.keys():
                    if k.startswith("error_") and "Controller" in k:
                        result_error_key = k
                        break
                if not result_error_key:
                    for k in result_data.keys():
                        if "error" in k and "Controller" in k:
                            result_error_key = k
                            break

                # プロットタイプに応じてデータを選択（コピーを作成）
                if plot_type == "position":
                    y_data = result_data.get(result_pos_key, np.array([])).copy()
                    y_label = "Position [m]"
                    title = "Position Comparison (Interactive)"
                elif plot_type == "velocity":
                    y_data = result_data.get(result_vel_key, np.array([])).copy()
                    y_label = "Velocity [m/s]"
                    title = "Velocity Comparison (Interactive)"
                elif plot_type == "cmd_thrust":
                    y_data = result_data.get(result_cmd_thrust_key, np.array([])).copy()
                    y_label = "Command Thrust [N]"
                    title = "Command Thrust (Controller Output) - Interactive"
                elif plot_type == "actual_thrust":
                    y_data = result_data.get(result_actual_thrust_key, np.array([])).copy()
                    y_label = "Actual Thrust [N]"
                    title = "Actual Thrust (Plant Input, After Delay) - Interactive"
                elif plot_type == "error":
                    y_data = result_data.get(result_error_key, np.array([])).copy()
                    y_label = "Position Error [m]"
                    title = "Position Error Comparison (Interactive)"
                else:
                    del result_data
                    continue

                # result_dataを削除
                del result_data

                if len(y_data) == 0:
                    continue

                # プロットに追加
                fig.add_trace(
                    go.Scatter(
                        x=time_data,
                        y=y_data,
                        mode="lines",
                        name=result_item["label"],
                        line=dict(
                            color=colors[plot_idx % len(colors)],
                            width=2,
                            dash=styles[plot_idx % len(styles)],
                        ),
                        hovertemplate="<b>%{fullData.name}</b><br>" + "Time: %{x:.4f} s<br>" + f"{y_label}: " + "%{y:.6f}<br>" + "<extra></extra>",
                    )
                )

            except Exception as exc:
                print(f"Error loading {result_item['name']}: {exc}")

        # 目標線を追加（位置プロットの場合）
        if plot_type == "position" and len(results_list) > 0:
            target_position = results_list[0]["config"].get("control", {}).get("target_position_m", 5.0)
            fig.add_hline(
                y=target_position,
                line_dash="dot",
                line_color="black",
                annotation_text="Target",
                annotation_position="right",
            )

        # ゼロ線を追加（誤差プロットの場合）
        if plot_type == "error":
            fig.add_hline(y=0, line_dash="dot", line_color="black", line_width=1)

        # レイアウト設定
        fig.update_layout(
            title=dict(text=title, font=dict(size=16, weight="bold")),
            xaxis_title="Time [s]",
            yaxis_title=y_label,
            hovermode="x unified",
            template="plotly_white",
            height=500,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            margin=dict(l=60, r=30, t=50, b=50),
        )

        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="LightGray")
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="LightGray")

        return fig

    # インタラクティブプロットを生成
    if plot_selector.value and selected_results:
        interactive_fig = generate_interactive_plot(selected_results, plot_selector.value)
    else:
        interactive_fig = mo.md("_Select results and a plot type above._")
    return (interactive_fig,)


@app.cell
def _(interactive_fig):
    """インタラクティブプロット表示"""
    interactive_fig
    return


@app.cell
def _(computed_metrics, mo, pd):
    """メトリクス表示"""

    if not computed_metrics or len(computed_metrics) == 0:
        metrics_display = mo.md("_No metrics available. Please select at least one result._")
        error_table = None
        transient_table = None
        integral_table = None
        control_table = None
        relative_table = None
    else:
        # メトリクスをDataFrameに変換（数値型で保持）
        rows = []
        for item in computed_metrics:
            metrics = item["metrics"]
            row = {
                "Simulation": item["label"],
                "Type": item["type"],
                "RMS Error": metrics["rms_error"],
                "Max Error": metrics["max_error"],
                "Mean |Error|": metrics["mean_abs_error"],
                "Std Error": metrics["std_error"],
                "Final Error": metrics["final_error"],
                "Overshoot": metrics["overshoot"],
                "Overshoot %": metrics["overshoot_percent"],
                "Rise Time": metrics["rise_time"] if metrics["rise_time"] is not None else None,
                "Settling 5%": (metrics["settling_time_5pct"] if metrics["settling_time_5pct"] is not None else None),
                "Settling 2%": (metrics["settling_time_2pct"] if metrics["settling_time_2pct"] is not None else None),
                "ISE": metrics["ise"],
                "IAE": metrics["iae"],
                "ITAE": metrics["itae"],
                "Max Thrust": metrics["max_thrust"],
                "Mean |Thrust|": metrics["mean_thrust"],
                "Control Effort": metrics["control_effort"],
                "Max Velocity": metrics["max_velocity"],
            }
            rows.append(row)

        df = pd.DataFrame(rows)

        # 各カテゴリの表を作成（数値は適切にフォーマット）
        error_table = df[["Simulation", "RMS Error", "Max Error", "Mean |Error|", "Std Error", "Final Error"]].copy()
        error_table["RMS Error"] = error_table["RMS Error"].apply(lambda x: f"{x:.6f}")
        error_table["Max Error"] = error_table["Max Error"].apply(lambda x: f"{x:.6f}")
        error_table["Mean |Error|"] = error_table["Mean |Error|"].apply(lambda x: f"{x:.6f}")
        error_table["Std Error"] = error_table["Std Error"].apply(lambda x: f"{x:.6f}")
        error_table["Final Error"] = error_table["Final Error"].apply(lambda x: f"{x:.6f}")

        transient_table = df[["Simulation", "Overshoot", "Overshoot %", "Rise Time", "Settling 5%", "Settling 2%"]].copy()
        transient_table["Overshoot"] = transient_table["Overshoot"].apply(lambda x: f"{x:.6f}")
        transient_table["Overshoot %"] = transient_table["Overshoot %"].apply(lambda x: f"{x:.2f}%")
        transient_table["Rise Time"] = transient_table["Rise Time"].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
        transient_table["Settling 5%"] = transient_table["Settling 5%"].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
        transient_table["Settling 2%"] = transient_table["Settling 2%"].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")

        integral_table = df[["Simulation", "ISE", "IAE", "ITAE"]].copy()
        integral_table["ISE"] = integral_table["ISE"].apply(lambda x: f"{x:.6f}")
        integral_table["IAE"] = integral_table["IAE"].apply(lambda x: f"{x:.6f}")
        integral_table["ITAE"] = integral_table["ITAE"].apply(lambda x: f"{x:.6f}")

        control_table = df[["Simulation", "Max Thrust", "Mean |Thrust|", "Control Effort", "Max Velocity"]].copy()
        control_table["Max Thrust"] = control_table["Max Thrust"].apply(lambda x: f"{x:.4f}")
        control_table["Mean |Thrust|"] = control_table["Mean |Thrust|"].apply(lambda x: f"{x:.4f}")
        control_table["Control Effort"] = control_table["Control Effort"].apply(lambda x: f"{x:.4f}")
        control_table["Max Velocity"] = control_table["Max Velocity"].apply(lambda x: f"{x:.6f}")

        # 相対比較テーブル（拡充版）
        if len(computed_metrics) >= 2:
            ref_metrics = computed_metrics[0]["metrics"]

            # 誤差メトリクスの比較
            error_rel_rows = []
            for item in computed_metrics[1:]:
                comp_metrics = item["metrics"]
                rel_row = {"Simulation": item["label"]}

                # RMS Error
                if ref_metrics["rms_error"] != 0:
                    rel_row["RMS Error Δ%"] = f"{((comp_metrics['rms_error'] - ref_metrics['rms_error']) / ref_metrics['rms_error'] * 100):+.2f}%"
                else:
                    rel_row["RMS Error Δ%"] = "N/A"

                # Max Error
                if ref_metrics["max_error"] != 0:
                    rel_row["Max Error Δ%"] = f"{((comp_metrics['max_error'] - ref_metrics['max_error']) / ref_metrics['max_error'] * 100):+.2f}%"
                else:
                    rel_row["Max Error Δ%"] = "N/A"

                # Mean Abs Error
                if ref_metrics["mean_abs_error"] != 0:
                    rel_row["Mean |Error| Δ%"] = f"{((comp_metrics['mean_abs_error'] - ref_metrics['mean_abs_error']) / ref_metrics['mean_abs_error'] * 100):+.2f}%"
                else:
                    rel_row["Mean |Error| Δ%"] = "N/A"

                # Final Error
                if ref_metrics["final_error"] != 0:
                    rel_row["Final Error Δ%"] = f"{((comp_metrics['final_error'] - ref_metrics['final_error']) / ref_metrics['final_error'] * 100):+.2f}%"
                else:
                    rel_row["Final Error Δ%"] = "N/A"

                error_rel_rows.append(rel_row)

            error_relative_table = pd.DataFrame(error_rel_rows) if error_rel_rows else None

            # 過渡応答の比較
            transient_rel_rows = []
            for item in computed_metrics[1:]:
                comp_metrics = item["metrics"]
                rel_row = {"Simulation": item["label"]}

                # Overshoot
                if ref_metrics["overshoot"] != 0:
                    rel_row["Overshoot Δ%"] = f"{((comp_metrics['overshoot'] - ref_metrics['overshoot']) / ref_metrics['overshoot'] * 100):+.2f}%"
                else:
                    rel_row["Overshoot Δ%"] = "N/A"

                # Rise Time
                if ref_metrics["rise_time"] is not None and comp_metrics["rise_time"] is not None and ref_metrics["rise_time"] != 0:
                    rel_row["Rise Time Δ%"] = f"{((comp_metrics['rise_time'] - ref_metrics['rise_time']) / ref_metrics['rise_time'] * 100):+.2f}%"
                else:
                    rel_row["Rise Time Δ%"] = "N/A"

                # Settling 5%
                if ref_metrics["settling_time_5pct"] is not None and comp_metrics["settling_time_5pct"] is not None and ref_metrics["settling_time_5pct"] != 0:
                    rel_row["Settling 5% Δ%"] = f"{((comp_metrics['settling_time_5pct'] - ref_metrics['settling_time_5pct']) / ref_metrics['settling_time_5pct'] * 100):+.2f}%"
                else:
                    rel_row["Settling 5% Δ%"] = "N/A"

                # Settling 2%
                if ref_metrics["settling_time_2pct"] is not None and comp_metrics["settling_time_2pct"] is not None and ref_metrics["settling_time_2pct"] != 0:
                    rel_row["Settling 2% Δ%"] = f"{((comp_metrics['settling_time_2pct'] - ref_metrics['settling_time_2pct']) / ref_metrics['settling_time_2pct'] * 100):+.2f}%"
                else:
                    rel_row["Settling 2% Δ%"] = "N/A"

                transient_rel_rows.append(rel_row)

            transient_relative_table = pd.DataFrame(transient_rel_rows) if transient_rel_rows else None

            # 積分性能指標の比較
            integral_rel_rows = []
            for item in computed_metrics[1:]:
                comp_metrics = item["metrics"]
                rel_row = {"Simulation": item["label"]}

                # ISE
                if ref_metrics["ise"] != 0:
                    rel_row["ISE Δ%"] = f"{((comp_metrics['ise'] - ref_metrics['ise']) / ref_metrics['ise'] * 100):+.2f}%"
                else:
                    rel_row["ISE Δ%"] = "N/A"

                # IAE
                if ref_metrics["iae"] != 0:
                    rel_row["IAE Δ%"] = f"{((comp_metrics['iae'] - ref_metrics['iae']) / ref_metrics['iae'] * 100):+.2f}%"
                else:
                    rel_row["IAE Δ%"] = "N/A"

                # ITAE
                if ref_metrics["itae"] != 0:
                    rel_row["ITAE Δ%"] = f"{((comp_metrics['itae'] - ref_metrics['itae']) / ref_metrics['itae'] * 100):+.2f}%"
                else:
                    rel_row["ITAE Δ%"] = "N/A"

                integral_rel_rows.append(rel_row)

            integral_relative_table = pd.DataFrame(integral_rel_rows) if integral_rel_rows else None

            # 制御入力の比較
            control_rel_rows = []
            for item in computed_metrics[1:]:
                comp_metrics = item["metrics"]
                rel_row = {"Simulation": item["label"]}

                # Max Thrust
                if ref_metrics["max_thrust"] != 0:
                    rel_row["Max Thrust Δ%"] = f"{((comp_metrics['max_thrust'] - ref_metrics['max_thrust']) / ref_metrics['max_thrust'] * 100):+.2f}%"
                else:
                    rel_row["Max Thrust Δ%"] = "N/A"

                # Mean Thrust
                if ref_metrics["mean_thrust"] != 0:
                    rel_row["Mean |Thrust| Δ%"] = f"{((comp_metrics['mean_thrust'] - ref_metrics['mean_thrust']) / ref_metrics['mean_thrust'] * 100):+.2f}%"
                else:
                    rel_row["Mean |Thrust| Δ%"] = "N/A"

                # Control Effort
                if ref_metrics["control_effort"] != 0:
                    rel_row["Control Effort Δ%"] = f"{((comp_metrics['control_effort'] - ref_metrics['control_effort']) / ref_metrics['control_effort'] * 100):+.2f}%"
                else:
                    rel_row["Control Effort Δ%"] = "N/A"

                # Max Velocity
                if ref_metrics["max_velocity"] != 0:
                    rel_row["Max Velocity Δ%"] = f"{((comp_metrics['max_velocity'] - ref_metrics['max_velocity']) / ref_metrics['max_velocity'] * 100):+.2f}%"
                else:
                    rel_row["Max Velocity Δ%"] = "N/A"

                control_rel_rows.append(rel_row)

            control_relative_table = pd.DataFrame(control_rel_rows) if control_rel_rows else None

            # 統合用（後方互換性のため残す）
            relative_table = error_relative_table
        else:
            error_relative_table = None
            transient_relative_table = None
            integral_relative_table = None
            control_relative_table = None
            relative_table = None

        metrics_display = mo.md(
            f"""
    ## 📊 Performance Metrics Comparison

    **Total simulations compared:** {len(computed_metrics)}
        """
        )
    return (
        control_relative_table,
        control_table,
        error_relative_table,
        error_table,
        integral_relative_table,
        integral_table,
        metrics_display,
        transient_relative_table,
        transient_table,
    )


@app.cell
def _(metrics_display):
    """メトリクスヘッダー表示"""
    metrics_display
    return


@app.cell
def _(error_table, mo):
    """誤差メトリクステーブル"""
    if error_table is not None:
        error_section = mo.vstack(
            [
                mo.md("### 📉 Error Metrics"),
                mo.ui.table(error_table, selection=None),
            ]
        )
    else:
        error_section = None
    return (error_section,)


@app.cell
def _(error_section):
    """誤差テーブル表示"""
    error_section
    return


@app.cell
def _(mo, transient_table):
    """過渡応答テーブル"""
    if transient_table is not None:
        transient_section = mo.vstack(
            [
                mo.md("### ⚡ Transient Response"),
                mo.ui.table(transient_table, selection=None),
            ]
        )
    else:
        transient_section = None
    return (transient_section,)


@app.cell
def _(transient_section):
    """過渡応答テーブル表示"""
    transient_section
    return


@app.cell
def _(integral_table, mo):
    """積分性能指標テーブル"""
    if integral_table is not None:
        integral_section = mo.vstack(
            [
                mo.md("### 📐 Integral Performance Indices"),
                mo.ui.table(integral_table, selection=None),
            ]
        )
    else:
        integral_section = None
    return (integral_section,)


@app.cell
def _(integral_section):
    """積分性能指標テーブル表示"""
    integral_section
    return


@app.cell
def _(control_table, mo):
    """制御入力統計テーブル"""
    if control_table is not None:
        control_section = mo.vstack(
            [
                mo.md("### 🎮 Control Input Statistics"),
                mo.ui.table(control_table, selection=None),
            ]
        )
    else:
        control_section = None
    return (control_section,)


@app.cell
def _(control_section):
    """制御入力統計テーブル表示"""
    control_section
    return


@app.cell
def _(
    computed_metrics,
    control_relative_table,
    error_relative_table,
    integral_relative_table,
    mo,
    transient_relative_table,
):
    """相対比較テーブル（拡充版）"""
    if len(computed_metrics) >= 2:
        # ヘッダー
        relative_header = mo.md(
            f"""
    ---

    ## 🔍 Relative Performance Analysis

    **Baseline:** {computed_metrics[0]["label"]}

    All percentages show the change relative to the baseline (positive = worse for errors, varies for other metrics).
        """
        )

        # 誤差メトリクスの相対比較
        error_relative_section = None
        if error_relative_table is not None:
            error_relative_section = mo.vstack(
                [
                    mo.md("### 📉 Error Metrics - Relative Change"),
                    mo.ui.table(error_relative_table, selection=None),
                ]
            )

        # 過渡応答の相対比較
        transient_relative_section = None
        if transient_relative_table is not None:
            transient_relative_section = mo.vstack(
                [
                    mo.md("### ⚡ Transient Response - Relative Change"),
                    mo.ui.table(transient_relative_table, selection=None),
                ]
            )

        # 積分性能指標の相対比較
        integral_relative_section = None
        if integral_relative_table is not None:
            integral_relative_section = mo.vstack(
                [
                    mo.md("### 📐 Integral Performance Indices - Relative Change"),
                    mo.ui.table(integral_relative_table, selection=None),
                ]
            )

        # 制御入力の相対比較
        control_relative_section = None
        if control_relative_table is not None:
            control_relative_section = mo.vstack(
                [
                    mo.md("### 🎮 Control Input Statistics - Relative Change"),
                    mo.ui.table(control_relative_table, selection=None),
                ]
            )
    else:
        relative_header = None
        error_relative_section = None
        transient_relative_section = None
        integral_relative_section = None
        control_relative_section = None
    return (
        control_relative_section,
        error_relative_section,
        integral_relative_section,
        relative_header,
        transient_relative_section,
    )


@app.cell
def _(relative_header):
    """相対比較ヘッダー表示"""
    relative_header
    return


@app.cell
def _(error_relative_section):
    """誤差相対比較表示"""
    error_relative_section
    return


@app.cell
def _(transient_relative_section):
    """過渡応答相対比較表示"""
    transient_relative_section
    return


@app.cell
def _(integral_relative_section):
    """積分性能相対比較表示"""
    integral_relative_section
    return


@app.cell
def _(control_relative_section):
    """制御入力相対比較表示"""
    control_relative_section
    return


@app.cell
def _(mo):
    """時刻範囲指定比較セクションヘッダー"""
    time_range_header = mo.md(
        """
    ---

    ## ⏱️ Time Range Trajectory Comparison

    Select a time range to analyze detailed trajectory differences between two selected simulations.
        """
    )
    return (time_range_header,)


@app.cell
def _(time_range_header):
    """時刻範囲ヘッダー表示"""
    time_range_header
    return


@app.cell
def _(load_hdf5_data, np, selected_results):
    """時刻範囲の最大値を計算"""
    if len(selected_results) >= 1:
        # 最初の結果から最大時刻を取得
        try:
            first_h5 = selected_results[0]["h5_file"]
            first_data = load_hdf5_data(first_h5)
            time_data_first = first_data.get("time_s", np.array([]))

            if len(time_data_first) > 0:
                max_time_calc = float(time_data_first[-1])
                min_time_calc = float(time_data_first[0])
            else:
                max_time_calc = 2.0
                min_time_calc = 0.0

            # 明示的に大きなデータを削除
            del first_data
            del time_data_first
        except Exception:
            max_time_calc = 2.0
            min_time_calc = 0.0
    else:
        max_time_calc = 2.0
        min_time_calc = 0.0
    return max_time_calc, min_time_calc


@app.cell
def _(max_time_calc, min_time_calc, mo, selected_results):
    """時刻範囲選択UI"""
    if len(selected_results) >= 2:
        # 開始時刻の入力（スライダーと数値入力）
        time_start_slider = mo.ui.slider(
            start=min_time_calc,
            stop=max_time_calc,
            step=0.01,
            value=min_time_calc,
            label="Start Time [s]",
            show_value=True,
        )

        time_start_number = mo.ui.number(
            start=min_time_calc,
            stop=max_time_calc,
            step=0.001,
            value=min_time_calc,
            label="Start Time (precise)",
        )

        # 終了時刻の入力（スライダーと数値入力）
        time_end_slider = mo.ui.slider(
            start=min_time_calc,
            stop=max_time_calc,
            step=0.01,
            value=max_time_calc,
            label="End Time [s]",
            show_value=True,
        )

        time_end_number = mo.ui.number(
            start=min_time_calc,
            stop=max_time_calc,
            step=0.001,
            value=max_time_calc,
            label="End Time (precise)",
        )

        # レイアウト：スライダーと数値入力を横に並べる
        time_range_ui = mo.vstack(
            [
                mo.md("**Select time range for comparison:**"),
                mo.md(f"_Use **sliders** for quick selection or **number inputs** for precise values (0.001s precision)._\n\n_Available time range: {min_time_calc:.3f}s to {max_time_calc:.3f}s_"),
                mo.hstack([time_start_slider, time_start_number], justify="start", widths=[3, 1]),
                mo.hstack([time_end_slider, time_end_number], justify="start", widths=[3, 1]),
            ]
        )
    else:
        time_start_slider = None
        time_end_slider = None
        time_start_number = None
        time_end_number = None
        time_range_ui = mo.md("_Select at least 2 results to enable time range comparison._")
    return (
        time_end_number,
        time_end_slider,
        time_range_ui,
        time_start_number,
        time_start_slider,
    )


@app.cell
def _(time_range_ui):
    """時刻範囲UI表示"""
    time_range_ui
    return


@app.cell
def _(
    load_hdf5_data,
    mo,
    np,
    selected_results,
    time_end_number,
    time_end_slider,
    time_start_number,
    time_start_slider,
):
    """時刻範囲内の軌跡誤差計算"""

    def calculate_trajectory_difference(result1, result2, t_start, t_end):
        """
        2つのシミュレーション結果の軌跡誤差を計算

        Args:
            result1: 基準となるシミュレーション結果
            result2: 比較するシミュレーション結果
            t_start: 開始時刻
            t_end: 終了時刻

        Returns:
            dict: 誤差統計情報
        """
        try:
            # データ読み込み
            data1 = load_hdf5_data(result1["h5_file"])
            data2 = load_hdf5_data(result2["h5_file"])
        except Exception:
            return None

        # キー検索関数
        def find_key_by_prefix_and_suffix(key_data, prefix, suffix):
            for k in key_data.keys():
                if k.startswith(prefix) and k.endswith(suffix):
                    return k
            return None

        def find_key_by_suffix(key_data, suffix):
            for k in key_data.keys():
                if k.endswith(suffix):
                    return k
            return None

        # 時刻データ取得
        time1 = data1.get("time_s", np.array([]))
        time2 = data2.get("time_s", np.array([]))

        if len(time1) == 0 or len(time2) == 0:
            return None

        # 位置データキーの検索
        pos_key1 = find_key_by_prefix_and_suffix(data1, "position_", "Spacecraft1DOF_0")
        if not pos_key1:
            pos_key1 = find_key_by_suffix(data1, "position_Spacecraft")

        pos_key2 = find_key_by_prefix_and_suffix(data2, "position_", "Spacecraft1DOF_0")
        if not pos_key2:
            pos_key2 = find_key_by_suffix(data2, "position_Spacecraft")

        if not pos_key1 or not pos_key2:
            return None

        # 位置データ取得
        pos1 = data1.get(pos_key1, np.array([]))
        pos2 = data2.get(pos_key2, np.array([]))

        if len(pos1) == 0 or len(pos2) == 0:
            return None

        # 速度データキーの検索
        vel_key1 = pos_key1.replace("position", "velocity")
        vel_key2 = pos_key2.replace("position", "velocity")

        vel1 = data1.get(vel_key1, np.array([]))
        vel2 = data2.get(vel_key2, np.array([]))

        # 時刻範囲でフィルタリング
        mask1 = (time1 >= t_start) & (time1 <= t_end)
        mask2 = (time2 >= t_start) & (time2 <= t_end)

        time1_filtered = time1[mask1]
        time2_filtered = time2[mask2]
        pos1_filtered = pos1[mask1]
        pos2_filtered = pos2[mask2]
        vel1_filtered = vel1[mask1]
        vel2_filtered = vel2[mask2]

        if len(time1_filtered) == 0 or len(time2_filtered) == 0:
            return None

        # 時刻を統一（線形補間）
        # result1の時刻を基準とする
        pos2_interp = np.interp(time1_filtered, time2_filtered, pos2_filtered)
        vel2_interp = np.interp(time1_filtered, time2_filtered, vel2_filtered) if len(vel2_filtered) > 0 else np.zeros_like(pos2_interp)

        # 誤差計算
        position_error = pos1_filtered - pos2_interp
        velocity_error = vel1_filtered - vel2_interp if len(vel1_filtered) > 0 else np.zeros_like(position_error)

        # データを間引く（プロット用に最大1000ポイント）
        max_plot_points = 1000
        if len(time1_filtered) > max_plot_points:
            # 均等に間引く
            indices = np.linspace(0, len(time1_filtered) - 1, max_plot_points, dtype=int)
            time_plot = time1_filtered[indices]
            pos1_plot = pos1_filtered[indices]
            pos2_plot = pos2_interp[indices]
            position_error_plot = position_error[indices]
            vel1_plot = vel1_filtered[indices]
            vel2_plot = vel2_interp[indices]
            velocity_error_plot = velocity_error[indices]
        else:
            time_plot = time1_filtered
            pos1_plot = pos1_filtered
            pos2_plot = pos2_interp
            position_error_plot = position_error
            vel1_plot = vel1_filtered
            vel2_plot = vel2_interp
            velocity_error_plot = velocity_error

        # 統計情報
        stats = {
            "n_samples": len(position_error),
            "time_range": (float(time1_filtered[0]), float(time1_filtered[-1])),
            # 位置誤差
            "pos_rmse": float(np.sqrt(np.mean(position_error**2))),
            "pos_max_error": float(np.max(np.abs(position_error))),
            "pos_mean_error": float(np.mean(position_error)),
            "pos_mean_abs_error": float(np.mean(np.abs(position_error))),
            "pos_std_error": float(np.std(position_error)),
            "pos_median_error": float(np.median(position_error)),
            "pos_min_error": float(np.min(position_error)),
            "pos_max_positive_error": float(np.max(position_error)),
            "pos_max_negative_error": float(np.min(position_error)),
            # 速度誤差
            "vel_rmse": float(np.sqrt(np.mean(velocity_error**2))),
            "vel_max_error": float(np.max(np.abs(velocity_error))),
            "vel_mean_error": float(np.mean(velocity_error)),
            "vel_mean_abs_error": float(np.mean(np.abs(velocity_error))),
            "vel_std_error": float(np.std(velocity_error)),
            # データ（プロット用） - 間引いたデータをリストに変換
            "time": time_plot.tolist(),
            "pos1": pos1_plot.tolist(),
            "pos2": pos2_plot.tolist(),
            "position_error": position_error_plot.tolist(),
            "vel1": vel1_plot.tolist(),
            "vel2": vel2_plot.tolist(),
            "velocity_error": velocity_error_plot.tolist(),
        }

        # 大きなデータを明示的に削除
        del data1, data2
        del time1, time2, pos1, pos2, vel1, vel2
        del time1_filtered, time2_filtered, pos1_filtered, pos2_filtered
        del vel1_filtered, vel2_filtered, pos2_interp, vel2_interp
        del position_error, velocity_error

        return stats

    # 計算実行
    if len(selected_results) >= 2 and time_start_slider is not None and time_end_slider is not None and time_start_number is not None and time_end_number is not None:
        # 数値入力を優先（より正確な値の入力が可能なため）
        # ユーザーが数値入力を使用した場合、その値を使用
        # そうでない場合はスライダーの値を使用
        t_start = time_start_number.value if time_start_number.value is not None else time_start_slider.value
        t_end = time_end_number.value if time_end_number.value is not None else time_end_slider.value

        # 時刻範囲の妥当性チェック
        if t_start >= t_end:
            trajectory_diff_stats = None
            trajectory_diff_message = mo.md(f"⚠️ **Invalid time range**: Start time ({t_start:.3f}s) must be less than end time ({t_end:.3f}s)")
        else:
            trajectory_diff_stats = calculate_trajectory_difference(selected_results[0], selected_results[1], t_start, t_end)

            if trajectory_diff_stats is None:
                trajectory_diff_message = mo.md("⚠️ **Error**: Could not calculate trajectory difference. Check data availability.")
            else:
                trajectory_diff_message = mo.md(
                    f"""
    **Comparing trajectories:**
    - **Baseline**: {selected_results[0]["label"]}
    - **Comparison**: {selected_results[1]["label"]}
    - **Time range**: {t_start:.3f}s to {t_end:.3f}s ({trajectory_diff_stats["n_samples"]} samples)
                    """
                )
    else:
        trajectory_diff_stats = None
        trajectory_diff_message = mo.md("_Waiting for time range selection..._")
    return trajectory_diff_message, trajectory_diff_stats


@app.cell
def _(trajectory_diff_message):
    """軌跡差分メッセージ表示"""
    trajectory_diff_message
    return


@app.cell
def _(mo, pd, selected_results, trajectory_diff_stats):
    """軌跡誤差統計テーブル作成"""
    if trajectory_diff_stats is not None and len(selected_results) >= 2:
        # 位置誤差テーブル
        pos_error_data = {
            "Metric": [
                "RMSE",
                "Max Absolute Error",
                "Mean Error (signed)",
                "Mean Absolute Error",
                "Std Deviation",
                "Median Error",
                "Max Positive Error",
                "Max Negative Error",
            ],
            "Value [m]": [
                f"{trajectory_diff_stats['pos_rmse']:.6f}",
                f"{trajectory_diff_stats['pos_max_error']:.6f}",
                f"{trajectory_diff_stats['pos_mean_error']:.6f}",
                f"{trajectory_diff_stats['pos_mean_abs_error']:.6f}",
                f"{trajectory_diff_stats['pos_std_error']:.6f}",
                f"{trajectory_diff_stats['pos_median_error']:.6f}",
                f"{trajectory_diff_stats['pos_max_positive_error']:.6f}",
                f"{trajectory_diff_stats['pos_max_negative_error']:.6f}",
            ],
        }

        pos_error_table_df = pd.DataFrame(pos_error_data)

        pos_error_table_section = mo.vstack(
            [
                mo.md("### 📏 Position Trajectory Error (Time Range)"),
                mo.ui.table(pos_error_table_df, selection=None),
            ]
        )

        # 速度誤差テーブル
        vel_error_data = {
            "Metric": [
                "RMSE",
                "Max Absolute Error",
                "Mean Error (signed)",
                "Mean Absolute Error",
                "Std Deviation",
            ],
            "Value [m/s]": [
                f"{trajectory_diff_stats['vel_rmse']:.6f}",
                f"{trajectory_diff_stats['vel_max_error']:.6f}",
                f"{trajectory_diff_stats['vel_mean_error']:.6f}",
                f"{trajectory_diff_stats['vel_mean_abs_error']:.6f}",
                f"{trajectory_diff_stats['vel_std_error']:.6f}",
            ],
        }

        vel_error_table_df = pd.DataFrame(vel_error_data)

        vel_error_table_section = mo.vstack(
            [
                mo.md("### 🚀 Velocity Trajectory Error (Time Range)"),
                mo.ui.table(vel_error_table_df, selection=None),
            ]
        )
    else:
        pos_error_table_section = None
        vel_error_table_section = None
    return pos_error_table_section, vel_error_table_section


@app.cell
def _(pos_error_table_section):
    """位置誤差テーブル表示"""
    pos_error_table_section
    return


@app.cell
def _(vel_error_table_section):
    """速度誤差テーブル表示"""
    vel_error_table_section
    return


@app.cell
def _(go, mo, selected_results, trajectory_diff_stats):
    """時刻範囲の軌跡比較プロット"""
    if trajectory_diff_stats is not None:
        # 位置比較プロット
        fig_pos_compare = go.Figure()

        # Baseline軌跡
        fig_pos_compare.add_trace(
            go.Scatter(
                x=trajectory_diff_stats["time"],
                y=trajectory_diff_stats["pos1"],
                mode="lines",
                name=f"Baseline: {selected_results[0]['name']}",
                line=dict(color="#1f77b4", width=2),
                hovertemplate="<b>Baseline</b><br>Time: %{x:.4f} s<br>Position: %{y:.6f} m<extra></extra>",
            )
        )

        # Comparison軌跡
        fig_pos_compare.add_trace(
            go.Scatter(
                x=trajectory_diff_stats["time"],
                y=trajectory_diff_stats["pos2"],
                mode="lines",
                name=f"Comparison: {selected_results[1]['name']}",
                line=dict(color="#ff7f0e", width=2, dash="dash"),
                hovertemplate="<b>Comparison</b><br>Time: %{x:.4f} s<br>Position: %{y:.6f} m<extra></extra>",
            )
        )

        fig_pos_compare.update_layout(
            title="Position Trajectory Comparison (Selected Time Range)",
            xaxis_title="Time [s]",
            yaxis_title="Position [m]",
            hovermode="x unified",
            template="plotly_white",
            height=400,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        )

        fig_pos_compare.update_xaxes(showgrid=True, gridwidth=1, gridcolor="LightGray")
        fig_pos_compare.update_yaxes(showgrid=True, gridwidth=1, gridcolor="LightGray")

        # 誤差プロット
        fig_error = go.Figure()

        fig_error.add_trace(
            go.Scatter(
                x=trajectory_diff_stats["time"],
                y=trajectory_diff_stats["position_error"],
                mode="lines",
                name="Position Error",
                line=dict(color="#d62728", width=2),
                fill="tozeroy",
                fillcolor="rgba(214, 39, 40, 0.2)",
                hovertemplate="<b>Position Error</b><br>Time: %{x:.4f} s<br>Error: %{y:.6f} m<extra></extra>",
            )
        )

        fig_error.add_hline(y=0, line_dash="dot", line_color="black", line_width=1)

        # RMSE線を追加
        rmse = trajectory_diff_stats["pos_rmse"]
        fig_error.add_hline(
            y=rmse,
            line_dash="dash",
            line_color="green",
            annotation_text=f"RMSE: {rmse:.6f}m",
            annotation_position="right",
        )
        fig_error.add_hline(
            y=-rmse,
            line_dash="dash",
            line_color="green",
        )

        fig_error.update_layout(
            title="Position Error (Baseline - Comparison)",
            xaxis_title="Time [s]",
            yaxis_title="Position Error [m]",
            hovermode="x unified",
            template="plotly_white",
            height=400,
        )

        fig_error.update_xaxes(showgrid=True, gridwidth=1, gridcolor="LightGray")
        fig_error.update_yaxes(showgrid=True, gridwidth=1, gridcolor="LightGray")

        trajectory_comparison_plots = mo.vstack(
            [
                mo.md("### 📊 Trajectory Comparison Plots"),
                fig_pos_compare,
                fig_error,
            ]
        )
    else:
        trajectory_comparison_plots = None
    return (trajectory_comparison_plots,)


@app.cell
def _(trajectory_comparison_plots):
    """軌跡比較プロット表示"""
    trajectory_comparison_plots
    return


@app.cell
def _(mo):
    """メトリクス定義"""
    definitions = mo.md(
        """
    ---

    ### 📖 Metric Definitions

    - **RMS Error**: Root Mean Square error - overall accuracy measure
    - **Max Error**: Maximum absolute deviation from target
    - **Mean |Error|**: Average absolute error magnitude
    - **Std Error**: Standard deviation of position error
    - **Final Error**: Average error in last 10% of simulation
    - **Overshoot**: Maximum position beyond target
    - **Rise Time**: Time from 10% to 90% of target position
    - **Settling Time (5%)**: Time to stay within 5% of target permanently
    - **Settling Time (2%)**: Time to stay within 2% of target permanently
    - **ISE**: Integral of Squared Error - penalizes large errors
    - **IAE**: Integral of Absolute Error - total error magnitude
    - **ITAE**: Integral of Time-weighted Absolute Error - penalizes persistent errors
    - **Control Effort**: Total variation of control input (sum of absolute changes)
    - **Max Thrust**: Maximum control input magnitude
    - **Mean |Thrust|**: Average control input magnitude
    - **Max Velocity**: Maximum velocity during trajectory
        """
    )
    return (definitions,)


@app.cell
def _(definitions):
    """定義表示"""
    definitions
    return


if __name__ == "__main__":
    app.run()
