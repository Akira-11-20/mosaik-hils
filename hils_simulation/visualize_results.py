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
        metrics["overshoot_percent"] = (
            (overshoot / target_position) * 100 if target_position != 0 else 0
        )

        # アンダーシュート
        undershoot = target_position - np.min(position)
        metrics["undershoot"] = undershoot
        metrics["undershoot_percent"] = (
            (undershoot / target_position) * 100 if target_position != 0 else 0
        )

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
def _(calculate_detailed_metrics, h5py, mo, np, plt, selected_results):
    """プロット生成とメトリクス計算"""

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

    def generate_comparison_plot_and_metrics(results_list):
        """比較プロットとメトリクスを生成する関数"""

        if len(results_list) == 0:
            return mo.md("⚠️ Please select at least one result"), []

        # メトリクスを保存するリスト
        all_metrics = []

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

                # 目標位置の取得
                target_position = result_item["config"].get("control", {}).get(
                    "target_position_m", 5.0
                )

                # メトリクス計算
                if (
                    len(time_data) > 0
                    and len(result_position) > 0
                    and len(result_velocity) > 0
                    and len(result_error) > 0
                    and len(result_thrust) > 0
                ):
                    metrics = calculate_detailed_metrics(
                        time_data,
                        result_position,
                        result_velocity,
                        result_error,
                        result_thrust,
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
        return fig, all_metrics

    # 関数を呼び出してプロットとメトリクスを生成
    plot_fig, computed_metrics = generate_comparison_plot_and_metrics(selected_results)
    return computed_metrics, plot_fig


@app.cell
def _(plot_fig):
    """プロット表示"""
    plot_fig
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
                "Settling 5%": (
                    metrics["settling_time_5pct"]
                    if metrics["settling_time_5pct"] is not None
                    else None
                ),
                "Settling 2%": (
                    metrics["settling_time_2pct"]
                    if metrics["settling_time_2pct"] is not None
                    else None
                ),
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
        error_table = df[
            ["Simulation", "RMS Error", "Max Error", "Mean |Error|", "Std Error", "Final Error"]
        ].copy()
        error_table["RMS Error"] = error_table["RMS Error"].apply(lambda x: f"{x:.6f}")
        error_table["Max Error"] = error_table["Max Error"].apply(lambda x: f"{x:.6f}")
        error_table["Mean |Error|"] = error_table["Mean |Error|"].apply(lambda x: f"{x:.6f}")
        error_table["Std Error"] = error_table["Std Error"].apply(lambda x: f"{x:.6f}")
        error_table["Final Error"] = error_table["Final Error"].apply(lambda x: f"{x:.6f}")

        transient_table = df[
            ["Simulation", "Overshoot", "Overshoot %", "Rise Time", "Settling 5%", "Settling 2%"]
        ].copy()
        transient_table["Overshoot"] = transient_table["Overshoot"].apply(lambda x: f"{x:.6f}")
        transient_table["Overshoot %"] = transient_table["Overshoot %"].apply(
            lambda x: f"{x:.2f}%"
        )
        transient_table["Rise Time"] = transient_table["Rise Time"].apply(
            lambda x: f"{x:.4f}" if pd.notna(x) else "N/A"
        )
        transient_table["Settling 5%"] = transient_table["Settling 5%"].apply(
            lambda x: f"{x:.4f}" if pd.notna(x) else "N/A"
        )
        transient_table["Settling 2%"] = transient_table["Settling 2%"].apply(
            lambda x: f"{x:.4f}" if pd.notna(x) else "N/A"
        )

        integral_table = df[["Simulation", "ISE", "IAE", "ITAE"]].copy()
        integral_table["ISE"] = integral_table["ISE"].apply(lambda x: f"{x:.6f}")
        integral_table["IAE"] = integral_table["IAE"].apply(lambda x: f"{x:.6f}")
        integral_table["ITAE"] = integral_table["ITAE"].apply(lambda x: f"{x:.6f}")

        control_table = df[
            ["Simulation", "Max Thrust", "Mean |Thrust|", "Control Effort", "Max Velocity"]
        ].copy()
        control_table["Max Thrust"] = control_table["Max Thrust"].apply(lambda x: f"{x:.4f}")
        control_table["Mean |Thrust|"] = control_table["Mean |Thrust|"].apply(
            lambda x: f"{x:.4f}"
        )
        control_table["Control Effort"] = control_table["Control Effort"].apply(
            lambda x: f"{x:.4f}"
        )
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
        relative_table,
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
