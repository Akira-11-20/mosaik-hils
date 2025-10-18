"""
3-Way Comparison Analysis: HILS vs RT vs Pure Python

3つのシミュレーション方式の制御性能を比較・可視化するスクリプト

使用方法:
    python compare_all.py results/20251017-123456/hils_data.h5 results_rt/20251017-123500/hils_data.h5 results_pure/20251017-123600/hils_data.h5

比較対象:
1. HILS: 通信遅延あり（Mosaikベース）
2. RT: 通信遅延なし（Mosaikベース）
3. Pure Python: Mosaikなしの理想的なシミュレーション

機能:
- 位置、速度、制御入力の3-way比較プロット
- 制御誤差の統計比較（RMS、最大値、整定時間）
- オーバーシュート、整定時間の評価
- 結果をPDF形式で保存
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np


def load_hdf5_data(hdf5_path: str) -> Dict[str, np.ndarray]:
    """
    HDF5ファイルからデータを読み込む

    Args:
        hdf5_path: HDF5ファイルパス

    Returns:
        データセットの辞書
    """
    data = {}
    with h5py.File(hdf5_path, "r") as f:
        # データは'data'グループ内にある
        if 'data' in f:
            data_group = f['data']
            for key in data_group.keys():
                data[key] = data_group[key][:]
        else:
            # 古い形式の互換性のため
            for key in f.keys():
                if isinstance(f[key], h5py.Dataset):
                    data[key] = f[key][:]
    return data


def load_simulation_config(result_dir: Path) -> Dict:
    """
    シミュレーション設定をJSONから読み込む

    Args:
        result_dir: 結果ディレクトリ

    Returns:
        設定辞書
    """
    config_path = result_dir / "simulation_config.json"
    if config_path.exists():
        with open(config_path, "r") as f:
            return json.load(f)
    return {}


def find_key_by_suffix(data: Dict[str, np.ndarray], suffix: str) -> str:
    """
    キーの接尾辞でデータセットキーを検索

    Args:
        data: データ辞書
        suffix: 検索する接尾辞

    Returns:
        マッチしたキー
    """
    for key in data.keys():
        if key.endswith(suffix):
            return key
    raise KeyError(f"No key found with suffix: {suffix}")


def calculate_metrics(
    time: np.ndarray,
    position: np.ndarray,
    velocity: np.ndarray,
    error: np.ndarray,
    target_position: float,
) -> Dict[str, float]:
    """
    制御性能指標を計算

    Args:
        time: 時刻配列 [s]
        position: 位置配列 [m]
        velocity: 速度配列 [m/s]
        error: 制御誤差配列 [m]
        target_position: 目標位置 [m]

    Returns:
        性能指標の辞書
    """
    # RMS誤差
    rms_error = np.sqrt(np.mean(error**2))

    # 最大誤差
    max_error = np.max(np.abs(error))

    # オーバーシュート
    overshoot = np.max(position) - target_position
    overshoot_percent = (overshoot / target_position) * 100 if target_position != 0 else 0

    # 整定時間（誤差が5%以内に収まる時刻）
    settling_threshold = 0.05 * target_position
    settled_indices = np.where(np.abs(error) <= settling_threshold)[0]
    if len(settled_indices) > 0:
        # 整定後に再び閾値を超えないかチェック
        for idx in settled_indices:
            if np.all(np.abs(error[idx:]) <= settling_threshold):
                settling_time = time[idx]
                break
        else:
            settling_time = None  # 整定しなかった
    else:
        settling_time = None

    # 最終誤差（最後の10%の平均）
    final_window = int(len(error) * 0.1)
    final_error = np.mean(np.abs(error[-final_window:]))

    return {
        "rms_error": rms_error,
        "max_error": max_error,
        "overshoot": overshoot,
        "overshoot_percent": overshoot_percent,
        "settling_time": settling_time,
        "final_error": final_error,
    }


def plot_3way_comparison(
    hils_data: Dict[str, np.ndarray],
    rt_data: Dict[str, np.ndarray],
    pure_data: Dict[str, np.ndarray],
    hils_config: Dict,
    rt_config: Dict,
    pure_config: Dict,
    output_dir: Path,
):
    """
    3つのシミュレーションの比較プロットを生成

    Args:
        hils_data: HILSデータ
        rt_data: RTデータ
        pure_data: Pure Pythonデータ
        hils_config: HILS設定
        rt_config: RT設定
        pure_config: Pure Python設定
        output_dir: 出力ディレクトリ
    """
    # 時刻データ
    hils_time = hils_data["time_s"]
    rt_time = rt_data["time_s"]
    pure_time = pure_data["time_s"]

    # HILSとRTのデータキーを検索
    pos_key_hils = find_key_by_suffix(hils_data, "Spacecraft1DOF_0")
    vel_key_hils = find_key_by_suffix(hils_data, "Spacecraft1DOF_0").replace("position", "velocity")
    thrust_key_hils = find_key_by_suffix(hils_data, "_thrust")
    error_key_hils = find_key_by_suffix(hils_data, "Controller_0").replace("command", "error")

    pos_key_rt = find_key_by_suffix(rt_data, "Spacecraft1DOF_0")
    vel_key_rt = find_key_by_suffix(rt_data, "Spacecraft1DOF_0").replace("position", "velocity")
    thrust_key_rt = find_key_by_suffix(rt_data, "_thrust")
    error_key_rt = find_key_by_suffix(rt_data, "Controller_0").replace("command", "error")

    # Pure Pythonのデータキーを検索（異なる命名規則）
    pos_key_pure = find_key_by_suffix(pure_data, "_Spacecraft")
    vel_key_pure = pos_key_pure.replace("position", "velocity")
    thrust_key_pure = find_key_by_suffix(pure_data, "_thrust")
    error_key_pure = find_key_by_suffix(pure_data, "_Controller")

    hils_pos = hils_data[pos_key_hils]
    hils_vel = hils_data[vel_key_hils]
    hils_thrust = hils_data[thrust_key_hils]
    hils_error = hils_data[error_key_hils]

    rt_pos = rt_data[pos_key_rt]
    rt_vel = rt_data[vel_key_rt]
    rt_thrust = rt_data[thrust_key_rt]
    rt_error = rt_data[error_key_rt]

    pure_pos = pure_data[pos_key_pure]
    pure_vel = pure_data[vel_key_pure]
    pure_thrust = pure_data[thrust_key_pure]
    pure_error = pure_data[error_key_pure]

    # 目標位置の取得
    target_position = hils_config.get("control", {}).get("target_position_m", 5.0)

    # 性能指標の計算
    hils_metrics = calculate_metrics(hils_time, hils_pos, hils_vel, hils_error, target_position)
    rt_metrics = calculate_metrics(rt_time, rt_pos, rt_vel, rt_error, target_position)
    pure_metrics = calculate_metrics(pure_time, pure_pos, pure_vel, pure_error, target_position)

    # プロット作成
    fig, axes = plt.subplots(4, 1, figsize=(14, 14))

    # 1. 位置比較
    ax = axes[0]
    ax.plot(hils_time, hils_pos, "b-", label="HILS (with delay)", linewidth=1.5, alpha=0.8)
    ax.plot(rt_time, rt_pos, "g--", label="RT (mosaik, no delay)", linewidth=1.5, alpha=0.8)
    ax.plot(pure_time, pure_pos, "r:", label="Pure Python (ideal)", linewidth=2, alpha=0.8)
    ax.axhline(target_position, color="k", linestyle=":", label="Target", linewidth=1)
    ax.set_xlabel("Time [s]", fontsize=11)
    ax.set_ylabel("Position [m]", fontsize=11)
    ax.set_title("Position Comparison: HILS vs RT vs Pure Python", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 2. 速度比較
    ax = axes[1]
    ax.plot(hils_time, hils_vel, "b-", label="HILS (with delay)", linewidth=1.5, alpha=0.8)
    ax.plot(rt_time, rt_vel, "g--", label="RT (mosaik, no delay)", linewidth=1.5, alpha=0.8)
    ax.plot(pure_time, pure_vel, "r:", label="Pure Python (ideal)", linewidth=2, alpha=0.8)
    ax.set_xlabel("Time [s]", fontsize=11)
    ax.set_ylabel("Velocity [m/s]", fontsize=11)
    ax.set_title("Velocity Comparison: HILS vs RT vs Pure Python", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 3. 制御入力比較
    ax = axes[2]
    ax.plot(hils_time, hils_thrust, "b-", label="HILS (with delay)", linewidth=1.5, alpha=0.8)
    ax.plot(rt_time, rt_thrust, "g--", label="RT (mosaik, no delay)", linewidth=1.5, alpha=0.8)
    ax.plot(pure_time, pure_thrust, "r:", label="Pure Python (ideal)", linewidth=2, alpha=0.8)
    ax.set_xlabel("Time [s]", fontsize=11)
    ax.set_ylabel("Thrust [N]", fontsize=11)
    ax.set_title("Control Input Comparison: HILS vs RT vs Pure Python", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 4. 制御誤差比較
    ax = axes[3]
    ax.plot(hils_time, hils_error, "b-", label="HILS (with delay)", linewidth=1.5, alpha=0.8)
    ax.plot(rt_time, rt_error, "g--", label="RT (mosaik, no delay)", linewidth=1.5, alpha=0.8)
    ax.plot(pure_time, pure_error, "r:", label="Pure Python (ideal)", linewidth=2, alpha=0.8)
    ax.axhline(0, color="k", linestyle=":", linewidth=1)
    ax.set_xlabel("Time [s]", fontsize=11)
    ax.set_ylabel("Position Error [m]", fontsize=11)
    ax.set_title("Control Error Comparison: HILS vs RT vs Pure Python", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存
    output_path = output_dir / "comparison_all.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"📊 Comparison plot saved: {output_path}")
    plt.close()

    # 性能指標の表示
    print("\n" + "=" * 80)
    print("Performance Metrics Comparison (3-Way)")
    print("=" * 80)

    # 設定の表示
    hils_comm = hils_config.get("communication", {})
    print(f"\nHILS Configuration:")
    print(f"  - Command delay: {hils_comm.get('cmd_delay_s', 0)*1000:.1f} ms")
    print(f"  - Command jitter: {hils_comm.get('cmd_jitter_s', 0)*1000:.1f} ms")
    print(f"  - Sense delay: {hils_comm.get('sense_delay_s', 0)*1000:.1f} ms")
    print(f"  - Sense jitter: {hils_comm.get('sense_jitter_s', 0)*1000:.1f} ms")

    print(f"\nRT Configuration:")
    print(f"  - No communication delays (Mosaik-based)")

    print(f"\nPure Python Configuration:")
    print(f"  - No Mosaik framework overhead (ideal)")

    # 性能指標の比較表
    print(f"\n{'Metric':<25} {'HILS':>15} {'RT':>15} {'Pure':>15}")
    print("-" * 80)

    metrics_to_compare = [
        ("RMS Error [m]", "rms_error"),
        ("Max Error [m]", "max_error"),
        ("Overshoot [m]", "overshoot"),
        ("Overshoot [%]", "overshoot_percent"),
        ("Final Error [m]", "final_error"),
    ]

    for label, key in metrics_to_compare:
        hils_val = hils_metrics[key]
        rt_val = rt_metrics[key]
        pure_val = pure_metrics[key]
        print(f"{label:<25} {hils_val:>15.4f} {rt_val:>15.4f} {pure_val:>15.4f}")

    # 整定時間の比較
    hils_settling = hils_metrics["settling_time"]
    rt_settling = rt_metrics["settling_time"]
    pure_settling = pure_metrics["settling_time"]

    hils_str = f"{hils_settling:.4f}" if hils_settling is not None else "N/A"
    rt_str = f"{rt_settling:.4f}" if rt_settling is not None else "N/A"
    pure_str = f"{pure_settling:.4f}" if pure_settling is not None else "N/A"
    print(f"{'Settling Time [s]':<25} {hils_str:>15} {rt_str:>15} {pure_str:>15}")

    print("=" * 80)

    # 相対比較（Pure Pythonを基準）
    print("\n" + "=" * 80)
    print("Relative Performance (% degradation from Pure Python baseline)")
    print("=" * 80)

    for label, key in metrics_to_compare[:2]:  # RMS and Max error only
        pure_val = pure_metrics[key]
        if pure_val != 0:
            hils_deg = ((hils_metrics[key] - pure_val) / pure_val) * 100
            rt_deg = ((rt_metrics[key] - pure_val) / pure_val) * 100
            print(f"{label:<25} HILS: {hils_deg:+.2f}%, RT: {rt_deg:+.2f}%")

    print("=" * 80)

    # 性能指標をJSONで保存
    comparison_metrics = {
        "hils": hils_metrics,
        "rt": rt_metrics,
        "pure_python": pure_metrics,
        "relative_to_pure": {
            "hils_rms_degradation_percent": ((hils_metrics["rms_error"] - pure_metrics["rms_error"]) / pure_metrics["rms_error"] * 100) if pure_metrics["rms_error"] != 0 else 0,
            "rt_rms_degradation_percent": ((rt_metrics["rms_error"] - pure_metrics["rms_error"]) / pure_metrics["rms_error"] * 100) if pure_metrics["rms_error"] != 0 else 0,
        },
    }

    metrics_path = output_dir / "comparison_all_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(comparison_metrics, f, indent=2)
    print(f"\n💾 Metrics saved: {metrics_path}")


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="Compare HILS, RT, and Pure Python simulation results")
    parser.add_argument("hils_h5", type=str, help="Path to HILS HDF5 data file")
    parser.add_argument("rt_h5", type=str, help="Path to RT HDF5 data file")
    parser.add_argument("pure_h5", type=str, help="Path to Pure Python HDF5 data file")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: current directory)",
    )
    args = parser.parse_args()

    # ファイルパスの確認
    hils_h5_path = Path(args.hils_h5)
    rt_h5_path = Path(args.rt_h5)
    pure_h5_path = Path(args.pure_h5)

    if not hils_h5_path.exists():
        print(f"❌ HILS HDF5 file not found: {hils_h5_path}")
        return

    if not rt_h5_path.exists():
        print(f"❌ RT HDF5 file not found: {rt_h5_path}")
        return

    if not pure_h5_path.exists():
        print(f"❌ Pure Python HDF5 file not found: {pure_h5_path}")
        return

    # 出力ディレクトリの設定
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path("comparison_results")

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("3-Way Comparison Analysis: HILS vs RT vs Pure Python")
    print("=" * 80)
    print(f"HILS data: {hils_h5_path}")
    print(f"RT data: {rt_h5_path}")
    print(f"Pure Python data: {pure_h5_path}")
    print(f"Output directory: {output_dir}")

    # データの読み込み
    print("\n📂 Loading data...")
    hils_data = load_hdf5_data(str(hils_h5_path))
    rt_data = load_hdf5_data(str(rt_h5_path))
    pure_data = load_hdf5_data(str(pure_h5_path))

    # 設定の読み込み
    hils_config = load_simulation_config(hils_h5_path.parent)
    rt_config = load_simulation_config(rt_h5_path.parent)
    pure_config = load_simulation_config(pure_h5_path.parent)

    # 比較プロットの生成
    print("\n📊 Generating comparison plots...")
    plot_3way_comparison(hils_data, rt_data, pure_data, hils_config, rt_config, pure_config, output_dir)

    print("\n✅ 3-way comparison analysis completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
