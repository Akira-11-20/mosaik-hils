"""
Pure Python vs RT (Mosaik) Comparison Analysis

Mosaikフレームワークのオーバーヘッドを評価するための比較スクリプト

使用方法:
    python compare_pure_rt.py results_pure/20251017-123456/hils_data.h5 results_rt/20251017-123500/hils_data.h5

比較対象:
1. Pure Python: Mosaikなしの素のPythonシミュレーション
2. RT (Mosaik): Mosaikベースだが通信遅延なし

評価項目:
- Mosaikフレームワークによる性能への影響
- 制御性能の違い（RMS誤差、オーバーシュート、整定時間）
- 両方とも通信遅延なし、制御周期10msで条件を揃えている
"""

import argparse
import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from hdf5_helper import load_hdf5_data


def load_hdf5_data(hdf5_path: str) -> Dict[str, np.ndarray]:
    """
    HDF5ファイルからデータを読み込む

    Args:
        hdf5_path: HDF5ファイルパス

    Returns:
        データセットの辞書
    """
    data = {}
    data_tmp = load_hdf5_data(hdf5_path)
    # データは'data'グループ内にある
    if "data" in f:
        data_group = f["data"]
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
        with open(config_path) as f:
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


def plot_comparison(
    pure_data: Dict[str, np.ndarray],
    rt_data: Dict[str, np.ndarray],
    pure_config: Dict,
    rt_config: Dict,
    output_dir: Path,
):
    """
    Pure PythonとRTの比較プロットを生成

    Args:
        pure_data: Pure Pythonデータ
        rt_data: RTデータ
        pure_config: Pure Python設定
        rt_config: RT設定
        output_dir: 出力ディレクトリ
    """
    # データの抽出
    pure_time = pure_data["time_s"]
    rt_time = rt_data["time_s"]

    # Pure Pythonのデータキー（直接指定）
    pure_pos = pure_data["position_Spacecraft"]
    pure_vel = pure_data["velocity_Spacecraft"]
    pure_thrust = pure_data["command_Controller_thrust"]
    pure_error = pure_data["error_Controller"]

    # RTのデータキー（直接指定）
    rt_pos = rt_data["position_EnvSim-0.Spacecraft1DOF_0"]
    rt_vel = rt_data["velocity_EnvSim-0.Spacecraft1DOF_0"]
    rt_thrust = rt_data["command_ControllerSim-0.PDController_0_thrust"]
    rt_error = rt_data["error_ControllerSim-0.PDController_0"]

    # 目標位置の取得
    target_position = pure_config.get("control", {}).get("target_position_m", 5.0)

    # 性能指標の計算
    pure_metrics = calculate_metrics(pure_time, pure_pos, pure_vel, pure_error, target_position)
    rt_metrics = calculate_metrics(rt_time, rt_pos, rt_vel, rt_error, target_position)

    # プロット作成
    fig, axes = plt.subplots(4, 1, figsize=(12, 12))

    # 1. 位置比較
    ax = axes[0]
    ax.plot(
        pure_time,
        pure_pos,
        "r-",
        label="Pure Python (no Mosaik)",
        linewidth=2,
        alpha=0.8,
    )
    ax.plot(rt_time, rt_pos, "b--", label="RT (Mosaik-based)", linewidth=2, alpha=0.8)
    ax.axhline(target_position, color="k", linestyle=":", label="Target")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Position [m]")
    ax.set_title(
        "Position Comparison: Pure Python vs RT (Mosaik)\nEvaluating Mosaik Framework Overhead",
        fontweight="bold",
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. 速度比較
    ax = axes[1]
    ax.plot(
        pure_time,
        pure_vel,
        "r-",
        label="Pure Python (no Mosaik)",
        linewidth=2,
        alpha=0.8,
    )
    ax.plot(rt_time, rt_vel, "b--", label="RT (Mosaik-based)", linewidth=2, alpha=0.8)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Velocity [m/s]")
    ax.set_title("Velocity Comparison: Pure Python vs RT (Mosaik)", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. 制御入力比較
    ax = axes[2]
    ax.plot(
        pure_time,
        pure_thrust,
        "r-",
        label="Pure Python (no Mosaik)",
        linewidth=2,
        alpha=0.8,
    )
    ax.plot(rt_time, rt_thrust, "b--", label="RT (Mosaik-based)", linewidth=2, alpha=0.8)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Thrust [N]")
    ax.set_title("Control Input Comparison: Pure Python vs RT (Mosaik)", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. 制御誤差比較
    ax = axes[3]
    ax.plot(
        pure_time,
        pure_error,
        "r-",
        label="Pure Python (no Mosaik)",
        linewidth=2,
        alpha=0.8,
    )
    ax.plot(rt_time, rt_error, "b--", label="RT (Mosaik-based)", linewidth=2, alpha=0.8)
    ax.axhline(0, color="k", linestyle=":", linewidth=1)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Position Error [m]")
    ax.set_title("Control Error Comparison: Pure Python vs RT (Mosaik)", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存
    output_path = output_dir / "comparison_pure_rt.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"📊 Comparison plot saved: {output_path}")
    plt.close()

    # 性能指標の表示
    print("\n" + "=" * 70)
    print("Performance Metrics Comparison: Pure Python vs RT (Mosaik)")
    print("=" * 70)

    print("\nPure Python Configuration:")
    print("  - No Mosaik framework")
    print("  - No communication delays")
    print(f"  - Control period: {pure_config.get('control', {}).get('control_period_s', 0) * 1000:.1f} ms")

    print("\nRT (Mosaik) Configuration:")
    print("  - Mosaik framework-based")
    print("  - No communication delays")
    print(f"  - Control period: {rt_config.get('control', {}).get('control_period_s', 0) * 1000:.1f} ms")

    # 性能指標の比較表
    print(f"\n{'Metric':<25} {'Pure Python':>15} {'RT (Mosaik)':>15} {'Difference':>15}")
    print("-" * 70)

    metrics_to_compare = [
        ("RMS Error [m]", "rms_error"),
        ("Max Error [m]", "max_error"),
        ("Overshoot [m]", "overshoot"),
        ("Overshoot [%]", "overshoot_percent"),
        ("Final Error [m]", "final_error"),
    ]

    for label, key in metrics_to_compare:
        pure_val = pure_metrics[key]
        rt_val = rt_metrics[key]
        diff = rt_val - pure_val
        print(f"{label:<25} {pure_val:>15.4f} {rt_val:>15.4f} {diff:>15.4f}")

    # 整定時間の比較
    pure_settling = pure_metrics["settling_time"]
    rt_settling = rt_metrics["settling_time"]

    pure_str = f"{pure_settling:.4f}" if pure_settling is not None else "N/A"
    rt_str = f"{rt_settling:.4f}" if rt_settling is not None else "N/A"

    if pure_settling is not None and rt_settling is not None:
        diff = rt_settling - pure_settling
        print(f"{'Settling Time [s]':<25} {pure_str:>15} {rt_str:>15} {diff:>15.4f}")
    else:
        print(f"{'Settling Time [s]':<25} {pure_str:>15} {rt_str:>15} {'N/A':>15}")

    print("=" * 70)

    # Mosaikフレームワークのオーバーヘッド評価
    print("\n" + "=" * 70)
    print("Mosaik Framework Overhead Analysis")
    print("=" * 70)

    rms_overhead = ((rt_metrics["rms_error"] - pure_metrics["rms_error"]) / pure_metrics["rms_error"]) * 100
    max_overhead = ((rt_metrics["max_error"] - pure_metrics["max_error"]) / pure_metrics["max_error"]) * 100

    print(f"\nRMS Error degradation due to Mosaik: {rms_overhead:+.2f}%")
    print(f"Max Error degradation due to Mosaik: {max_overhead:+.2f}%")

    if pure_metrics["overshoot"] != 0:
        overshoot_overhead = ((rt_metrics["overshoot"] - pure_metrics["overshoot"]) / pure_metrics["overshoot"]) * 100
        print(f"Overshoot degradation due to Mosaik: {overshoot_overhead:+.2f}%")

    print("\n📌 Key Findings:")
    print("   Both simulations use the same control period (10ms) and no communication delays.")
    print("   Differences in performance are attributed to Mosaik framework overhead.")

    if abs(rms_overhead) < 1.0:
        print("   ✓ Mosaik overhead is negligible (<1% RMS error difference).")
    elif abs(rms_overhead) < 5.0:
        print(f"   ⚠ Mosaik introduces minor overhead ({abs(rms_overhead):.1f}% RMS error difference).")
    else:
        print(f"   ⚠ Mosaik introduces significant overhead ({abs(rms_overhead):.1f}% RMS error difference).")

    print("=" * 70)

    # 性能指標をJSONで保存
    comparison_metrics = {
        "pure_python": pure_metrics,
        "rt_mosaik": rt_metrics,
        "mosaik_overhead": {
            "rms_error_degradation_percent": rms_overhead,
            "max_error_degradation_percent": max_overhead,
        },
    }

    metrics_path = output_dir / "comparison_pure_rt_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(comparison_metrics, f, indent=2)
    print(f"\n💾 Metrics saved: {metrics_path}")


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="Compare Pure Python and RT (Mosaik) simulation results to evaluate Mosaik framework overhead")
    parser.add_argument("pure_h5", type=str, help="Path to Pure Python HDF5 data file")
    parser.add_argument("rt_h5", type=str, help="Path to RT (Mosaik) HDF5 data file")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: comparison_results)",
    )
    args = parser.parse_args()

    # ファイルパスの確認
    pure_h5_path = Path(args.pure_h5)
    rt_h5_path = Path(args.rt_h5)

    if not pure_h5_path.exists():
        print(f"❌ Pure Python HDF5 file not found: {pure_h5_path}")
        return

    if not rt_h5_path.exists():
        print(f"❌ RT HDF5 file not found: {rt_h5_path}")
        return

    # 出力ディレクトリの設定
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path("comparison_results")

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Pure Python vs RT (Mosaik) Comparison Analysis")
    print("Evaluating Mosaik Framework Overhead")
    print("=" * 70)
    print(f"Pure Python data: {pure_h5_path}")
    print(f"RT (Mosaik) data: {rt_h5_path}")
    print(f"Output directory: {output_dir}")

    # データの読み込み
    print("\n📂 Loading data...")
    pure_data = load_hdf5_data(str(pure_h5_path))
    rt_data = load_hdf5_data(str(rt_h5_path))

    # 設定の読み込み
    pure_config = load_simulation_config(pure_h5_path.parent)
    rt_config = load_simulation_config(rt_h5_path.parent)

    # 比較プロットの生成
    print("\n📊 Generating comparison plots...")
    plot_comparison(pure_data, rt_data, pure_config, rt_config, output_dir)

    print("\n✅ Comparison analysis completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
