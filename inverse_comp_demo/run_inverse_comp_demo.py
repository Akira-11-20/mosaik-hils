"""Minimal HILS-like setup that highlights inverse transfer compensation.

Run this script to compare a fixed-delay measurement with and without inverse
compensation. Reported metrics include RMSE, cross-power phase-derived delay,
correlation-based lag estimates, and 90% step-rise timing. A plot overlays the true plant output, delayed
measurement, and compensated signal.

逆伝達補償(inverse transfer compensation)のデモンストレーション用スクリプト。
固定遅延を持つ測定信号に対して、補償あり・なしの比較を行い、以下の指標を評価します：
- RMSE（二乗平均平方根誤差）
- クロスパワースペクトルから得られる位相遅延
- 相互相関に基づく遅延推定
- ステップ応答の立ち上がり時間（90%到達時間）
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class SimulationConfig:
    """シミュレーション設定を保持するデータクラス"""

    # 時間関連パラメータ
    Ts: float = 0.01  # サンプリング周期 [s] - 制御ループの時間刻み幅
    Tend: float = 20.0  # シミュレーション終了時刻 [s]

    # システム特性パラメータ
    tau: float = 0.15  # 通信遅延時間 [s] - センサから制御器への伝送遅延

    # 2次系パラメータ（質量-バネ-ダンパー系）
    # 差分方程式: x[k] = a1*x[k-1] + a2*x[k-2] + b0*u[k] + b1*u[k-1]
    # 連続系: m*d²x/dt² + c*dx/dt + k*x = u
    # 以下のパラメータは減衰振動系を想定（ζ < 1, 固有周波数 ω_n ≈ 2π rad/s）
    a1: float = 1.8  # 2次系係数1（極の位置を決定）
    a2: float = -0.85  # 2次系係数2（極の位置を決定）
    b0: float = 0.005  # 入力ゲイン（現在ステップ）
    b1: float = 0.005  # 入力ゲイン（1ステップ前）

    # ノイズパラメータ
    noise_std: float = 0.00  # ガウスノイズの標準偏差 - 測定ノイズのシミュレーション

    # 入力信号パラメータ（正弦波）
    sine_amp: float = 0.2  # 正弦波の振幅
    sine_freq_hz: float = 0.3  # 正弦波の周波数 [Hz]

    # 入力信号パラメータ（ステップ）
    step_time: float = 2.0  # ステップ入力を加える時刻 [s]
    step_amp: float = 0.5  # ステップ入力の大きさ

    @property
    def sample_count(self) -> int:
        """総サンプル数を計算 (シミュレーション時間をサンプリング周期で割る)"""
        return int(self.Tend / self.Ts)

    @property
    def delay_samples(self) -> int:
        """遅延時間をサンプル数に変換 (τ / Ts を四捨五入)"""
        return int(round(self.tau / self.Ts))

    @property
    def inverse_gain(self) -> float:
        """逆伝達補償のゲイン - 遅延サンプル数 d と同じ値を使用"""
        return float(self.delay_samples)


def build_input_signal(cfg: SimulationConfig) -> Tuple[np.ndarray, np.ndarray]:
    """入力信号を生成（正弦波 + ステップ入力）

    Args:
        cfg: シミュレーション設定

    Returns:
        t: 時刻配列 [s]
        u: 入力信号配列（正弦波とステップの重ね合わせ）
    """
    t = np.arange(cfg.sample_count) * cfg.Ts  # 時刻配列を生成
    sine = cfg.sine_amp * np.sin(2.0 * np.pi * cfg.sine_freq_hz * t)  # 正弦波成分
    step = cfg.step_amp * (t >= cfg.step_time).astype(float)  # ステップ成分（t≥step_timeで1）
    u = sine + step  # 正弦波とステップを足し合わせた入力信号
    return t, u


def propagate_second_order(u: np.ndarray, cfg: SimulationConfig) -> np.ndarray:
    """2次系の伝播（プラント動特性のシミュレーション）

    差分方程式: x[k] = a1*x[k-1] + a2*x[k-2] + b0*u[k] + b1*u[k-1]

    これは連続時間の2次系（質量-バネ-ダンパー系など）を離散化したもの:
        m*d²x/dt² + c*dx/dt + k*x = u

    パラメータ設定により以下の挙動が得られる:
    - a1, a2 が減衰振動の極を持つ → 振動しながら収束
    - 安定条件: |a2| < 1, |a1| < 1 + a2

    Args:
        u: 入力信号配列
        cfg: シミュレーション設定（a1, a2, b0, b1パラメータを含む）

    Returns:
        x: プラント出力（2次系を通過した信号）
    """
    x = np.zeros_like(u)  # 出力配列を初期化

    # 初期値（最初の2ステップは特別処理）
    if len(u) > 0:
        x[0] = cfg.b0 * u[0]  # k=0: 過去データなし
    if len(u) > 1:
        x[1] = cfg.a1 * x[0] + cfg.b0 * u[1] + cfg.b1 * u[0]  # k=1: x[k-2]なし

    # k >= 2 から通常の2次差分方程式を適用
    for k in range(2, len(u)):
        x[k] = cfg.a1 * x[k - 1] + cfg.a2 * x[k - 2] + cfg.b0 * u[k] + cfg.b1 * u[k - 1]
    return x


def apply_delay_and_noise(signal: np.ndarray, delay_samples: int, noise_std: float) -> np.ndarray:
    """信号に遅延とノイズを付加（通信遅延とセンサノイズのシミュレーション）

    FIFOバッファ（deque）を使用して、delay_samples分の遅延を実現します。
    例: delay_samples=3の場合、現在の出力は3サンプル前の入力に相当します。

    Args:
        signal: 入力信号配列
        delay_samples: 遅延サンプル数
        noise_std: ガウスノイズの標準偏差

    Returns:
        delayed: 遅延とノイズが付加された信号
    """
    if delay_samples < 0:
        raise ValueError("delay_samples must be non-negative")
    # FIFOバッファを作成（初期値0で埋める）
    buffer = deque([0.0] * delay_samples, maxlen=delay_samples or 1)
    delayed = np.zeros_like(signal)
    for k, value in enumerate(signal):
        buffer.append(value)  # 新しい値をバッファに追加（最古値は自動的に削除）
        # バッファの最古値（delay_samples前の値）を取り出してノイズを加える
        delayed[k] = buffer[0] + np.random.randn() * noise_std
    return delayed


def apply_inverse_compensation(y: np.ndarray, gain: float) -> np.ndarray:
    """逆伝達補償を適用して遅延を補償

    補償式: y_comp[k] = a·y[k] - (a-1)·y[k-1]
    ここで a = delay_samples (遅延サンプル数)

    理論的背景:
    - 遅延は z^(-d) として表現される（z変換）
    - その逆伝達関数 z^d を近似的に実現するため、1次の差分補償を使用
    - a が大きいほど補償が強くなるが、ノイズ増幅のリスクも増加

    Args:
        y: 遅延した測定信号
        gain: 逆補償ゲイン（通常は遅延サンプル数 d に設定）

    Returns:
        y_comp: 逆補償された信号（遅延が部分的にキャンセルされる）
    """
    if len(y) == 0:
        return np.array([])
    y_comp = np.zeros_like(y)
    prev = y[0]  # 前サンプルの値を保持
    y_comp[0] = prev  # 初期値はそのまま
    for k in range(1, len(y)):
        curr = y[k]
        # 逆補償式を適用: 現在値を強調し、前値を引き算することで先読み効果を得る
        y_comp[k] = gain * curr - (gain - 1.0) * prev
        prev = curr  # 次のループのために前値を更新
    return y_comp


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    """二乗平均平方根誤差（Root Mean Square Error）を計算

    RMSE = sqrt(mean((a - b)^2))
    信号の全体的な追従性能を評価する指標

    Args:
        a: 基準信号（真値）
        b: 比較対象信号（推定値や測定値）

    Returns:
        RMSE値（小さいほど良い追従性能を示す）
    """
    return float(np.sqrt(np.mean((a - b) ** 2)))


def estimate_lag(ref: np.ndarray, sig: np.ndarray, Ts: float) -> Tuple[int, float]:
    """相互相関を用いて2つの信号間の遅延を推定

    手順:
    1. 各信号の差分（微分近似）を計算 → 定数オフセットの影響を除去
    2. 差分信号から平均を引く → DC成分を除去
    3. 相互相関を計算（mode="full"で全範囲の遅延を探索）
    4. 相互相関が最大となる遅延を求める

    Args:
        ref: 基準信号（真値）
        sig: 比較対象信号（遅延が含まれる可能性のある信号）
        Ts: サンプリング周期 [s]

    Returns:
        lag_samples: 推定された遅延サンプル数（正の値は sig が ref より遅れている）
        lag_ms: 推定された遅延時間 [ms]
    """
    # 差分を計算（1階微分の近似）
    ref_d = np.diff(ref, prepend=ref[0])
    sig_d = np.diff(sig, prepend=sig[0])
    # 相互相関を計算（平均を引いてゼロ平均にする）
    c = np.correlate(sig_d - np.mean(sig_d), ref_d - np.mean(ref_d), mode="full")
    # 相互相関が最大となるラグを求める（中心からのオフセット）
    lag_samples = int(np.argmax(c) - (len(ref) - 1))
    lag_ms = lag_samples * Ts * 1000.0  # サンプル数から時間 [ms] に変換
    return lag_samples, lag_ms


def step_rise_time(t: np.ndarray, sig: np.ndarray, cfg: SimulationConfig, baseline: float) -> float:
    """ステップ応答の立ち上がり時間を計算（90%到達時刻）

    ステップ入力後、信号が最終値の90%に到達する時刻を求めます。
    これは過渡応答の速さを評価する標準的な指標です。

    Args:
        t: 時刻配列 [s]
        sig: 評価対象の信号
        cfg: シミュレーション設定（ステップ時刻と振幅を含む）
        baseline: ステップ前のベースライン値

    Returns:
        90%値に到達した時刻 [s]（到達しない場合は NaN）
    """
    idx_step = np.searchsorted(t, cfg.step_time)  # ステップ開始時刻のインデックス
    target = baseline + cfg.step_amp  # 最終目標値
    threshold = baseline + 0.9 * cfg.step_amp  # 90%到達判定の閾値
    # ステップ時刻以降で90%値を超える最初の時刻を探す
    for idx in range(idx_step, len(sig)):
        if sig[idx] >= threshold:
            return t[idx]
    return float("nan")  # 到達しなかった場合


def compute_metrics(
    t: np.ndarray, x: np.ndarray, y_delayed: np.ndarray, y_comp: np.ndarray, cfg: SimulationConfig
) -> Dict[str, float]:
    """補償あり・なしの両方について各種評価指標を計算

    以下の3種類の指標を計算します：
    1. RMSE: 全体的な追従誤差
    2. 相互相関ベース遅延推定: 信号の時間的ずれ
    3. 位相遅延（周波数領域）: 正弦波成分の遅れ
    4. ステップ応答立ち上がり時間: 過渡応答の速さ

    Args:
        t: 時刻配列
        x: 真の信号（基準）
        y_delayed: 遅延のみの信号（補償なし）
        y_comp: 逆補償後の信号
        cfg: シミュレーション設定

    Returns:
        各種評価指標を格納した辞書
    """
    metrics: Dict[str, float] = {}

    # 1. RMSEを計算（補償なし vs 補償あり）
    metrics["rmse_no_comp"] = rmse(x, y_delayed)
    metrics["rmse_inverse"] = rmse(x, y_comp)

    # 2. 相互相関による遅延推定（補償なし vs 補償あり）
    lag_no_samples, lag_no_ms = estimate_lag(x, y_delayed, cfg.Ts)
    lag_inv_samples, lag_inv_ms = estimate_lag(x, y_comp, cfg.Ts)
    metrics["lag_no_samples"] = lag_no_samples
    metrics["lag_no_ms"] = lag_no_ms
    metrics["lag_inv_samples"] = lag_inv_samples
    metrics["lag_inv_ms"] = lag_inv_ms

    # 3. 周波数領域での位相遅延（補償なし vs 補償あり）
    phase_no, phase_time_no = phase_delay_at_freq(x, y_delayed, cfg)
    phase_inv, phase_time_inv = phase_delay_at_freq(x, y_comp, cfg)
    metrics["phase_delay_no_rad"] = phase_no
    metrics["phase_delay_no_time"] = phase_time_no
    metrics["phase_delay_inv_rad"] = phase_inv
    metrics["phase_delay_inv_time"] = phase_time_inv

    # 4. ステップ応答の立ち上がり時間（補償なし vs 補償あり）
    baseline = x[np.searchsorted(t, cfg.step_time) - 1]  # ステップ前の値
    rise_true = step_rise_time(t, x, cfg, baseline)  # 真値の立ち上がり
    rise_no = step_rise_time(t, y_delayed, cfg, baseline)  # 補償なし
    rise_inv = step_rise_time(t, y_comp, cfg, baseline)  # 補償あり
    metrics["rise_time_true"] = rise_true
    metrics["rise_time_no"] = rise_no
    metrics["rise_time_inv"] = rise_inv
    metrics["rise_delay_no"] = rise_no - rise_true  # 真値からの遅れ（補償なし）
    metrics["rise_delay_inv"] = rise_inv - rise_true  # 真値からの遅れ（補償あり）
    return metrics


def phase_delay_at_freq(
    ref: np.ndarray, sig: np.ndarray, cfg: SimulationConfig
) -> Tuple[float, float]:
    """特定周波数におけるクロススペクトルの位相遅延を計算

    手順:
    1. FFTで周波数領域に変換
    2. 目標周波数（正弦波の周波数）に最も近いビンを探す
    3. クロススペクトル（sig の FFT × ref の FFT の共役）を計算
    4. その位相角から時間遅延を算出

    理論: 位相遅延 Δφ = -ω·Δt  →  Δt = -Δφ / ω

    Args:
        ref: 基準信号（真値）
        sig: 比較対象信号（遅延が含まれる）
        cfg: シミュレーション設定（正弦波周波数を含む）

    Returns:
        phase_rad: 位相差 [rad]（負の値は sig が遅れている）
        time_delay: 時間遅延 [s]（正の値は sig が遅れている）
    """
    # FFTの周波数軸を生成
    freqs = np.fft.rfftfreq(len(ref), cfg.Ts)
    target = cfg.sine_freq_hz  # 評価対象の周波数
    if target <= 0.0:
        return 0.0, 0.0
    # 目標周波数に最も近いビンのインデックスを取得
    idx = int(np.argmin(np.abs(freqs - target)))
    # クロススペクトルを計算（sig × conj(ref)）
    cross = np.fft.rfft(sig)[idx] * np.conj(np.fft.rfft(ref)[idx])
    # 位相角を取得
    phase_rad = float(np.angle(cross))
    # 位相角から時間遅延に変換
    time_delay = float(phase_rad / (2.0 * np.pi * target))
    return phase_rad, time_delay


def plot_results(
    t: np.ndarray,
    x: np.ndarray,
    y_delayed: np.ndarray,
    y_comp: np.ndarray,
    cfg: SimulationConfig,
) -> None:
    """結果を可視化（真値、遅延信号、補償後信号を重ねてプロット）

    3つの信号を同じグラフに描画して逆補償の効果を視覚的に確認します：
    - 真値（x）: プラントの実際の出力
    - 遅延信号（y_delayed）: 通信遅延が入った測定値（補償なし）
    - 補償後信号（y_comp）: 逆伝達補償を適用した結果

    Args:
        t: 時刻配列
        x: 真値
        y_delayed: 補償なし信号
        y_comp: 補償あり信号
        cfg: シミュレーション設定
    """
    plt.figure(figsize=(10, 5))
    # 遅延信号（補償なし）を半透明でプロット
    plt.plot(t, y_delayed, label=f"Delayed measurement (d={cfg.delay_samples})", alpha=0.7)
    # 補償後の信号を太線でプロット
    plt.plot(t, y_comp, label="After inverse compensation", linewidth=2)
    # 真値を太線でプロット（これが理想的な追従対象）
    plt.plot(t, x, label="True plant output x[k]", linewidth=2)
    plt.xlabel("Time [s]")
    plt.ylabel("Signal")
    plt.title("Inverse Compensation Demo (2nd-order system)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    # スクリプトと同じディレクトリに保存
    output_path = Path(__file__).resolve().parent / "inverse_comp_demo.png"
    plt.savefig(output_path, dpi=200)
    print(f"Plot saved to: {output_path}")
    plt.show()


def main() -> None:
    """メイン実行関数 - 逆伝達補償デモの全体フロー

    処理の流れ:
    1. シミュレーション設定の読み込み
    2. 入力信号の生成（正弦波 + ステップ）
    3. プラント応答の計算（2次系 - 質量-バネ-ダンパー系）
    4. 遅延とノイズの付加
    5. 逆伝達補償の適用
    6. 各種評価指標の計算
    7. 結果の表示とプロット

    重要: 逆補償は2次系の動特性には依存せず、純粋な通信遅延のみを補償します
    """
    np.random.seed(42)  # 再現性のため乱数シードを固定
    cfg = SimulationConfig()  # デフォルト設定を使用

    # ステップ1: 入力信号を生成（正弦波 + ステップ）
    t, u = build_input_signal(cfg)

    # ステップ2: 2次系のプラント応答を計算
    x = propagate_second_order(u, cfg)

    # ステップ3: 遅延とノイズを付加（HILS通信遅延のシミュレーション）
    y_delayed = apply_delay_and_noise(x, cfg.delay_samples, cfg.noise_std)

    # ステップ4: 逆伝達補償を適用して遅延をキャンセル
    y_comp = apply_inverse_compensation(y_delayed, cfg.inverse_gain)

    # ステップ5: 評価指標を計算
    metrics = compute_metrics(t, x, y_delayed, y_comp, cfg)

    # ステップ6: 結果を表示
    print("=== Mini HILS inverse compensation demo ===")
    print(f"Sampling period Ts: {cfg.Ts:.3f} s  |  Delay τ: {cfg.tau * 1000:.1f} ms")
    print(f"Delay samples d: {cfg.delay_samples}  |  Inverse gain a: {cfg.inverse_gain:.1f}")
    print()
    print(f"RMSE (no compensation)  : {metrics['rmse_no_comp']:.4f}")
    print(f"RMSE (inverse)          : {metrics['rmse_inverse']:.4f}")
    print(
        f"Lag  (no compensation)  : {metrics['lag_no_samples']} samples (~{metrics['lag_no_ms']:.1f} ms)"
    )
    print(
        f"Lag  (inverse)          : {metrics['lag_inv_samples']} samples (~{metrics['lag_inv_ms']:.1f} ms)"
    )
    print(
        f"Phase delay (no comp)   : {metrics['phase_delay_no_time'] * 1000:.1f} ms"
        f" ({metrics['phase_delay_no_rad']:.3f} rad)"
    )
    print(
        f"Phase delay (inverse)   : {metrics['phase_delay_inv_time'] * 1000:.1f} ms"
        f" ({metrics['phase_delay_inv_rad']:.3f} rad)"
    )
    print()
    print(
        f"Rise time (true)        : {metrics['rise_time_true']:.3f} s"
        f"  | reference step @ {cfg.step_time:.1f} s"
    )
    print(f"Rise delay (no comp)    : {metrics['rise_delay_no']:.3f} s")
    print(f"Rise delay (inverse)    : {metrics['rise_delay_inv']:.3f} s")

    # ステップ7: 結果をプロット
    plot_results(t, x, y_delayed, y_comp, cfg)


if __name__ == "__main__":
    # スクリプトとして直接実行された場合のみmain関数を実行
    # （モジュールとしてインポートされた場合は実行されない）
    main()
