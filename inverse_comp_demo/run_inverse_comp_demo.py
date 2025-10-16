"""Minimal HILS-like setup that highlights inverse transfer compensation.

Run this script to compare a fixed-delay measurement with and without inverse
compensation. Reported metrics include RMSE, cross-power phase-derived delay,
correlation-based lag estimates, and 90% step-rise timing. A plot overlays the true plant output, delayed
measurement, and compensated signal.
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
    Ts: float = 0.01
    Tend: float = 20.0
    tau: float = 0.15
    alpha: float = 0.05
    noise_std: float = 0.00
    sine_amp: float = 0.2
    sine_freq_hz: float = 0.3
    step_time: float = 2.0
    step_amp: float = 0.5

    @property
    def sample_count(self) -> int:
        return int(self.Tend / self.Ts)

    @property
    def delay_samples(self) -> int:
        return int(round(self.tau / self.Ts))

    @property
    def inverse_gain(self) -> float:
        return float(self.delay_samples)


def build_input_signal(cfg: SimulationConfig) -> Tuple[np.ndarray, np.ndarray]:
    t = np.arange(cfg.sample_count) * cfg.Ts
    sine = cfg.sine_amp * np.sin(2.0 * np.pi * cfg.sine_freq_hz * t)
    step = cfg.step_amp * (t >= cfg.step_time).astype(float)
    u = sine + step
    return t, u


def propagate_first_order(u: np.ndarray, cfg: SimulationConfig) -> np.ndarray:
    x = np.zeros_like(u)
    for k in range(1, len(u)):
        x[k] = (1.0 - cfg.alpha) * x[k - 1] + cfg.alpha * u[k - 1]
    return x


def apply_delay_and_noise(
    signal: np.ndarray, delay_samples: int, noise_std: float
) -> np.ndarray:
    if delay_samples < 0:
        raise ValueError("delay_samples must be non-negative")
    buffer = deque([0.0] * delay_samples, maxlen=delay_samples or 1)
    delayed = np.zeros_like(signal)
    for k, value in enumerate(signal):
        buffer.append(value)
        delayed[k] = buffer[0] + np.random.randn() * noise_std
    return delayed


def apply_inverse_compensation(y: np.ndarray, gain: float) -> np.ndarray:
    if len(y) == 0:
        return np.array([])
    y_comp = np.zeros_like(y)
    prev = y[0]
    y_comp[0] = prev
    for k in range(1, len(y)):
        curr = y[k]
        y_comp[k] = gain * curr - (gain - 1.0) * prev
        prev = curr
    return y_comp


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def estimate_lag(ref: np.ndarray, sig: np.ndarray, Ts: float) -> Tuple[int, float]:
    ref_d = np.diff(ref, prepend=ref[0])
    sig_d = np.diff(sig, prepend=sig[0])
    c = np.correlate(
        sig_d - np.mean(sig_d), ref_d - np.mean(ref_d), mode="full"
    )
    lag_samples = int(np.argmax(c) - (len(ref) - 1))
    lag_ms = lag_samples * Ts * 1000.0
    return lag_samples, lag_ms


def step_rise_time(
    t: np.ndarray, sig: np.ndarray, cfg: SimulationConfig, baseline: float
) -> float:
    idx_step = np.searchsorted(t, cfg.step_time)
    target = baseline + cfg.step_amp
    threshold = baseline + 0.9 * cfg.step_amp
    for idx in range(idx_step, len(sig)):
        if sig[idx] >= threshold:
            return t[idx]
    return float("nan")


def compute_metrics(
    t: np.ndarray, x: np.ndarray, y_delayed: np.ndarray, y_comp: np.ndarray, cfg: SimulationConfig
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    metrics["rmse_no_comp"] = rmse(x, y_delayed)
    metrics["rmse_inverse"] = rmse(x, y_comp)

    lag_no_samples, lag_no_ms = estimate_lag(x, y_delayed, cfg.Ts)
    lag_inv_samples, lag_inv_ms = estimate_lag(x, y_comp, cfg.Ts)
    metrics["lag_no_samples"] = lag_no_samples
    metrics["lag_no_ms"] = lag_no_ms
    metrics["lag_inv_samples"] = lag_inv_samples
    metrics["lag_inv_ms"] = lag_inv_ms

    phase_no, phase_time_no = phase_delay_at_freq(x, y_delayed, cfg)
    phase_inv, phase_time_inv = phase_delay_at_freq(x, y_comp, cfg)
    metrics["phase_delay_no_rad"] = phase_no
    metrics["phase_delay_no_time"] = phase_time_no
    metrics["phase_delay_inv_rad"] = phase_inv
    metrics["phase_delay_inv_time"] = phase_time_inv

    baseline = x[np.searchsorted(t, cfg.step_time) - 1]
    rise_true = step_rise_time(t, x, cfg, baseline)
    rise_no = step_rise_time(t, y_delayed, cfg, baseline)
    rise_inv = step_rise_time(t, y_comp, cfg, baseline)
    metrics["rise_time_true"] = rise_true
    metrics["rise_time_no"] = rise_no
    metrics["rise_time_inv"] = rise_inv
    metrics["rise_delay_no"] = rise_no - rise_true
    metrics["rise_delay_inv"] = rise_inv - rise_true
    return metrics


def phase_delay_at_freq(
    ref: np.ndarray, sig: np.ndarray, cfg: SimulationConfig
) -> Tuple[float, float]:
    freqs = np.fft.rfftfreq(len(ref), cfg.Ts)
    target = cfg.sine_freq_hz
    if target <= 0.0:
        return 0.0, 0.0
    idx = int(np.argmin(np.abs(freqs - target)))
    cross = np.fft.rfft(sig)[idx] * np.conj(np.fft.rfft(ref)[idx])
    phase_rad = float(np.angle(cross))
    time_delay = float(phase_rad / (2.0 * np.pi * target))
    return phase_rad, time_delay


def plot_results(
    t: np.ndarray,
    x: np.ndarray,
    y_delayed: np.ndarray,
    y_comp: np.ndarray,
    cfg: SimulationConfig,
) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(t, y_delayed, label=f"Delayed measurement (d={cfg.delay_samples})", alpha=0.7)
    plt.plot(t, y_comp, label="After inverse compensation", linewidth=2)
    plt.plot(t, x, label="True plant output x[k]", linewidth=2)
    plt.xlabel("Time [s]")
    plt.ylabel("Signal")
    plt.title("Inverse Compensation Demo")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    output_path = Path(__file__).resolve().parent / "inverse_comp_demo.png"
    plt.savefig(output_path, dpi=200)
    print(f"Plot saved to: {output_path}")
    plt.show()


def main() -> None:
    np.random.seed(42)
    cfg = SimulationConfig()

    t, u = build_input_signal(cfg)
    x = propagate_first_order(u, cfg)
    y_delayed = apply_delay_and_noise(x, cfg.delay_samples, cfg.noise_std)
    y_comp = apply_inverse_compensation(y_delayed, cfg.inverse_gain)

    metrics = compute_metrics(t, x, y_delayed, y_comp, cfg)

    print("=== Mini HILS inverse compensation demo ===")
    print(f"Sampling period Ts: {cfg.Ts:.3f} s  |  Delay τ: {cfg.tau*1000:.1f} ms")
    print(f"Delay samples d: {cfg.delay_samples}  |  Inverse gain a: {cfg.inverse_gain:.1f}")
    print()
    print(f"RMSE (no compensation)  : {metrics['rmse_no_comp']:.4f}")
    print(f"RMSE (inverse)          : {metrics['rmse_inverse']:.4f}")
    print(f"Lag  (no compensation)  : {metrics['lag_no_samples']} samples (~{metrics['lag_no_ms']:.1f} ms)")
    print(f"Lag  (inverse)          : {metrics['lag_inv_samples']} samples (~{metrics['lag_inv_ms']:.1f} ms)")
    print(
        f"Phase delay (no comp)   : {metrics['phase_delay_no_time']*1000:.1f} ms"
        f" ({metrics['phase_delay_no_rad']:.3f} rad)"
    )
    print(
        f"Phase delay (inverse)   : {metrics['phase_delay_inv_time']*1000:.1f} ms"
        f" ({metrics['phase_delay_inv_rad']:.3f} rad)"
    )
    print()
    print(
        f"Rise time (true)        : {metrics['rise_time_true']:.3f} s"
        f"  | reference step @ {cfg.step_time:.1f} s"
    )
    print(
        f"Rise delay (no comp)    : {metrics['rise_delay_no']:.3f} s"
    )
    print(
        f"Rise delay (inverse)    : {metrics['rise_delay_inv']:.3f} s"
    )

    plot_results(t, x, y_delayed, y_comp, cfg)


if __name__ == "__main__":
    main()
