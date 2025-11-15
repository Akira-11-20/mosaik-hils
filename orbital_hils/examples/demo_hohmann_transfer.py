"""
ホーマン遷移デモスクリプト

400km円軌道から600km円軌道への遷移をシミュレーション。

実行方法:
    cd orbital_hils
    uv run python examples/demo_hohmann_transfer.py
"""

import sys
from pathlib import Path

import numpy as np

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.orbital_parameters import CONFIG_ISS, CelestialBodyConstants
from models.thrust_model import HohmannThrustModel


def demo_hohmann_transfer():
    """ホーマン遷移のデモ"""
    print("=" * 70)
    print("Hohmann Transfer Demonstration")
    print("=" * 70)
    print("\n📡 Mission: Transfer from 400km to 600km circular orbit\n")

    # 物理定数
    constants = CelestialBodyConstants()
    mu = constants.MU_EARTH
    radius_earth = constants.RADIUS_EARTH

    # 初期軌道（400km円軌道）
    initial_altitude = 400e3  # 400km
    target_altitude = 600e3  # 600km

    # 衛星パラメータ
    spacecraft_mass = 500.0  # kg
    max_thrust = 1.0  # N

    # ホーマン遷移モデルを作成
    print("🛠️  Creating Hohmann transfer model...")
    hohmann_model = HohmannThrustModel(
        mu=mu,
        initial_altitude=initial_altitude,
        target_altitude=target_altitude,
        radius_body=radius_earth,
        spacecraft_mass=spacecraft_mass,
        max_thrust=max_thrust,
        start_time=10.0,  # 10秒後に開始
    )

    # 初期状態（400km円軌道）
    r_initial = radius_earth + initial_altitude
    v_circular = np.sqrt(mu / r_initial)

    # ECI座標系での初期位置・速度
    position = np.array([r_initial, 0.0, 0.0])
    velocity = np.array([0.0, v_circular, 0.0])

    print(f"\n📍 Initial state:")
    print(f"   Position: {position / 1e3} km")
    print(f"   Velocity: {velocity} m/s")
    print(f"   Orbital speed: {v_circular:.2f} m/s")

    # 遷移状態を取得
    status = hohmann_model.get_status()

    print(f"\n🚀 Hohmann transfer parameters:")
    print(f"   ΔV1 (first burn):  {status['delta_v1']:+.2f} m/s")
    print(f"   ΔV2 (second burn): {status['delta_v2']:+.2f} m/s")
    print(f"   Total ΔV:          {status['total_delta_v']:.2f} m/s")
    print(f"   Transfer time:     {status['transfer_time']:.2f} s ({status['transfer_time'] / 60:.2f} min)")
    print(f"   Burn1 duration:    {status['burn1_duration']:.2f} s")
    print(f"   Burn2 duration:    {status['burn2_duration']:.2f} s")

    # 簡易的なタイムラインシミュレーション
    print(f"\n⏱️  Transfer timeline:")
    print(f"   t = 10.0s - {10.0 + status['burn1_duration']:.2f}s : First burn (velocity increase)")
    print(
        f"   t = {10.0 + status['burn1_duration']:.2f}s - {10.0 + status['transfer_time']:.2f}s : Coast phase (elliptical transfer)"
    )
    print(
        f"   t = {10.0 + status['transfer_time']:.2f}s - {10.0 + status['transfer_time'] + status['burn2_duration']:.2f}s : Second burn (circularization)"
    )

    # 推力計算のテスト（各フェーズ）
    print(f"\n🔥 Thrust calculation test:")

    # フェーズ1: 第1バーン前（ゼロ推力）
    t1 = 5.0
    thrust1 = hohmann_model.calculate_thrust(position, velocity, time=t1)
    print(f"   t={t1:.2f}s (before transfer): thrust = {thrust1} N")

    # フェーズ2: 第1バーン中
    t2 = 12.0
    thrust2 = hohmann_model.calculate_thrust(position, velocity, time=t2)
    print(f"   t={t2:.2f}s (first burn):      thrust = {thrust2} N (magnitude: {np.linalg.norm(thrust2):.3f} N)")

    # フェーズ3: コースト中（ゼロ推力）
    t3 = 1000.0
    thrust3 = hohmann_model.calculate_thrust(position, velocity, time=t3)
    print(f"   t={t3:.2f}s (coast phase):     thrust = {thrust3} N")

    # フェーズ4: 第2バーン中
    t4 = 10.0 + status["transfer_time"] + 5.0
    thrust4 = hohmann_model.calculate_thrust(position, velocity, time=t4)
    print(f"   t={t4:.2f}s (second burn):     thrust = {thrust4} N (magnitude: {np.linalg.norm(thrust4):.3f} N)")

    # フェーズ5: 遷移完了後（ゼロ推力）
    t5 = 10.0 + status["transfer_time"] + status["burn2_duration"] + 100.0
    thrust5 = hohmann_model.calculate_thrust(position, velocity, time=t5)
    print(f"   t={t5:.2f}s (completed):        thrust = {thrust5} N")

    print(f"\n✅ Hohmann transfer demo completed!")
    print(f"\n💡 To run a full HILS simulation with Hohmann transfer:")
    print(f"   1. Update config/orbital_parameters.py to use HohmannThrustModel")
    print(f"   2. Run: cd orbital_hils && uv run python main.py")


if __name__ == "__main__":
    demo_hohmann_transfer()
