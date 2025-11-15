"""
OrbitalEnvSimulator - 軌道力学環境シミュレーター

地球周回軌道の二体問題をシミュレート。
ニュートンの万有引力の法則に基づく軌道運動を計算。

実装内容:
- 二体問題の運動方程式（r̈ = -μ/r³ × r）
- 3次元位置・速度ベクトル
- 制御力（スラスタ推力）の入力
- RK4法による数値積分
"""

import mosaik_api
import numpy as np

meta = {
    "type": "time-based",
    "models": {
        "OrbitalSpacecraft": {
            "public": True,
            "params": [
                "mass",
                "mu",  # 中心天体の重力定数
                "initial_position",  # [x, y, z] or list
                "initial_velocity",  # [vx, vy, vz] or list
                "radius_earth",  # 地球半径 [m]
            ],
            "attrs": [
                # 入力
                "force_x",  # 推力 X成分 [N]
                "force_y",  # 推力 Y成分 [N]
                "force_z",  # 推力 Z成分 [N]
                "norm_force",  # 推力ノルム [N]
                # 出力（位置）
                "position_x",  # 位置 X [m]
                "position_y",  # 位置 Y [m]
                "position_z",  # 位置 Z [m]
                "position_norm",  # 動径距離 |r| [m]
                # 出力（速度）
                "velocity_x",  # 速度 X [m/s]
                "velocity_y",  # 速度 Y [m/s]
                "velocity_z",  # 速度 Z [m/s]
                "velocity_norm",  # 速度ノルム |v| [m/s]
                # 出力（加速度）
                "acceleration_x",  # 加速度 X [m/s^2]
                "acceleration_y",  # 加速度 Y [m/s^2]
                "acceleration_z",  # 加速度 Z [m/s^2]
                # 軌道要素
                "altitude",  # 高度（地表からの距離） [m]
                "semi_major_axis",  # 軌道長半径 [m]
                "eccentricity",  # 離心率
                "specific_energy",  # 比エネルギー [J/kg]
            ],
        },
    },
}


class OrbitalEnvSimulator(mosaik_api.Simulator):
    """
    軌道力学環境シミュレーター

    二体問題の運動方程式:
        r̈ = -μ/r³ × r + F/m

    ここで:
        r: 位置ベクトル [m]
        μ: 中心天体の重力定数 [m³/s²]
        F: 制御力（推力） [N]
        m: 衛星質量 [kg]

    入力:
        force_x, force_y, force_z: 推力ベクトル [N]

    出力:
        position_x, position_y, position_z: 位置 [m]
        velocity_x, velocity_y, velocity_z: 速度 [m/s]
        acceleration_x, acceleration_y, acceleration_z: 加速度 [m/s^2]
        軌道要素（altitude, semi_major_axis, eccentricity, specific_energy）
    """

    def __init__(self):
        super().__init__(meta)
        self.entities = {}
        self.step_size = 1  # デフォルト: 1ステップ
        self.time = 0

    def init(self, sid, time_resolution=0.001, step_size=1):
        """
        初期化

        Args:
            sid: シミュレーターID
            time_resolution: 時間解像度 [s]（例: 0.001 = 1ms）
            step_size: ステップサイズ [時間単位]
        """
        self.sid = sid
        self.time_resolution = time_resolution
        self.step_size = step_size
        return self.meta

    def create(
        self,
        num,
        model,
        mass=500.0,
        mu=3.986004418e14,  # 地球の重力定数 [m³/s²]
        initial_position=None,
        initial_velocity=None,
        radius_earth=6378137.0,  # 地球半径 [m]
    ):
        """
        軌道衛星エンティティの作成

        Args:
            num: 作成数
            model: モデル名
            mass: 衛星質量 [kg]
            mu: 中心天体の重力定数 [m³/s²] (デフォルト: 地球)
            initial_position: 初期位置 [x, y, z] [m]
            initial_velocity: 初期速度 [vx, vy, vz] [m/s]
            radius_earth: 地球半径 [m] (高度計算用)

        デフォルト軌道:
            高度400km円軌道 (ISS相当)
        """
        entities = []

        # デフォルト: 高度400kmの円軌道
        if initial_position is None:
            altitude = 400e3  # 400 km
            r = radius_earth + altitude
            initial_position = [r, 0.0, 0.0]

        if initial_velocity is None:
            r = np.linalg.norm(initial_position)
            v_circular = np.sqrt(mu / r)  # 円軌道速度
            initial_velocity = [0.0, v_circular, 0.0]

        for i in range(num):
            eid = f"{model}_{i}"

            # 状態ベクトル（位置・速度）
            position = np.array(initial_position, dtype=float)
            velocity = np.array(initial_velocity, dtype=float)

            self.entities[eid] = {
                "mass": mass,
                "mu": mu,
                "radius_earth": radius_earth,
                "position": position.copy(),
                "velocity": velocity.copy(),
                "acceleration": np.zeros(3),
                "force": np.zeros(3),
            }

            entities.append({"eid": eid, "type": model})

            r_norm = np.linalg.norm(position)
            v_norm = np.linalg.norm(velocity)
            altitude = r_norm - radius_earth

            print(f"[OrbitalEnvSim] Created {eid}:")
            print(f"  Mass: {mass} kg")
            print(f"  Position: {position} m")
            print(f"  Velocity: {velocity} m/s")
            print(f"  Altitude: {altitude / 1e3:.2f} km")
            print(f"  Orbital radius: {r_norm / 1e3:.2f} km")
            print(f"  Orbital velocity: {v_norm:.2f} m/s")

        return entities

    def step(self, time, inputs, max_advance=None):
        """
        シミュレーションステップ（RK4法による軌道積分）

        Args:
            time: 現在時刻 [時間単位]
            inputs: 入力データ
        """
        self.time = time
        dt = self.step_size * self.time_resolution

        for eid, entity in self.entities.items():
            # 入力: 推力の受信
            force = np.zeros(3)
            if eid in inputs:
                for axis, attr in enumerate(["force_x", "force_y", "force_z"]):
                    if attr in inputs[eid]:
                        force_value = list(inputs[eid][attr].values())[0]
                        force[axis] = force_value if force_value is not None else 0.0

            entity["force"] = force

            # RK4法による数値積分
            r = entity["position"]
            v = entity["velocity"]

            # RK4積分で状態を更新
            new_r, new_v = self._rk4_step(r, v, force, entity, dt)

            # 状態更新
            entity["position"] = new_r
            entity["velocity"] = new_v
            entity["acceleration"] = self._acceleration(new_r, new_v, force, entity)

        return time + self.step_size

    def _rk4_step(self, r, v, force, entity, dt):
        """
        RK4法による1ステップの積分

        Args:
            r: 位置ベクトル [m]
            v: 速度ベクトル [m/s]
            force: 推力ベクトル [N]
            entity: エンティティデータ
            dt: 時間ステップ [s]

        Returns:
            (new_r, new_v): 更新後の位置・速度ベクトル
        """
        # k1
        k1_v = self._acceleration(r, v, force, entity)
        k1_r = v

        # k2
        k2_v = self._acceleration(r + 0.5 * dt * k1_r, v + 0.5 * dt * k1_v, force, entity)
        k2_r = v + 0.5 * dt * k1_v

        # k3
        k3_v = self._acceleration(r + 0.5 * dt * k2_r, v + 0.5 * dt * k2_v, force, entity)
        k3_r = v + 0.5 * dt * k2_v

        # k4
        k4_v = self._acceleration(r + dt * k3_r, v + dt * k3_v, force, entity)
        k4_r = v + dt * k3_v

        # 状態更新
        new_r = r + (dt / 6.0) * (k1_r + 2 * k2_r + 2 * k3_r + k4_r)
        new_v = v + (dt / 6.0) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)

        return new_r, new_v

    def _acceleration(self, r, v, force, entity):
        """
        加速度の計算

        Args:
            r: 位置ベクトル [m]
            v: 速度ベクトル [m/s] (未使用だが将来の擾乱用に残す)
            force: 推力ベクトル [N]
            entity: エンティティデータ

        Returns:
            加速度ベクトル [m/s²]
        """
        r_norm = np.linalg.norm(r)
        mu = entity["mu"]
        mass = entity["mass"]

        # 重力加速度: a_grav = -μ/r³ × r
        a_grav = -mu / (r_norm**3) * r

        # 推力加速度: a_thrust = F/m
        a_thrust = force / mass

        return a_grav + a_thrust

    def get_data(self, outputs):
        """
        データ取得

        Args:
            outputs: 要求データ仕様
        """
        data = {}

        for eid, attrs in outputs.items():
            if eid not in self.entities:
                continue

            entity = self.entities[eid]
            data[eid] = {}

            # 現在の状態
            r = entity["position"]
            v = entity["velocity"]
            a = entity["acceleration"]

            # ノルム計算
            r_norm = np.linalg.norm(r)
            v_norm = np.linalg.norm(v)

            # 軌道要素の計算
            mu = entity["mu"]
            radius_earth = entity["radius_earth"]

            # 比エネルギー: ε = v²/2 - μ/r
            specific_energy = 0.5 * v_norm**2 - mu / r_norm

            # 軌道長半径: a = -μ/(2ε)
            if specific_energy < 0:  # 楕円軌道
                semi_major_axis = -mu / (2 * specific_energy)
            else:
                semi_major_axis = float("inf")  # 双曲線軌道

            # 角運動量ベクトル: h = r × v
            h = np.cross(r, v)
            np.linalg.norm(h)

            # 離心率ベクトル: e = (v × h)/μ - r/|r|
            e_vec = np.cross(v, h) / mu - r / r_norm
            eccentricity = np.linalg.norm(e_vec)

            # 高度
            altitude = r_norm - radius_earth

            # データマッピング
            attr_map = {
                # 位置
                "position_x": r[0],
                "position_y": r[1],
                "position_z": r[2],
                "position_norm": r_norm,
                # 速度
                "velocity_x": v[0],
                "velocity_y": v[1],
                "velocity_z": v[2],
                "velocity_norm": v_norm,
                # 加速度
                "acceleration_x": a[0],
                "acceleration_y": a[1],
                "acceleration_z": a[2],
                #推力
                "force_x": entity["force"][0],
                "force_y": entity["force"][1],
                "force_z": entity["force"][2],
                "norm_force": np.linalg.norm(entity["force"]),
                # 軌道要素
                "altitude": altitude,
                "semi_major_axis": semi_major_axis,
                "eccentricity": eccentricity,
                "specific_energy": specific_energy,
            }

            for attr in attrs:
                data[eid][attr] = attr_map.get(attr, None)

        return data


if __name__ == "__main__":
    mosaik_api.start_simulator(OrbitalEnvSimulator())
