# HILS遅延・ジッタ補償シミュレーション 実装計画書

## 1. プロジェクト概要

### 1.1 目的
Mosaik上で「通信遅延・ジッタの影響と補償手法」を検証するHILSシミュレーション環境を構築する。将来的に実際のスラスタハードウェアを統合し、完全なHILSシステムとして運用できるよう、標準化されたインターフェースで設計する。

**最終ゴール**: PlantSimを実機スラスタ試験装置に置き換え、実測推力値を用いた宇宙環境シミュレーションを実現する。

### 1.2 検証項目
- 通信遅延（固定遅延 + ジッター）が制御性能に与える影響
- パケットロス・順序入れ替えの影響
- 補償手法の効果検証
  - ZOH（Zero-Order Hold）
  - 線形補間
  - 先行送出（Command Advance）
  - Nowcasting（軽量予測）
  - チャネル等化（複数信号の時刻揃え）

### 1.3 成果物
- 各シミュレータの実装（ControllerSim, BridgeSim, PlantSim, LoggingSim）
- 統計データ（実効遅延、ジッター、欠損率、制御誤差）
- 可視化（時系列プロット、統計グラフ、トポロジ図）
- 技術レポート（補償手法の定量評価）

---

## 2. システムアーキテクチャ

### 2.1 全体構成図（本物のHILSアーキテクチャ）

```
┌─────────────────────────────────────────────────────────┐
│                    ControllerSim                        │
│  ┌────────────────────────────────────────────────┐    │
│  │  PD制御器                                       │    │
│  │  - 姿勢誤差計算                                 │    │
│  │  - トルク/推力コマンド生成                      │    │
│  └────────────────────────────────────────────────┘    │
│                         ↓ cmd_thrust/torque             │
│  ┌────────────────────────────────────────────────┐    │
│  │  6DoF運動シミュレーション                       │    │
│  │  - 姿勢伝播（四元数）                           │    │
│  │  - 軌道伝播（位置・速度）                       │    │
│  │  - ★実測推力を使用（thrust_actual）            │    │
│  └────────────────────────────────────────────────┘    │
│                         ↓                               │
│  ┌────────────────────────────────────────────────┐    │
│  │  宇宙環境モデル                                 │    │
│  │  - 重力（2体問題）                              │    │
│  │  - 重力勾配トルク                               │    │
│  │  - 大気ドラッグ（任意）                         │    │
│  │  - 太陽輻射圧（任意）                           │    │
│  └────────────────────────────────────────────────┘    │
│                         ↓ quat, omega, pos, vel         │
└─────────────────────────────────────────────────────────┘
                          ↓ cmd_thrust/torque
                          ↓
                 ┌────────────────┐
                 │   BridgeSim    │
                 │                │
                 │  [遅延/補償]   │
                 │  - コマンド経路│
                 │  - センシング  │
                 │    経路        │
                 └────────────────┘
                          ↓ cmd_thrust_eff/torque_eff
                          ↓
                 ┌────────────────┐
                 │   PlantSim     │◄─── 将来：実機スラスタ試験装置
                 │                │
                 │ (スラスタ試験台)│
                 │  - 推力計測    │
                 │  - トルク計測  │
                 └────────────────┘
                          ↓ thrust_actual, torque_actual
                          ↓
                 ┌────────────────┐
                 │   BridgeSim    │
                 │  [遅延/補償]   │
                 └────────────────┘
                          ↓ thrust_actual_delayed
                          ↓
                 ┌─────────────────────────────────────┐
                 │  ControllerSim (6DoF運動部)         │
                 │  ★実測推力で運動方程式を更新       │
                 └─────────────────────────────────────┘

                          ↓ 全データ
                 ┌────────────────┐
                 │  LoggingSim    │
                 │  (統計・記録)  │
                 └────────────────┘
```

**データフローの説明**:
1. **ControllerSim**: 制御則でコマンド計算 → BridgeSim
2. **BridgeSim**: 遅延・ジッター適用 → PlantSim
3. **PlantSim**: 実際のスラスタ出力を計測（将来は実機） → BridgeSim
4. **BridgeSim**: 遅延・ジッター適用 → ControllerSim
5. **ControllerSim（6DoF部）**: ★実測推力を使って運動方程式を更新 ← **これがHILSの本質！**
6. **ControllerSim**: 更新された姿勢・速度で制御則を実行（ループ）

### 2.2 Mosaikシミュレーション設定

#### 時間設定
```python
TIME_RESOLUTION = 0.01  # 10 ms = 0.01 秒（mosaikの時間単位）
CONTROLLER_STEP = 1      # 1 step = 10 ms (100 Hz)
PLANT_STEP = 1           # 1 step = 10 ms
BRIDGE_STEP = 1          # 1 step = 10 ms（補間精度重視なら0.5も可）
LOGGING_STEP = 1         # 1 step = 10 ms
ENV_STEP = 100           # 100 steps = 1秒（環境更新は低頻度）
SIMULATION_DURATION = 30000  # 30000 steps = 300秒
```

#### Mosaikの制約と対応
| 制約 | 対応策 |
|------|--------|
| 時刻は整数型 | TIME_RESOLUTION=0.01で10ms単位を表現 |
| 循環依存の解決 | `time_shifted=True`または適切なstep_size設計 |
| 同一ステップ内の双方向通信不可 | BridgeSimで1ステップ遅延を許容 |
| タイムスタンプ伝搬なし | BridgeSimから明示的にts_*属性を出力 |

---

## 3. 各シミュレータの詳細設計

### 3.1 ControllerSim（統合制御・運動シミュレータ）

#### 役割
**制御器、6DoF運動シミュレーション、環境モデルを統合したシミュレータ**。PlantSimからの実測推力を使って宇宙環境をシミュレーションする。

#### 入力（Inputs）
```python
thrust_actual: [3]     # 実測推力 [N] (Body frame) ← PlantSimから（遅延込み）
torque_actual: [3]     # 実測トルク [N·m] (Body frame) ← PlantSimから（遅延込み）
```

#### 出力（Outputs）
```python
# 制御コマンド（PlantSim向け）
cmd_thrust: [3]        # 推力コマンド [N] (Body frame)
cmd_torque: [3]        # トルクコマンド [N·m] (Body frame)
ts_cmd: float          # コマンド生成時刻（mosaikステップ数）

# 運動状態（可視化・ロギング向け）
quat: [4]              # 四元数（Body→Inertial）
omega: [3]             # 角速度 [rad/s] (Body frame)
pos: [3]               # 位置 [m] (Inertial frame)
vel: [3]               # 速度 [m/s] (Inertial frame)
```

#### パラメータ
```python
kp_attitude: float = 10.0      # 姿勢比例ゲイン
kd_attitude: float = 2.0       # 姿勢微分ゲイン
target_quat: [4] = [1,0,0,0]   # 目標姿勢（初期値：慣性系一致）
max_torque: float = 0.1        # 最大トルク制限 [N·m]
```

#### 実装クラス
```python
class ControllerSimulator(mosaik_api.Simulator):
    def __init__(self):
        self.step_size = 1  # 10 ms
        self.entities = {}

    def create(self, num, model, **params):
        # 統合エンティティを作成
        # - 制御器パラメータ（kp, kd）
        # - 衛星パラメータ（mass, inertia）
        # - 初期状態（quat, omega, pos, vel）

    def step(self, time, inputs, max_advance):
        # 1. PlantSimから実測推力を取得（thrust_actual, torque_actual）
        # 2. ★実測推力を使って6DoF運動方程式を更新
        #    - 姿勢伝播（四元数）
        #    - 軌道伝播（位置・速度）
        #    - 環境外乱（重力、SRP、ドラッグ等）
        # 3. 更新された姿勢・角速度でPD制御則を実行
        #    τ = -Kp * q_err - Kd * ω
        # 4. 新しいコマンドを出力
        return time + self.step_size
```

#### 実装の詳細

##### 1. 運動方程式の更新（実測推力を使用）
```python
def _update_dynamics(self, thrust_actual, torque_actual, dt):
    """実測推力/トルクを使って運動状態を更新"""

    # Body frameからInertial frameへの変換
    R_BI = quaternion_to_rotation_matrix(self.quat)
    thrust_inertial = R_BI @ thrust_actual

    # 並進運動（重力含む）
    r = np.linalg.norm(self.pos)
    g_accel = -MU_EARTH / r**3 * self.pos
    total_accel = thrust_inertial / self.mass + g_accel

    self.vel += total_accel * dt
    self.pos += self.vel * dt

    # 回転運動（ジャイロトルク含む）
    I = np.diag(self.inertia)
    I_omega = I @ self.omega
    gyro_torque = np.cross(self.omega, I_omega)

    d_omega = np.linalg.solve(I, torque_actual - gyro_torque)
    self.omega += d_omega * dt

    # 四元数更新
    omega_quat = [0, *self.omega]
    q_dot = 0.5 * quaternion_multiply(omega_quat, self.quat)
    self.quat += q_dot * dt
    self.quat /= np.linalg.norm(self.quat)  # 正規化
```

##### 2. 制御則（更新された状態を使用）
```python
def _compute_control(self):
    """PD制御則でコマンドを計算"""

    # 四元数誤差
    q_err = quaternion_multiply(self.target_quat, quaternion_conjugate(self.quat))
    q_err_vector = q_err[1:4]

    # PD制御
    torque = -self.kp_attitude * q_err_vector - self.kd_attitude * self.omega

    # 飽和処理
    torque = np.clip(torque, -self.max_torque, self.max_torque)

    # 推力（姿勢制御のみの場合は0、位置制御なら別途計算）
    thrust = np.array([0.0, 0.0, 0.0])

    return thrust, torque
```

---

### 3.2 BridgeSim（遅延・補償シミュレータ）

#### 役割
**HILS特有の通信遅延・ジッター・補償を一手に引き受ける時間整合ノード**。

#### 入力（Inputs）

**コマンドチャネル（Controller → Plant）**
```python
cmd_thrust: [3]     # 推力コマンド [N]
cmd_torque: [3]     # トルクコマンド [N·m]
ts_cmd: float       # コマンド生成時刻
```

**センシングチャネル（Plant → Controller）**
```python
thrust_actual: [3]     # 実測推力 [N] (Body frame)
torque_actual: [3]     # 実測トルク [N·m] (Body frame)
ts_sense: float        # 計測時刻
```

#### 出力（Outputs）

**Plant向け（遅延・補償適用後）**
```python
cmd_thrust_eff: [3]     # 実効推力コマンド [N]
cmd_torque_eff: [3]     # 実効トルクコマンド [N·m]
ts_cmd_sent: float      # 送信時刻（実際の到着時刻）
delay_cmd: float        # 実際に発生した遅延 [s]
```

**Controller向け（時刻揃え後）**
```python
thrust_actual_delayed: [3]  # 時刻揃え後の実測推力 [N]
torque_actual_delayed: [3]  # 時刻揃え後の実測トルク [N·m]
ts_sense_aligned: float     # 揃えた時刻
delay_sense: float          # センシング遅延 [s]
```

**統計情報（Logging向け）**
```python
stats: dict = {
    'cmd_delay_mean': float,      # コマンド遅延平均 [s]
    'cmd_delay_std': float,       # コマンド遅延標準偏差（ジッター） [s]
    'sense_delay_mean': float,    # センシング遅延平均 [s]
    'sense_delay_std': float,     # センシング遅延標準偏差 [s]
    'cmd_drop_count': int,        # コマンドドロップ数
    'sense_drop_count': int,      # センシングドロップ数
    'cmd_total_count': int,       # 総コマンド数
    'sense_total_count': int,     # 総センシング数
}
```

#### パラメータ
```python
# 遅延設定
delay_cmd_s: float = 0.05           # コマンド経路の基本遅延 [s] (50 ms)
jitter_cmd_std: float = 0.01        # コマンドジッター標準偏差 [s] (10 ms)
delay_sense_s: float = 0.03         # センシング経路の基本遅延 [s] (30 ms)
jitter_sense_std: float = 0.005     # センシングジッター標準偏差 [s] (5 ms)

# パケットロス
drop_prob: float = 0.01             # ドロップ確率 (1%)
preserve_order: bool = True         # パケット順序保持（False=順序入れ替え許可）

# 補間・補償
interp_mode: str = 'zoh'            # 補間モード: 'zoh', 'linear', 'fir'
cmd_advance_s: float = 0.0          # 先行送出時間 [s] (0 = 無効)
nowcast_horizon_s: float = 0.0      # Nowcasting予測時間 [s] (0 = 無効)
equalize_to_s: float = 0.0          # チャネル等化目標時刻 [s] (0 = 無効)

# FIRフィルタ（interp_mode='fir'時）
fir_taps: int = 5                   # FIRタップ数
```

#### 内部状態
```python
# キュー（遅延バッファ）
self.cmd_queue: List[Tuple[float, dict]]      # [(arrival_time, data), ...]
self.sense_queue: List[Tuple[float, dict]]    # [(arrival_time, data), ...]

# ZOH（Zero-Order Hold）状態
self.zoh_cmd: dict = {}    # 最後に有効だったコマンド
self.zoh_sense: dict = {}  # 最後に有効だったセンシング値

# 線形補間用履歴
self.sense_history: deque = deque(maxlen=2)   # [(time, data), ...]

# FIRフィルタバッファ
self.fir_buffer: deque = deque(maxlen=fir_taps)

# Nowcasting予測器
self.predictor: SimplePredictor = None  # 線形外挿または定速度モデル

# 統計情報
self.stats_window: deque = deque(maxlen=100)  # 移動窓統計用
```

#### 実装ロジック

##### step()メインフロー
```python
def step(self, time, inputs, max_advance):
    # 1. 入力受信と遅延適用
    self._receive_inputs(time, inputs)

    # 2. 時刻timeに到着すべきパケットを取り出し
    cmd_available = self._retrieve_packets(self.cmd_queue, time)
    sense_available = self._retrieve_packets(self.sense_queue, time)

    # 3. ドロップ判定
    cmd_available = self._apply_packet_loss(cmd_available)
    sense_available = self._apply_packet_loss(sense_available)

    # 4. 補間・補償処理
    cmd_eff = self._compensate_cmd(cmd_available, time)
    sense_aligned = self._compensate_sense(sense_available, time)

    # 5. 統計更新
    self._update_statistics(time)

    # 6. 出力データをcache（mosaik get_data用）
    self._cache_outputs(cmd_eff, sense_aligned)

    return time + self.step_size
```

##### 遅延適用
```python
def _receive_inputs(self, time, inputs):
    for eid, attrs in inputs.items():
        if 'cmd_thrust' in attrs:
            # コマンド経路の遅延
            delay = self.delay_cmd_s + np.random.normal(0, self.jitter_cmd_std)
            delay = max(0, delay)  # 負の遅延を防ぐ
            arrival_time = time + delay / TIME_RESOLUTION  # mosaikステップ数に変換

            data = {
                'cmd_thrust': attrs['cmd_thrust'][src_id],
                'cmd_torque': attrs['cmd_torque'][src_id],
                'ts_sent': time,
            }
            self.cmd_queue.append((arrival_time, data))

        if 'quat' in attrs:
            # センシング経路の遅延
            delay = self.delay_sense_s + np.random.normal(0, self.jitter_sense_std)
            delay = max(0, delay)
            arrival_time = time + delay / TIME_RESOLUTION

            data = {
                'quat': attrs['quat'][src_id],
                'omega': attrs['omega'][src_id],
                # ... 他のセンサ値
                'ts_measured': time,
            }
            self.sense_queue.append((arrival_time, data))
```

##### パケット取り出し
```python
def _retrieve_packets(self, queue, time):
    available = []
    remaining = []

    for arrival_time, data in queue:
        if arrival_time <= time:
            available.append((arrival_time, data))
        else:
            remaining.append((arrival_time, data))

    # 順序保持オプション
    if self.preserve_order:
        available.sort(key=lambda x: x[0])

    queue[:] = remaining
    return available
```

##### 補償処理（コマンド）
```python
def _compensate_cmd(self, cmd_available, time):
    if not cmd_available:
        # パケット到着なし → ZOH
        return self.zoh_cmd

    # 最新のパケットを使用
    latest = cmd_available[-1][1]

    # 先行送出（Command Advance）
    if self.cmd_advance_s > 0:
        # 簡易実装: 現在のコマンドを未来時刻用として扱う
        # （本格実装では予測軌道を使用）
        latest['ts_cmd_sent'] = time + self.cmd_advance_s / TIME_RESOLUTION
    else:
        latest['ts_cmd_sent'] = time

    # ZOH状態を更新
    self.zoh_cmd = latest

    return latest
```

##### 補償処理（センシング）
```python
def _compensate_sense(self, sense_available, time):
    if not sense_available:
        # パケット到着なし → 補間モードに応じた処理
        if self.interp_mode == 'zoh':
            return self.zoh_sense
        elif self.interp_mode == 'linear' and len(self.sense_history) >= 2:
            # 線形外挿
            return self._linear_extrapolate(time)
        else:
            return self.zoh_sense

    # 最新データを履歴に追加
    latest = sense_available[-1][1]
    self.sense_history.append((time, latest))

    # Nowcasting（予測補正）
    if self.nowcast_horizon_s > 0:
        latest = self._apply_nowcasting(latest, time)

    # チャネル等化（複数信号の時刻揃え）
    if self.equalize_to_s > 0:
        latest = self._apply_equalization(latest, time)

    # ZOH状態を更新
    self.zoh_sense = latest

    return latest
```

##### Nowcasting（簡易実装）
```python
def _apply_nowcasting(self, data, time):
    # 定速度モデル: x(t+Δt) = x(t) + v(t) * Δt
    if len(self.sense_history) < 2:
        return data  # 履歴不足

    (t1, data1), (t2, data2) = self.sense_history[-2:]
    dt = (t2 - t1) * TIME_RESOLUTION

    if dt <= 0:
        return data

    # 速度推定
    vel_est = (data2['pos'] - data1['pos']) / dt

    # 未来予測
    forecast_horizon = self.nowcast_horizon_s
    data_forecast = data.copy()
    data_forecast['pos'] = data['pos'] + vel_est * forecast_horizon
    data_forecast['ts_predicted'] = time + forecast_horizon / TIME_RESOLUTION

    return data_forecast
```

##### 統計更新
```python
def _update_statistics(self, time):
    # 移動窓で統計計算
    # cmd_delay_mean, cmd_delay_std, drop_rateなどを計算
    # self.cached_statsに保存
    pass
```

---

### 3.3 PlantSim（スラスタ試験台シミュレータ）

#### 役割
**スラスタ試験装置のシミュレータ。コマンドを受けて推力・トルクを計測する**。将来的に実機ハードウェアに置き換え可能なインターフェースを提供。

#### 入力（Inputs）
```python
cmd_thrust_eff: [3]     # 実効推力コマンド [N] (Body frame) ← BridgeSimから（遅延込み）
cmd_torque_eff: [3]     # 実効トルクコマンド [N·m] (Body frame)
```

#### 出力（Outputs）
```python
thrust_actual: [3]      # 実測推力 [N] (Body frame) ★これをControllerSimに送る
torque_actual: [3]      # 実測トルク [N·m] (Body frame)
ts_sense: float         # 計測時刻（現在時刻）
```

#### パラメータ
```python
# スラスタ特性
thrust_gain: [3] = [1.0, 1.0, 1.0]    # 推力ゲイン（理想値からのずれ）
torque_gain: [3] = [1.0, 1.0, 1.0]    # トルクゲイン
thrust_noise_std: float = 0.001       # 推力計測ノイズ [N]
torque_noise_std: float = 0.0001      # トルク計測ノイズ [N·m]

# 応答遅れ（1次遅れ系）
time_constant: float = 0.01           # 時定数 [s] (10ms)

# 飽和・不感帯
thrust_deadzone: float = 0.001        # 不感帯 [N]
thrust_saturation: float = 1.0        # 飽和値 [N]
```

#### 内部状態
```python
self.thrust_state: np.ndarray = [0, 0, 0]   # 現在の推力状態（1次遅れ用）
self.torque_state: np.ndarray = [0, 0, 0]   # 現在のトルク状態
```

#### 実装（スラスタ応答モデル）

##### 1次遅れ系応答
```python
def _simulate_thruster_response(self, cmd_thrust, cmd_torque, dt):
    """1次遅れ + ノイズ + 飽和でスラスタ応答を模擬"""

    # 1次遅れ系: dx/dt = (u - x) / τ
    tau = self.time_constant

    # 推力応答
    thrust_target = cmd_thrust * self.thrust_gain
    self.thrust_state += (thrust_target - self.thrust_state) / tau * dt

    # 不感帯・飽和
    thrust_actual = np.clip(self.thrust_state, -self.thrust_saturation, self.thrust_saturation)
    thrust_actual[np.abs(thrust_actual) < self.thrust_deadzone] = 0

    # 計測ノイズ
    thrust_actual += np.random.normal(0, self.thrust_noise_std, 3)

    # トルクも同様
    torque_target = cmd_torque * self.torque_gain
    self.torque_state += (torque_target - self.torque_state) / tau * dt
    torque_actual = self.torque_state + np.random.normal(0, self.torque_noise_std, 3)

    return thrust_actual, torque_actual
```

#### 実機ハードウェア差し替え用インターフェース
```python
# 将来的な実装例
class ThrusterHardwareAdapter(mosaik_api.Simulator):
    """実機スラスタ試験装置へのアダプタ"""
    def __init__(self):
        self.serial_port = None      # シリアル通信
        self.daq_device = None       # DAQデバイス（推力計測）
        self.load_cell = None        # ロードセル

    def init(self, sid, **params):
        # シリアルポート接続（コマンド送信用）
        self.serial_port = serial.Serial('/dev/ttyUSB0', 115200)

        # DAQデバイス初期化（推力計測用）
        self.daq_device = nidaqmx.Task()
        self.daq_device.ai_channels.add_ai_voltage_chan("Dev1/ai0:2")
        pass

    def step(self, time, inputs, max_advance):
        # cmd_thrust_eff, cmd_torque_effをシリアル送信
        # DAQから推力・トルクを計測
        # thrust_actual, torque_actualとして返す
        pass
```

---

### 3.4 （削除：SpaceEnvSimはControllerSim内に統合）

宇宙環境モデル（重力、重力勾配、大気ドラッグ、太陽輻射圧）は**ControllerSim内の6DoF運動シミュレーション部**に統合されています。

---

### 3.4 LoggingSim（ロギング・統計シミュレータ）

#### 役割
全信号のタイムスタンプを収集し、遅延統計・制御性能指標を計算。

#### 入力（Inputs）
```python
# 全シミュレータから全属性を受信（connect_many_to_one）
# 例:
# - ControllerSim: cmd_thrust, cmd_torque, quat, omega, pos, vel
# - BridgeSim: cmd_thrust_eff, thrust_actual_delayed, stats, delay_cmd, delay_sense
# - PlantSim: thrust_actual, torque_actual
```

#### 出力（Outputs）
なし（データ収集専用）

#### 内部状態
```python
self.log_data: List[dict] = []  # 全データポイント
self.statistics: dict = {}      # 統計情報
```

#### 実装
```python
def step(self, time, inputs, max_advance):
    for eid, attrs in inputs.items():
        for attr, sources in attrs.items():
            for src_full_id, value in sources.items():
                record = {
                    'time': time * TIME_RESOLUTION,  # 秒単位に変換
                    'src': src_full_id,
                    'attr': attr,
                    'value': value,
                }
                self.log_data.append(record)

    return time + self.step_size

def finalize(self):
    # シミュレーション終了後の統計計算
    df = pd.DataFrame(self.log_data)

    # 遅延統計
    delay_stats = df[df['attr'] == 'delay_cmd'].describe()

    # 制御性能（姿勢誤差RMS）
    quat_data = df[df['attr'] == 'quat']
    attitude_error_rms = self._compute_attitude_error_rms(quat_data)

    # HDF5/CSV保存
    df.to_hdf(self.output_path / 'full_log.h5', key='data')

    # 統計サマリ保存
    with open(self.output_path / 'statistics.json', 'w') as f:
        json.dump({
            'delay_stats': delay_stats.to_dict(),
            'attitude_error_rms': attitude_error_rms,
            # ...
        }, f, indent=2)
```

---

## 4. 実装フェーズ

### Phase 0: 準備（1日）
- [ ] ディレクトリ構造作成
  ```
  src/simulators/
  ├── controller_simulator.py
  ├── bridge_simulator.py
  ├── plant_simulator.py
  ├── space_env_simulator.py
  └── logging_simulator.py
  ```
- [ ] 共通ユーティリティ実装
  - 四元数演算（quaternion_multiply, quaternion_conjugate, quat_to_euler）
  - 座標変換（body_to_inertial, inertial_to_body）
- [ ] テスト環境準備（pytest設定）

### Phase 1: 基本ループ実装（2-3日）
**目標**: ControllerSim → BridgeSim → PlantSim → BridgeSim → ControllerSim のループを動作させる

#### 1.1 PlantSim（簡易版）
- [ ] 6DoF運動方程式（オイラー積分）
- [ ] 四元数更新ロジック
- [ ] 重力のみ（外乱なし）
- [ ] ユニットテスト（軌道安定性確認）

#### 1.2 ControllerSim（PD制御）
- [ ] 四元数誤差計算
- [ ] PD制御則
- [ ] トルク飽和処理
- [ ] ユニットテスト（ゼロ姿勢誤差時のトルク確認）

#### 1.3 BridgeSim（ZOHのみ）
- [ ] 固定遅延のみ実装（ジッターなし）
- [ ] ZOH補間
- [ ] パケットキュー基本構造
- [ ] ユニットテスト（遅延時間の検証）

#### 1.4 main.py統合
- [ ] TIME_RESOLUTION=0.01設定
- [ ] 循環接続（time_shifted対応）
- [ ] LoggingSimの仮実装（CSV保存のみ）
- [ ] 10秒間のテスト実行

### Phase 2: 遅延・ジッター実装（2日）
- [ ] BridgeSim: ガウシアンジッター追加
- [ ] BridgeSim: パケットロス実装
- [ ] BridgeSim: 順序保持/非保持オプション
- [ ] LoggingSim: 遅延統計計算（平均、標準偏差）
- [ ] 可視化: 遅延時系列プロット（matplotlib）

**検証項目**:
- 遅延50ms、ジッター10msでの姿勢制御安定性
- パケットロス1%の影響

### Phase 3: 補償手法実装（3-4日）

#### 3.1 線形補間
- [ ] sense_history管理
- [ ] 線形補間ロジック
- [ ] 外挿時の安全性確認

#### 3.2 先行送出（Command Advance）
- [ ] タイムスタンプ調整
- [ ] PlantSim側の未来コマンド処理

#### 3.3 Nowcasting（簡易予測）
- [ ] 定速度モデル実装
- [ ] 予測誤差の記録
- [ ] （発展）線形回帰ベース予測

#### 3.4 チャネル等化
- [ ] 複数信号の時刻差計算
- [ ] 等化目標時刻への補正

**検証項目**:
- 各補償手法のON/OFF比較
- 姿勢誤差RMSの定量評価

### Phase 4: 統計・可視化強化（2日）
- [ ] LoggingSim: HDF5保存
- [ ] 統計サマリJSON出力
- [ ] プロット自動生成
  - 時系列（姿勢、角速度、トルク、遅延）
  - ヒストグラム（遅延分布）
  - 散布図（遅延 vs 制御誤差）
- [ ] mosaik-webでのリアルタイム表示最適化

### Phase 5: 外乱・拡張機能（1-2日、任意）
- [ ] SpaceEnvSim実装
- [ ] PlantSim: 重力勾配トルク
- [ ] PlantSim: 大気ドラッグ
- [ ] 軌道伝播精度向上（RK4積分）

### Phase 6: S2E差し替え準備（1日）
- [ ] S2EAdapterSimのスケルトン実装
- [ ] 通信プロトコル定義（TCP/UDP）
- [ ] インターフェース互換性テスト

---

## 5. テスト計画

### 5.1 ユニットテスト
各シミュレータを単体でテスト。

```python
# tests/test_controller_simulator.py
def test_zero_error_zero_torque():
    """姿勢誤差ゼロ時、トルクもゼロになることを確認"""
    controller = PDController(kp=10, kd=2)
    quat = [1, 0, 0, 0]
    omega = [0, 0, 0]
    torque = controller.compute_torque(quat, omega)
    assert np.allclose(torque, [0, 0, 0])

# tests/test_bridge_simulator.py
def test_delay_timing():
    """遅延時間が正しく適用されることを確認"""
    bridge = HilsBridge(delay_cmd_s=0.05, jitter_cmd_std=0)
    # ... テストロジック
```

### 5.2 統合テスト
main.pyを実行し、期待される動作を確認。

```python
# tests/test_integration.py
def test_basic_loop():
    """基本ループが300ステップ実行できることを確認"""
    world = create_test_world()
    world.run(until=300)
    assert world.sim_progress == 300
```

### 5.3 性能ベンチマーク
- 10000ステップの実行時間
- メモリ使用量
- リアルタイムファクター達成度

---

## 6. 検証シナリオ

### シナリオ1: 基本遅延影響評価
**目的**: 遅延が制御性能に与える影響を定量化

**設定**:
- 遅延: 0ms, 20ms, 50ms, 100ms, 200ms
- ジッター: 0ms
- パケットロス: 0%
- 補償: なし（ZOHのみ）

**評価指標**:
- 姿勢誤差RMS
- 整定時間
- オーバーシュート

### シナリオ2: ジッター影響評価
**目的**: ジッターの影響を評価

**設定**:
- 遅延: 50ms（固定）
- ジッター: 0ms, 5ms, 10ms, 20ms, 50ms
- パケットロス: 0%
- 補償: ZOH vs 線形補間

**評価指標**:
- 制御入力の滑らかさ（ジャーク）
- 姿勢誤差の分散

### シナリオ3: パケットロス影響
**目的**: パケットロスの許容限界を評価

**設定**:
- 遅延: 50ms
- ジッター: 10ms
- パケットロス: 0%, 1%, 5%, 10%, 20%
- 補償: ZOH vs 線形補間

**評価指標**:
- 制御安定性（発散の有無）
- 姿勢誤差RMS

### シナリオ4: 補償手法比較
**目的**: 各補償手法の効果を比較

**設定**:
- 遅延: 50ms
- ジッター: 10ms
- パケットロス: 1%
- 補償:
  1. ZOH（ベースライン）
  2. 線形補間
  3. 先行送出（20ms）
  4. Nowcasting（50ms horizon）
  5. 線形補間 + 先行送出

**評価指標**:
- 姿勢誤差RMS比較
- 計算コスト（実行時間）

### シナリオ5: 最悪ケース
**目的**: 極端な条件での安定性確認

**設定**:
- 遅延: 200ms
- ジッター: 50ms
- パケットロス: 10%
- 補償: 全手法適用

**評価指標**:
- 制御安定性（発散しないか）
- 姿勢誤差の上限

---

## 7. 成果物

### 7.1 コード
```
mosaik-hils/
├── src/simulators/
│   ├── controller_simulator.py     (300行)
│   ├── bridge_simulator.py         (500行)
│   ├── plant_simulator.py          (300行)
│   ├── space_env_simulator.py      (150行、任意)
│   └── logging_simulator.py        (200行)
├── src/utils/
│   ├── quaternion.py               (四元数演算)
│   └── coordinates.py              (座標変換)
├── tests/
│   ├── test_controller.py
│   ├── test_bridge.py
│   ├── test_plant.py
│   └── test_integration.py
├── scenarios/
│   ├── scenario1_delay_impact.py
│   ├── scenario2_jitter.py
│   ├── scenario3_packet_loss.py
│   ├── scenario4_compensation.py
│   └── scenario5_worst_case.py
├── main.py                         (更新)
└── docs/
    └── hils_delay_compensation_plan.md (本ファイル)
```

### 7.2 データ
各シナリオ実行後に生成:
```
logs/YYYYMMDD-HHMMSS/
├── full_log.h5                     # 全時系列データ
├── statistics.json                 # 統計サマリ
├── plots/
│   ├── timeseries_attitude.png
│   ├── timeseries_delay.png
│   ├── histogram_delay.png
│   └── scatter_delay_vs_error.png
└── execution_graph.png             (mosaikグラフ)
```

### 7.3 レポート
```markdown
# HILS遅延補償検証レポート

## エグゼクティブサマリ
- 遅延50msまでは補償なしでも制御可能
- ジッター10ms以上で線形補間が有効
- 先行送出で姿勢誤差RMSを30%改善
- パケットロス5%を超えるとNowcasting必須

## 詳細結果
（各シナリオの定量結果）

## 推奨事項
（実システムへの適用指針）
```

---

## 8. マイルストーン

| フェーズ | 期間 | 完了条件 |
|---------|------|---------|
| Phase 0 | 1日 | ディレクトリ・ユーティリティ完成 |
| Phase 1 | 3日 | 基本ループが10秒間動作 |
| Phase 2 | 2日 | 遅延・ジッターのプロット生成 |
| Phase 3 | 4日 | 4種類の補償手法が動作 |
| Phase 4 | 2日 | 自動プロット・統計JSON生成 |
| Phase 5 | 2日 | SpaceEnvSim統合（任意） |
| Phase 6 | 1日 | S2EAdapterスケルトン完成 |
| **合計** | **15日** | 全シナリオ実行可能 |

---

## 9. リスクと対策

### リスク1: Mosaikの循環依存でデッドロック
**対策**:
- `time_shifted=True`の適切な使用
- step_sizeを1以上に保つ
- 必要に応じてBridgeSimのバッファ戦略を調整

### リスク2: 数値積分の不安定化
**対策**:
- オイラー法で不安定なら4次Runge-Kutta法に変更
- 時間ステップを5ms（STEP_SIZE=0.5）に短縮
- 四元数の正規化を毎ステップ実施

### リスク3: 大量データでHDF5保存が遅延
**対策**:
- バッファリング書き込み（100ステップごと）
- 圧縮オプション（gzip level 4）
- 必要に応じてダウンサンプリング

### リスク4: 補償ロジックのデバッグ困難
**対策**:
- 段階的実装（ZOH → 線形 → 予測）
- 詳細ログ出力（DEBUG_MODE=True）
- ユニットテストの徹底

---

## 10. 将来の拡張

### 短期（3ヶ月）
- [ ] S2E実機統合
- [ ] Simulink Co-simulationサポート（FMI/FMU）
- [ ] GPU加速（大規模アンサンブル実行）

### 中期（6ヶ月）
- [ ] 機械学習ベースNowcasting（LSTM）
- [ ] マルチエージェント（複数衛星のフォーメーション）
- [ ] クラウド実行対応（AWS/Azure）

### 長期（1年）
- [ ] デジタルツイン連携（実機テレメトリとの同期）
- [ ] VR/AR可視化
- [ ] 認証取得（NASA/JAXA標準準拠）

---

## 11. 参考文献

1. Mosaik Documentation: https://mosaik.readthedocs.io/
2. "Quaternion-based Attitude Control for Spacecraft" (Wie, 1985)
3. "Hardware-in-the-Loop Simulation: A Survey" (Fathy et al., 2006)
4. "Predictive Compensation for Communication Delays in Networked Control Systems" (Liu & Goldsmith, 2004)
5. S2E Documentation: https://github.com/ut-issl/s2e-core

---

## 12. 連絡先・問い合わせ

プロジェクトリード: [あなたの名前]
技術質問: GitHub Issues
実装レビュー: Pull Request

---

**更新履歴**
- 2025-01-12: 初版作成
