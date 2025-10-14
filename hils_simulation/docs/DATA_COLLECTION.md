# HDF5データ収集機能

## 概要

HILS シミュレーションの全データを HDF5 形式で記録する機能です。シミュレーション終了後、データを解析・可視化できます。

## 機能

### 自動データ収集

シミュレーション実行時、以下のデータが自動的に記録されます：

**Controller (制御器)**
- `command`: 制御コマンド（JSON/dict形式: `{thrust, duration}`）
- `error`: 位置誤差 [m]

**Bridge (通信ブリッジ)**
- `stats`: 通信統計情報（遅延、パケットロス等）

**Plant (推力測定器)**
- `measured_thrust`: 測定推力 [N]
- `status`: 動作状態（"idle", "thrusting"）

**Env (宇宙機環境)**
- `position`: 位置 [m]
- `velocity`: 速度 [m/s]
- `acceleration`: 加速度 [m/s²]
- `force`: 作用力 [N]

**Time (時刻情報)**
- `time_ms`: シミュレーション時刻 [ms]
- `time_s`: 実時間 [秒]

### データ形式

HDF5ファイル構造:
```
hils_data.h5
├── attributes (metadata)
│   ├── created_at: ISO timestamp
│   ├── num_samples: データポイント数
│   └── time_resolution: 時間解像度 [秒]
└── data/ (group)
    ├── time_ms: (N,) float64
    ├── time_s: (N,) float64
    ├── position_EnvSim-0.Spacecraft1DOF_0: (N,) float64
    ├── velocity_EnvSim-0.Spacecraft1DOF_0: (N,) float64
    ├── acceleration_EnvSim-0.Spacecraft1DOF_0: (N,) float64
    ├── force_EnvSim-0.Spacecraft1DOF_0: (N,) float64
    ├── command_ControllerSim-0.PDController_0: (N,) object (JSON)
    ├── command_ControllerSim-0.PDController_0_thrust: (N,) float64
    ├── command_ControllerSim-0.PDController_0_duration: (N,) float64
    ├── error_ControllerSim-0.PDController_0: (N,) float64
    ├── measured_thrust_PlantSim-0.ThrustStand_0: (N,) float64
    ├── status_PlantSim-0.ThrustStand_0: (N,) object
    ├── stats_BridgeSim-0.CommBridge_0: (N,) float64
    └── stats_BridgeSim-1.CommBridge_0: (N,) float64
```

## 使用方法

### 1. シミュレーション実行

```bash
cd hils_simulation
uv run python main_hils.py
```

実行すると、`results/YYYYMMDD-HHMMSS/` ディレクトリに以下が生成されます：
- `hils_data.h5`: HDF5データファイル
- `*_dataflowGraph_*.png`: データフロー図
- `*_executionGraph.png`: 実行グラフ
- `*_executiontime.png`: 実行時間グラフ

### 2. データ解析

解析スクリプトを使用してデータを可視化：

```bash
# 統計情報のみ表示
uv run python analyze_data.py results/20251013-183849/hils_data.h5 --no-plot

# グラフを画面表示
uv run python analyze_data.py results/20251013-183849/hils_data.h5

# グラフをファイル保存
uv run python analyze_data.py results/20251013-183849/hils_data.h5 --save-plots
```

### 3. カスタム解析

Python で直接 HDF5 ファイルを読み込んで解析：

```python
import h5py
import matplotlib.pyplot as plt

# HDF5ファイルを開く
with h5py.File('results/20251013-183849/hils_data.h5', 'r') as f:
    # メタデータ
    print(f"Created: {f.attrs['created_at']}")
    print(f"Samples: {f.attrs['num_samples']}")

    # データ取得
    time_s = f['data/time_s'][:]
    position = f['data/position_EnvSim-0.Spacecraft1DOF_0'][:]
    velocity = f['data/velocity_EnvSim-0.Spacecraft1DOF_0'][:]
    thrust = f['data/command_ControllerSim-0.PDController_0_thrust'][:]

    # プロット
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.plot(time_s, position)
    plt.xlabel('Time [s]')
    plt.ylabel('Position [m]')
    plt.title('Spacecraft Position')
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(time_s, velocity)
    plt.xlabel('Time [s]')
    plt.ylabel('Velocity [m/s]')
    plt.title('Spacecraft Velocity')
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(time_s, thrust)
    plt.xlabel('Time [s]')
    plt.ylabel('Thrust [N]')
    plt.title('Thrust Command')
    plt.grid(True)

    plt.tight_layout()
    plt.show()
```

## 解析スクリプトの出力

`analyze_data.py` は以下の8つのグラフを生成します：

1. **Spacecraft Position**: 位置の時系列
2. **Spacecraft Velocity**: 速度の時系列
3. **Spacecraft Acceleration**: 加速度の時系列
4. **Thrust Command vs Measured**: 指令推力と測定推力の比較
5. **Position Error**: 制御誤差の時系列
6. **Applied Force**: 宇宙機に作用する力の時系列
7. **Phase Plane**: 位相平面（位置-速度）
8. **Bridge Statistics**: 通信遅延統計

サマリー統計情報：
```
======================================================================
SUMMARY STATISTICS
======================================================================

📍 Position [position_EnvSim-0.Spacecraft1DOF_0]:
   Initial: 0.000000 m
   Final:   0.000416 m
   Max:     0.000416 m
   Min:     0.000000 m

🚀 Velocity [velocity_EnvSim-0.Spacecraft1DOF_0]:
   Initial: 0.000000 m/s
   Final:   0.012800 m/s
   Max:     0.012800 m/s
   Min:     0.000000 m/s

⚡ Acceleration [acceleration_EnvSim-0.Spacecraft1DOF_0]:
   Mean:    0.064000 m/s²
   Max:     0.200000 m/s²
   Min:     0.000000 m/s²
   Std:     0.093295 m/s²

🔥 Thrust Command [command_ControllerSim-0.PDController_0_thrust]:
   Mean:    20.000 N
   Max:     20.000 N
   Min:     19.999 N
   Std:     0.000 N

📉 Position Error [error_ControllerSim-0.PDController_0]:
   Initial: 10.000000 m
   Final:   10.000000 m
   Mean:    10.000000 m (MAE)
   RMS:     10.000000 m

⏱️  Simulation Duration:
   Total:   0.199 s
   Steps:   200

======================================================================
```

## データ収集の仕組み

### DataCollectorSimulator

`data_collector.py` に実装された Mosaik シミュレーター。

**特徴:**
- 任意の属性を動的に受け入れる（`any_inputs: True`）
- dict/numeric/string/None 型を自動判別
- dict 型（例: command）は JSON 文字列として保存し、各要素も個別に記録
- 1ms 毎にデータポイントを収集（`step_size=1`）
- シミュレーション終了時に HDF5 ファイルを自動保存（`finalize()`）

### main_hils.py での設定

```python
# DataCollector の起動
data_collector_sim = world.start("DataCollector", step_size=1)
collector = data_collector_sim.Collector(output_dir=str(run_dir))

# 全エンティティからデータを収集
mosaik.util.connect_many_to_one(world, [controller], collector, "command", "error")
mosaik.util.connect_many_to_one(world, [bridge_cmd], collector, "stats")
mosaik.util.connect_many_to_one(world, [plant], collector, "measured_thrust", "status")
mosaik.util.connect_many_to_one(world, [bridge_sense], collector, "stats")
mosaik.util.connect_many_to_one(
    world, [spacecraft], collector, "position", "velocity", "acceleration", "force"
)
```

## コマンドタイミングの解析

制御コマンドがいつ送られたかを確認する例：

```python
import h5py
import json

with h5py.File('results/20251013-183849/hils_data.h5', 'r') as f:
    time_ms = f['data/time_ms'][:]
    command_json = f['data/command_ControllerSim-0.PDController_0'][:]
    thrust = f['data/command_ControllerSim-0.PDController_0_thrust'][:]

    # コマンドが変化したタイミングを検出
    thrust_changes = []
    for i in range(1, len(thrust)):
        if abs(thrust[i] - thrust[i-1]) > 0.001:
            thrust_changes.append((time_ms[i], thrust[i]))

    print("コマンド変化タイミング:")
    for t, cmd in thrust_changes[:10]:  # 最初の10件
        print(f"  t={t:.0f}ms: thrust={cmd:.3f}N")
```

## 注意事項

1. **ファイルサイズ**: 長時間シミュレーションでは HDF5 ファイルが大きくなります
   - 1ms 解像度、500ms シミュレーション → 約 50KB
   - 1ms 解像度、5000ms（5秒）シミュレーション → 約 500KB
   - データ圧縮を追加したい場合は `data_collector.py` の `create_dataset()` に `compression="gzip"` を追加

2. **メモリ使用量**: データは全てメモリ上に保持されます
   - 極めて長時間のシミュレーションでは要注意
   - 必要に応じてストリーミング書き込みに変更可能

3. **h5py の依存関係**: HDF5 機能には `h5py` が必要
   ```bash
   uv add h5py
   ```

## 今後の拡張

- [ ] データ圧縮オプション（gzip）
- [ ] リアルタイムストリーミング書き込み
- [ ] NetCDF4 形式のサポート
- [ ] 遅延補償データの追加（将来実装時）
- [ ] パケットロスイベントの詳細記録
- [ ] Jupyter Notebook による対話的解析例

## 関連ファイル

- [`simulators/data_collector.py`](simulators/data_collector.py): DataCollector 実装
- [`main_hils.py`](main_hils.py): メインシナリオ（データ収集設定を含む）
- [`analyze_data.py`](analyze_data.py): データ解析・可視化スクリプト
- [`results/`](../results/): シミュレーション結果の保存先

## 参考

- [HDF5 公式ドキュメント](https://www.hdfgroup.org/solutions/hdf5/)
- [h5py ユーザーガイド](https://docs.h5py.org/en/stable/)
- [Mosaik ドキュメント](https://mosaik.readthedocs.io/)
