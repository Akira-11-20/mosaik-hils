# Parameter Loading in Sweep Scripts

## 問題と解決策 (Problem & Solution)

### 以前の問題 (Previous Issue)

スイープスクリプトが環境変数を設定してから`load_config_from_env()`を呼び出していましたが、この関数は`.env`ファイルから読み込むため、すでに設定された環境変数が無視されていました。

The sweep script was setting environment variables and then calling `load_config_from_env()`, but this function reads from the `.env` file, ignoring already-set environment variables.

**結果**: `MAX_THRUST`などのパラメータが常に`.env`の値（100.0 N）ではなく、ハードコードされたデフォルト値（1.0 N）になっていました。

**Result**: Parameters like `MAX_THRUST` were always using hardcoded default values (1.0 N) instead of `.env` values (100.0 N).

### 新しいアプローチ (New Approach)

`hils_simulation`のスイープスクリプトと同じパターンを採用：

Following the same pattern as `hils_simulation` sweep scripts:

1. **環境変数を設定** - Set environment variables first:
   ```python
   # シナリオがget_env_param()で読み込むパラメータ用
   for key, value in config.items():
       os.environ[key] = str(value)
   ```

2. **ベース設定を読み込み** - Load base configuration from `.env`:
   ```python
   orbital_config = load_config_from_env()  # 環境変数が優先される
   ```

3. **OrbitalSimulationConfigの属性を直接上書き** - Directly override config attributes:
   ```python
   for key, value in config.items():
       if key == "MAX_THRUST":
           orbital_config.spacecraft.max_thrust = float(value)
       elif key == "SPACECRAFT_MASS":
           orbital_config.spacecraft.mass = float(value)
       # ... etc
   ```

4. **変更された設定をシナリオに渡す** - Pass modified config to scenario:
   ```python
   scenario = HohmannScenario(config=orbital_config)
   ```

**重要**: 2段階のパラメータ設定が必要です：
- **環境変数**: `INVERSE_COMPENSATION`, `PLANT_TIME_CONSTANT`などシナリオが`get_env_param()`で読むパラメータ
- **Config属性**: `MAX_THRUST`, `SPACECRAFT_MASS`など`OrbitalSimulationConfig`に含まれるパラメータ

### サポートされているスイープパラメータ (Supported Sweep Parameters)

以下のパラメータをスイープ設定で上書き可能:

The following parameters can be overridden in sweep configurations:

#### Spacecraft Parameters
- `SPACECRAFT_MASS` - 衛星質量 [kg]
- `MAX_THRUST` - 最大推力 [N]
- `SPECIFIC_IMPULSE` - 比推力 [s]

#### Orbital Parameters
- `ALTITUDE_KM` - 軌道高度 [km]
- `ECCENTRICITY` - 離心率 [-]
- `INCLINATION_DEG` - 軌道傾斜角 [deg]
- `RAAN_DEG` - 昇交点赤経 [deg]
- `ARG_PERIAPSIS_DEG` - 近地点引数 [deg]
- `TRUE_ANOMALY_DEG` - 真近点角 [deg]

#### Simulation Parameters
- `SIMULATION_TIME` - シミュレーション時間 [s]
- `TIME_RESOLUTION` - 時間解像度 [s]

#### Controller Type
- `CONTROLLER_TYPE` - コントローラタイプ (`"zero"`, `"pd"`, `"hohmann"`)

**注意**: `PLANT_TIME_CONSTANT`, `INVERSE_COMPENSATION`などのパラメータは、シナリオ内で`get_env_param()`を使用して環境変数から直接読み込まれるため、スイープ設定に含めるとそちらが優先されます。

**Note**: Parameters like `PLANT_TIME_CONSTANT`, `INVERSE_COMPENSATION` are read directly from environment variables using `get_env_param()` in the scenario, so including them in the sweep config will override the `.env` values.

## 使用例 (Usage Example)

### Example 1: Default Parameters from .env

```python
from scripts.sweeps.run_parameter_sweep import ParameterSweepConfig, run_sweep

# MAX_THRUSTは指定しない → .envから100.0 Nが使われる
sweep_params = {
    "CONTROLLER_TYPE": ["hohmann"],
    "INVERSE_COMPENSATION": [True, False],
}

config = ParameterSweepConfig(
    sweep_params=sweep_params,
    base_env_file=".env",
    output_base_dir="results_sweep",
    description="Hohmann with InvComp (MAX_THRUST from .env)",
)

run_sweep(config)
```

**Result**: Uses `MAX_THRUST=100.0` from `.env` file.

### Example 2: Override Specific Parameters

```python
# MAX_THRUSTを明示的に上書き
sweep_params = {
    "CONTROLLER_TYPE": ["hohmann"],
    "MAX_THRUST": [50.0, 100.0, 200.0],  # Sweep different thrust values
    "SPACECRAFT_MASS": [500.0, 1000.0],
}

config = ParameterSweepConfig(
    sweep_params=sweep_params,
    base_env_file=".env",
    output_base_dir="results_sweep",
    description="Thrust and Mass Sweep",
)

run_sweep(config)
```

**Result**: Sweeps through combinations of thrust (50, 100, 200 N) and mass (500, 1000 kg).

### Example 3: Mixed Parameters

```python
sweep_params = {
    # These will override .env values
    "ALTITUDE_KM": [400.0, 500.0, 600.0],
    "MAX_THRUST": [100.0],  # Fixed thrust

    # These are read from environment variables by the scenario
    "PLANT_TIME_CONSTANT": [10.0, 50.0, 100.0],
    "INVERSE_COMPENSATION": [True, False],
}
```

**Result**: Altitude and thrust from sweep config, plant/compensation from environment variables.

## 検証方法 (Verification)

スイープ実行時の出力で設定を確認:

Check configuration in sweep output:

```bash
cd orbital_hils
uv run python scripts/sweeps/run_parameter_sweep.py
```

出力例:
```
[HohmannScenario] Configuration:
  Controller type: hohmann
  Initial altitude: 408.00 km
  Target altitude: 1000.00 km
  Transfer start time: 100.00 s
  Max thrust: 100.00 N  ← Should be 100.0 if loading from .env
  Spacecraft mass: 500.00 kg
  Plant time constant: 100.00 s
  Plant noise std: 0.0000
```

## 参考 (Reference)

- Updated implementation: [`run_parameter_sweep.py` lines 228-272](../run_parameter_sweep.py#L228-L272)
- Reference pattern: `hils_simulation/scripts/sweeps/run_delay_sweep_advanced.py`
- Test script: [`test_config_override.py`](test_config_override.py)
