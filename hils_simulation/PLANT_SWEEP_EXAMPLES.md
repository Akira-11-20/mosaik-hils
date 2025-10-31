# Plant Time Constant Sweep Examples

## Overview

`run_delay_sweep_advanced.py` now supports sweeping plant time constant (actuator lag) parameters in addition to communication delays and inverse compensation gains.

## New Parameters

### `DelayConfig` Class

```python
DelayConfig(
    cmd_delay: float,              # Command path delay [ms]
    sense_delay: float,            # Sensing path delay [ms]
    use_inverse_comp: bool = False,
    comp_gain: Optional[float] = None,
    plant_time_constant: Optional[float] = None,  # NEW: Plant lag time constant [ms]
    plant_enable_lag: Optional[bool] = None,      # NEW: Enable/disable plant lag
    label: Optional[str] = None
)
```

## Usage Examples

### Example 1: Plant Time Constant Sweep (No Communication Delay)

Test different actuator dynamics without communication delays:

```python
configs = [
    DelayConfig(0.0, 0.0, plant_time_constant=100.0),   # Fast actuator (τ=100ms)
    DelayConfig(0.0, 0.0, plant_time_constant=500.0),   # Medium actuator (τ=500ms)
    DelayConfig(0.0, 0.0, plant_time_constant=1000.0),  # Slow actuator (τ=1000ms)
    DelayConfig(0.0, 0.0, plant_enable_lag=False),      # Ideal actuator (no lag)
]
```

**Use case:** Compare control performance with different actuator response times.

### Example 2: Combined Delay and Plant Lag Sweep

Test interaction between communication delay and actuator lag:

```python
configs = [
    # 20ms comm delay with different actuator speeds
    DelayConfig(20.0, 20.0, plant_time_constant=100.0),
    DelayConfig(20.0, 20.0, plant_time_constant=500.0),
    DelayConfig(20.0, 20.0, plant_time_constant=1000.0),

    # 50ms comm delay with different actuator speeds
    DelayConfig(50.0, 50.0, plant_time_constant=100.0),
    DelayConfig(50.0, 50.0, plant_time_constant=500.0),
    DelayConfig(50.0, 50.0, plant_time_constant=1000.0),
]
```

**Use case:** Study how communication delay and actuator lag compound.

### Example 3: Actuator Lag with Inverse Compensation

Test if inverse compensation helps when actuator has lag:

```python
configs = [
    # Baseline: no delay, ideal actuator
    DelayConfig(0.0, 0.0, plant_enable_lag=False),

    # Communication delay only
    DelayConfig(50.0, 50.0, plant_enable_lag=False),

    # Actuator lag only
    DelayConfig(0.0, 0.0, plant_time_constant=500.0),

    # Both delays
    DelayConfig(50.0, 50.0, plant_time_constant=500.0),

    # Both delays + inverse compensation
    DelayConfig(50.0, 50.0, plant_time_constant=500.0, use_inverse_comp=True),
]
```

**Use case:** Determine if inverse compensation is effective when actuator has intrinsic lag.

### Example 4: Full Parametric Study

Comprehensive study of all parameters:

```python
# Communication delays to test
delays = [0.0, 20.0, 50.0]

# Plant time constants to test
plant_taus = [100.0, 500.0, 1000.0]

# Generate all combinations
configs = []
for delay in delays:
    for tau in plant_taus:
        # Without compensation
        configs.append(DelayConfig(delay, delay, plant_time_constant=tau))

        # With compensation (only if delay > 0)
        if delay > 0:
            configs.append(DelayConfig(delay, delay, plant_time_constant=tau, use_inverse_comp=True))
```

**Use case:** Full factorial design for statistical analysis.

## Output Files

Each configuration creates a separate result directory with the label format:

```
delay{comm_delay}ms_{comp_status}_tau{plant_tau}ms
```

Examples:
- `delay0ms_nocomp_tau100ms`
- `delay20ms_comp_tau500ms`
- `delay0ms_nocomp_nolag` (ideal actuator)

## Plant Lag Implementation

The plant lag is implemented as a 1st-order system:

```
τ * dy/dt + y = u

where:
  τ = time constant [ms]
  u = ideal thrust command (input)
  y = actual thrust (output with lag)
```

Discrete implementation with sub-stepping for accuracy:
```python
for each 1ms Plant update:
    for 10 sub-steps of 0.1ms:
        y = y + (dt_sub/τ) * (u - y)
```

## Expected Behavior

### Time Constant Effects

- **τ = 100ms** (fast): Quick response, reaches 63% in 100ms, 95% in 300ms
- **τ = 500ms** (medium): Moderate lag, reaches 63% in 500ms, 95% in 1500ms
- **τ = 1000ms** (slow): Significant lag, reaches 63% in 1000ms, 95% in 3000ms
- **no lag** (ideal): Instantaneous response, actual = commanded

### Step Response Characteristics

For a step input of magnitude A:
- At t = τ: output ≈ 0.632 × A (63.2%)
- At t = 3τ: output ≈ 0.950 × A (95%)
- At t = 5τ: output ≈ 0.993 × A (99.3%)

## Analysis Recommendations

1. **Compare position tracking error** between different time constants
2. **Measure overshoot and settling time** for each configuration
3. **Analyze thrust saturation** - slower actuators may limit control authority
4. **Check stability margins** - large τ can destabilize control loops
5. **Visualize thrust comparison** - use updated visualize_results.py to see:
   - Command thrust (blue)
   - Measured thrust (green, ideal)
   - Actual thrust (red, with lag)

## Running the Sweep

1. Edit `run_delay_sweep_advanced.py`
2. Uncomment Example 5 or 6 in `main()`
3. Run:
   ```bash
   uv run python run_delay_sweep_advanced.py
   ```

4. Results will be saved to separate directories in `results/`

5. Use `visualize_results.py` to compare:
   ```bash
   marimo edit visualize_results.py
   ```

## Notes

- Plant simulation period is 1ms by default
- DataCollector samples at 0.1ms (will show step-like updates)
- Sub-stepping ensures accurate 1st-order lag computation
- Setting `plant_enable_lag=False` bypasses all lag (ideal actuator)
- Default τ from `.env` is 500.0ms if not specified
