# Scripts Directory

This directory contains organized simulation scripts for running experiments, analyzing data, and debugging the HILS system.

## Directory Structure

```
scripts/
├── sweeps/          # Parameter sweep experiments
├── analysis/        # Data analysis and visualization
├── debug/           # Debug and testing scripts
└── README.md        # This file
```

## Parameter Sweep Scripts (`sweeps/`)

Scripts for running multiple simulations with varying parameters.

### `run_delay_sweep.py`
Basic delay sweep - tests various communication delays with/without compensation.

**Usage:**
```bash
cd hils_simulation
uv run python scripts/sweeps/run_delay_sweep.py
```

### `run_delay_sweep_advanced.py`
Advanced delay sweep with fine-grained control over:
- Separate command/sense delay configurations
- Custom inverse compensation gain per delay
- Plant time constant (actuator lag) configuration

**Usage:**
```bash
cd hils_simulation
uv run python scripts/sweeps/run_delay_sweep_advanced.py
```

### `run_gain_sweep.py`
Inverse compensation gain sweep - tests different compensation gain values.

**Usage:**
```bash
cd hils_simulation
uv run python scripts/sweeps/run_gain_sweep.py
```

### `test_plant_sweep.py`
Plant time constant sweep - tests different actuator dynamics with varying time constants.

**Usage:**
```bash
cd hils_simulation
uv run python scripts/sweeps/test_plant_sweep.py
```

**Features:**
- Tests multiple plant time constants (τ)
- Supports time constant variability (standard deviation)
- Compares with/without inverse compensation
- Configurable compensation gains per test case

## Analysis Scripts (`analysis/`)

Scripts for analyzing simulation results and visualizing data.

### `visualize_results.py`
Comprehensive visualization tool for simulation results.

**Usage:**
```bash
cd hils_simulation
uv run python scripts/analysis/visualize_results.py results/YYYYMMDD-HHMMSS/hils_data.h5
```

### `analyze_plant_delay.py`
Analyze plant actuator delays and their effects on system performance.

**Usage:**
```bash
cd hils_simulation
uv run python scripts/analysis/analyze_plant_delay.py
```

### `analyze_integral_windup.py`
Analyze integral windup effects in the PID controller.

**Usage:**
```bash
cd hils_simulation
uv run python scripts/analysis/analyze_integral_windup.py
```

## Debug Scripts (`debug/`)

Scripts for testing and debugging specific components.

### `test_inverse_timing.py`
Quick test to verify inverse compensator timing.

**Usage:**
```bash
cd hils_simulation
uv run python scripts/debug/test_inverse_timing.py
```

### `test_plant_lag.py`
Test plant first-order lag dynamics.

**Usage:**
```bash
cd hils_simulation
uv run python scripts/debug/test_plant_lag.py
```

### `debug_inverse_comp.py`
Debug inverse compensator behavior and verify compensation logic.

**Usage:**
```bash
cd hils_simulation
uv run python scripts/debug/debug_inverse_comp.py
```

### `check_command_discreteness.py`
Check command signal discreteness and timing.

**Usage:**
```bash
cd hils_simulation
uv run python scripts/debug/check_command_discreteness.py
```

### `check_update_timing.py`
Verify update timing across different simulators.

**Usage:**
```bash
cd hils_simulation
uv run python scripts/debug/check_update_timing.py
```

## General Usage Notes

1. **Working Directory**: Always run scripts from the `hils_simulation/` directory:
   ```bash
   cd hils_simulation
   uv run python scripts/<category>/<script_name>.py
   ```

2. **Environment Configuration**: Configure parameters via `.env` file in `hils_simulation/` directory:
   ```bash
   # Example .env
   SIMULATION_TIME=2.0
   KP=15.0
   CMD_DELAY=20.0
   PLANT_TIME_CONSTANT=50.0
   PLANT_TIME_CONSTANT_STD=5.0  # Add variability
   ```

3. **Results**: Most scripts save results to organized directories:
   - HILS simulations: `results/YYYYMMDD-HHMMSS/`
   - RT simulations: `results_rt/YYYYMMDD-HHMMSS/`
   - Pure Python: `results_pure/YYYYMMDD-HHMMSS/`

4. **Data Format**: Simulation data is saved in HDF5 format (`hils_data.h5`) with configuration files (`simulation_config.json`).

## Adding New Scripts

When adding new scripts:

1. Place them in the appropriate category directory
2. Add path adjustment at the top:
   ```python
   import sys
   from pathlib import Path
   sys.path.insert(0, str(Path(__file__).parent.parent.parent))
   ```
3. Document usage in this README
4. Follow naming conventions:
   - Sweep scripts: `run_<parameter>_sweep.py`
   - Analysis scripts: `analyze_<aspect>.py`
   - Debug scripts: `test_<component>.py` or `debug_<component>.py`

## Related Documentation

- [Main README](../README.md) - Project overview
- [V2 Architecture](../docs/V2_ARCHITECTURE.md) - System architecture details
- [CLAUDE.md](../.claude/CLAUDE.md) - Development commands and guidelines
