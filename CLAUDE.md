# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Running the Simulation
```bash
# Run the main HILS simulation
uv run python main.py

# Run without official mosaik-web (only custom dashboard)
SKIP_MOSAIK_WEBVIS=1 uv run python main.py

# Run custom dashboard in standalone mode for testing
python test_server.py
```

### Code Quality
```bash
# Run linter
uv run ruff check

# Auto-fix linting issues
uv run ruff check --fix

# Format code
uv run ruff format
```

### Package Management
```bash
# Install dependencies
uv sync

# Add new dependency
uv add <package-name>

# Reinstall all dependencies
uv sync --reinstall
```

## Architecture Overview

This is a **Mosaik-based HILS (Hardware-in-the-Loop Simulation)** system with dual web visualization capabilities:

### Core Components

1. **main.py** - Orchestrates the entire co-simulation scenario
   - Configures and starts all simulators
   - Defines entity connections using mosaik.util patterns
   - Manages timestamped log directories under `logs/YYYYMMDD-HHMMSS/`
   - Supports optional official mosaik-web via environment variable

2. **Simulation Modules** (all inherit from `mosaik_api.Simulator`):
   - **numerical_simulator.py** - Generates sine wave signals
   - **hardware_simulator.py** - Simulates sensors (random 0.5-1.5V) and actuators
   - **data_collector.py** - Saves all data to HDF5 format with automatic plotting
   - **custom_web_server.py** - Provides custom WebSocket-enabled dashboard

3. **Web Visualization Stack**:
   - **Official mosaik-web** - Standard visualization at http://localhost:8002
   - **Custom Dashboard** - Enhanced Japanese UI at http://localhost:8003
     - Real-time WebSocket data streaming (port 8004)
     - Chart.js time-series plotting
     - D3.js system topology diagram
     - Data export and control features

### Data Flow Architecture

The simulation follows a **many-to-one connection pattern**:
```
NumericalModel → HardwareInterface (actuator_command)
NumericalModel → DataCollector (output)
HardwareInterface → DataCollector (sensor_value, actuator_command)
Both → CustomWebDashboard (all attributes)
Both → WebVis (visualization)
```

### Key Design Patterns

- **Timestamped Logging**: Each run creates `logs/YYYYMMDD-HHMMSS/` with HDF5 data and execution graphs
- **Dual Visualization**: Both standard mosaik-web and custom dashboard run simultaneously
- **WebSocket Streaming**: Custom dashboard receives real-time simulation data via WebSocket
- **Threaded Web Servers**: HTTP static file server + WebSocket server run in separate threads
- **Environment-based Config**: Use `SKIP_MOSAIK_WEBVIS=1` to disable official web visualization

### File Structure Notes

- `web_visualization/` contains the custom dashboard frontend (HTML/CSS/JS)
- `logs/` directories contain HDF5 files, execution graphs, and timing plots
- `pyproject.toml` uses uv for dependency management with ruff for linting
- All simulators use mosaik-api version patterns for compatibility

### Development Notes

- The system runs in debug mode (`debug=True`) which enables execution graph tracking but may slow performance
- Real-time factor is set to 0.5 (faster than real-time)
- Simulation runs for 300 steps by default
- WebSocket reconnection logic is implemented in the dashboard for robust connectivity
- HDF5 output structure: `/steps` group with columns like `time`, `output_NumSim_0`, `sensor_value_HW_0`

When extending this system:
1. New simulators should follow the mosaik_api.Simulator pattern
2. Add to sim_config in main.py with appropriate connection setup
3. Custom dashboard can be extended by modifying web_visualization/ files
4. Data collection automatically includes new simulator outputs when connected properly