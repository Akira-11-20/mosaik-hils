# Measurement Delay Estimation for Kalman Filter

Implementation of delay estimation methods for networked control systems based on:
"Measurement Delay Estimation for Kalman Filter in Networked Control Systems"

## Project Structure

```
delay_estimation/
├── config/              # Configuration and parameters
├── estimators/          # Delay estimation algorithms
├── scenarios/           # Test scenarios
├── simulators/          # System simulators (plant, network, etc.)
├── utils/               # Utility functions
├── docs/                # Documentation and references
├── results/             # Simulation results
└── main.py              # Main entry point
```

## Setup

From the project root:
```bash
cd delay_estimation
uv run python main.py
```

## Features

- Kalman filter with delay estimation
- Network delay modeling
- Comparison with standard Kalman filter
- Visualization of estimation performance

## Dependencies

Shared with main project via `pyproject.toml`:
- numpy
- scipy
- matplotlib
- h5py (for data storage)
