# Analysis of Inverse Compensation with Plant Noise

This directory contains Monte Carlo simulation results studying the effect of plant noise on inverse compensation performance.

## Experiment Setup

- **Fixed Parameters:**
  - Plant time constant (τ) = 100 ms
  - Inverse compensation enabled (post-plant position)
  - Inverse compensation gain = 8
  - Control gains: Kp=15, Ki=5, Kd=8
  - Target position = 5.0 m
  - Simulation time = 5.0 s

- **Variable Parameters:**
  - Plant noise standard deviation: 0, 5, 10, 15 ms
  - Monte Carlo runs: 50 per noise level
  - Total simulations: 200 (+ 1 baseline RT)

## Key Findings

### RMSE from Baseline RT
- **0 ms noise:** 8.53 ± 5.37 mm
- **5 ms noise:** 9.07 ± 4.91 mm
- **10 ms noise:** 10.16 ± 5.09 mm
- **15 ms noise:** 11.50 ± 5.09 mm

**Observation:** RMSE increases approximately linearly with noise level (~0.2 mm per ms of noise).

### Settling Time (5% threshold)
- **0 ms noise:** 1.195 ± 0.004 s
- **5 ms noise:** 1.195 ± 0.004 s
- **10 ms noise:** 1.195 ± 0.004 s
- **15 ms noise:** 1.196 ± 0.004 s

**Observation:** Settling time is remarkably consistent across all noise levels, indicating robust convergence.

### Maximum Overshoot
- **0 ms noise:** 4.13 ± 0.01%
- **5 ms noise:** 4.14 ± 0.02%
- **10 ms noise:** 4.14 ± 0.05%
- **15 ms noise:** 4.13 ± 0.07%

**Observation:** Overshoot remains consistent (~4.1%), with slightly increased variation at higher noise.

### Integral Absolute Error (IAE)
- **0 ms noise:** 3.213 ± 0.005 m·s
- **5 ms noise:** 3.215 ± 0.009 m·s
- **10 ms noise:** 3.214 ± 0.016 m·s
- **15 ms noise:** 3.212 ± 0.024 m·s

**Observation:** IAE shows minimal variation, with slightly increased std dev at higher noise.

## Analysis Scripts

### 1. `analyze_inverse_comp_noise.py`
Main analysis script that calculates:
- RMSE, settling time, overshoot for each run
- Statistical aggregation (mean, std dev)
- Box plots showing Monte Carlo distributions
- Error bar plots showing trends

**Outputs:**
- `inverse_comp_noise_metrics.png` - Error metrics vs noise level
- `inverse_comp_noise_distributions.png` - Box plots of distributions
- `inverse_comp_noise_bar_chart.png` - Bar chart comparison

### 2. `create_comprehensive_analysis.py`
Detailed analysis including:
- Position traces comparison
- Deviation from baseline plots
- Multiple error metrics (RMSE, MAE, Max Dev, IAE, ISE)
- Statistical summaries

**Outputs:**
- `position_traces_by_noise.png` - Sample position trajectories
- `deviation_from_baseline_by_noise.png` - Deviation plots with mean ± std
- `statistical_comparison.png` - Comprehensive metric comparison

## Running the Analysis

```bash
# Basic analysis
uv run python analyze_inverse_comp_noise.py

# Comprehensive analysis
uv run python create_comprehensive_analysis.py
```

## Interpretation

1. **Robustness:** The inverse compensator shows good robustness to plant noise up to 15 ms std dev
   - Control performance degrades gradually
   - No instability or dramatic failure modes observed

2. **Noise Impact:**
   - Primary impact is on position tracking accuracy (RMSE increases)
   - Transient response (settling time, overshoot) remains largely unaffected
   - Error accumulation (IAE) shows minimal change

3. **Design Implications:**
   - Plant noise up to 15 ms can be tolerated with τ=100ms
   - Inverse compensation gain of 8 provides good balance
   - No need for adaptive compensation at these noise levels

## Comparison with 16_tau_noise

The `16_tau_noise` directory contains a full 2D sweep:
- Variable tau: 50, 100, 150, 200 ms
- Variable noise: 0, 5, 10, 15 ms
- **Without** inverse compensation

This directory (17_noise_inverse_heatmap) focuses on:
- Fixed tau = 100 ms
- Variable noise: 0, 5, 10, 15 ms
- **With** inverse compensation enabled

Direct comparison shows inverse compensation reduces RMSE by approximately 40-60% compared to uncompensated case at similar noise levels.

## Data Format

Each simulation directory contains:
- `hils_data.h5` - Complete simulation data
- `simulation_config.json` - Parameter configuration
- `Bridge_cmd_0_events.jsonl` - Command bridge events
- `Bridge_sense_0_events.jsonl` - Sensing bridge events
- `dataflowGraph_custom.png` - Mosaik data flow visualization

## Notes

- Baseline RT has no plant lag (τ=0) and no delays
- All simulations use same control gains and spacecraft parameters
- Monte Carlo variations come from plant noise only (Gaussian distribution)
- Results demonstrate statistical consistency across 50 runs per condition
