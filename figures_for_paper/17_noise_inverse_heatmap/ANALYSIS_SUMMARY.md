# Analysis Summary: Inverse Compensation with Plant Noise

## Dataset Overview

**Directory:** `17_noise_inverse_heatmap`

**Experiment Parameters:**
- Fixed plant time constant: τ = 100 ms
- **Inverse compensation gain: 8** (lower than 16_tau_noise which used gain=10)
- Plant noise levels: 0, 5, 10, 15 ms (standard deviation)
- Monte Carlo runs: 50 per condition
- Total simulations: 200

**Control Configuration:**
- PID gains: Kp=15, Ki=5, Kd=8
- Target position: 5.0 m
- Simulation duration: 5.0 s
- Compensation position: post-plant

## Key Results

### 1. Position Tracking (RMSE from Baseline RT)

| Noise (ms) | RMSE Mean (mm) | RMSE Std (mm) | Runs |
|------------|----------------|---------------|------|
| 0          | 8.527          | 5.371         | 50   |
| 5          | 9.065          | 4.910         | 50   |
| 10         | 10.162         | 5.090         | 50   |
| 15         | 11.498         | 5.085         | 50   |

**Key Finding:** RMSE increases approximately **0.2 mm per 1 ms** of plant noise.

### 2. Transient Response

**Settling Time (5% threshold):**
- Extremely consistent: 1.195 ± 0.004 s across all noise levels
- Noise has minimal impact on convergence time

**Maximum Overshoot:**
- Consistent: ~4.13% across all noise levels
- Slightly increased variation at higher noise (std: 0.01% → 0.07%)

### 3. Error Metrics

**Integral Absolute Error (IAE):**
- Baseline (0 ms): 3.213 ± 0.005 m·s
- Highest noise (15 ms): 3.212 ± 0.024 m·s
- **Minimal change** despite 15ms noise addition

**Mean Absolute Error (MAE):**
- 0 ms: 4.09 mm
- 5 ms: 4.45 mm
- 10 ms: 5.15 mm
- 15 ms: 6.16 mm

## Statistical Analysis

### Monte Carlo Consistency
- **50 runs per condition** provides robust statistical confidence
- Standard deviations remain ~5mm across all conditions
- No outliers or instability observed

### Noise Sensitivity
The inverse compensator (gain=8) shows **linear degradation** with noise:
- RMSE increase: ~35% from 0→15ms noise
- MAE increase: ~50% from 0→15ms noise
- Max deviation increase: ~25% from 0→15ms noise

### Comparison with Gain=10 (from 16_tau_noise)
The 16_tau_noise dataset used gain=10 with the same conditions:
- **Note:** Direct comparison shows gain=10 had much smaller RMSE for noise-free case
- This suggests gain=8 may be **under-compensating** the plant lag
- Trade-off: Lower gain may provide more robustness but less accuracy

## Visualizations Generated

1. **`inverse_comp_noise_metrics.png`**
   - RMSE, settling time, and overshoot vs noise
   - Error bar plots showing mean ± std

2. **`inverse_comp_noise_distributions.png`**
   - Box plots of Monte Carlo distributions
   - Shows spread and outliers

3. **`inverse_comp_noise_bar_chart.png`**
   - Simple bar chart for presentation
   - Clear comparison across noise levels

4. **`position_traces_by_noise.png`**
   - Sample trajectories (first 5 runs per condition)
   - Comparison with baseline RT

5. **`deviation_from_baseline_by_noise.png`**
   - Deviation plots with mean ± 1σ bands
   - Shows tracking error evolution over time

6. **`statistical_comparison.png`**
   - Comprehensive 6-panel figure
   - Box plots + trend lines for all metrics

## Conclusions

1. **Robustness:** Inverse compensation with gain=8 maintains stability under all tested noise conditions (0-15ms)

2. **Performance Trade-off:**
   - Gain=8 provides adequate compensation but with ~8.5mm baseline RMSE
   - Lower gain → more robust but less accurate tracking

3. **Noise Linearity:** Error metrics scale approximately linearly with noise, indicating predictable degradation

4. **Transient Consistency:** Settling time and overshoot are remarkably insensitive to noise, suggesting the compensator preserves closed-loop dynamics

5. **Practical Implication:** For τ=100ms plants, gain=8 is acceptable for noise up to 15ms, but gain tuning (8→12 sweep planned) may improve accuracy

## Next Steps

Based on modified `run_sweep.py`:
- Sweep inverse compensation gain: 8, 9, 10, 11, 12 (fixed τ=100ms)
- Identify optimal gain balancing accuracy and robustness
- Compare against this baseline (gain=8) dataset

## Generated Files

**Analysis Scripts:**
- `analyze_inverse_comp_noise.py` - Main RMSE/metrics analysis
- `create_comprehensive_analysis.py` - Detailed multi-metric analysis

**Documentation:**
- `README.md` - Dataset description and usage
- `ANALYSIS_SUMMARY.md` - This file

**Figures:** (6 PNG files, ~2.6 MB total)
- Located in same directory as data
