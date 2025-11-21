# Formation Flying Comparison Results

This directory contains comparison results for formation flying simulations with different configurations:

## Scenarios

1. **001_baseline_tau=0.0** - Baseline scenario with no communication delay (τ=0)
2. **002_inv_comp=True_gain=100.0_tau=100.0_controller_type=formation** - With inverse compensation (τ=100ms, gain=100)
3. **003_inv_comp=False_gain=100.0_tau=100.0_controller_type=formation** - Without inverse compensation (τ=100ms)

## Generated Comparison Files

### Figures

- **[formation_3d_relative.png](formation_3d_relative.png)** - 3D visualization of relative position trajectories (Chaser-Target)
  - Shows the relative motion of the chaser spacecraft with respect to the target
  - Includes markers for initial positions (circles) and final positions (triangles)
  - Target position (origin) marked with a gold star

- **[formation_relative_2d_planes.png](formation_relative_2d_planes.png)** - 3-panel 2D projections in ECI frame:
  - (a) XY Plane (Orbital Plane) - In-plane relative motion
  - (b) XZ Plane - Radial-tangential motion
  - (c) YZ Plane - Cross-track motion
  - All planes show initial (circles) and final (triangles) positions

- **[formation_relative_rt_plane.png](formation_relative_rt_plane.png)** - RT plane in RTN (Hill's frame):
  - In-plane relative motion: Radial (R) vs Along-track (T)
  - RTN frame is physically meaningful for formation flying dynamics
  - R (Radial): Earth center to satellite direction
  - T (Tangential): Orbital velocity direction
  - Shows the most important dynamics for formation flying control

- **[formation_relative_rt_plane_zoomed.png](formation_relative_rt_plane_zoomed.png)** - RT plane zoomed (25m × 25m):
  - Close-up view of relative motion near target
  - Same RT plane but with fixed 25m × 25m range
  - Better visualization of fine-scale trajectory differences
  - Clearly shows convergence behavior near target position

- **[formation_distance_thrust.png](formation_distance_thrust.png)** - 5-panel comparison:
  - (a) Relative distance between Chaser and Target over time
  - (b) Relative distance deviation from baseline (full time)
  - (c) Relative distance deviation from baseline (time ≥ 60 min, zoomed)
  - (d) Thrust magnitude comparison
  - (e) Thrust deviation from baseline

- **[formation_altitude.png](formation_altitude.png)** - 2-panel comparison:
  - (a) Chaser altitude over time
  - (b) Chaser altitude deviation from baseline

### Data Files

- **[comparison_summary.txt](comparison_summary.txt)** - Text summary with error metrics (RMSE, MAE, Max Error)
- **[comparison_metrics.csv](comparison_metrics.csv)** - CSV file with quantitative comparison metrics

## Key Findings

### Baseline (τ=0ms)
- Final relative distance: 0.002034 m
- Final relative position: (1.456, 1.219, 0.728) mm
- Chaser final altitude: 408.000 km

### With Inverse Compensation (τ=100ms, gain=100)
- **Perfect tracking** of baseline scenario
- RMSE: 0.000000 m (all components)
- Inverse compensation fully eliminates the effect of 100ms communication delay

### Without Inverse Compensation (τ=100ms)
- **Significant degradation** in formation flying performance
- Relative distance RMSE: **19.52 m** (vs baseline)
- Max deviation: **107.05 m**
- Position errors: X=18.49 m, Y=18.88 m, Z=10.29 m
- Thrust RMSE: 3.22 N

### Conclusion

The inverse compensator with gain=100 **perfectly compensates** for the 100ms communication delay in the formation flying scenario, maintaining identical performance to the zero-delay baseline. Without compensation, the delay causes substantial errors in relative position tracking (up to 107m maximum deviation).

## Running the Comparison

To regenerate the comparison figures:

```bash
cd /home/akira/mosaik-hils/figures_for_paper/15_formation
uv run python compare_formation.py
```

## Script Details

The comparison script ([compare_formation.py](compare_formation.py)) performs:

1. **Data loading** from HDF5 files in each scenario directory
2. **Parameter extraction** from directory names (tau, inv_comp, gain, controller_type)
3. **Error metrics calculation** (RMSE, MAE, Max Error) against baseline
4. **Visualization generation**:
   - 3D relative trajectories
   - Time-series comparisons
   - Deviation plots
5. **Statistical summary** export (TXT and CSV formats)

## Notes

- Time is displayed in **minutes** for better readability of long-duration simulations
- The baseline (τ=0) is shown as a thick black dashed line in all plots
- Inverse compensation results show zero error, indicating perfect delay compensation
- The script is based on similar comparison scripts in other directories (14_hohmann, etc.)
