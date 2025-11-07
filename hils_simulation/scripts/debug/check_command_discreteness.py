"""
Check how discrete command changes affect inverse compensation
"""

import h5py
import numpy as np

filepath = "results/20251022-013456_inverse_comp/hils_data.h5"

with h5py.File(filepath, "r") as f:
    time_s = f["data"]["time_s"][:]
    controller_thrust = f["data"]["command_ControllerSim-0.PIDController_0_thrust"][:]
    compensated_thrust = f["data"]["compensated_output_InverseCompSim-0.cmd_compensator_thrust"][:]

# Find when controller output changes (non-continuous updates)
controller_changes = np.diff(controller_thrust)
change_indices = np.where(np.abs(controller_changes) > 1e-3)[0]

print("=" * 80)
print("CONTROLLER OUTPUT UPDATE PATTERN")
print("=" * 80)

# Check update intervals
if len(change_indices) > 1:
    update_intervals = np.diff(change_indices)
    print(f"Number of controller updates: {len(change_indices)}")
    print("Update interval stats:")
    print(f"  Mean: {np.mean(update_intervals):.1f} samples ({np.mean(update_intervals) * 0.0001:.6f}s)")
    print(f"  Std: {np.std(update_intervals):.1f} samples")
    print(f"  Min: {np.min(update_intervals)} samples")
    print(f"  Max: {np.max(update_intervals)} samples")

    # Expected update interval = control period / time resolution = 0.01s / 0.0001s = 100 samples
    print("\nExpected update interval: 100 samples (10ms control period)")

print("\n" + "=" * 80)
print("SAMPLE UPDATES AROUND DIVERGENCE (2.7-2.85s)")
print("=" * 80)

# Sample at 2.7-2.85s (around divergence)
start_idx = np.searchsorted(time_s, 2.7)
end_idx = np.searchsorted(time_s, 2.85)

print(f"{'Time [s]':>10} | {'Controller [N]':>15} | {'Compensated [N]':>16} | {'Ratio':>10} | {'Change':>8}")
print("-" * 80)

prev_c = controller_thrust[start_idx]
for i in range(start_idx, min(start_idx + 40, end_idx), 4):  # Every 4th sample
    t = time_s[i]
    c_thrust = controller_thrust[i]
    comp_thrust = compensated_thrust[i]

    # Check if this is an update point
    change_marker = "*" if abs(c_thrust - prev_c) > 1e-3 else ""

    # Avoid division by zero
    ratio = comp_thrust / c_thrust if abs(c_thrust) > 1e-6 else 0

    print(f"{t:10.4f} | {c_thrust:15.3f} | {comp_thrust:16.3f} | {ratio:10.3f} | {change_marker:>8}")
    prev_c = c_thrust

print("\n" + "=" * 80)
print("COMPENSATION AT UPDATE POINTS")
print("=" * 80)

# Look at what happens exactly at controller update points
sample_updates = change_indices[change_indices > start_idx][:10]  # First 10 updates after 2.7s

print(
    f"{'Time [s]':>10} | {'Ctrl[k-1]':>12} | {'Ctrl[k]':>12} | {'Delta':>12} | {'Comp[k]':>12} | {'Expected':>12} | {'Actual Gain':>12}"
)
print("-" * 130)

gain = 15.0
for idx in sample_updates:
    t = time_s[idx]
    prev_val = controller_thrust[idx - 1]
    curr_val = controller_thrust[idx]
    delta = curr_val - prev_val
    comp_val = compensated_thrust[idx]

    # Expected from formula: comp = gain * curr - (gain-1) * prev
    expected_comp = gain * curr_val - (gain - 1) * prev_val

    # Actual gain if we solve: comp = gain * curr - (gain-1) * prev
    # Rearranging: gain = (comp - curr) / (curr - prev) + 1
    if abs(delta) > 1e-6:
        actual_gain = (comp_val - curr_val) / delta + 1
    else:
        actual_gain = 0

    print(
        f"{t:10.4f} | {prev_val:12.3f} | {curr_val:12.3f} | {delta:12.3f} | {comp_val:12.3f} | {expected_comp:12.3f} | {actual_gain:12.3f}"
    )

print("\n" + "=" * 80)
print("ZERO-ORDER HOLD EFFECT")
print("=" * 80)

# Check what happens between controller updates (zero-order hold)
print("When controller output is constant (zero-order hold), compensation should reduce to identity...")
print("But does it?")
print()

# Find a stretch where controller is constant
constant_start = 27000  # Around 2.7s
constant_end = constant_start + 100

if controller_thrust[constant_start] == controller_thrust[constant_start + 50]:
    print(f"Checking samples {constant_start} to {constant_start + 20}:")
    print(f"{'Sample':>8} | {'Time [s]':>10} | {'Controller':>12} | {'Compensated':>13} | {'Prev Ctrl':>12}")
    print("-" * 70)

    for i in range(constant_start, constant_start + 20, 2):
        t = time_s[i]
        curr = controller_thrust[i]
        comp = compensated_thrust[i]
        prev = controller_thrust[max(0, i - 1)]

        print(f"{i:8d} | {t:10.4f} | {curr:12.3f} | {comp:13.3f} | {prev:12.3f}")
else:
    print("Controller is changing in this region")
