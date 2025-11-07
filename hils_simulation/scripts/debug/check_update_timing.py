"""Check when controller and compensator actually update"""

import h5py
import numpy as np

filepath = "results/20251022-032735_inverse_comp/hils_data.h5"

with h5py.File(filepath, "r") as f:
    time_s = f["data"]["time_s"][:]
    controller_thrust = f["data"]["command_ControllerSim-0.PIDController_0_thrust"][:]
    compensated_thrust = f["data"]["compensated_output_InverseCompSim-0.cmd_compensator_thrust"][:]

# Find when controller actually updates (changes value)
controller_changes = np.diff(controller_thrust)
change_indices = np.where(np.abs(controller_changes) > 1e-6)[0] + 1  # +1 because diff shifts index

print("=" * 80)
print("CONTROLLER UPDATE TIMING")
print("=" * 80)
print(f"Total samples: {len(time_s)}")
print(f"Controller updates: {len(change_indices)}")
print()

print("First 20 controller updates:")
print(f"{'Index':>8} | {'Time [s]':>10} | {'Controller [N]':>15} | {'Compensated [N]':>16}")
print("-" * 70)

for i, idx in enumerate(change_indices[:20]):
    print(f"{idx:8d} | {time_s[idx]:10.4f} | {controller_thrust[idx]:15.6f} | {compensated_thrust[idx]:16.6f}")

# Check update interval
if len(change_indices) > 1:
    intervals = np.diff(change_indices)
    print(f"\n{'=' * 80}")
    print("UPDATE INTERVAL ANALYSIS")
    print(f"{'=' * 80}")
    print(f"Update intervals (first 10): {intervals[:10]}")
    print("Expected interval: 100 samples (10ms)")
    print(f"Actual mean interval: {np.mean(intervals):.1f} samples")
    print(f"Actual std interval: {np.std(intervals):.1f} samples")

# Now check compensator updates
comp_changes = np.diff(compensated_thrust)
comp_change_indices = np.where(np.abs(comp_changes) > 1e-6)[0] + 1

print(f"\n{'=' * 80}")
print("COMPENSATOR UPDATE TIMING")
print(f"{'=' * 80}")
print(f"Compensator updates: {len(comp_change_indices)}")

if len(comp_change_indices) > 1:
    comp_intervals = np.diff(comp_change_indices)
    print(f"Update intervals (first 20): {comp_intervals[:20]}")
    print(f"Mean interval: {np.mean(comp_intervals):.1f} samples")

# Check if compensator updates at every step
consecutive_updates = 0
for i in range(1, min(100, len(compensated_thrust))):
    if compensated_thrust[i] != compensated_thrust[i - 1]:
        consecutive_updates += 1

print(f"\nConsecutive compensator changes in first 100 samples: {consecutive_updates}")
if consecutive_updates > 90:
    print("⚠ WARNING: Compensator appears to update EVERY step!")
else:
    print(f"✓ Compensator updates periodically ({consecutive_updates}/100 changes)")
