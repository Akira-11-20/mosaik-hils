"""
位置データの比較プロット作成スクリプト

2つのHDF5ファイルからposition_EnvSimデータを読み込んで重ねてプロット

Data 1: 遅延なし
Data 2: 50ms遅延あり
"""

import h5py
import matplotlib.pyplot as plt
from pathlib import Path

# ファイルパス
file1 = "/home/akira/mosaik-hils/hils_simulation/results/20251016-123939/hils_data.h5"  # 遅延なし
file2 = "/home/akira/mosaik-hils/hils_simulation/results/20251016-124128/hils_data.h5"  # 50ms遅延

# データ読み込み
with h5py.File(file1, "r") as f1:
    time1 = f1["data"]["time_s"][:]
    position1 = f1["data"]["position_EnvSim-0.Spacecraft1DOF_0"][:]

with h5py.File(file2, "r") as f2:
    time2 = f2["data"]["time_s"][:]
    position2 = f2["data"]["position_EnvSim-0.Spacecraft1DOF_0"][:]

# プロット作成
plt.figure(figsize=(12, 6))

plt.plot(
    time1, position1, label="No Delay (0ms)", linewidth=2, alpha=0.8, color="tab:blue"
)
plt.plot(
    time2,
    position2,
    label="With Delay (50ms)",
    linewidth=2,
    alpha=0.8,
    color="tab:orange",
)

plt.xlabel("Time [s]", fontsize=12)
plt.ylabel("Position [m]", fontsize=12)
plt.title(
    "Position Comparison - Effect of Communication Delay",
    fontsize=14,
    fontweight="bold",
)
plt.legend(fontsize=11, loc="best", framealpha=0.9)
plt.grid(True, alpha=0.3)

# 遅延情報をテキストボックスで追加
textstr = "Data 1: No delay\nData 2: 50ms delay"
props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
plt.text(
    0.02,
    0.98,
    textstr,
    transform=plt.gca().transAxes,
    fontsize=10,
    verticalalignment="top",
    bbox=props,
)

plt.tight_layout()

# 保存
output_path = Path("/home/akira/mosaik-hils/hils_simulation/position_comparison.png")
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"✅ Plot saved to: {output_path}")

# 統計情報を表示
print(f"\nData 1 (No Delay - 20251016-123939):")
print(f"  Time range: {time1[0]:.3f}s - {time1[-1]:.3f}s")
print(f"  Position range: {position1.min():.3f}m - {position1.max():.3f}m")
print(f"  Final position: {position1[-1]:.3f}m")

print(f"\nData 2 (50ms Delay - 20251016-124128):")
print(f"  Time range: {time2[0]:.3f}s - {time2[-1]:.3f}s")
print(f"  Position range: {position2.min():.3f}m - {position2.max():.3f}m")
print(f"  Final position: {position2[-1]:.3f}m")

print(f"\nDifference:")
print(f"  Final position difference: {abs(position2[-1] - position1[-1]):.3f}m")
print(f"  Max position difference: {abs(position2.max() - position1.max()):.3f}m")

plt.close()
