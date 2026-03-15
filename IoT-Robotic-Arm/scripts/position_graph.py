import json
import matplotlib.pyplot as plt
import numpy as np

with open("saved_positions.json", "r") as f:
    data = json.load(f)

indices = list(range(len(data)))
initial_angles = [d["initial_s1_angle"] for d in data]
saved_angles = [d["saved_s1_angle"] for d in data]
angle_diff = [d["saved_s1_angle"] - d["initial_s1_angle"] for d in data]
center_x = [d["center_x"] for d in data]
clicked_x = [d["clicked_x"] for d in data]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].plot(indices, initial_angles, "b-o", label="Initial S1 Angle")
axes[0, 0].plot(indices, saved_angles, "r-o", label="Saved S1 Angle")
axes[0, 0].set_xlabel("Data Point Index")
axes[0, 0].set_ylabel("Angle (degrees)")
axes[0, 0].set_title("Initial vs Saved Position")
axes[0, 0].legend()
axes[0, 0].grid(True)

axes[0, 1].bar(
    indices, angle_diff, color=["green" if d >= 0 else "red" for d in angle_diff]
)
axes[0, 1].axhline(y=0, color="black", linestyle="-", linewidth=0.5)
axes[0, 1].set_xlabel("Data Point Index")
axes[0, 1].set_ylabel("Angle Difference (degrees)")
axes[0, 1].set_title("Difference: Saved - Initial Position")
axes[0, 1].grid(True, axis="y")

axes[1, 0].scatter(indices, center_x, c="blue", label="Center X", marker="o", s=50)
axes[1, 0].scatter(indices, clicked_x, c="red", label="Clicked X", marker="x", s=50)
axes[1, 0].set_xlabel("Data Point Index")
axes[1, 0].set_ylabel("X Position")
axes[1, 0].set_title("Center X vs Clicked X")
axes[1, 0].legend()
axes[1, 0].grid(True)

axes[1, 1].scatter(center_x, clicked_x, c="purple", alpha=0.7, s=80)
axes[1, 1].plot(
    [min(center_x), max(center_x)],
    [min(center_x), max(center_x)],
    "k--",
    label="y=x line",
)
axes[1, 1].set_xlabel("Center X")
axes[1, 1].set_ylabel("Clicked X")
axes[1, 1].set_title("Center X vs Clicked X Scatter")
axes[1, 1].legend()
axes[1, 1].grid(True)

plt.tight_layout()
plt.savefig("position_analysis.png", dpi=150)
plt.show()

print(f"Total data points: {len(data)}")
print(f"Average angle difference: {np.mean(angle_diff):.2f}")
print(f"Clicked X range: {min(clicked_x)} - {max(clicked_x)}")
