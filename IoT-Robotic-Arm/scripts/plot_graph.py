import json
import matplotlib.pyplot as plt

with open("scripts/saved_positions.json", "r") as f:
    data = json.load(f)

x_diff = [d["center_x"] - d["clicked_x"] for d in data]
y_diff = [d["initial_s1_angle"] - d["saved_s1_angle"] for d in data]

fig, ax = plt.subplots(figsize=(10, 6))

ax.scatter(x_diff, y_diff, color="blue", alpha=0.7)

ax.axhline(y=0, color="red", linestyle="--", linewidth=1, label="Zero angle diff")
ax.axvline(x=0, color="green", linestyle="--", linewidth=1, label="Zero position diff")

ax.set_xlabel("center_x - clicked_x")
ax.set_ylabel("initial_s1_angle - saved_s1_angle")
ax.set_title("Position Difference vs Angle Difference")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("scripts/position_difference_graph.png", dpi=150)
