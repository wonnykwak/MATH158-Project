import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load and preprocess green data (in feet)
df = pd.read_csv("results/gpr_green_preds.csv")

# Filter to within 70 feet, then convert to yards
df = df[df["holedis"] <= 70/3]
#df["holedis"] = df["holedis"] / 3.0  # Convert feet to yards

r_vals = df["holedis"].values
z_vals = df["pred"].values

# Create polar grid
theta = np.linspace(0, 2 * np.pi, 360)
R, T = np.meshgrid(r_vals, theta)
Z = np.tile(z_vals, (len(theta), 1))
X = R * np.cos(T)
Y = R * np.sin(T)

# Set up plot
fig, ax = plt.subplots(figsize=(6.5, 6.5), subplot_kw={'aspect': 'equal'})
mesh = ax.pcolormesh(X, Y, Z, shading='auto', cmap='inferno_r')

# Guide circles every 5 yards
r_max = r_vals.max()
for rad in range(5, int(r_max) + 5, 5):
    circle = plt.Circle((0, 0), radius=rad, color='black', fill=False, linewidth=0.8, linestyle='--')
    ax.add_patch(circle)

# Add pin flag
ax.text(0, 0, "X", fontsize=16, ha='center', va='center')

# Titles and aesthetics
ax.set_title("GPR Heatmap: Green (≤ 70 ft / 23 yards)", fontsize=14)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlim(-r_max, r_max)
ax.set_ylim(-r_max, r_max)

# Colorbar
cbar = plt.colorbar(mesh, ax=ax, orientation='horizontal', pad=0.08)
cbar.set_label("Predicted strokes to hole out")

plt.tight_layout()
plt.savefig("results/gpr_green_polar.png")
plt.show()
