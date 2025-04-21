import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Lie types and display labels
lies = {
    "fairway": "Fairway",
    "rough": "Rough",
    "sand": "Sand",
}

# Load data and determine shared scales
r_max, z_min, z_max = 0, float('inf'), float('-inf')
r_dict = {}

for lie in lies:
    df = pd.read_csv(f"results/gpr_{lie}_preds.csv")
    r_vals = df["holedis"].values
    z_vals = df["pred"].values
    r_dict[lie] = (r_vals, z_vals)
    r_max = max(r_max, r_vals.max())
    z_min = min(z_min, z_vals.min())
    z_max = max(z_max, z_vals.max())

# --------- Set up layout ---------
fig = plt.figure(figsize=(14, 6.5))
gs = gridspec.GridSpec(2, 3, height_ratios=[9, 1])  # 2 rows: 3 plots on top, 1 colorbar on bottom
axes = [fig.add_subplot(gs[0, i], aspect='equal') for i in range(3)]

# Plot each lie
for ax, (lie, title) in zip(axes, lies.items()):
    r_vals, z_vals = r_dict[lie]
    theta = np.linspace(0, 2 * np.pi, 360)
    R, T = np.meshgrid(r_vals, theta)
    Z = np.tile(z_vals, (len(theta), 1))
    X = R * np.cos(T)
    Y = R * np.sin(T)

    mesh = ax.pcolormesh(X, Y, Z, shading='auto', cmap='inferno_r', vmin=z_min, vmax=z_max)

    for rad in range(5, int(r_max) + 5, 5):
        circle = plt.Circle((0, 0), radius=rad, color='black', fill=False, linewidth=0.8, linestyle='--')
        ax.add_patch(circle)

    # Add a little pin flag in the center
    ax.text(0, 0, "X", fontsize=16, ha='center', va='center')
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-r_max, r_max)
    ax.set_ylim(-r_max, r_max)

# Add a shared horizontal colorbar
cbar_ax = fig.add_subplot(gs[1, :])  # Bottom row spans all columns
cbar = fig.colorbar(mesh, cax=cbar_ax, orientation='horizontal')
cbar.set_label('Predicted strokes to hole out')

fig.suptitle("GPR Heatmaps by Lie Type", fontsize=16, y=0.97)
plt.savefig("results/gpr_polar_faceted.png")
plt.show()
