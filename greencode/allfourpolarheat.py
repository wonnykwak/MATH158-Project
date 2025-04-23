import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Lie types and display labels (including green!)
lies = {
    "fairway": "Fairway",
    "rough": "Rough",
    "sand": "Sand",
    "green": "Green (≤ 70 ft)"
}

# Load data and determine shared scales
r_max, z_min, z_max = 0, float('inf'), float('-inf')
r_dict = {}

for lie in lies:
    df = pd.read_csv(f"results/gpr_{lie}_preds.csv")
    
    if lie == "green":
        df = df[df["holedis"] <= 70/3]

    r_vals = df["holedis"].values
    z_vals = df["pred"].values
    r_dict[lie] = (r_vals, z_vals)
    r_max = max(r_max, r_vals.max())
    z_min = min(z_min, z_vals.min())
    z_max = max(z_max, z_vals.max())

# --------- Set up 2x2 layout ---------
fig = plt.figure(figsize=(13, 11))
gs = gridspec.GridSpec(3, 2, height_ratios=[6, 6, 1])  # 3rd row for colorbar
axes = [fig.add_subplot(gs[i // 2, i % 2], aspect='equal') for i in range(4)]

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

    ax.text(0, 0, "X", fontsize=16, ha='center', va='center')
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-r_max, r_max)
    ax.set_ylim(-r_max, r_max)

# Shared colorbar
cbar_ax = fig.add_subplot(gs[2, :])
cbar = fig.colorbar(mesh, cax=cbar_ax, orientation='horizontal')
cbar.set_label("Predicted strokes to hole out")

fig.suptitle("GPR Heatmaps by Lie Type", fontsize=16, y=0.94)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("results/gpr_polar_faceted_with_green.png")
plt.show()
