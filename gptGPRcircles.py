import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

lies = ['fairway', 'rough', 'sand']
n_angles = 200
theta = np.linspace(0, 2 * np.pi, n_angles)

fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw={'aspect': 'equal'})
pcs = []

for ax, lie in zip(axes, lies):
    try:
        df = pd.read_csv(f"results/gpr_{lie}_preds.csv")
    except FileNotFoundError:
        ax.set_title(f"{lie.capitalize()} (Missing)")
        ax.axis('off')
        continue

    r = df["holedis"].values
    z = df["pred"].values

    R, T = np.meshgrid(r, theta)
    Z = np.tile(z, (n_angles, 1))

    X = R * np.cos(T)
    Y = R * np.sin(T)

    pc = ax.pcolormesh(X, Y, Z, shading='auto', cmap='YlGn')
    pcs.append(pc)
    ax.set_title(lie.capitalize(), fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-max(r), max(r))
    ax.set_ylim(-max(r), max(r))

# Add colorbar if at least one plot succeeded
if pcs:
    cbar = fig.colorbar(pcs[0], ax=axes, shrink=0.8, location='right')
    cbar.set_label("Predicted strokes to hole out")

fig.suptitle("GPR Heatmaps by Lie (Top-Down View)", fontsize=16)
plt.tight_layout()
plt.subplots_adjust(top=0.88)
plt.show()
