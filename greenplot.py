import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon, Circle

fig, ax = plt.subplots(figsize=(8, 8))

# Background rough (white)
ax.set_facecolor("white")

# Water (blue)
water = Polygon([[-30, -20], [-25, 30], [-15, 35], [-20, -25]], color="skyblue", alpha=0.6, zorder=1)
ax.add_patch(water)

# Green (light green)
green = Circle((0, 0), radius=6, color="lightgreen", zorder=2)
ax.add_patch(green)

# Bunker (yellow)
bunker = Polygon([[4, -3], [8, 0], [5, 4], [2, 1]], color="goldenrod", zorder=2)
ax.add_patch(bunker)

# Circles every 5 yards
for r in np.arange(5, 35, 5):
    circle = Circle((0, 0), r, color="black", fill=False, lw=1, zorder=3)
    ax.add_patch(circle)

# GPR overlay (example using meshgrid X, Y and Z)
# ax.contourf(X, Y, Z, cmap="inferno", levels=20, alpha=0.6, zorder=4)

# Pin marker
ax.plot(0, 0, marker="x", color="black", markersize=10, zorder=5)

ax.set_xlim(-35, 35)
ax.set_ylim(-35, 35)
ax.set_aspect('equal')
plt.title("GPR Overlay on Custom Hole Design")
plt.show()
