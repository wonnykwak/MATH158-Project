from shapely.geometry import Point, Polygon as SPolygon
from shapely.geometry import LineString
from shapely.affinity import scale, rotate
from shapely.plotting import plot_polygon, plot_points
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
import matplotlib.cm as cm
import math 
import os
import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor 
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import contextlib, io
from scipy.stats import gaussian_kde

PIN = (0.0, 140.0)
PAR3_DIST = 140
circle = Point(*PIN).buffer(15.0)
green_poly = SPolygon(scale(circle, xfact=1.5, yfact = 1.0, origin = PIN))        # 30‑yd radius
sand_right_angle_line = LineString([
    (15, -25 + PAR3_DIST),
    (30, -13+ PAR3_DIST)
])
rounded_sand = sand_right_angle_line.buffer(8.0, join_style=2 )
sand_poly  = SPolygon(rounded_sand.exterior.coords)     # simple triangle bunker
#rotated_oval = rotate(scale(circle, xfact=1.0, yfact=3.0, origin=PIN), angle=120, origin=PIN)
water_right_angle_line = LineString([
    (-34, -15+ PAR3_DIST),
    (-34, 8+ PAR3_DIST),
    (-28, 20+ PAR3_DIST)
])
rounded_water = water_right_angle_line.buffer(10.0, join_style=2)
water_poly = SPolygon(rounded_water.exterior.coords)
#water_poly = SPolygon([(-35, -10), (-35, 10),(-20,  10), (-20, -10)])    

x,y = sand_poly.exterior.xy
plt.plot(x,y)
# plt.show()
print(green_poly)


fig, ax = plt.subplots(figsize=(7, 7))

# plot_polygon(green_poly, ax=ax, facecolor='palegreen', edgecolor='green',  lw=2, label='Green')
# plot_polygon(sand_poly,  ax=ax, facecolor='khaki',     edgecolor='peru',   label='Bunker')
# plot_polygon(water_poly, ax=ax, facecolor='lightblue', edgecolor='navy',   label='Water')
# #plot_points([Point(*PIN)], ax=ax, marker='*', color='red', markersize=150, label='Pin')

# ax.set_aspect('equal', 'box')
# ax.set_xlim(-45, 45); ax.set_ylim(0, 180)
# ax.set_xlabel("x (yd)"); ax.set_ylabel("y (yd)")
# ax.set_title("Toy green with water + sand hazards")
# ax.legend(loc="upper right")
# plt.grid(ls=":", alpha=.3)
# plt.tight_layout(); plt.show()



lies = ['fairway', 'sand', 'dprough', 'rough', 'green']  # will handle green differently
os.makedirs("results", exist_ok=True)

gpr_by_lie = {}

for lie in lies:
    print(f"GPR on {lie}...")

    df = pd.read_csv(f"data/strokes_summary_{lie}.csv")

    # input features (no expansion)
    X = df[['holedis']].values
    y = df['avg_strokes'].values

    # Define kernel with default tuning
    kernel = RBF(length_scale_bounds=(5, 100.0)) + WhiteKernel(noise_level=0.05, noise_level_bounds=(0.05, 1.0))

    gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-2, normalize_y=True)
    gpr.fit(X, y)
    gpr_by_lie[lie] = gpr  
    print(f"Optimised kernel for {lie}: {gpr.kernel_}")



def euc_dist(point1, point2):
    distance = math.sqrt(sum([(a - b) ** 2 for a, b in zip(point1, point2)]))
    return distance

def expected_strokes(x, y):
    dist = math.dist(PIN, (x,y))
    print(dist)
    d2 = np.array([[dist]], dtype=float)
    cur_point = Point(x,y)
    if(green_poly.contains(cur_point)): #on green
        return gpr_by_lie['green'].predict(d2,return_std=False, return_cov=False)[0]
    elif(sand_poly.contains(cur_point)): #in bunker
        return gpr_by_lie['sand'].predict(d2,return_std=False, return_cov=False)[0] 
    elif(water_poly.contains(cur_point)): #in water hazard 
        return gpr_by_lie['fairway'].predict(d2,return_std=False, return_cov=False)[0] + 1
    else: #rough situation
        return gpr_by_lie['rough'].predict(d2,return_std=False, return_cov=False)[0]
vec_exp = np.vectorize(expected_strokes)

   # > inside rough fringe
print("On green :", expected_strokes(0,1))
print("Water:", expected_strokes(-34, -10)) 
print("Bunker :", expected_strokes(15, -25))

df = pd.read_csv("data/15-may-fede.csv")
df["Carry Flat - Length"] = pd.to_numeric(df["Carry Flat - Length"], errors='coerce')
df["Carry Flat - Side"]   = pd.to_numeric(df["Carry Flat - Side"],   errors='coerce')

shots = (
    df.loc[119:157, ["Carry Flat - Side", "Carry Flat - Length"]]
      .dropna()
      .rename(columns={"Carry Flat - Side": "x",
                       "Carry Flat - Length": "y"})
)

# 2.  Call your expected_strokes for every shot --------------------
#     (suppress the debug prints so the loop doesn’t spam stdout)
estimates = []
buf = io.StringIO()

for x, y in zip(shots["x"], shots["y"]):
    est = expected_strokes(x, y)
    estimates.append(est)

shots["exp_strokes"] = estimates

print(shots)
fig, ax = plt.subplots(figsize=(7, 7))
plot_polygon(green_poly, ax=ax, fc="palegreen", ec="green", lw=2, label="Green")
plot_polygon(sand_poly,  ax=ax, fc="khaki",     ec="peru",  label="Bunker")
plot_polygon(water_poly, ax=ax, fc="lightblue", ec="navy",  label="Water")
ax.plot(*PIN, "r*", ms=12, label="Pin")

norm   = plt.Normalize(shots.exp_strokes.min(), shots.exp_strokes.max())
colors = cm.viridis(norm(shots.exp_strokes.values))
ax.scatter(shots.x, shots.y, c=colors, s=45, edgecolor="k", alpha=.9, label="Shots")

ax.set_aspect("equal", "box")
ax.set_xlim(-45, 45); ax.set_ylim(40, 180)
ax.set_xlabel("Carry‑side offset (yd)")
ax.set_ylabel("Carry distance (yd)")
ax.set_title("9‑iron dispersion coloured by expected strokes")
fig.colorbar(cm.ScalarMappable(norm=norm, cmap="viridis"),
             ax=ax, label="Expected strokes")
ax.legend(loc="upper right")
ax.grid(ls=":", alpha=.3); plt.tight_layout()
plt.savefig("results/shot_overlay_expected_strokes.png", dpi=150)
plt.show()

# 4.  Histogram + KDE ---------------------------------------------
fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(shots.exp_strokes, bins=10, color="steelblue", alpha=.6,
        density=True, label="Histogram")

xs  = np.linspace(shots.exp_strokes.min()-0.2,
                  shots.exp_strokes.max()+0.2, 200)
kde = gaussian_kde(shots.exp_strokes)(xs)
ax.plot(xs, kde, color="darkblue", lw=2, label="KDE")

ax.set_xlabel("Expected strokes to hole‑out")
ax.set_ylabel("Density")
ax.set_title("Distribution of expected strokes (9‑iron sample)")
ax.legend(); plt.tight_layout()
plt.savefig("results/expected_strokes_distribution.png", dpi=150)
plt.show()

print(f"Mean expected strokes : {shots.exp_strokes.mean():.2f}")
print(f"Median expected strokes: {shots.exp_strokes.median():.2f}")