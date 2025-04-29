from shapely.geometry import Point, Polygon as SPolygon
from shapely.geometry import LineString
from shapely.affinity import scale, rotate
from shapely.plotting import plot_polygon, plot_points
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, Ellipse
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

plot_polygon(green_poly, ax=ax, facecolor='palegreen', edgecolor='green',  lw=2, label='Green')
plot_polygon(sand_poly,  ax=ax, facecolor='khaki',     edgecolor='peru',   label='Bunker')
plot_polygon(water_poly, ax=ax, facecolor='lightblue', edgecolor='navy',   label='Water')
#plot_points([Point(*PIN)], ax=ax, marker='*', color='red', markersize=150, label='Pin')

ax.set_aspect('equal', 'box')
ax.set_xlim(-45, 45); ax.set_ylim(40, 180)
ax.set_xlabel("x (yd)"); ax.set_ylabel("y (yd)")
ax.set_title("Par-3 with water + sand hazards")
ax.legend(loc="upper right")
plt.grid(ls=":", alpha=.3)
plt.tight_layout(); plt.show()



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

#    # > some test cases
# print("On green :", expected_strokes(0,1))
# print("Water:", expected_strokes(-34, -10)) 
# print("Bunker :", expected_strokes(15, -25))

files = ["data/15-may-fede.csv", "data/fede8ironhalf7iron.csv"]

dfs = []
for f in files: #taking two datasets for 9iron, 8iron and 7iron shots
    df_tmp = pd.read_csv(f)

    # change from string to number
    df_tmp["Carry Flat - Side"]   = pd.to_numeric(df_tmp["Carry Flat - Side"],   errors="coerce")
    df_tmp["Carry Flat - Length"] = pd.to_numeric(df_tmp["Carry Flat - Length"], errors="coerce")

    dfs.append(df_tmp)

df_all = pd.concat(dfs, ignore_index=True) #reset row values from 0 onwards

clubs = ["9 iron", "8 iron", "7 iron"]
df_all["Club"] = df_all["Club"].str.lower() #reformat the club column
print(df_all["Club"])
#filtering shots that we want (9,8,7 iron shots)
shots = (
    df_all[df_all["Club"].isin(clubs)]
        [["Club", "Carry Flat - Side", "Carry Flat - Length"]] #checking the x,y coordinate of landing spot of ball
        .dropna() 
        .rename(columns={"Carry Flat - Side": "x", #set x and y coordinates
                         "Carry Flat - Length": "y",
                         "Club": "club"}) # club for coloring
        .reset_index(drop=True) #start row from 0 after selection from df_all
)

#expected strokes for each club shot
estimates = []
for x, y in zip(shots["x"], shots["y"]):
    est = expected_strokes(x, y) #calculate expected strokes
    estimates.append(est)

shots["exp_strokes"] = estimates

print(shots)

#CODE FOR PLOTTING ON MOCK-UP GREEN
fig, ax = plt.subplots(figsize=(7, 7)) #plotting the dispersion of club shots
# course details color 
plot_polygon(green_poly, ax=ax, fc="palegreen", ec="green", label="Green")
plot_polygon(sand_poly,  ax=ax, fc="khaki",     ec="peru",  label="Bunker")
plot_polygon(water_poly, ax=ax, fc="lightblue", ec="navy",  label="Water")
ax.plot(*PIN, "r*", ms=12, label="Pin")

# expected strokes heat chart
norm   = plt.Normalize(shots.exp_strokes.min(), shots.exp_strokes.max()) #maps [min, max] -> [0,1]
cmap   = cm.viridis #found online, used for gradient shading

# different shapes for type of club shot 
marker_map = {
    "9 iron": "o",      
    "8 iron": "s",  #square
    "7 iron": "^",  #triangle
}
#for all club shots, plot 
for club, grp in shots.groupby("club"):
    ax.scatter(grp.x, grp.y, c=cmap(norm(grp.exp_strokes.values)), marker=marker_map.get(club, "o"),
        s=35, 
        edgecolor="k", #add black outline
        label=club.title(), #adds club label to the key
    )

# color bar for expected strokes (found online)
fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label="Expected strokes")

ax.set_aspect("equal", "box")
ax.set_xlim(-50, 50); ax.set_ylim(40, 180)
ax.set_xlabel("Divergence from Aimpoint (yd)")
ax.set_ylabel("Carry distance (yd)")
ax.set_title("7, 8 and 9-iron dispersion")
ax.legend(loc="upper right") #key/legend for markers, club names, etc
ax.grid(ls=":", alpha=.3) #light grid for reference
plt.savefig("results/shot_overlay_expected_strokes.png", dpi=150)
plt.show()

print(f"Mean expected strokes : {shots.exp_strokes.mean():.2f}")
print(f"Median expected strokes: {shots.exp_strokes.median():.2f}")


#CONE APPROACH: Setting target direction from -10 degrees to 10 degrees away from the pin location
def rotate_xy(x, y, theta):
    #using rotation matrix R_theta
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    new_coord = (x * cos_t - y * sin_t, x * sin_t + y * cos_t) #comes from rotation matrix * [x,y]
    return new_coord

#penalty function for expected number of strokes against target value
def playoff_loss(exp_strokes, target, k = 3):
    #setting exponential penalty for exp strokes > our target value
    #if exp strokes = target (lets say birdie) then our function returns 1
    #if exp strokes > target then penalty > 1 and increases exponentially
    #if exp strokes < target then penalty < 1 but small reward for this
    return np.mean(np.exp(k * (exp_strokes - target)))

#plots club dispersion 
def plot_cone_on_course(df_c, best_angle_deg, club, cone_deg=10, cone_len=200):
    shot_angles = np.degrees(np.arctan2(df_c.x.values, df_c.y.values))
    
    #mock up hole plot
    fig, ax = plt.subplots(figsize=(7, 7))
    plot_polygon(green_poly, ax=ax, fc='palegreen',  ec='green', lw=2, label='Green')
    plot_polygon(sand_poly,  ax=ax, fc='khaki',      ec='peru',  label='Bunker')
    plot_polygon(water_poly, ax=ax, fc='lightblue',  ec='navy',  label='Water')
    ax.plot(*PIN, 'r*', ms=12, label='Pin')

    # plot the shot dispersion 
    ax.scatter(df_c['x'], df_c['y'], s=28, c='seagreen')
    
    # cone boundaries and shading (origin = tee pointed up from (0,0))
    y_line = np.linspace(0, cone_len, 220)
    for bd in (-cone_deg, cone_deg): #from -10 degrees to +10 degrees about 0
        rad = np.deg2rad(bd)
        ax.plot(y_line * np.sin(rad), y_line * np.cos(rad), 'k--', lw=1.2)
    left_x  = y_line * np.sin(np.deg2rad(-cone_deg))
    right_x = y_line * np.sin(np.deg2rad(+cone_deg))
    ax.fill_betweenx(y_line, left_x, right_x, color='grey', alpha=0.10, label=f'±{cone_deg}° cone')

    # find and note best aim line 
    best_rad = np.deg2rad(best_angle_deg)
    ax.plot([0, cone_len * np.sin(best_rad)],
            [0, cone_len * np.cos(best_rad)],
            color='purple', lw=3,
            label=f'Best aimpoint {best_angle_deg:+.0f}°')
    #all similar to ax configs as before
    ax.set_aspect('equal', 'box')
    ax.set_xlim(-70,  70)
    ax.set_ylim(  0, cone_len)
    ax.set_xlabel('Divergence from Aimpoint (yd)')
    ax.set_ylabel('Carry distance (yd)')
    ax.set_title(f'{club.title()} shots with ±{cone_deg}° cone')
    ax.grid(ls=':', alpha=.3)
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

    #save to file path (ex: 7 iron -> 7_iron_cone_plot.png)
    club_filename = club.lower().replace(' ', '_') 
    save_path = f"results/{club_filename}_cone_plot.png"
    fig.savefig(save_path, dpi=150) #ensure good quality for png
    plt.close(fig)

for club, df_c in shots.groupby("club"): # for each iron
    print(f"\n━━━━━━━━━━━━━━━━━━  {club.title()}  ━━━━━━━━━━━━━━━━━━")

    # Mean expected strokes for each club (df_c includes (x,y) coords + exp_strokes)
    club_mean = df_c.exp_strokes.mean()
    club_med  = df_c.exp_strokes.median()
    print(f" Mean & Median exp. strokes) : {club_mean:.2f}  /  {club_med:.2f}")

    # Checking expected shots from dispersion of shots: 
    # essentially changing the landing location of all shots via rotation then calculating expected strokes
    results = []
    for deg in range(-10, 11): # negative 10 degrees to +10 degrees if pin is at 0 
        theta = np.deg2rad(deg)
        xs_r, ys_r = rotate_xy(df_c.x.values, df_c.y.values, theta) #rotate based rad value of degree
        est_rotated = []
        for x_rot, y_rot in zip(xs_r, ys_r): #goes through all shots but rotated
            stroke = expected_strokes(x_rot, y_rot)
            est_rotated.append(stroke)
        mean_exp = np.mean(est_rotated)
        
        results.append({"angle_deg": deg,
                        "mean_exp": mean_exp})

    angle_df = pd.DataFrame(results)
    best = angle_df.loc[angle_df["mean_exp"].idxmin()] #finds min aka best expected strokes of aimpoint degree

    plot_cone_on_course(df_c, best["angle_deg"], club) # plots cone (degree spread) and best degree on mockup hole
    print("±10° mean exp. strokes by angle")
    print(angle_df.to_string(index=False, formatters={"mean_exp": "{:.2f}".format}))
    print(f"Optimal within cone : {best.angle_deg:+.0f}°  "f"(mean {best.mean_exp:.2f})")

    angle_df.to_csv(f"results/{club.replace(' ','_')}_aim_cone.csv", index=False)

    # “must-birdie” loss calculated (if not birdie situation, then add higher penalty)
    need_birdie = 2
    cannot_par = 3
    loss = playoff_loss(df_c.exp_strokes.values, need_birdie, k = 3)
    print(f"Play-off weighted loss (birdie target) : {loss:.3f}")
    
    