import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("data/15-may-fede.csv")

#print(df.head())
print(df.columns)
df["Carry Flat - Length"] = pd.to_numeric(df["Carry Flat - Length"], errors='coerce')
df["Carry Flat - Side"] = pd.to_numeric(df["Carry Flat - Side"], errors='coerce')

nineiron_colCF_length_side = df.loc[119:157, ["Carry Flat - Length", "Carry Flat - Side"]]

nineiron_colCF_length_side_sorted = nineiron_colCF_length_side.sort_values(by = "Carry Flat - Length", ascending=True)

print(nineiron_colCF_length_side.describe())
# 3. Make a scatter plot
plt.figure(figsize=(7, 6))
plt.scatter(
    nineiron_colCF_length_side_sorted["Carry Flat - Side"].values,
    nineiron_colCF_length_side_sorted["Carry Flat - Length"].values,
    alpha=0.8
)

# 4. Label and style the plot
plt.xlabel("Carry Flat - Side")  # downrange distance
plt.ylabel("Carry Flat - Distance")   # lateral dispersion from aim line
plt.title("9-Iron Shot Dispersion")
plt.axhline(0, color='black', linewidth=1)
plt.axvline(0, color='black', linewidth=1)

# Draw reference lines at x=0 (target carry distance = 0) and y=0 (aim line)
plt.ylim(80, 160)            # 2) Zoom in to your data range for length
plt.xlim(-35, 35)  
# Optional: Make x and y scales the same so circles appear round
plt.axis("equal")
plt.grid(True)
plt.show()