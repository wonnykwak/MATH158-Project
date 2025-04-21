# prepare_avg_data.py
import pandas as pd
import numpy as np

lie_type = 'rough'  
bin_width = 1
max_distance = 70

df = pd.read_csv(f"data/shots_under70_{lie_type}.csv")


df["bin"] = pd.cut(df["holedis"], bins=np.arange(0, max_distance + bin_width, bin_width))

# mean holedis and average strokes to hole out
summary = df.groupby("bin").agg(
    holedis=("holedis", "mean"),
    avg_strokes=("strokes_remaining", "mean"),
    count=("strokes_remaining", "count")
).dropna().reset_index(drop=True)

# Save the binned summary data
summary.to_csv(f"data/strokes_summary_{lie_type}.csv", index=False)

print(f"Saved summary for {lie_type}.")
