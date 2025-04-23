import pandas as pd
import numpy as np

lie_types = ['green']
bin_width = 1  # in yards
max_distance = 70/3  # in yards

for lie_type in lie_types:
    # Load yard-based green data
    df = pd.read_csv(f"data/putts_under90ft_{lie_type}.csv")  # file has distances in yards now

    # Bin by holedis (which is now in yards)
    df["bin"] = pd.cut(df["holedis"], bins=np.arange(0, max_distance + bin_width, bin_width))

    # mean holedis and average strokes to hole out
    summary = df.groupby("bin").agg(
        holedis=("holedis", "mean"),
        avg_strokes=("strokes_remaining", "mean"),
        count=("strokes_remaining", "count")
    ).dropna().reset_index(drop=True)

    # Save the binned summary data
    summary.to_csv(f"data/strokes_summary_{lie_type}_yards.csv", index=False)

    print(f"Saved yard-based summary for {lie_type}.")
