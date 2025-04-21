
import pandas as pd


lie_type = 'green'    


df = pd.read_csv(f"shots_under70_{lie_type}.csv")



# Group by exact distance
summary = df.groupby("holedis").agg(
    avg_strokes=("strokes_remaining", "mean"),
    count=("strokes_remaining", "count")
).reset_index()



# -------- SAVE --------
summary.to_csv(f"strokes_summary_{lie_type}.csv", index=False)
print(f"Saved summary for {lie_type}.")
