import pandas as pd

'''
Splitting ppdata into separate CSVs by lie type, keeping only shots within 70 yards
'''

# Load data
ppdata = pd.read_csv("ppdata.csv")

# Map startpos codes to readable lie labels
lie_map = {
    0: "Tee",
    1: "Fairway",
    2: "Rough",
    3: "Sand",
    4: "Green",
    6: "DpRough"
}
ppdata["lie"] = ppdata["startpos"].map(lie_map)

# Clean data- remove rows without essential info
ppdata = ppdata[ppdata["stroke"].notna() & ppdata["holedis"].notna()]

# Compute strokes remaining to hole out
ppdata["total_strokes"] = ppdata.groupby(["roundid", "holeid"])["stroke"].transform("max")
ppdata["strokes_remaining"] = ppdata["total_strokes"] - ppdata["stroke"] + 1

# Keep only shots within 70 yards
ppdata = ppdata[ppdata["holedis"] <= 70]

# Save one CSV per lie type
for lie_type in ppdata["lie"].dropna().unique():
    lie_df = ppdata[ppdata["lie"] == lie_type]
    filename = f"shots_under70_{lie_type.lower()}.csv"
    lie_df.to_csv(filename, index=False)

print("Raw shot files saved by lie.")
