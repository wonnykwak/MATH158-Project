import pandas as pd

'''
Extract and save green shots within 90 feet, converting holedis to yards
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

# Clean data: keep only rows with stroke and holedis
ppdata = ppdata[ppdata["stroke"].notna() & ppdata["holedis"].notna()]

# Compute strokes remaining to hole out
ppdata["total_strokes"] = ppdata.groupby(["roundid", "holeid"])["stroke"].transform("max")
ppdata["strokes_remaining"] = ppdata["total_strokes"] - ppdata["stroke"] + 1

# Filter for green shots within 90 feet
green_df = ppdata[(ppdata["lie"] == "Green") & (ppdata["holedis"] <= 70)]

# Convert holedis from feet to yards
green_df["holedis"] = green_df["holedis"] / 3
green_df["holedis_unit"] = "yards"  # optional column

# Save to CSV
green_df.to_csv("data/putts_under70ft_green.csv", index=False)

print("Green shots under 70 feet processed and saved.")
