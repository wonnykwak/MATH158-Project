import pandas as pd
import matplotlib.pyplot as plt

# Lie types and display names
lies = {
    "fairway": "Fairway",
    "rough": "Rough",
    # "dprough": "Deep Rough",
    "sand": "Sand"
}

# Optional: custom colours for each lie
colours = {
    "fairway": "limegreen",
    "rough": "darkgreen",
    # "dprough": "saddlebrown",
    "sand": "goldenrod"
}

plt.figure(figsize=(10, 6))

for lie, label in lies.items():
    # Load GPR predictions
    pred_df = pd.read_csv(f"results/gpr_{lie}_preds.csv")

    # Load binned average data
    avg_df = pd.read_csv(f"data/strokes_summary_{lie}.csv")

    # GPR line + band
    plt.plot(pred_df["holedis"], pred_df["pred"], label=label, color=colours[lie])
    plt.fill_between(pred_df["holedis"],
                     pred_df["pred"] - pred_df["std"],
                     pred_df["pred"] + pred_df["std"],
                     color=colours[lie],
                     alpha=0.2)

    # Binned points
    plt.scatter(avg_df["holedis"], avg_df["avg_strokes"], color=colours[lie], s=30, alpha=0.8, edgecolor='k', linewidth=0.5)


plt.xlabel("Distance to hole (yards)")
plt.ylabel("Predicted strokes to hole out")
plt.title("GPR Prediction Curves and Binned Averages by Lie Type")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/gpr_all_lies_overlay.png")
plt.show()
