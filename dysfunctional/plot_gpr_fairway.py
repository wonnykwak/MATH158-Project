
import pandas as pd
import matplotlib.pyplot as plt



# Load GPR predictions for fairway
df = pd.read_csv("results/gpr_fairway_preds.csv")

# Plot
plt.figure(figsize=(8, 5))
plt.plot(df["holedis"], df["pred"], label="GPR prediction", lw=2)
plt.fill_between(
    df["holedis"],
    df["pred"] - df["std"],
    df["pred"] + df["std"],
    color="lightblue",
    alpha=0.4,
    label="±1 std. dev"
)

plt.xlabel("Distance to hole (yards)")
plt.ylabel("Predicted strokes to hole out")
plt.title("GPR Prediction: Fairway")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
