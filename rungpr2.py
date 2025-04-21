import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor 
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

lies = ['fairway', 'sand', 'dprough', 'rough']  # will handle green differently

os.makedirs("results", exist_ok=True)

for lie in lies:
    print(f"GPR on {lie}...")

    df = pd.read_csv(f"data/strokes_summary_{lie}.csv")

    # input features (no expansion)
    X = df[['holedis']].values
    y = df['avg_strokes'].values

    # Define kernel with default tuning
    kernel = RBF() + WhiteKernel()
    gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True)
    gpr.fit(X, y)

    print(f"Optimised kernel for {lie}: {gpr.kernel_}")

    # Generate prediction curve
    X_grid = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
    y_pred, std_pred = gpr.predict(X_grid, return_std=True)

    preds = pd.DataFrame({
        'holedis': X_grid.flatten(),
        'pred': y_pred,
        'std': std_pred
    })
    preds.to_csv(f"results/gpr_{lie}_preds.csv", index=False)

    # --------- Plot ---------
    plt.figure(figsize=(8, 5))

    # Plot prediction + confidence band
    plt.plot(preds["holedis"], preds["pred"], label="GPR prediction, RBF+WHITE", lw=2)
    plt.fill_between(preds["holedis"],
                     preds["pred"] - preds["std"],
                     preds["pred"] + preds["std"],
                     color="lightblue", alpha=0.4, label="±1 std. dev")

    # Plot original points
    plt.scatter(df["holedis"], df["avg_strokes"], color="black", s=40, label="Binned avg data")

    plt.title(f"GPR Prediction - {lie.capitalize()}")
    plt.xlabel("Distance to hole (yards)")
    plt.ylabel("Predicted strokes to hole out")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Save and/or show
    plot_path = f"results/gpr_{lie}_plot.png"
    plt.savefig(plot_path)
    plt.close()

    print(f"Saved predictions and plot for {lie}.\n")
