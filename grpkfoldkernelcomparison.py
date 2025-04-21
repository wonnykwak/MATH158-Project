import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

lies = ['fairway', 'rough', 'sand', 'dprough']

os.makedirs("results", exist_ok=True)

for lie in lies:
    print(f"\n--- GPR Comparison on {lie} ---")
    
    df = pd.read_csv(f"data/strokes_summary_{lie}.csv")
    X = df[['holedis']].values
    y = df['avg_strokes'].values

    # Define kernels
    kernel_rbf = RBF()
    kernel_rbf_white = kernel_rbf + WhiteKernel()

    kernels = {
        "RBF": kernel_rbf,
        "RBF + WhiteKernel": kernel_rbf_white
    }

    # Cross-validation setup
    n_splits = min(5, max(2, len(X) // 5))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    mse_results = {}

    for name, kernel in kernels.items():
        mses = []
        for train_idx, val_idx in kf.split(X):
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            model = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            mses.append(mean_squared_error(y_val, y_pred))

        avg_mse = np.mean(mses)
        mse_results[name] = avg_mse
        print(f"{name}: avg MSE = {avg_mse:.4f}")

    # Select best kernel
    best_kernel_name = min(mse_results, key=mse_results.get)
    print(f"→ Best kernel for {lie}: {best_kernel_name}")

    best_kernel = kernels[best_kernel_name]
    final_model = GaussianProcessRegressor(kernel=best_kernel, alpha=1e-6, normalize_y=True)
    final_model.fit(X, y)

    # Predict full curve
    X_grid = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
    y_pred, std_pred = final_model.predict(X_grid, return_std=True)

    preds = pd.DataFrame({
        'holedis': X_grid.flatten(),
        'pred': y_pred,
        'std': std_pred
    })
    preds.to_csv(f"results/gpr_{lie}_preds.csv", index=False)

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(preds["holedis"], preds["pred"], label="GPR prediction", lw=2)
    plt.fill_between(preds["holedis"],
                     preds["pred"] - preds["std"],
                     preds["pred"] + preds["std"],
                     color="lightblue", alpha=0.4, label="±1 std. dev")
    plt.scatter(df["holedis"], df["avg_strokes"], color="black", s=40, label="Binned avg data")
    plt.title(f"GPR - {lie.capitalize()} (Best: {best_kernel_name})")
    plt.xlabel("Distance to hole (yards)")
    plt.ylabel("Predicted strokes to hole out")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(f"results/gpr_{lie}_plot.png")
    plt.close()
