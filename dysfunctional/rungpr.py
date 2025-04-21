import pandas as pd
import numpy as np
import os
from sklearn.gaussian_process import GaussianProcessRegressor 
from sklearn.gaussian_process.kernels import RBF 
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

lies = ['fairway'] # will handle green differently
gammas = [0.4, 0.7, .8, .9, 1.0, 1.1, 1.2,1.3, 1.5]
n_splits = 3 # default, will override if needed based on data

os.makedirs("results", exist_ok=True)

for lie in lies:
    print(f"GPR on {lie}...")

    df = pd.read_csv(f"data/strokes_summary_{lie}.csv")

    # input features (no expansion)
    X = df[['holedis']].values
    y = df['avg_strokes'].values

    best_gamma = None
    best_mse = np.inf
    mse_per_gamma = []
    # fit model -> cross val -> measure MSE -> keep best MSE

    for gamma in gammas:
        print(f"Testing gamma {gamma}...")
        #define kernel -> RBF as suggested by chandler
        kernel = RBF(length_scale=gamma,  length_scale_bounds="fixed") 
        # 2/3 for training, 1/3 for validation 
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=47)
        mses = []

        for train_idx, val_idx in kf.split(X):
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            #train GPR with current gamma
            gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True)
            # use sample weight to give more importance to points based on count
            gpr.fit(X_train, y_train)
            # predict validation set
            y_pred = gpr.predict(X_val)

            # calculate MSE
            mse = mean_squared_error(y_val, y_pred)
            mses.append(mse)

        avg_mse = np.mean(mses)
        mse_per_gamma.append((gamma, avg_mse))

        if avg_mse < best_mse:
            best_mse = avg_mse
            best_gamma = gamma

    print(f"Best gamma for {lie}: {best_gamma:.4f}")
    # after best gamma found -> fit the GPR model on all data with the best setting
    best_kernel = RBF(length_scale=np.sqrt(best_gamma))
    gpr = GaussianProcessRegressor(kernel=best_kernel, alpha=1e-6, normalize_y=True)
    gpr.fit(X, y)

    X_grid = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
    y_pred, std_pred = gpr.predict(X_grid, return_std=True)

    preds = pd.DataFrame({
        'holedis': X_grid.flatten(),
        'pred': y_pred,
        'std': std_pred
    })
    preds.to_csv(f"results/gpr_{lie}_preds.csv", index=False)

    # Save MSE vs gamma curve
    mse_df = pd.DataFrame(mse_per_gamma, columns=['gamma', 'mse'])
    mse_df.to_csv(f"results/gpr_{lie}_mse.csv", index=False)
