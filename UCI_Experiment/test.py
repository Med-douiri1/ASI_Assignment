import os
import numpy as np
from keras.models import load_model

# Paths and settings (here for the example of the energy dataset)
data_path = "UCI_Datasets/energy/data/"
models_path = "saved_models_energy"
splits = 20
T = 100  # MC dropout samples

# we load the full dataset once 
print("Loading the dataset...")
data = np.loadtxt(os.path.join(data_path, "data.txt"))
features = np.loadtxt(os.path.join(data_path, "index_features.txt")).astype(int)
target = int(np.loadtxt(os.path.join(data_path, "index_target.txt")))
X = data[:, features]
y = data[:, target]

# we'll store results here 
rmse_list = []
mc_rmse_list = []
ll_list = []

# we loop over saved models for each split
for i in range(splits):
    print(f"\n Evaluating on split {i}")
    # we load the trained model and normalization parameters for this split
    model = load_model(os.path.join(models_path, f"model_split{i}.h5"))
    norm = np.load(os.path.join(models_path, f"norm_split{i}.npz"))
    X_mean, X_std = norm["X_mean"], norm["X_std"]
    y_mean, y_std = norm["y_mean"], norm["y_std"]
    tau = float(norm["tau"])

    # we load and normalize the test set
    test_idx = np.loadtxt(os.path.join(data_path, f"index_test_{i}.txt")).astype(int)
    X_test = (X[test_idx] - X_mean) / X_std
    y_test = y[test_idx]

    # we do one forward pass 
    pred_std = model.predict(X_test, batch_size=128).squeeze()
    pred_std = pred_std * y_std + y_mean
    rmse = np.sqrt(np.mean((pred_std - y_test) ** 2))

    # MC Dropout: forward T times with dropout active
    all_preds = np.array([
        model.predict(X_test, batch_size=128).squeeze()
        for _ in range(T)
    ])
    all_preds = all_preds * y_std + y_mean
    mean_pred = all_preds.mean(axis=0)
    mc_rmse = np.sqrt(np.mean((mean_pred - y_test) ** 2))

    # we compute the predictive log-likelihood using MC samples
    ll = np.log(np.mean(np.exp(-0.5 * tau * (all_preds - y_test[np.newaxis, :])**2), axis=0)).mean()
    ll += 0.5 * np.log(tau) - 0.5 * np.log(2 * np.pi)

    # Save results
    rmse_list.append(rmse)
    mc_rmse_list.append(mc_rmse)
    ll_list.append(ll)

    # we print the results for this split
    print(f"  RMSE (one pass):        {rmse:.4f}")
    print(f"  RMSE (MC dropout):      {mc_rmse:.4f}")
    print(f"  Log-Likelihood:         {ll:.4f}")

# we print the final report: average metrics across all splits with standard error 
print(f"Avg RMSE (standard):      {np.mean(rmse_list):.4f} ± {np.std(rmse_list)/np.sqrt(splits):.4f}")
print(f"Avg RMSE (MC dropout):    {np.mean(mc_rmse_list):.4f} ± {np.std(mc_rmse_list)/np.sqrt(splits):.4f}")
print(f"Avg Log-Likelihood:       {np.mean(ll_list):.4f} ± {np.std(ll_list)/np.sqrt(splits):.4f}")

