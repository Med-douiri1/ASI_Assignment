import os
import numpy as np
from keras import Input, Model
from keras.layers import Dense, Dropout
from keras.regularizers import l2
from keras.optimizers import Adam
from tqdm import tqdm

# --- Settings you can tweak ---
epochs = 400   # Number of training epochs
hidden_layers = [50]   # One hidden layer with 50 units
dropout_options = [0.005, 0.01, 0.05, 0.1]   #Dropout rates to test 
tau_options = [0.25, 0.5, 0.75]   # these values change depending on the dataset here this is the example specific to the energy daset
num_splits = 20
data_folder = "UCI_Datasets/energy/data/" # this is the example of the energy dataset
models_folder = "saved_models_energy" # Path to save the models after training

# create the folder for saving models if it doesn't exist
if not os.path.exists(models_folder):
    os.makedirs(models_folder)

# we load the dataset and feature/target indices
data = np.loadtxt(os.path.join(data_folder, "data.txt"))
feat_idx = np.loadtxt(os.path.join(data_folder, "index_features.txt")).astype(int)
target_idx = int(np.loadtxt(os.path.join(data_folder, "index_target.txt")))
X = data[:, feat_idx]
y = data[:, target_idx]

# Main loop: we train a model for each data split 
for split in tqdm(range(num_splits), desc="Splits"):
    # Load the train/test indices
    train_idx = np.loadtxt(os.path.join(data_folder, f"index_train_{split}.txt")).astype(int)
    test_idx = np.loadtxt(os.path.join(data_folder, f"index_test_{split}.txt")).astype(int)

    # we split training data into train/validation (80/20)
    X_train_full, y_train_full = X[train_idx], y[train_idx]
    val_cut = int(0.8 * len(X_train_full))
    X_train, y_train = X_train_full[:val_cut], y_train_full[:val_cut]
    X_val, y_val = X_train_full[val_cut:], y_train_full[val_cut:]

    # we normalize inputs and outputs
    X_mean, X_std = X_train.mean(0), X_train.std(0)
    X_std[X_std == 0] = 1
    X_train = (X_train - X_mean) / X_std
    X_val = (X_val - X_mean) / X_std
    y_mean, y_std = y_train.mean(), y_train.std()
    y_train = (y_train - y_mean) / y_std
    y_val = (y_val - y_mean) / y_std

    # we do search for best dropout and tau
    best_ll = float("-inf")    # Best log-likelihood seen so far
    best_model = None
    best_dropout, best_tau = None, None

    for dropout in tqdm(dropout_options):
        for tau in tau_options:
            # we compute L2 regularization coefficient 
            N = len(X_train)
            lengthscale = 1e-2
            reg = (lengthscale**2) * (1 - dropout) / (2.0 * N * tau)

            # we build the model with current dropout and regularization
            inp = Input(shape=(X_train.shape[1],))
            x = Dropout(dropout)(inp, training=True)
            for h in hidden_layers:
                x = Dense(h, activation='relu', kernel_regularizer=l2(reg))(x)
                x = Dropout(dropout)(x, training=True)
            out = Dense(1, kernel_regularizer=l2(reg))(x)
            model = Model(inp, out)
            model.compile(optimizer=Adam(), loss="mse")
            model.fit(X_train, y_train, epochs=epochs, batch_size=128, verbose=0)

            # we estimate log-likelihood using MC sampling 
            T = 100
            preds = []
            for _ in range(T):
                pred = model.predict(X_val, batch_size=128, verbose=0).squeeze()
                preds.append(pred)
            preds = np.array(preds)

            # we compute predictive log-likelihood (as in the paper)
            ll = np.log(np.mean(np.exp(-0.5 * tau * (preds - y_val[np.newaxis, :])**2), axis=0)).mean()
            ll += 0.5 * np.log(tau) - 0.5 * np.log(2 * np.pi)

            # we update best model if log-likelihood improved
            if ll > best_ll:
                best_ll = ll
                best_model = model
                best_dropout = dropout
                best_tau = tau

    print(f"Best dropout: {best_dropout}, Best tau: {best_tau}")

    # we retrain the best model on the full training data 
    X_train_full = (X_train_full - X_mean) / X_std
    y_train_full = (y_train_full - y_mean) / y_std
    N = len(X_train_full)
    reg = 1e-4 * (1 - best_dropout) / (2.0 * N * best_tau)

    inp = Input(shape=(X_train_full.shape[1],))
    x = Dropout(best_dropout)(inp, training=True)
    for h in hidden_layers:
        x = Dense(h, activation='relu', kernel_regularizer=l2(reg))(x)
        x = Dropout(best_dropout)(x, training=True)
    out = Dense(1, kernel_regularizer=l2(reg))(x)
    final_model = Model(inp, out)
    final_model.compile(optimizer=Adam(), loss="mse")
    final_model.fit(X_train_full, y_train_full, epochs=epochs, batch_size=128, verbose=0)

    # we save the final model and normalization statistics for testing
    final_model.save(os.path.join(models_folder, f"model_split{split}.h5"))
    np.savez(os.path.join(models_folder, f"norm_split{split}.npz"),
             X_mean=X_mean, X_std=X_std, y_mean=y_mean, y_std=y_std,
             dropout=best_dropout, tau=best_tau)

