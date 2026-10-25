"""
NumPy Multiple Linear Regression GD

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - shuffle_xy
def shuffle_xy(X, y, seed=42):
    """Randomly permute feature rows and targets together.

    Parameters
    ----------
    X : np.ndarray, shape (n, d)
        Feature matrix.
    y : np.ndarray, shape (n,)
        Target vector.
    seed : int, optional
        RNG seed for reproducibility (default 42).

    Returns
    -------
    X_shuffled : np.ndarray, shape (n, d)
    y_shuffled : np.ndarray, shape (n,)
    """
    # TODO: Return (X, y) under one shared seeded row permutation
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(X))
    return X[permutation], y[permutation]

# Step 2 - split_train_val_test
def split_train_val_test(X, y, train_frac=0.6, val_frac=0.2):
    # TODO: Slice already-shuffled data into contiguous train/val/test partitions...
    n = len(X)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac) 

    X_tr = X[:n_train]
    y_tr = y[:n_train]

    X_va = X[n_train : n_train + n_val]
    y_va = y[n_train : n_train + n_val]

    X_te = X[n_train + n_val:] 
    y_te = y[n_train + n_val:] 

    return X_tr, y_tr, X_va, y_va, X_te, y_te

# Step 3 - compute_feature_stats
import numpy as np

def compute_feature_stats(X):
    # TODO: Compute per-feature mean and std; replace std of 0 with 1
    col_means = np.mean(X, axis = 0)
    col_std = np.std(X, axis = 0) 
    # build corrected std array, keep positive values, swaps 0 for 1s

    col_std = np.where(col_std == 0, 1.0, col_std)

    # returns mean and std
    return (col_means, col_std)

# Step 4 - standardize_features
def standardize_features(X, mean, std):
    # TODO: Apply z-score normalization using precomputed training mean and std.
    Z_score_norm = (X - mean) / std 
    return Z_score_norm

# Step 5 - add_bias_column
import numpy as np

def add_bias_column(X):
    # TODO: Prepend a column of ones to feature matrix X
    n = X.shape[0]
    bias_col = np.ones((n, 1))
    X = np.hstack([bias_col, X])
    return X

# Step 6 - prepare_design_matrix
def prepare_design_matrix(X, mean, std):
    # TODO: Standardize features then add the bias column to form the design matrix.
    X_std = standardize_features(X, mean, std)
    X_combined = add_bias_column(X_std) 

    return X_combined

# Step 7 - predict_linear
import numpy as np
def predict_linear(X, weights):
    """Compute linear predictions y_hat = X @ weights.

    Args:
        X: Design matrix of shape (n, d_in), often including a bias column.
        weights: Weight vector of shape (d_in,).

    Returns:
        Predicted targets of shape (n,).
    """
    # TODO: Return the predicted target vector from X and weights
    return np.dot(X, weights)

# Step 8 - mse_loss
def mse_loss(y_true, y_pred):
    # TODO: Return the average of squared residuals as a scalar float.
    arr = (y_true - y_pred) ** 2
    return np.mean(arr)

# Step 9 - mse_gradient
def mse_gradient(X, y_true, y_pred):
    # TODO: Return the analytic MSE gradient w.r.t. weights: (2/n) X^T (y_pred - y_true)
    n = X.shape[0]
    residual = y_pred - y_true 
    return np.dot(X.T, residual) * 2.0 / n

# Step 10 - normal_equation
def normal_equation(X, y):
    # TODO: Solve for the closed-form least-squares weights via the normal equation.
    A = X.T @ X
    b = X.T @ y 
    return np.linalg.solve(A, b)

# Step 11 - initialize_weights
def initialize_weights(n_features, seed=None):
    # TODO: Return (n_features,) weights sampled from N(0, 0.01)
    if seed != None:
        rng = np.random.seed(seed)
    return np.random.normal(loc = 0, scale = 0.01, size = n_features)

# Step 12 - gd_step
def gd_step(X, y, weights, lr):
    """Run one full-batch gradient descent update on the weights.

    Args:
        X: Design matrix of shape (n, d_in).
        y: Target vector of shape (n,).
        weights: Current weight vector of shape (d_in,).
        lr: Learning rate (float).

    Returns:
        Updated weight vector of shape (d_in,).
    """
    # TODO: return the updated weight vector after one MSE gradient step
    y_pred = predict_linear(X, weights)
    grad = mse_gradient(X, y, y_pred)

    return weights - lr * grad

# Step 13 - epoch_train_val_losses
def epoch_train_val_losses(X_train, y_train, X_val, y_val, weights):
    """Evaluate MSE on train and validation sets for the current weights.

    Args:
        X_train: Training design matrix of shape (n_tr, d_in).
        y_train: Training targets of shape (n_tr,).
        X_val: Validation design matrix of shape (n_va, d_in).
        y_val: Validation targets of shape (n_va,).
        weights: Weight vector of shape (d_in,).

    Returns:
        (train_loss, val_loss) as plain floats.
    """
    # TODO: return the pair (train_loss, val_loss) as MSE floats
    train_pred = predict_linear(X_train, weights)
    val_pred = predict_linear(X_val, weights) # * 
    # these are the y_pred

    # y_true vs y_pred in description
    train_loss = mse_loss(y_train, train_pred) 
    val_loss = mse_loss(y_val, val_pred)

    return (train_loss, val_loss)

# Step 14 - update_early_stop_state
def update_early_stop_state(val_loss, best_val_loss, wait, weights, best_weights, patience):
    # TODO: Update best weights and patience counter; signal stop when val loss stalls...
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        wait = 0
        best_weights = weights.copy()
        stop = False
    else:
        wait = wait + 1
        stop = wait >= patience
    
    return (best_val_loss, wait, best_weights, stop)

# Step 15 - init_training_state
def init_training_state(n_features, seed=None):
    # TODO: Build the initial training-state dictionary for the GD epoch loop.
    weights = initialize_weights(n_features, seed=seed)
    # starting weight vec

    # build the dict
    state = {
        "weights": weights, # put vector under weights
        "best_weights": weights.copy(), # ind copy
        "best_val_loss": np.inf, # init best val loss
        "wait": 0, 
        "train_losses": [],
        "val_losses": [],
        "stopped": False 
    }
    
    return state

# Step 16 - run_one_epoch (not yet solved)
# TODO: implement

# Step 17 - train_batch_gd (not yet solved)
# TODO: implement

# Step 18 - mean_absolute_error (not yet solved)
# TODO: implement

# Step 19 - root_mean_squared_error (not yet solved)
# TODO: implement

# Step 20 - r_squared (not yet solved)
# TODO: implement

# Step 21 - evaluate_regression (not yet solved)
# TODO: implement

# Step 22 - learning_curve_data (not yet solved)
# TODO: implement

# Step 23 - weights_l2_distance (not yet solved)
# TODO: implement

# Step 24 - create_lr_model (not yet solved)
# TODO: implement

# Step 25 - fit_lr_model (not yet solved)
# TODO: implement

# Step 26 - predict_lr_model (not yet solved)
# TODO: implement

# Step 27 - score_lr_model (not yet solved)
# TODO: implement

# Step 28 - compare_with_normal_equation (not yet solved)
# TODO: implement

