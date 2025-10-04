import os
import sys
import glob
import json
from itertools import product

import numpy as np
import joblib
from tensorflow.keras.models import load_model

# ensure package imports work if repo structure as before
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from src.training.generate_training_data import simulate_gbm_paths

# --- CONFIG (adjust if your training used different values) ---
# These must match how you generated training data.
TRAINING_N_STEPS = 50  # how many steps used when generating full paths for training
TRAINING_PREFIX_LEN = 10  # how many prefix steps the NN expects
# ----------------------------------------------------------------


def build_features_from_prefix(prefix, K, T, r, sigma, option_type):
    """
    Given a prefix array shape (n_paths, prefix_len) and contract params,
    create the same feature vector as in training:
      [prefix (prefix_len) | min | max | mean | std | realized_vol | params(K,T,r,sigma,opt_flag)]
    Note: does not scale features (scaling is applied later).
    """
    prefix_min = np.min(prefix, axis=1, keepdims=True)
    prefix_max = np.max(prefix, axis=1, keepdims=True)
    prefix_mean = np.mean(prefix, axis=1, keepdims=True)
    prefix_std = np.std(prefix, axis=1, keepdims=True)
    realized_vol = np.std(
        np.diff(np.log(prefix + 1e-12), axis=1), axis=1, keepdims=True
    )

    opt_flag = {"call": 1.0, "put": 0.0, "digital": 2.0, "asian": 3.0}.get(
        option_type, 1.0
    )
    n_paths = prefix.shape[0]
    params = np.array([K, T, r, sigma, opt_flag])[None, :].repeat(n_paths, axis=0)

    X_unscaled = np.hstack(
        [prefix, prefix_min, prefix_max, prefix_mean, prefix_std, realized_vol, params]
    )
    return X_unscaled


def hybrid_pricing_nn_mc(
    model,
    scaler,
    s0,
    K,
    T,
    r,
    sigma,
    option_type="call",
    training_n_steps=TRAINING_N_STEPS,
    prefix_len=TRAINING_PREFIX_LEN,
    n_paths=10000,
):
    """
    Hybrid pricing using NN:
      - Simulate full (training_n_steps) GBM paths, then take the first prefix_len steps as prefix
        so discretization matches training data (CRITICAL).
      - Construct features identical to training.
      - Scale first (num_features - 1) columns with provided scaler (option flag left unscaled).
      - Predict using `model` (expects model.predict to return discounted payoff).
      - Return mean(predictions) (NO additional discounting).
    """
    if scaler is None:
        raise ValueError(
            "scaler must be provided and should be the exact scaler used during training."
        )

    # Simulate GBM with same n_steps used in training so prefix timing matches training.
    paths = simulate_gbm_paths(s0, r, sigma, T, training_n_steps, n_paths)
    # Use first prefix_len columns (same as training)
    prefix = paths[:, :prefix_len]

    # Build unscaled features exactly as training did
    X_unscaled = build_features_from_prefix(prefix, K, T, r, sigma, option_type)

    # Scale first (all except final option_flag column)
    X_scaled = scaler.transform(X_unscaled[:, :-1])
    X = np.hstack([X_scaled, X_unscaled[:, -1].reshape(-1, 1)])

    # Predict (model outputs discounted payoff if training labels were discounted)
    y_pred = model.predict(X, verbose=0).flatten()
    return float(np.mean(y_pred))


def standard_mc_price(
    s0, K, T, r, sigma, option_type, n_steps=50, n_paths=10000, discount=True
):
    """
    Standard Monte Carlo price using full path.
    """
    paths = simulate_gbm_paths(s0, r, sigma, T, n_steps, n_paths)
    if option_type == "call":
        payoff = np.maximum(paths[:, -1] - K, 0.0)
    elif option_type == "put":
        payoff = np.maximum(K - paths[:, -1], 0.0)
    elif option_type == "digital":
        payoff = (paths[:, -1] > K).astype(float)
    elif option_type == "asian":
        payoff = np.maximum(np.mean(paths, axis=1) - K, 0.0)
    else:
        payoff = np.maximum(paths[:, -1] - K, 0.0)

    avg_payoff = np.mean(payoff)
    return np.exp(-r * T) * avg_payoff if discount else avg_payoff


# -------------------------
# Diagnostic helpers
# -------------------------
def inspect_scaler(scaler):
    """
    Print scaler stats to verify it matches training scaler.
    """
    print("SCALER INFO")
    if hasattr(scaler, "mean_"):
        print(f"  mean_ (first 5): {scaler.mean_[:5]}")
    if hasattr(scaler, "scale_"):
        print(f"  scale_ (first 5): {scaler.scale_[:5]}")
    if hasattr(scaler, "var_"):
        print(f"  var_ (first 5): {scaler.var_[:5]}")
    print(f"  n_features_in_: {getattr(scaler, 'n_features_in_', 'Unknown')}")


def sanity_check_on_saved_training_data(saved_npz_path, model, scaler, n_check=1000):
    """
    If you saved training data X/y, use this to check model vs training labels on a subset.
    saved_npz_path: path to npz file created by generate_training_data (contains X and y).
    """
    if not os.path.exists(saved_npz_path):
        raise FileNotFoundError(f"Training data file not found: {saved_npz_path}")

    d = np.load(saved_npz_path)
    X = d["X"]
    y = d["y"]
    print(f"Loaded training data X.shape={X.shape}, y.shape={y.shape}")

    # Use a small subset
    X_sub = X[:n_check]
    y_sub = y[:n_check]

    y_pred = model.predict(X_sub, verbose=0).flatten()
    mse = np.mean((y_pred - y_sub) ** 2)
    mae = np.mean(np.abs(y_pred - y_sub))
    print(f"Prediction on training subset: MSE={mse:.6e}, MAE={mae:.6e}")
    # show some sample comparisons
    for i in range(min(5, len(y_sub))):
        print(
            f"  sample {i:2d}: y_true={y_sub[i]:.6f}, y_pred={y_pred[i]:.6f}, err={y_pred[i]-y_sub[i]:.6f}"
        )


# -------------------------
# Evaluation routine
# -------------------------
def evaluate_all_models(
    model_dir="src/models",
    scaler_path="data/processed/scalers/feature_scaler.pkl",
    param_grid=None,
    training_n_steps=TRAINING_N_STEPS,
    prefix_len=TRAINING_PREFIX_LEN,
    n_steps_mc=50,
    n_paths=5000,  # default smaller for speed — increase to reduce MC noise
):
    """
    Evaluate models found in model_dir against MC prices.
    Important: scaler_path must point to the same scaler used during training (pickle/joblib).
    param_grid is a dict: { 'r': [...], 'sigma': [...], 'T': [...], 'K': [...], 'option_type': [...] }
    """
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(
            f"Scaler not found at {scaler_path}. Save and provide the scaler used at training time."
        )

    scaler = joblib.load(scaler_path)
    inspect_scaler(scaler)

    # find models
    model_files = glob.glob(os.path.join(model_dir, "*.h5"))
    if not model_files:
        print("No .h5 model files found in", model_dir)
        return {}

    print(f"Found models: {[os.path.basename(m) for m in model_files]}")

    # default grid if not provided
    if param_grid is None:
        param_grid = {
            "r": [0.03],
            "sigma": [0.2],
            "T": [1.0],
            "K": [100],
            "option_type": ["call"],
        }

    combos = list(
        product(
            param_grid["r"],
            param_grid["sigma"],
            param_grid["T"],
            param_grid["K"],
            param_grid["option_type"],
        )
    )
    print(f"Total param combos to evaluate per model: {len(combos)}")

    all_results = {}
    for model_path in model_files:
        model_name = os.path.basename(model_path).replace(".h5", "")
        print(f"\n{'='*60}\nEvaluating model: {model_name}\n{'='*60}")
        model = load_model(model_path, compile=False)

        # basic shape check
        expected_in = model.input_shape[1]
        if expected_in != (
            prefix_len + 4 + 1 + 5
        ):  # prefix_len + [min,max,mean,std] + realized_vol + params(5)
            print(
                f"WARNING: model input width {expected_in} doesn't match expected feature length {(prefix_len + 4 + 1 + 5)}. Continue but results may be invalid."
            )

        model_results = []
        for r, sigma, T, K, option_type in combos:
            try:
                nn_mc_price = hybrid_pricing_nn_mc(
                    model=model,
                    scaler=scaler,
                    s0=100,
                    K=K,
                    T=T,
                    r=r,
                    sigma=sigma,
                    option_type=option_type,
                    training_n_steps=training_n_steps,
                    prefix_len=prefix_len,
                    n_paths=n_paths,
                )
                mc_price = standard_mc_price(
                    s0=100,
                    K=K,
                    T=T,
                    r=r,
                    sigma=sigma,
                    option_type=option_type,
                    n_steps=n_steps_mc,
                    n_paths=n_paths,
                    discount=True,
                )

                abs_error = abs(nn_mc_price - mc_price)
                pct_error = 100 * abs_error / (mc_price if mc_price != 0 else 1.0)

                model_results.append(
                    {
                        "r": r,
                        "sigma": sigma,
                        "T": T,
                        "K": K,
                        "option_type": option_type,
                        "nn_mc": nn_mc_price,
                        "mc": mc_price,
                        "abs_error": abs_error,
                        "pct_error": pct_error,
                    }
                )
            except Exception as e:
                print(
                    f"Error for combo r={r},sigma={sigma},T={T},K={K},opt={option_type}: {e}"
                )
                continue

        all_results[model_name] = model_results

        # quick summary
        if model_results:
            abs_errors = np.array([x["abs_error"] for x in model_results])
            pct_errors = np.array([x["pct_error"] for x in model_results])
            print(
                f"Model {model_name} summary: mean_AE={abs_errors.mean():.6f}, median_AE={np.median(abs_errors):.6f}, mean_PE={pct_errors.mean():.2f}%"
            )

    # optionally dump results
    with open("model_evaluation_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("Saved results to model_evaluation_results.json")
    return all_results


# Quick CLI
if __name__ == "__main__":
    results = evaluate_all_models()
    print("Done.")
