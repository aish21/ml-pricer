import os
import json
import math
import time
from typing import Tuple
import numpy as np

OUT_DIR = "C:\\Users\\aisha\\OneDrive\\Desktop\\GitHub\\neural-pricer\\data\\raw"
OUT_BASENAME = "training_data_phoenix"
OUT_FILE = os.path.join(OUT_DIR, OUT_BASENAME + ".npz")
METADATA_FILE = os.path.join(OUT_DIR, OUT_BASENAME + "_metadata.json")

SEED = 12345

# Number of independent parameter samples (not MC paths)
# e.g., if you set param_samples=5000 and n_paths_per_param=200 you'll produce 1e6 simulated paths
PARAM_SAMPLES = 10000

# MC paths per parameter vector used to compute the single label (option price)
# Keep this reasonably high to make the labels low-noise (you can set lower to produce more diverse labels cheaply)
N_PATHS_PER_PARAM = 500

# Paths discretization for payoff (must be used also in inference)
N_STEPS = 252  # trading days per year, for realism

# Observation schedule (quarterly / obs_count)
OBS_COUNT = 6  # e.g., for T=1.0, 6 obs (every 2 months) - choose as you wish
OBS_INDICES = None  # will be computed per-sample as indices into path length

# Whether to log-transform the target for training (training script must match)
USE_LOG_TARGET = True

# Phoenix parameter ranges (we sample uniformly across these ranges)
S0 = 100.0
R_RANGE = (0.0, 0.05)  # risk-free rate
SIGMA_RANGE = (0.10, 0.35)  # volatility
T_RANGE = (0.5, 2.0)  # maturities in years
AUTOCALL_FRAC_RANGE = (1.00, 1.15)  # autocall barrier as fraction of S0 (e.g., 1.05)
COUPON_BARRIER_FRAC_RANGE = (0.60, 1.00)  # coupon qualification barrier
COUPON_RATE_RANGE = (
    0.005,
    0.05,
)  # coupon per observation as decimal (e.g., 0.02 -> 2%)
KNOCKIN_FRAC_RANGE = (0.5, 0.9)  # knock-in barrier fraction of S0

# Save sub-sampling of param combinations cap to avoid runaway disk usage
MAX_TOTAL_SAMPLES = (
    200_000  # final rows (each row corresponds to one param vector + label)
)


def simulate_gbm_paths(
    s0: float,
    r: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int,
    rng: np.random.RandomState,
):
    """
    Vectorized GBM path simulation using log-Euler (exact increments).
    Returns paths array shape (n_paths, n_steps+1)
    """
    dt = T / n_steps
    Z = rng.randn(n_paths, n_steps)
    increments = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    log_paths = np.cumsum(increments, axis=1)
    log_paths = np.hstack([np.zeros((n_paths, 1)), log_paths]) + math.log(s0)
    return np.exp(log_paths)  # (n_paths, n_steps+1)


def phoenix_payoff_from_paths(
    paths: np.ndarray,
    autocall_frac: float,
    coupon_barrier_frac: float,
    coupon_rate: float,
    knockin_frac: float,
    r: float,
    T: float,
    obs_indices: np.ndarray,
    exact_discount_on_call=True,
) -> Tuple[np.ndarray, dict]:
    """
    Vectorized phoenix payoff across many simulated paths.
    - paths: (n_paths, n_steps+1)
    - obs_indices: array of observation indices into path columns (integers)
    Returns:
     - discounted_payoffs: shape (n_paths,)
     - diagnostics dict (e.g., per-path call times, knocked_in flags)
    Payoff assumptions (common phoenix/autocall):
     - On observation if underlying >= autocall barrier: autocalled, pay outstanding coupons (if any) + principal (100)
       we assume principal = 1.0 (you may rescale externally).
     - Coupons accrue at coupon_rate each observation where underlying >= coupon_barrier_frac*S0.
       If coupon paid at call, they are paid only on call; otherwise we assume coupons are paid on observation (set later).
     - If knocked-in (ever below knockin barrier at any time): payoff at maturity depends on final underlying (we use worst-case:
       if knocked-in, assume redemption is min(1, S_T / S0) i.e. linear).
    """
    n_paths, n_time = paths.shape
    S0 = paths[0, 0]  # assuming same across simulation
    autocall_level = autocall_frac * S0
    coupon_level = coupon_barrier_frac * S0
    knockin_level = knockin_frac * S0

    # Precompute booleans
    knocked_in = np.any(paths < knockin_level, axis=1)  # (n_paths,)
    # For each observation index, did path meet autocall?
    obs_prices = paths[:, obs_indices]  # (n_paths, obs_count)
    meets_autocall = obs_prices >= autocall_level  # bool
    meets_coupon = obs_prices >= coupon_level  # bool

    # For each path, find first observation index that triggers autocall (if any)
    first_call_pos = np.argmax(
        meets_autocall, axis=1
    )  # if none true, argmax returns 0 -> we need mask
    has_call = np.any(meets_autocall, axis=1)

    # Compute coupon accrual up to call or until maturity:
    # coupon accrual per observation = coupon_rate (as decimal fraction of notional)
    # We will treat coupons as paid only at call (typical Phoenix paid at call) — this is a choice; you can change.
    cumulative_coupon_counts = np.cumsum(
        meets_coupon, axis=1
    )  # number of coupon-qualifying obs up to each obs
    # coupon counts up to call (for called paths), or to final obs for non-called
    coupon_count_at_call = np.where(
        has_call,
        cumulative_coupon_counts[np.arange(n_paths), first_call_pos],
        cumulative_coupon_counts[:, -1],
    )

    # Principal redemption:
    # If autocall at obs k -> redeem principal = 1.0 (notional)
    # If not called:
    #   if knocked_in: payoff = min(1.0, S_T / S0)  (i.e. possibly principal loss)
    #   else: payoff = 1.0 (principal returned)
    S_T = paths[:, -1]
    principal = np.where(
        has_call, 1.0, np.where(knocked_in, np.minimum(1.0, S_T / S0), 1.0)
    )

    # Time of payment: if called at obs index k, compute t_call as obs_index / n_steps * T
    # otherwise t_call = T
    obs_time_fracs = obs_indices / float(n_time - 1)  # in [0,1]
    t_call = np.where(has_call, obs_time_fracs[first_call_pos], 1.0) * T
    # total payoff per path BEFORE discount: principal + coupon_rate * coupon_count_at_call
    payoff_nominal = principal + coupon_rate * coupon_count_at_call

    # Discount each payoff to time 0 using r and actual payment time
    if exact_discount_on_call:
        discounted = np.exp(-r * t_call) * payoff_nominal
    else:
        # fallback: discount everything at maturity
        discounted = np.exp(-r * T) * payoff_nominal

    diagnostics = {
        "has_call_mean": float(np.mean(has_call)),
        "knocked_in_mean": float(np.mean(knocked_in)),
        "avg_coupon_count": float(np.mean(coupon_count_at_call)),
    }
    return discounted, diagnostics


PHOENIX_FEATURE_ORDER = [
    "S0",
    "r",
    "sigma",
    "T",
    "autocall_barrier_frac",
    "coupon_barrier_frac",
    "coupon_rate",
    "knock_in_frac",
]


def build_feature_vector(
    s0, r, sigma, T, autocall_frac, coupon_frac, coupon_rate, knockin_frac
):
    """
    Returns numpy array of shape (len(PHOENIX_FEATURE_ORDER),)
    """
    return np.array(
        [s0, r, sigma, T, autocall_frac, coupon_frac, coupon_rate, knockin_frac],
        dtype=float,
    )


def generate_dataset(
    seed=SEED,
    param_samples=PARAM_SAMPLES,
    n_paths_per_param=N_PATHS_PER_PARAM,
    n_steps=N_STEPS,
    obs_count=OBS_COUNT,
    max_total_samples=MAX_TOTAL_SAMPLES,
    out_file=OUT_FILE,
    metadata_file=METADATA_FILE,
):
    rng = np.random.RandomState(seed)
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    X_list = []
    y_list = []
    diagnostics_agg = []

    sample_count = 0
    start = time.time()

    # sample parameter vectors uniformly in ranges
    for i in range(param_samples):
        if sample_count >= max_total_samples:
            break

        # sample params
        r = float(rng.uniform(*R_RANGE))
        sigma = float(rng.uniform(*SIGMA_RANGE))
        T = float(rng.uniform(*T_RANGE))
        autocall_frac = float(rng.uniform(*AUTOCALL_FRAC_RANGE))
        coupon_barrier_frac = float(rng.uniform(*COUPON_BARRIER_FRAC_RANGE))
        coupon_rate = float(rng.uniform(*COUPON_RATE_RANGE))
        knockin_frac = float(rng.uniform(*KNOCKIN_FRAC_RANGE))
        s0 = float(np.random.uniform(80, 120))

        # observation indices (integer indices into time grid)
        # We define observation indices based on n_steps so they align with simulated paths
        obs_indices = np.linspace(0, n_steps, obs_count + 1, dtype=int)[
            1:
        ]  # excludes t=0
        # simulate paths for this param vector
        paths = simulate_gbm_paths(s0, r, sigma, T, n_steps, n_paths_per_param, rng)

        # compute discounted payoffs per path
        payoffs, diag = phoenix_payoff_from_paths(
            paths,
            autocall_frac,
            coupon_barrier_frac,
            coupon_rate,
            knockin_frac,
            r,
            T,
            obs_indices,
            exact_discount_on_call=True,
        )

        # label = Monte Carlo estimate (mean discounted payoff)
        label = float(np.mean(payoffs))

        # append
        feat = build_feature_vector(
            s0,
            r,
            sigma,
            T,
            autocall_frac,
            coupon_barrier_frac,
            coupon_rate,
            knockin_frac,
        )
        X_list.append(feat)
        y_list.append(label)
        diagnostics_agg.append(diag)
        sample_count += 1

        if (i + 1) % 100 == 0:
            elapsed = time.time() - start
            print(
                f"Generated {i+1} param-samples, collected {sample_count} labels, elapsed {elapsed:.1f}s"
            )

    X = np.vstack(X_list).astype(float)
    y = np.array(y_list).astype(float)

    # optional: log-transform target to stabilize distribution (training must use same flag)
    if USE_LOG_TARGET:
        y_saved = np.log1p(y)
    else:
        y_saved = y

    # Save arrays and metadata
    np.savez(out_file, X=X, y=y_saved)
    metadata = {
        "generated_at": time.time(),
        "param_samples_requested": param_samples,
        "collected_samples": X.shape[0],
        "n_paths_per_param": n_paths_per_param,
        "n_steps": n_steps,
        "obs_count": obs_count,
        "feature_order": PHOENIX_FEATURE_ORDER,
        "use_log_target": bool(USE_LOG_TARGET),
        "notes": "Phoenix generator: payoff uses exact discounting at call time; coupons paid at call; principal loss if knocked-in.",
    }
    with open(metadata_file, "w") as fh:
        json.dump(metadata, fh, indent=2)

    print("Saved X,y to", out_file)
    print("Saved metadata to", metadata_file)
    print("X shape:", X.shape, "y shape:", y.shape)
    return X, y, metadata


if __name__ == "__main__":
    generate_dataset()

    # sanity checks
    data = np.load("../../data/raw/training_data_phoenix.npz")
    X = data["X"]
    y = data["y"]

    print("X shape:", X.shape, "y shape:", y.shape)
    meta = json.load(open("../../data/raw/training_data_phoenix_metadata.json"))
    print("Metadata:", meta)
