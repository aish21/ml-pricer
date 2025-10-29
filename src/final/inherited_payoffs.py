from .payoffs import PhoenixPayoff, AccumulatorPayoff
import numpy as np
import math


class StepDownPhoenixPayoff(PhoenixPayoff):
    """Phoenix payoff with step-down autocall barrier."""

    def compute_payoff(self, paths, params, r, T):
        n_paths, n_points = paths.shape
        n_steps = n_points - 1
        obs_count = int(params.get("obs_count", 6))
        obs_idx = np.linspace(0, n_steps, obs_count + 1, dtype=int)[1:]

        base_autocall_b = params["S0"] * params["autocall_barrier_frac"]
        stepdown_rate = params.get("stepdown_rate", 0.02)
        coupon_rate = params["coupon_rate"]
        knockin_b = params["S0"] * params["knock_in_frac"]

        payoffs = np.zeros(n_paths, dtype=np.float64)

        for i in range(n_paths):
            path = paths[i]
            knocked_in = np.any(path < knockin_b)
            call_idx = None

            for j, idx in enumerate(obs_idx):
                # barrier decreases each observation
                autocall_b = base_autocall_b * (1 - j * stepdown_rate)
                if path[idx] >= autocall_b:
                    call_idx = idx
                    break

            if call_idx is not None:
                t_call = (call_idx / n_steps) * T
                payoff = 1.0 + coupon_rate * (j + 1)
                payoffs[i] = payoff * math.exp(-r * t_call)
            else:
                t_maturity = T
                payoff = (
                    (path[-1] / path[0])
                    if knocked_in
                    else 1.0 + coupon_rate * obs_count
                )
                payoffs[i] = payoff * math.exp(-r * t_maturity)

        return payoffs


class ReverseAccumulatorPayoff(AccumulatorPayoff):
    """Reverse accumulator - accumulates when price is OUTSIDE barriers."""

    def compute_payoff(self, paths, params, r, T):
        n_paths, n_points = paths.shape
        n_steps = n_points - 1

        S0 = params["S0"]
        upper_barrier = S0 * params["upper_barrier_frac"]
        lower_barrier = S0 * params["lower_barrier_frac"]
        participation_rate = params["participation_rate"]
        obs_frequency = params.get("obs_frequency", 0.25)

        n_obs = max(2, int(T / obs_frequency))
        obs_idx = np.linspace(0, n_steps, n_obs + 1, dtype=int)[1:]

        payoffs = np.zeros(n_paths, dtype=np.float64)

        for i in range(n_paths):
            path = paths[i]
            accumulated_value = 0.0
            count = 0

            for idx in obs_idx:
                S_t = path[idx]
                if S_t >= upper_barrier or S_t <= lower_barrier:
                    # reverse accumulation logic
                    discounted_price = S_t * (1 + participation_rate)
                    accumulated_value += discounted_price
                    count += 1

            payoffs[i] = (
                (accumulated_value / count) * math.exp(-r * T) * count / n_obs
                if count > 0
                else 0.0
            )

        return payoffs
