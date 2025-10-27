import time
from typing import Dict, Any, List, Optional
import numpy as np

from .payoffs import BasePayoff
from .data_generator import simulate_gbm_paths


class Evaluator:
    """Evaluate models against MC baseline."""

    def __init__(self, payoff: BasePayoff, n_steps: int = 252, verbose: bool = True):
        self.payoff = payoff
        self.n_steps = n_steps
        self.verbose = verbose

    def evaluate_case(
        self,
        params: Dict[str, Any],
        model,
        scaler,
        n_paths_list: List[int],
        use_log_target: bool = True,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Evaluate a single parameter set vs MC with different path counts."""
        results = {"params": params, "per_npaths": {}}

        feature_order = self.payoff.get_feature_order()
        feat = np.array([[params[f] for f in feature_order]])
        feat_s = scaler.transform(feat)

        t0 = time.time()
        pred_raw = model.predict(feat_s)[0]
        t_model = time.time() - t0

        if use_log_target:
            model_price = float(np.expm1(pred_raw))
        else:
            model_price = float(pred_raw)

        for n_paths in n_paths_list:
            t_mc0 = time.time()

            paths = simulate_gbm_paths(
                params["S0"],
                params["r"],
                params["sigma"],
                params["T"],
                self.n_steps,
                n_paths,
                seed=seed,
            )

            payoffs = self.payoff.compute_payoff(
                paths, params, params["r"], params["T"]
            )

            mc_time = time.time() - t_mc0
            mc_price = float(np.mean(payoffs))
            mc_std = float(np.std(payoffs))

            abs_error = abs(model_price - mc_price)
            rel_error = abs_error / abs(mc_price) if mc_price != 0 else None
            speedup = mc_time / t_model if t_model > 0 else float("inf")

            results["per_npaths"][str(n_paths)] = {
                "MC": {
                    "price": mc_price,
                    "std": mc_std,
                    "time": mc_time,
                    "n_paths": n_paths,
                },
                "Model": {
                    "price": model_price,
                    "time": t_model,
                    "abs_error": abs_error,
                    "rel_error": rel_error,
                    "speedup": speedup,
                },
            }

            if self.verbose:
                print(
                    f"  n_paths={n_paths}: MC={mc_price:.6f} "
                    f"(±{mc_std:.4f}, t={mc_time:.3f}s), "
                    f"Model={model_price:.6f} (t={t_model:.4f}s), "
                    f"err={abs_error:.6f}, speedup={speedup:.1f}x"
                )

        return results

    def evaluate_multiple_cases(
        self,
        test_cases: List[Dict[str, Any]],
        model,
        scaler,
        n_paths_list: List[int],
        use_log_target: bool = True,
        seed: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Evaluate multiple test cases."""
        results = []

        for i, test_case in enumerate(test_cases):
            if self.verbose:
                print(f"Evaluating case {i+1}/{len(test_cases)}")

            res = self.evaluate_case(
                test_case, model, scaler, n_paths_list, use_log_target, seed
            )
            results.append(res)

        return results

    @staticmethod
    def summarize_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Summarize evaluation results across all test cases."""
        summary = {
            "n_test_cases": len(results),
            "errors_by_npaths": {},
            "times_by_npaths": {},
            "speedups_by_npaths": {},
        }

        for case in results:
            for n_paths_str, data in case["per_npaths"].items():
                n_paths = int(n_paths_str)

                abs_error = data["Model"]["abs_error"]
                rel_error = data["Model"]["rel_error"]
                speedup = data["Model"]["speedup"]
                mc_time = data["MC"]["time"]
                model_time = data["Model"]["time"]

                if n_paths not in summary["errors_by_npaths"]:
                    summary["errors_by_npaths"][n_paths] = {
                        "abs": [],
                        "rel": [],
                    }
                    summary["times_by_npaths"][n_paths] = {
                        "mc": [],
                        "model": [],
                    }
                    summary["speedups_by_npaths"][n_paths] = []

                summary["errors_by_npaths"][n_paths]["abs"].append(abs_error)
                if rel_error is not None:
                    summary["errors_by_npaths"][n_paths]["rel"].append(rel_error)
                summary["times_by_npaths"][n_paths]["mc"].append(mc_time)
                summary["times_by_npaths"][n_paths]["model"].append(model_time)
                summary["speedups_by_npaths"][n_paths].append(speedup)

        for n_paths in summary["errors_by_npaths"]:
            summary["errors_by_npaths"][n_paths] = {
                "abs_mean": np.mean(summary["errors_by_npaths"][n_paths]["abs"]),
                "abs_std": np.std(summary["errors_by_npaths"][n_paths]["abs"]),
                "rel_mean": (
                    np.mean(summary["errors_by_npaths"][n_paths]["rel"])
                    if summary["errors_by_npaths"][n_paths]["rel"]
                    else None
                ),
                "rel_std": (
                    np.std(summary["errors_by_npaths"][n_paths]["rel"])
                    if summary["errors_by_npaths"][n_paths]["rel"]
                    else None
                ),
            }
            summary["times_by_npaths"][n_paths] = {
                "mc_mean": np.mean(summary["times_by_npaths"][n_paths]["mc"]),
                "model_mean": np.mean(summary["times_by_npaths"][n_paths]["model"]),
            }
            summary["speedups_by_npaths"][n_paths] = {
                "mean": np.mean(summary["speedups_by_npaths"][n_paths]),
                "std": np.std(summary["speedups_by_npaths"][n_paths]),
            }

        return summary
