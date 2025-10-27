import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np

from .payoffs import BasePayoff
from .data_generator import DataGenerator
from .model_trainer import ModelTrainer
from .evaluator import Evaluator


class PricingPipeline:
    """Orchestrate the pricing pipeline."""

    def __init__(
        self,
        payoff: BasePayoff,
        n_steps: int = 252,
        use_log_target: bool = True,
        verbose: bool = True,
    ):
        self.payoff = payoff
        self.n_steps = n_steps
        self.use_log_target = use_log_target
        self.verbose = verbose

        self.data_generator = DataGenerator(payoff, n_steps, verbose)
        self.trainer = ModelTrainer(use_log_target, verbose=verbose)
        self.evaluator = Evaluator(payoff, n_steps, verbose)

    def run_full_pipeline(
        self,
        n_samples: int,
        n_paths_per_sample: int,
        n_trials: int = 30,
        test_cases: Optional[List[Dict[str, Any]]] = None,
        n_paths_list: Optional[List[int]] = None,
        output_dir: Optional[Path] = None,
        data_file: Optional[Path] = None,
        model_file: Optional[Path] = None,
        scaler_file: Optional[Path] = None,
        results_file: Optional[Path] = None,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Run the complete pipeline."""
        start_time = time.time()
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            if data_file is None:
                data_file = output_dir / "training_data.npz"
            if model_file is None:
                model_file = output_dir / "model.joblib"
            if scaler_file is None:
                scaler_file = output_dir / "scaler.joblib"
            if results_file is None:
                results_file = output_dir / "results.json"

        # Generate or load training data
        if data_file and data_file.exists():
            if self.verbose:
                print(f"Loading training data from {data_file}")
            X, y, meta = DataGenerator.load(data_file)
        else:
            if self.verbose:
                print("Generating training data...")
            X, y = self.data_generator.generate(
                n_samples, n_paths_per_sample, seed=seed
            )
            if data_file:
                self.data_generator.save(X, y, data_file)

        # Train model
        if self.verbose:
            print("\nTraining model...")
        train_info = self.trainer.train(
            X, y, self.payoff.get_feature_order(), n_trials=n_trials
        )

        # Save model
        if model_file and scaler_file:
            self.trainer.save(model_file, scaler_file)

        # Setup evaluation
        if test_cases is None:
            # Default test cases (can be customized per payoff type)
            test_cases = self._get_default_test_cases()

        if n_paths_list is None:
            n_paths_list = [500, 2000, 8000]

        # Evaluate
        if self.verbose:
            print("\nEvaluating model...")
        eval_results = self.evaluator.evaluate_multiple_cases(
            test_cases,
            train_info["model"],
            train_info["scaler"],
            n_paths_list,
            self.use_log_target,
            seed=seed,
        )

        # Summarize results
        summary = self.evaluator.summarize_results(eval_results)

        # Create final output
        output = {
            "config": {
                "payoff_type": self.payoff.__class__.__name__,
                "n_samples": n_samples,
                "n_paths_per_sample": n_paths_per_sample,
                "n_steps": self.n_steps,
                "n_trials": n_trials,
                "use_log_target": self.use_log_target,
            },
            "training": {
                "metrics": train_info["metrics"],
                "optuna": train_info["optuna_study"],
                "feature_importance": train_info["feature_importance"],
            },
            "evaluation": {
                "test_cases": test_cases,
                "results": eval_results,
                "summary": summary,
            },
            "timing": {
                "total_seconds": time.time() - start_time,
            },
        }

        # Save results
        if results_file:
            # Convert numpy types to native Python types for JSON
            output_serializable = self._make_serializable(output)
            with open(results_file, "w") as f:
                json.dump(output_serializable, f, indent=2)
            if self.verbose:
                print(f"\nResults saved to {results_file}")

        elapsed = time.time() - start_time
        if self.verbose:
            print(f"\nPipeline complete in {elapsed:.1f}s")

        return output

    def _get_default_test_cases(self) -> List[Dict[str, Any]]:
        """Get default test cases."""
        payoff_type = self.payoff.__class__.__name__

        if payoff_type == "PhoenixPayoff":
            return [
                {
                    "S0": 100.0,
                    "r": 0.03,
                    "sigma": 0.2,
                    "T": 1.0,
                    "autocall_barrier_frac": 1.05,
                    "coupon_barrier_frac": 1.0,
                    "coupon_rate": 0.02,
                    "knock_in_frac": 0.7,
                    "obs_count": 6,
                },
                {
                    "S0": 100.0,
                    "r": 0.01,
                    "sigma": 0.25,
                    "T": 2.0,
                    "autocall_barrier_frac": 1.0,
                    "coupon_barrier_frac": 0.95,
                    "coupon_rate": 0.015,
                    "knock_in_frac": 0.6,
                    "obs_count": 8,
                },
            ]
        elif payoff_type == "AccumulatorPayoff":
            return [
                {
                    "S0": 100.0,
                    "r": 0.03,
                    "sigma": 0.2,
                    "T": 1.0,
                    "upper_barrier_frac": 1.05,
                    "lower_barrier_frac": 0.95,
                    "participation_rate": 2.0,
                    "obs_frequency": 0.25,
                },
                {
                    "S0": 100.0,
                    "r": 0.02,
                    "sigma": 0.25,
                    "T": 2.0,
                    "upper_barrier_frac": 1.06,
                    "lower_barrier_frac": 0.94,
                    "participation_rate": 2.5,
                    "obs_frequency": 0.5,
                },
            ]
        elif payoff_type == "BarrierOptionPayoff":
            return [
                {
                    "S0": 100.0,
                    "r": 0.03,
                    "sigma": 0.2,
                    "T": 1.0,
                    "K": 100.0,
                    "barrier_frac": 0.8,
                    "option_type": 1.0,
                },
                {
                    "S0": 100.0,
                    "r": 0.02,
                    "sigma": 0.25,
                    "T": 1.5,
                    "K": 95.0,
                    "barrier_frac": 0.75,
                    "option_type": 0.0,
                },
            ]
        elif payoff_type == "DecumulatorPayoff":
            return [
                {
                    "S0": 100.0,
                    "r": 0.03,
                    "sigma": 0.2,
                    "T": 1.0,
                    "upper_barrier_frac": 1.05,
                    "lower_barrier_frac": 0.95,
                    "participation_rate": 2.0,
                    "obs_frequency": 0.25,
                },
                {
                    "S0": 100.0,
                    "r": 0.02,
                    "sigma": 0.25,
                    "T": 2.0,
                    "upper_barrier_frac": 1.06,
                    "lower_barrier_frac": 0.94,
                    "participation_rate": 2.5,
                    "obs_frequency": 0.5,
                },
            ]
        else:
            # Generic default
            return [
                {"S0": 100.0, "r": 0.03, "sigma": 0.2, "T": 1.0},
            ]

    def _make_serializable(self, obj):
        """Convert numpy types to native Python types."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(
            obj,
            (
                np.integer,
                np.int_,
                np.intc,
                np.intp,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ),
        ):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
            return float(obj)
        else:
            return obj
