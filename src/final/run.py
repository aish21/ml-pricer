from pathlib import Path
from .payoffs import (
    PhoenixPayoff,
    AccumulatorPayoff,
    BarrierOptionPayoff,
    DecumulatorPayoff,
)
from .pipeline import PricingPipeline


def run_phoenix():
    """Phoenix structured product."""
    print("=" * 80)
    print("Phoenix Payoff")
    print("=" * 80)

    payoff = PhoenixPayoff()
    pipeline = PricingPipeline(
        payoff=payoff, n_steps=252, use_log_target=True, verbose=True
    )

    results = pipeline.run_full_pipeline(
        n_samples=1000,
        n_paths_per_sample=500,
        n_trials=10,
        output_dir=Path("../final/results/phoenix"),
        seed=42,
    )

    print("\nTraining Metrics:")
    print(f"  RMSE: {results['training']['metrics']['rmse']:.6f}")
    print(f"  MAE: {results['training']['metrics']['mae']:.6f}")
    print(f"  R2: {results['training']['metrics']['r2']:.4f}")

    print("\nEvaluation Summary:")
    summary = results["evaluation"]["summary"]
    for n_paths in sorted(summary["speedups_by_npaths"].keys()):
        speedup = summary["speedups_by_npaths"][n_paths]["mean"]
        abs_err = summary["errors_by_npaths"][n_paths]["abs_mean"]
        print(f"  {n_paths} paths: speedup={speedup:.1f}x, abs_error={abs_err:.6f}")


def run_accumulator():
    """Accumulator payoff."""
    print("\n" + "=" * 80)
    print("Accumulator Payoff")
    print("=" * 80)

    payoff = AccumulatorPayoff()
    pipeline = PricingPipeline(
        payoff=payoff, n_steps=252, use_log_target=True, verbose=True
    )

    results = pipeline.run_full_pipeline(
        n_samples=5000,
        n_paths_per_sample=1000,
        n_trials=20,
        output_dir=Path("../final/results/accumulator"),
        seed=42,
    )

    print("\nTraining Metrics:")
    print(f"  RMSE: {results['training']['metrics']['rmse']:.6f}")
    print(f"  MAE: {results['training']['metrics']['mae']:.6f}")
    print(f"  R2: {results['training']['metrics']['r2']:.4f}")

    print("\nEvaluation Summary:")
    summary = results["evaluation"]["summary"]
    for n_paths in sorted(summary["speedups_by_npaths"].keys()):
        speedup = summary["speedups_by_npaths"][n_paths]["mean"]
        abs_err = summary["errors_by_npaths"][n_paths]["abs_mean"]
        print(f"  {n_paths} paths: speedup={speedup:.1f}x, abs_error={abs_err:.6f}")


def run_barrier():
    """Barrier option."""
    print("\n" + "=" * 80)
    print("Barrier Option Payoff")
    print("=" * 80)

    payoff = BarrierOptionPayoff()
    pipeline = PricingPipeline(
        payoff=payoff, n_steps=252, use_log_target=True, verbose=True
    )

    results = pipeline.run_full_pipeline(
        n_samples=5000,
        n_paths_per_sample=1000,
        n_trials=20,
        output_dir=Path("../final/results/barrier"),
        seed=42,
    )

    print("\nTraining Metrics:")
    print(f"  RMSE: {results['training']['metrics']['rmse']:.6f}")
    print(f"  MAE: {results['training']['metrics']['mae']:.6f}")
    print(f"  R2: {results['training']['metrics']['r2']:.4f}")

    print("\nEvaluation Summary:")
    summary = results["evaluation"]["summary"]
    for n_paths in sorted(summary["speedups_by_npaths"].keys()):
        speedup = summary["speedups_by_npaths"][n_paths]["mean"]
        abs_err = summary["errors_by_npaths"][n_paths]["abs_mean"]
        print(f"  {n_paths} paths: speedup={speedup:.1f}x, abs_error={abs_err:.6f}")


def run_decumulator():
    """Decumulator payoff."""
    print("\n" + "=" * 80)
    print("Decumulator Payoff")
    print("=" * 80)

    payoff = DecumulatorPayoff()
    pipeline = PricingPipeline(
        payoff=payoff, n_steps=252, use_log_target=True, verbose=True
    )

    results = pipeline.run_full_pipeline(
        n_samples=5000,
        n_paths_per_sample=1000,
        n_trials=20,
        output_dir=Path("/results/decumulator"),
        seed=42,
    )

    print("\nTraining Metrics:")
    print(f"  RMSE: {results['training']['metrics']['rmse']:.6f}")
    print(f"  MAE: {results['training']['metrics']['mae']:.6f}")
    print(f"  R2: {results['training']['metrics']['r2']:.4f}")

    print("\nEvaluation Summary:")
    summary = results["evaluation"]["summary"]
    for n_paths in sorted(summary["speedups_by_npaths"].keys()):
        speedup = summary["speedups_by_npaths"][n_paths]["mean"]
        abs_err = summary["errors_by_npaths"][n_paths]["abs_mean"]
        print(f"  {n_paths} paths: speedup={speedup:.1f}x, abs_error={abs_err:.6f}")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("Running All Available Payoffs")
    print("=" * 80 + "\n")

    run_phoenix()
    run_accumulator()
    run_barrier()
    run_decumulator()

    print("\n" + "=" * 80)
    print("All payoffs completed!")
    print("=" * 80)
