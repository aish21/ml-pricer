from pathlib import Path
from src.final.payoffs import (
    PhoenixPayoff,
    AccumulatorPayoff,
    BarrierOptionPayoff,
    DecumulatorPayoff,
)
from src.final.pipeline import PricingPipeline


def run_pipeline(name, payoff, n_samples, n_paths_per_sample, n_trials, output_subdir):
    """Generic runner for any payoff type."""
    print("\n" + "=" * 80)
    print(f"{name} Payoff")
    print("=" * 80)

    pipeline = PricingPipeline(
        payoff=payoff, n_steps=252, use_log_target=True, verbose=True
    )

    results = pipeline.run_full_pipeline(
        n_samples=n_samples,
        n_paths_per_sample=n_paths_per_sample,
        n_trials=n_trials,
        output_dir=Path(f"final/results/{output_subdir}"),
        seed=42,
    )

    metrics = results["training"]["metrics"]
    summary = results["evaluation"]["summary"]

    print("\nTraining Metrics:")
    print(f"  RMSE: {metrics['rmse']:.6f}")
    print(f"  MAE: {metrics['mae']:.6f}")
    print(f"  R2: {metrics['r2']:.4f}")

    print("\nEvaluation Summary:")
    for n_paths in sorted(summary["speedups_by_npaths"].keys()):
        s = summary["speedups_by_npaths"][n_paths]["mean"]
        e = summary["errors_by_npaths"][n_paths]["abs_mean"]
        print(f"  {n_paths} paths: speedup={s:.1f}x, abs_error={e:.6f}")

    print("=" * 80 + "\n")


def run_phoenix():
    run_pipeline("Phoenix", PhoenixPayoff(), 5000, 4000, 20, "phoenix")


def run_accumulator():
    run_pipeline("Accumulator", AccumulatorPayoff(), 5000, 4000, 20, "accumulator")


def run_barrier():
    run_pipeline("Barrier Option", BarrierOptionPayoff(), 8000, 3000, 25, "barrier")


def run_decumulator():
    run_pipeline("Decumulator", DecumulatorPayoff(), 5000, 4000, 20, "decumulator")


def run_phoenix_stepdown():
    from src.final.inherited_payoffs import StepDownPhoenixPayoff

    run_pipeline(
        "Step-Down Phoenix", StepDownPhoenixPayoff(), 3000, 4000, 15, "phoenix_stepdown"
    )


def run_reverse_accumulator():
    from src.final.inherited_payoffs import ReverseAccumulatorPayoff

    run_pipeline(
        "Reverse Accumulator",
        ReverseAccumulatorPayoff(),
        3000,
        4000,
        15,
        "reverse_accumulator",
    )


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("Running All Payoffs (Extended Training)")
    print("=" * 80 + "\n")

    run_phoenix()
    run_accumulator()
    run_barrier()
    run_decumulator()
    run_reverse_accumulator()
    run_phoenix_stepdown()

    print("\n" + "=" * 80)
    print("All payoffs completed!")
    print("=" * 80)
