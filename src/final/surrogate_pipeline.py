import argparse
import json
from pathlib import Path

from .surrogate_data import (
    PhoenixDatasetConfig,
    generate_phoenix_surrogate_dataset,
    load_phoenix_surrogate_dataset,
    save_phoenix_surrogate_dataset,
)
from .surrogate_trainer import (
    PhoenixSurrogateTrainingConfig,
    train_phoenix_surrogate,
)


DEFAULT_OUTPUT_ROOT = Path("data") / "surrogates" / "phoenix-v4"


def _dataset_config(
    args: argparse.Namespace,
    *,
    role: str,
    n_contracts: int | None = None,
    markets_per_contract: int | None = None,
    seed_offset: int = 0,
    sampling_profile: str | None = None,
) -> PhoenixDatasetConfig:
    return PhoenixDatasetConfig(
        n_contracts=n_contracts or args.n_contracts,
        markets_per_contract=markets_per_contract or args.markets_per_contract,
        paths_per_replication=args.paths_per_replication,
        label_replications=args.label_replications,
        dataset_seed=args.dataset_seed + seed_offset,
        label_seed=args.label_seed + seed_offset,
        sampling_method=args.sampling_method,
        dataset_role=role,
        sampling_profile=sampling_profile or args.sampling_profile,
    )


def _training_config(args: argparse.Namespace) -> PhoenixSurrogateTrainingConfig:
    candidate_layouts = (
        tuple(
            tuple(int(width) for width in layout.split(","))
            for layout in args.candidate_layouts
        )
        if args.candidate_search
        else ()
    )
    return PhoenixSurrogateTrainingConfig(
        hidden_layer_sizes=tuple(args.hidden_layers),
        candidate_hidden_layer_sizes=candidate_layouts,
        candidate_seed_offsets=(
            tuple(args.candidate_seed_offsets) if args.candidate_search else (0,)
        ),
        max_iter=args.max_iter,
        random_state=args.training_seed,
        train_lightgbm_baseline=not args.skip_lightgbm,
        greek_validation_cases=args.greek_validation_cases,
        greek_validation_paths=args.greek_validation_paths,
    )


def _save_generated_dataset(dataset, output_root: Path) -> Path:
    dataset_directory = output_root / "datasets"
    dataset_name = dataset.metadata["dataset_id"].removeprefix("sha256:") + ".npz"
    dataset_path = dataset_directory / dataset_name
    save_phoenix_surrogate_dataset(dataset, dataset_path)
    return dataset_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate and train the monitored Phoenix payoff-aware v4 model."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_dataset_arguments(command_parser: argparse.ArgumentParser) -> None:
        command_parser.add_argument("--n-contracts", type=int, default=1024)
        command_parser.add_argument("--markets-per-contract", type=int, default=6)
        command_parser.add_argument("--paths-per-replication", type=int, default=1024)
        command_parser.add_argument("--label-replications", type=int, default=2)
        command_parser.add_argument("--dataset-seed", type=int, default=42)
        command_parser.add_argument("--label-seed", type=int, default=7301)
        command_parser.add_argument(
            "--sampling-method",
            choices=("sobol", "antithetic"),
            default="sobol",
        )
        command_parser.add_argument(
            "--sampling-profile",
            choices=("balanced", "low_vol_barrier_focus"),
            default="low_vol_barrier_focus",
        )

    def add_training_arguments(command_parser: argparse.ArgumentParser) -> None:
        command_parser.add_argument(
            "--hidden-layers", type=int, nargs="+", default=[128, 128]
        )
        command_parser.add_argument("--max-iter", type=int, default=1000)
        command_parser.add_argument("--training-seed", type=int, default=42)
        command_parser.add_argument(
            "--candidate-search",
            action=argparse.BooleanOptionalAction,
            default=True,
        )
        command_parser.add_argument(
            "--candidate-layouts",
            nargs="+",
            default=["192,192", "256,256", "256,128,64"],
            help="comma-separated payoff-aware hidden-layer layouts",
        )
        command_parser.add_argument(
            "--candidate-seed-offsets",
            type=int,
            nargs="+",
            default=[0, 101],
        )
        command_parser.add_argument("--skip-lightgbm", action="store_true")
        command_parser.add_argument("--greek-validation-cases", type=int, default=16)
        command_parser.add_argument("--greek-validation-paths", type=int, default=4096)

    generate = subparsers.add_parser("generate", help="Generate a versioned dataset")
    add_dataset_arguments(generate)
    generate.add_argument(
        "--dataset-role", choices=("development", "audit"), default="development"
    )
    generate.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)

    train = subparsers.add_parser("train", help="Train from an existing dataset")
    train.add_argument("dataset", type=Path)
    train.add_argument("--audit-dataset", type=Path, required=True)
    train.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    add_training_arguments(train)

    full = subparsers.add_parser("full", help="Generate data and train the model")
    add_dataset_arguments(full)
    full.add_argument("--audit-contracts", type=int, default=256)
    full.add_argument("--audit-markets-per-contract", type=int, default=4)
    full.add_argument("--audit-seed-offset", type=int, default=1_000_003)
    add_training_arguments(full)
    full.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    output_root = Path(args.output_root)
    if args.command == "generate":
        dataset = generate_phoenix_surrogate_dataset(
            _dataset_config(args, role=args.dataset_role)
        )
        dataset_path = _save_generated_dataset(dataset, output_root)
        print(
            json.dumps(
                {
                    "dataset_id": dataset.metadata["dataset_id"],
                    "dataset_path": str(dataset_path),
                    "split_counts": dataset.metadata["split_counts"],
                },
                indent=2,
            ),
            flush=True,
        )
        return 0

    if args.command == "train":
        dataset = load_phoenix_surrogate_dataset(args.dataset)
        audit_dataset = load_phoenix_surrogate_dataset(args.audit_dataset)
    else:
        dataset = generate_phoenix_surrogate_dataset(
            _dataset_config(args, role="development")
        )
        dataset_path = _save_generated_dataset(dataset, output_root)
        print(f"[PhoenixSurrogatePipeline] dataset saved to {dataset_path}", flush=True)
        audit_dataset = generate_phoenix_surrogate_dataset(
            _dataset_config(
                args,
                role="audit",
                n_contracts=args.audit_contracts,
                markets_per_contract=args.audit_markets_per_contract,
                seed_offset=args.audit_seed_offset,
                sampling_profile="balanced",
            )
        )
        audit_path = _save_generated_dataset(audit_dataset, output_root)
        print(
            f"[PhoenixSurrogatePipeline] sealed audit dataset saved to {audit_path}",
            flush=True,
        )
    manifest = train_phoenix_surrogate(
        dataset=dataset,
        audit_dataset=audit_dataset,
        output_root=output_root / "artifacts",
        config=_training_config(args),
    )
    print(
        json.dumps(
            {
                "artifact_id": manifest["artifact_id"],
                "deployment_status": manifest["deployment_status"],
                "selected_strategy": manifest["selected_strategy"],
                "selected_candidate": manifest["selected_candidate"],
                "audit_metrics": manifest["audit_evaluation"]["price_metrics"],
                "acceptance": manifest["acceptance"],
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
