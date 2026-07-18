import src.final.surrogate_pipeline as surrogate_pipeline
from src.final.surrogate_pipeline import _build_parser, _dataset_config


def test_full_pipeline_defaults_keep_development_and_audit_budgets_separate():
    args = _build_parser().parse_args(["full"])

    development = _dataset_config(args, role="development")
    audit = _dataset_config(
        args,
        role="audit",
        n_contracts=args.audit_contracts,
        markets_per_contract=args.audit_markets_per_contract,
        seed_offset=args.audit_seed_offset,
        sampling_profile="balanced",
        paths_per_replication=args.audit_paths_per_replication,
        label_replications=args.audit_label_replications,
    )

    assert development.label_replications == 2
    assert development.paths_per_replication == 1024
    assert development.sampling_profile == "low_vol_barrier_focus"
    assert audit.label_replications == 8
    assert audit.paths_per_replication == 256
    assert audit.label_replications * audit.paths_per_replication == 2048
    assert audit.dataset_seed == 9_000_073
    assert audit.label_seed == 9_007_332
    assert audit.sampling_profile == "balanced"


def test_event_conditioned_research_command_has_no_audit_or_artifact_arguments():
    args = _build_parser().parse_args(["research-events", "development-dataset.npz"])

    assert args.command == "research-events"
    assert str(args.dataset) == "development-dataset.npz"
    assert args.report is None
    assert not hasattr(args, "audit_dataset")
    assert not hasattr(args, "output_root")


def test_hazard_commands_keep_labels_and_training_development_only():
    generate_args = _build_parser().parse_args(
        ["hazard-generate", "development-dataset.npz"]
    )
    train_args = _build_parser().parse_args(
        [
            "research-hazards",
            "development-dataset.npz",
            "hazard-dataset.npz",
        ]
    )

    assert generate_args.command == "hazard-generate"
    assert generate_args.workers == 1
    assert train_args.command == "research-hazards"
    assert train_args.hazard_max_iter == 800
    assert train_args.training_seed == 143
    assert not hasattr(train_args, "audit_dataset")
    assert not hasattr(train_args, "output_root")


def test_research_hazard_command_executes_without_an_output_root(monkeypatch, capsys):
    base = object()
    hazard = object()
    monkeypatch.setattr(
        surrogate_pipeline,
        "load_phoenix_surrogate_dataset",
        lambda _path: base,
    )
    monkeypatch.setattr(
        surrogate_pipeline,
        "load_phoenix_hazard_dataset",
        lambda _path, *, base: hazard,
    )
    monkeypatch.setattr(
        surrogate_pipeline,
        "train_phoenix_observation_hazard_candidate",
        lambda _dataset, _config: (None, {"status": "research_only"}),
    )

    result = surrogate_pipeline.main(
        [
            "research-hazards",
            "development-dataset.npz",
            "hazard-dataset.npz",
        ]
    )

    assert result == 0
    assert '"status": "research_only"' in capsys.readouterr().out
