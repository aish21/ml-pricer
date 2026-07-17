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
    assert audit.dataset_seed == 5_000_053
    assert audit.label_seed == 5_007_312
    assert audit.sampling_profile == "balanced"
