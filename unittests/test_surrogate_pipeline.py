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


def test_hybrid_command_uses_the_frozen_v7_layout_and_no_audit():
    args = _build_parser().parse_args(
        [
            "research-hybrid",
            "development.npz",
            "hazards.npz",
        ]
    )

    assert args.hidden_layers == [256, 128, 64]
    assert args.training_seed == 143
    assert args.validation_seed == 42
    assert args.max_iter == 1000
    assert not hasattr(args, "audit_dataset")
    assert not hasattr(args, "output_root")


def test_price_first_command_keeps_weight_selection_inside_training_data():
    args = _build_parser().parse_args(
        [
            "research-price-first",
            "development.npz",
            "hazards.npz",
        ]
    )

    assert args.hidden_layers == [256, 128, 64]
    assert args.auxiliary_loss_weights == [0.0, 0.03, 0.1]
    assert args.internal_folds == 3
    assert args.training_seed == 143
    assert args.validation_seed == 42
    assert not hasattr(args, "audit_dataset")
    assert not hasattr(args, "output_root")


def test_price_first_audit_command_freezes_all_model_and_gate_settings():
    args = _build_parser().parse_args(
        [
            "audit-price-first",
            "development.npz",
            "hazards.npz",
            "audit.npz",
            "--report",
            "audit-report.json",
        ]
    )

    assert args.command == "audit-price-first"
    assert not hasattr(args, "training_seed")
    assert not hasattr(args, "auxiliary_loss_weights")
    assert not hasattr(args, "acceptance_audit_mae")
    assert not hasattr(args, "output_root")


def test_price_first_package_command_only_accepts_frozen_inputs_and_output_root():
    args = _build_parser().parse_args(
        [
            "package-price-first",
            "development.npz",
            "hazards.npz",
            "audit-report.json",
            "--output-root",
            "artifacts",
        ]
    )

    assert args.command == "package-price-first"
    assert args.audit_report.name == "audit-report.json"
    assert str(args.output_root) == "artifacts"
    assert not hasattr(args, "training_seed")
    assert not hasattr(args, "audit_dataset")


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


def test_research_hybrid_command_executes_without_an_output_root(monkeypatch, capsys):
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
        "train_phoenix_event_summary_hybrid_candidate",
        lambda _dataset, _config, **_kwargs: (
            None,
            {"status": "research_only"},
        ),
    )

    result = surrogate_pipeline.main(
        [
            "research-hybrid",
            "development-dataset.npz",
            "hazard-dataset.npz",
        ]
    )

    assert result == 0
    assert '"status": "research_only"' in capsys.readouterr().out


def test_research_price_first_executes_without_an_output_root(monkeypatch, capsys):
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
        "train_phoenix_price_first_candidate",
        lambda _dataset, _config: (
            None,
            {"status": "research_only"},
        ),
    )

    result = surrogate_pipeline.main(
        [
            "research-price-first",
            "development-dataset.npz",
            "hazard-dataset.npz",
        ]
    )

    assert result == 0
    assert '"status": "research_only"' in capsys.readouterr().out


def test_price_first_audit_executes_without_mutable_settings(
    monkeypatch,
    tmp_path,
    capsys,
):
    base = object()
    hazard = object()
    audit = object()
    loaded = iter([base, audit])
    monkeypatch.setattr(
        surrogate_pipeline,
        "load_phoenix_surrogate_dataset",
        lambda _path: next(loaded),
    )
    monkeypatch.setattr(
        surrogate_pipeline,
        "load_phoenix_hazard_dataset",
        lambda _path, *, base: hazard,
    )
    monkeypatch.setattr(
        surrogate_pipeline,
        "audit_frozen_phoenix_price_first_candidate",
        lambda **_kwargs: {
            "audit_dataset_id": "sha256:audit",
            "audit_decision": "passed",
            "audit_evaluation": {"price_metrics": {"mae": 0.01}},
            "acceptance": {"passed": True},
        },
    )
    report_path = tmp_path / "audit-report.json"

    result = surrogate_pipeline.main(
        [
            "audit-price-first",
            "development-dataset.npz",
            "hazard-dataset.npz",
            "audit-dataset.npz",
            "--report",
            str(report_path),
        ]
    )

    assert result == 0
    assert report_path.exists()
    assert '"audit_decision": "passed"' in capsys.readouterr().out


def test_price_first_audit_refuses_to_overwrite_a_consumed_report(
    tmp_path,
):
    report_path = tmp_path / "audit-report.json"
    report_path.write_text("{}\n", encoding="utf-8")

    try:
        surrogate_pipeline.main(
            [
                "audit-price-first",
                "development-dataset.npz",
                "hazard-dataset.npz",
                "audit-dataset.npz",
                "--report",
                str(report_path),
            ]
        )
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("consumed audit report was overwritten")

    assert report_path.read_text(encoding="utf-8") == "{}\n"


def test_price_first_package_executes_with_the_sealed_report(
    monkeypatch,
    tmp_path,
    capsys,
):
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
        "package_audit_approved_price_first_artifact",
        lambda **_kwargs: {
            "artifact_id": "sha256:" + "a" * 64,
            "deployment_status": "shadow_approved",
            "model_version": "phoenix-price-first-multitask-v1",
            "numpy_parity": {"passed": True},
        },
    )

    result = surrogate_pipeline.main(
        [
            "package-price-first",
            "development-dataset.npz",
            "hazard-dataset.npz",
            "sealed-audit-report.json",
            "--output-root",
            str(tmp_path),
        ]
    )

    assert result == 0
    assert '"deployment_status": "shadow_approved"' in capsys.readouterr().out
