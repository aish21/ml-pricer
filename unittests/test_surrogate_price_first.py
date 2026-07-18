import numpy as np
import pytest

from src.final.surrogate_price_first import (
    PHOENIX_PRICE_FIRST_EVENT_TARGET_NAMES,
    PhoenixPriceFirstTrainingConfig,
    price_first_event_targets,
    train_phoenix_price_first_candidate,
)
from unittests.test_surrogate_hazard import small_hazard_dataset


def test_price_first_targets_mask_timing_and_remove_redundant_coupon_total():
    dataset = small_hazard_dataset()

    targets, weights = price_first_event_targets(dataset)

    assert (
        targets.shape
        == weights.shape
        == (
            len(dataset.base.y),
            len(PHOENIX_PRICE_FIRST_EVENT_TARGET_NAMES),
        )
    )
    assert "expected_coupon_count" not in PHOENIX_PRICE_FIRST_EVENT_TARGET_NAMES
    assert np.array_equal(weights[:, 0], weights[:, 1])
    assert np.all((weights[:, :2] >= 0.0) & (weights[:, :2] <= 1.0))
    assert np.all(weights[:, 2:] == 1.0)


def test_price_first_config_rejects_duplicate_auxiliary_weights():
    with pytest.raises(RuntimeError, match="auxiliary loss weights"):
        PhoenixPriceFirstTrainingConfig(
            auxiliary_loss_weights=(0.03, 0.03),
        )


def test_price_first_candidate_selects_weight_inside_training_groups_only():
    pytest.importorskip("torch")
    dataset = small_hazard_dataset()

    model, report = train_phoenix_price_first_candidate(
        dataset,
        PhoenixPriceFirstTrainingConfig(
            hidden_layer_sizes=(8,),
            auxiliary_head_width=4,
            batch_size=32,
            epochs=3,
            internal_selection_folds=2,
            auxiliary_loss_weights=(0.0, 0.1),
            selection_validation_folds=2,
            selection_validation_repeats=2,
        ),
        verbose=False,
    )
    predictions = model.predict(dataset.base.X[:3])
    internal = report["internal_auxiliary_weight_selection"]

    assert predictions.shape == (3,)
    assert np.all(np.isfinite(predictions))
    assert internal["split"] == "train"
    assert internal["validation_or_test_rows_used"] is False
    assert report["price_inference"] == "independent focused direct-price head"
    assert report["runtime_eligible"] is False
    assert report["audit_evaluated"] is False
    assert report["deployment_status"] == "research_only"
    assert report["selection"]["policy"] == "price-first-development-comparison-v1"
    assert report["selected_auxiliary_loss_weight"] in (0.0, 0.1)

    replay, replay_report = train_phoenix_price_first_candidate(
        dataset,
        PhoenixPriceFirstTrainingConfig(
            hidden_layer_sizes=(8,),
            auxiliary_head_width=4,
            batch_size=32,
            epochs=3,
            internal_selection_folds=2,
            auxiliary_loss_weights=(0.0, 0.1),
            selection_validation_folds=2,
            selection_validation_repeats=2,
        ),
        verbose=False,
    )

    assert replay_report["selected_auxiliary_loss_weight"] == (
        report["selected_auxiliary_loss_weight"]
    )
    assert np.array_equal(
        replay.predict(dataset.base.X[:3]),
        predictions,
    )
