import numpy as np

from src.final.surrogate_hybrid import (
    PHOENIX_EVENT_SUMMARY_TARGET_NAMES,
    summarize_phoenix_observation_events,
    train_phoenix_event_summary_hybrid_candidate,
)
from src.final.surrogate_trainer import PhoenixSurrogateTrainingConfig
from unittests.test_surrogate_hazard import small_hazard_dataset


def test_event_summaries_are_bounded_and_coupon_masses_reconcile():
    dataset = small_hazard_dataset()

    summaries = summarize_phoenix_observation_events(dataset)

    assert summaries.shape == (
        len(dataset.base.y),
        len(PHOENIX_EVENT_SUMMARY_TARGET_NAMES),
    )
    assert np.all((summaries[:, :3] >= 0.0) & (summaries[:, :3] <= 1.0))
    assert np.all(summaries[:, 3:] >= 0.0)
    assert np.allclose(summaries[:, 3], summaries[:, 4] + summaries[:, 5])


def test_hybrid_candidate_keeps_price_direct_and_research_only():
    dataset = small_hazard_dataset()

    model, report = train_phoenix_event_summary_hybrid_candidate(
        dataset,
        PhoenixSurrogateTrainingConfig(
            hidden_layer_sizes=(8,),
            max_iter=30,
            train_lightgbm_baseline=False,
            greek_validation_cases=0,
            selection_validation_folds=2,
            selection_validation_repeats=2,
        ),
        hidden_layer_sizes=(8,),
        random_state=143,
    )
    predictions = model.predict(dataset.base.X[:3])

    assert predictions.shape == (3,)
    assert np.all(np.isfinite(predictions))
    assert report["price_inference"] == "independent direct-price output"
    assert report["runtime_eligible"] is False
    assert report["audit_evaluated"] is False
    assert report["deployment_status"] == "research_only"
    assert report["selection"]["policy"] == (
        "event-summary-hybrid-development-comparison-v1"
    )
    assert len(report["candidate_models"]) == 2
