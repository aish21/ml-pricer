import numpy as np

from src.final.model_trainer import split_train_validation_test


def test_train_validation_and_test_partitions_are_disjoint():
    sample_ids = np.arange(100)
    X = np.column_stack([sample_ids, sample_ids * 2])
    y = sample_ids.astype(float)

    X_train, X_val, X_test, _, _, _ = split_train_validation_test(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    train_ids = set(X_train[:, 0])
    validation_ids = set(X_val[:, 0])
    test_ids = set(X_test[:, 0])

    assert train_ids.isdisjoint(validation_ids)
    assert train_ids.isdisjoint(test_ids)
    assert validation_ids.isdisjoint(test_ids)
    assert train_ids | validation_ids | test_ids == set(sample_ids)
