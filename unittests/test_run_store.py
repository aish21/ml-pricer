import time

from app.services.run_store import get_run, list_recent_runs, save_run


def test_run_store_saves_and_fetches_run(tmp_path):
    db_path = tmp_path / "runs.sqlite3"

    run_id = save_run(
        "phoenix",
        {"params": {"S0": 100}},
        {"price": 0.99, "model": "LightGBM surrogate", "latency_ms": 12},
        db_path=db_path,
    )

    run = get_run(run_id, db_path=db_path)
    assert run is not None
    assert run["run_id"] == run_id
    assert run["product_key"] == "phoenix"
    assert run["request_payload"]["params"]["S0"] == 100
    assert run["result_payload"]["price"] == 0.99
    assert run["run_type"] == "price"
    assert run["parent_run_id"] is None


def test_run_store_missing_run_returns_none(tmp_path):
    db_path = tmp_path / "runs.sqlite3"
    assert get_run("missing", db_path=db_path) is None


def test_run_store_recent_runs_returns_latest_first(tmp_path):
    db_path = tmp_path / "runs.sqlite3"
    first = save_run("phoenix", {"seq": 1}, {"price": 0.98}, db_path=db_path)
    time.sleep(0.001)
    second = save_run("phoenix", {"seq": 2}, {"price": 0.99}, db_path=db_path)

    runs = list_recent_runs(limit=10, db_path=db_path)
    assert [run["run_id"] for run in runs] == [second, first]


def test_run_store_saves_and_fetches_scenario_metadata(tmp_path):
    db_path = tmp_path / "runs.sqlite3"
    base_id = save_run("phoenix", {"seq": 1}, {"price": 0.98}, db_path=db_path)
    scenario_id = save_run(
        "phoenix",
        {"seq": 2},
        {"base_price": 0.98, "shocked_price": 0.95},
        db_path=db_path,
        run_type="scenario",
        parent_run_id=base_id,
    )

    scenario = get_run(scenario_id, db_path=db_path)
    assert scenario["run_type"] == "scenario"
    assert scenario["parent_run_id"] == base_id
