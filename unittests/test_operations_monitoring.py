from datetime import datetime, timezone

import pytest

from app.services.operations_monitoring import (
    OPERATIONS_MONITORING_VERSION,
    OperationsMonitor,
)


def test_operations_monitor_reports_bounded_latency_and_error_metrics():
    ticks = iter([10.0, 15.0])
    monitor = OperationsMonitor(
        window_size=2,
        clock=lambda: next(ticks),
        started_at=datetime(2026, 7, 20, 8, 0, tzinfo=timezone.utc),
    )
    for status_code, duration_ms in ((200, 10.0), (404, 20.0), (503, 30.0)):
        monitor.request_started()
        monitor.request_finished(
            status_code=status_code,
            duration_ms=duration_ms,
        )

    snapshot = monitor.snapshot()

    assert snapshot["version"] == OPERATIONS_MONITORING_VERSION
    assert snapshot["uptime_seconds"] == 5.0
    assert snapshot["requests"] == {
        "total": 3,
        "in_flight": 0,
        "status_classes": {
            "1xx": 0,
            "2xx": 1,
            "3xx": 0,
            "4xx": 1,
            "5xx": 1,
        },
        "server_error_rate": pytest.approx(1 / 3),
    }
    assert snapshot["latency_ms"]["window_size"] == 2
    assert snapshot["latency_ms"]["mean"] == 25.0
    assert snapshot["latency_ms"]["p50"] == 25.0
    assert snapshot["latency_ms"]["p95"] == 29.5
    assert snapshot["latency_ms"]["maximum"] == 30.0
