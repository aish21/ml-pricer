"""Small, dependency-free process metrics for the research API."""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import Callable


OPERATIONS_MONITORING_VERSION = "operations-monitoring-v1"


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


class OperationsMonitor:
    """Track bounded request, error, and latency metrics for one API process."""

    def __init__(
        self,
        *,
        window_size: int = 1_000,
        clock: Callable[[], float] = time.monotonic,
        started_at: datetime | None = None,
    ) -> None:
        if window_size < 1:
            raise ValueError("window_size must be positive")
        self._clock = clock
        self._started_clock = clock()
        self._started_at = (started_at or datetime.now(timezone.utc)).astimezone(
            timezone.utc
        )
        self._lock = threading.Lock()
        self._total_requests = 0
        self._in_flight = 0
        self._status_classes = {f"{value}xx": 0 for value in range(1, 6)}
        self._durations_ms: deque[float] = deque(maxlen=window_size)

    def request_started(self) -> None:
        with self._lock:
            self._in_flight += 1

    def request_finished(self, *, status_code: int, duration_ms: float) -> None:
        status_class = f"{max(1, min(int(status_code) // 100, 5))}xx"
        with self._lock:
            self._in_flight = max(0, self._in_flight - 1)
            self._total_requests += 1
            self._status_classes[status_class] += 1
            self._durations_ms.append(max(0.0, float(duration_ms)))

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            total_requests = self._total_requests
            in_flight = self._in_flight
            status_classes = dict(self._status_classes)
            durations = list(self._durations_ms)
        failures = status_classes["5xx"]
        return {
            "version": OPERATIONS_MONITORING_VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "started_at": self._started_at.isoformat(),
            "uptime_seconds": max(0.0, self._clock() - self._started_clock),
            "requests": {
                "total": total_requests,
                "in_flight": in_flight,
                "status_classes": status_classes,
                "server_error_rate": (
                    failures / total_requests if total_requests else 0.0
                ),
            },
            "latency_ms": {
                "window_size": len(durations),
                "mean": sum(durations) / len(durations) if durations else None,
                "p50": _percentile(durations, 0.50),
                "p95": _percentile(durations, 0.95),
                "maximum": max(durations) if durations else None,
            },
        }


operations_monitor = OperationsMonitor()
