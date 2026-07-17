import argparse
import json
from pathlib import Path

from app.services.surrogate_monitoring import (
    DEFAULT_SURROGATE_MONITORING_DB,
    SurrogateMonitoringSettings,
    replay_surrogate_shadow_observations,
)
from app.services.surrogate_service import (
    DEFAULT_SURROGATE_ROOT,
    SurrogateSettings,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Replay stored Phoenix shadow observations through a selected artifact."
        )
    )
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument(
        "--monitoring-db", type=Path, default=DEFAULT_SURROGATE_MONITORING_DB
    )
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_SURROGATE_ROOT)
    parser.add_argument("--allow-unapproved", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = replay_surrogate_shadow_observations(
        limit=args.limit,
        monitoring_settings=SurrogateMonitoringSettings(
            enabled=True,
            db_path=args.monitoring_db,
        ),
        surrogate_settings=SurrogateSettings(
            enabled=True,
            artifact_root=args.artifact_root,
            allow_unapproved=args.allow_unapproved,
        ),
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0 if result["n_replayed"] == result["n_successful"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
