import argparse
import json
from pathlib import Path

from app.services.surrogate_monitoring import (
    DEFAULT_SURROGATE_MONITORING_DB,
    SurrogateMonitoringSettings,
)
from app.services.surrogate_promotion import get_surrogate_promotion_readiness


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the frozen Phoenix shadow promotion-readiness policy. "
            "This command never changes the runtime policy."
        )
    )
    parser.add_argument(
        "--monitoring-db",
        type=Path,
        default=DEFAULT_SURROGATE_MONITORING_DB,
    )
    parser.add_argument("--limit", type=int, default=100_000)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = get_surrogate_promotion_readiness(
        limit=args.limit,
        settings=SurrogateMonitoringSettings(
            enabled=True,
            db_path=args.monitoring_db,
        ),
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
