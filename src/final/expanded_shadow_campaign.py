"""Run the daily out-of-time evidence campaign for expanded shadow models.

The campaign freezes one research-market calibration per underlier, applies a
predeclared payoff-boundary grid, computes a high-path Monte Carlo reference,
and records the shadow comparison with durable provenance.  It is deliberately
separate from interactive traffic and safe to rerun.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from app.services.expanded_shadow_monitoring import (
    DEFAULT_DB as DEFAULT_MONITORING_DB,
    expanded_shadow_case_exists,
    get_expanded_shadow_readiness,
    get_expanded_shadow_summary,
    record_expanded_shadow_observation,
)
from app.services.expanded_shadow_service import (
    evaluate_expanded_shadow,
    get_expanded_shadow_status,
)
from app.services.market_snapshot_store import (
    DEFAULT_MARKET_SNAPSHOT_STORE,
    get_research_market_snapshot,
    save_research_market_snapshot,
)
from app.services.pricing_service import (
    price_barrier_reverse_convertible_with_term_structure,
    price_phoenix_v3_with_term_structure,
)
from app.services.product_registry import REPO_ROOT
from app.services.research_market_data import get_research_market_data_service
from src.final.barrier_reverse_convertible import BarrierReverseConvertibleV1Contract
from src.final.market import EquityMarketSegment, EquityMarketTermStructure
from src.final.phoenix_contract import PhoenixSingleV3Contract


CAMPAIGN_VERSION = "expanded-shadow-out-of-time-v1"
CONTRACT_GRID_VERSION = "expanded-shadow-boundary-grid-v1"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "expanded_shadow_campaigns"
MINIMUM_REFERENCE_PATHS = 4_096


@dataclass(frozen=True)
class CampaignUnderlier:
    symbol: str
    underlier_type: str

    def __post_init__(self) -> None:
        symbol = self.symbol.strip().upper()
        underlier_type = self.underlier_type.strip().lower()
        if not symbol or len(symbol) > 32 or not symbol.isprintable():
            raise ValueError("campaign symbol is invalid")
        if underlier_type not in {"equity", "etf"}:
            raise ValueError("campaign underlier type must be equity or etf")
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "underlier_type", underlier_type)

    def to_dict(self) -> dict[str, str]:
        return {"symbol": self.symbol, "underlier_type": self.underlier_type}


DEFAULT_UNDERLIERS = (
    CampaignUnderlier("SPY", "etf"),
    CampaignUnderlier("QQQ", "etf"),
    CampaignUnderlier("IWM", "etf"),
    CampaignUnderlier("DIA", "etf"),
    CampaignUnderlier("XLK", "etf"),
    CampaignUnderlier("XLF", "etf"),
    CampaignUnderlier("GLD", "etf"),
    CampaignUnderlier("AAPL", "equity"),
    CampaignUnderlier("MSFT", "equity"),
    CampaignUnderlier("NVDA", "equity"),
    CampaignUnderlier("AMZN", "equity"),
    CampaignUnderlier("JPM", "equity"),
)


@dataclass(frozen=True)
class CampaignConfig:
    campaign_date: date
    underliers: tuple[CampaignUnderlier, ...] = DEFAULT_UNDERLIERS
    reference_paths: int = MINIMUM_REFERENCE_PATHS
    seed: int = 20260722
    calibration_maturity_years: float = 2.0
    require_research_ready: bool = True
    output_dir: Path = DEFAULT_OUTPUT_DIR
    monitoring_db: Path = DEFAULT_MONITORING_DB
    snapshot_db: Path = DEFAULT_MARKET_SNAPSHOT_STORE

    def __post_init__(self) -> None:
        if not isinstance(self.campaign_date, date):
            raise ValueError("campaign_date must be a date")
        if not 1 <= len(self.underliers) <= 50:
            raise ValueError("campaign must contain between 1 and 50 underliers")
        if len({item.symbol for item in self.underliers}) != len(self.underliers):
            raise ValueError("campaign underliers must be unique")
        if not MINIMUM_REFERENCE_PATHS <= self.reference_paths <= 20_000:
            raise ValueError("reference_paths must be between 4096 and 20000")
        if isinstance(self.seed, bool) or not 0 <= self.seed < 2**32:
            raise ValueError("campaign seed must be a uint32")
        if not math.isclose(self.calibration_maturity_years, 2.0, abs_tol=1e-12):
            raise ValueError("the v1 campaign calibration maturity must be 2 years")


@dataclass(frozen=True)
class CampaignCase:
    product_key: str
    template_id: str
    expected_region: str
    contract: PhoenixSingleV3Contract | BarrierReverseConvertibleV1Contract


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _digest(value: Any) -> str:
    return f"sha256:{hashlib.sha256(_canonical_bytes(value)).hexdigest()}"


def _even_schedule(maturity: float, count: int) -> tuple[float, ...]:
    values = [maturity * index / count for index in range(1, count + 1)]
    values[-1] = maturity
    return tuple(values)


def _linear_schedule(first: float, final: float, count: int) -> tuple[float, ...]:
    return tuple(float(value) for value in np.linspace(first, final, count))


def _phoenix_specs() -> tuple[dict[str, Any], ...]:
    return (
        {
            "id": "broad-high-short",
            "region": "broad",
            "spot": 1.35,
            "maturity": 0.5,
            "count": 2,
        },
        {
            "id": "broad-high-memory",
            "region": "broad",
            "spot": 1.27,
            "maturity": 1.0,
            "count": 4,
            "memory": True,
            "unpaid": 2,
        },
        {
            "id": "broad-low-seasoned",
            "region": "broad",
            "spot": 0.72,
            "maturity": 1.0,
            "count": 4,
            "prior_ki": True,
        },
        {
            "id": "broad-low-long",
            "region": "broad",
            "spot": 0.70,
            "maturity": 2.0,
            "count": 8,
            "memory": True,
            "unpaid": 1,
        },
        {
            "id": "first-autocall-standard",
            "region": "first_autocall",
            "spot": 1.10,
            "maturity": 1.0,
            "count": 4,
        },
        {
            "id": "first-autocall-low",
            "region": "first_autocall",
            "spot": 1.02,
            "maturity": 2.0,
            "count": 8,
            "first": 1.02,
            "final": 0.90,
        },
        {
            "id": "final-autocall-standard",
            "region": "final_autocall",
            "spot": 0.95,
            "maturity": 1.0,
            "count": 4,
        },
        {
            "id": "final-autocall-memory",
            "region": "final_autocall",
            "spot": 0.90,
            "maturity": 2.0,
            "count": 8,
            "first": 1.05,
            "final": 0.90,
            "memory": True,
            "unpaid": 3,
        },
        {
            "id": "coupon-standard",
            "region": "coupon",
            "spot": 0.80,
            "maturity": 0.5,
            "count": 2,
        },
        {
            "id": "coupon-memory",
            "region": "coupon",
            "spot": 0.85,
            "maturity": 1.0,
            "count": 4,
            "coupon_barrier": 0.85,
            "memory": True,
            "unpaid": 2,
        },
        {
            "id": "knock-in-standard",
            "region": "knock_in",
            "spot": 0.60,
            "maturity": 1.0,
            "count": 4,
        },
        {
            "id": "knock-in-seasoned",
            "region": "knock_in",
            "spot": 0.65,
            "maturity": 2.0,
            "count": 8,
            "knock_in": 0.65,
            "prior_ki": True,
        },
    )


def _brc_specs() -> tuple[dict[str, Any], ...]:
    return (
        {
            "id": "broad-high-short",
            "region": "broad",
            "spot": 1.35,
            "maturity": 0.5,
            "count": 2,
        },
        {
            "id": "broad-high-long",
            "region": "broad",
            "spot": 1.25,
            "maturity": 2.0,
            "count": 8,
        },
        {
            "id": "broad-low-standard",
            "region": "broad",
            "spot": 0.75,
            "maturity": 1.0,
            "count": 4,
        },
        {
            "id": "broad-low-seasoned",
            "region": "broad",
            "spot": 0.80,
            "maturity": 2.0,
            "count": 8,
            "prior_ki": True,
        },
        {
            "id": "strike-090",
            "region": "strike",
            "spot": 0.90,
            "strike": 0.90,
            "maturity": 0.5,
            "count": 2,
        },
        {
            "id": "strike-095",
            "region": "strike",
            "spot": 0.95,
            "strike": 0.95,
            "maturity": 1.0,
            "count": 4,
        },
        {
            "id": "strike-100",
            "region": "strike",
            "spot": 1.00,
            "strike": 1.00,
            "maturity": 1.0,
            "count": 4,
        },
        {
            "id": "strike-110",
            "region": "strike",
            "spot": 1.10,
            "strike": 1.10,
            "maturity": 2.0,
            "count": 8,
        },
        {
            "id": "knock-in-055",
            "region": "knock_in",
            "spot": 0.55,
            "knock_in": 0.55,
            "maturity": 0.5,
            "count": 2,
        },
        {
            "id": "knock-in-060",
            "region": "knock_in",
            "spot": 0.60,
            "knock_in": 0.60,
            "maturity": 1.0,
            "count": 4,
        },
        {
            "id": "knock-in-070",
            "region": "knock_in",
            "spot": 0.70,
            "knock_in": 0.70,
            "maturity": 2.0,
            "count": 8,
        },
        {
            "id": "knock-in-seasoned",
            "region": "knock_in",
            "spot": 0.65,
            "knock_in": 0.65,
            "maturity": 1.0,
            "count": 4,
            "prior_ki": True,
        },
    )


def build_campaign_cases(market: EquityMarketTermStructure) -> tuple[CampaignCase, ...]:
    cases: list[CampaignCase] = []
    for spec in _phoenix_specs():
        maturity = float(spec["maturity"])
        count = int(spec["count"])
        first = float(spec.get("first", 1.10))
        final = float(spec.get("final", 0.95))
        coupon_barrier = float(spec.get("coupon_barrier", 0.80))
        knock_in = float(spec.get("knock_in", 0.60))
        memory = bool(spec.get("memory", False))
        contract = PhoenixSingleV3Contract(
            reference_level=market.spot / float(spec["spot"]),
            maturity_years=maturity,
            observation_times_years=_even_schedule(maturity, count),
            autocall_barrier_fracs=_linear_schedule(first, final, count),
            coupon_barrier_frac=coupon_barrier,
            coupon_rate=0.02 if maturity <= 1.0 else 0.015,
            knock_in_frac=knock_in,
            prior_knock_in_breached=bool(spec.get("prior_ki", False)),
            memory_coupon=memory,
            unpaid_coupon_count=int(spec.get("unpaid", 0)) if memory else 0,
        )
        cases.append(
            CampaignCase("phoenix_v3", str(spec["id"]), str(spec["region"]), contract)
        )
    for spec in _brc_specs():
        maturity = float(spec["maturity"])
        count = int(spec["count"])
        contract = BarrierReverseConvertibleV1Contract(
            reference_level=market.spot / float(spec["spot"]),
            maturity_years=maturity,
            coupon_times_years=_even_schedule(maturity, count),
            coupon_rate_per_period=0.02 if count <= 4 else 0.0125,
            strike_frac=float(spec.get("strike", 1.0)),
            knock_in_frac=float(spec.get("knock_in", 0.60)),
            prior_knock_in_breached=bool(spec.get("prior_ki", False)),
        )
        cases.append(
            CampaignCase(
                "barrier_reverse_convertible",
                str(spec["id"]),
                str(spec["region"]),
                contract,
            )
        )
    return tuple(cases)


def campaign_plan(config: CampaignConfig) -> dict[str, Any]:
    dummy_time = datetime.combine(
        config.campaign_date, datetime.min.time(), tzinfo=timezone.utc
    )
    dummy_market = EquityMarketTermStructure(
        symbol="PLAN",
        underlier_type="etf",
        currency="USD",
        valuation_time=dummy_time,
        market_data_time=dummy_time,
        spot=100.0,
        segments=(EquityMarketSegment(2.0, 0.03, 0.01, 0.20),),
        calendar="WEEKDAYS",
        day_count="ACT/365F",
        source=CAMPAIGN_VERSION,
    )
    cases = build_campaign_cases(dummy_market)
    by_product: dict[str, dict[str, Any]] = {}
    for product in {case.product_key for case in cases}:
        product_cases = [case for case in cases if case.product_key == product]
        regions: dict[str, int] = {}
        for case in product_cases:
            regions[case.expected_region] = regions.get(case.expected_region, 0) + 1
        by_product[product] = {
            "cases_per_underlier": len(product_cases),
            "regions": regions,
        }
    return {
        "campaign_version": CAMPAIGN_VERSION,
        "contract_grid_version": CONTRACT_GRID_VERSION,
        "campaign_date": config.campaign_date.isoformat(),
        "reference_paths": config.reference_paths,
        "underliers": [item.to_dict() for item in config.underliers],
        "products": by_product,
        "total_cases": len(cases) * len(config.underliers),
    }


def _market_from_payload(payload: Mapping[str, Any]) -> EquityMarketTermStructure:
    def timestamp(name: str) -> datetime:
        return datetime.fromisoformat(str(payload[name]).replace("Z", "+00:00"))

    return EquityMarketTermStructure(
        symbol=str(payload["symbol"]),
        underlier_type=str(payload["underlier_type"]),
        currency=str(payload["currency"]),
        valuation_time=timestamp("valuation_time"),
        market_data_time=timestamp("market_data_time"),
        spot=float(payload["spot"]),
        segments=tuple(
            EquityMarketSegment(
                end_time_years=float(item["end_time_years"]),
                risk_free_rate=float(item["risk_free_rate"]),
                dividend_yield=float(item["dividend_yield"]),
                volatility=float(item["volatility"]),
            )
            for item in payload["segments"]
        ),
        calendar=str(payload["calendar"]),
        day_count=str(payload["day_count"]),
        source=str(payload["source"]),
    )


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _seed(config: CampaignConfig, symbol: str, case: CampaignCase) -> int:
    digest = hashlib.sha256(
        f"{config.seed}|{config.campaign_date}|{symbol}|{case.product_key}|{case.template_id}".encode()
    ).digest()
    return int.from_bytes(digest[:4], "big")


def _price_case(
    market: EquityMarketTermStructure,
    case: CampaignCase,
    *,
    reference_paths: int,
    seed: int,
) -> dict[str, Any]:
    if case.product_key == "phoenix_v3":
        return price_phoenix_v3_with_term_structure(
            market, case.contract, n_paths=reference_paths, seed=seed
        )
    return price_barrier_reverse_convertible_with_term_structure(
        market, case.contract, n_paths=reference_paths, seed=seed
    )


def _campaign_identity(config: CampaignConfig, plan: Mapping[str, Any]) -> str:
    runtime = get_expanded_shadow_status().get("products") or {}
    return _digest(
        {
            "campaign_version": CAMPAIGN_VERSION,
            "contract_grid_version": CONTRACT_GRID_VERSION,
            "campaign_date": config.campaign_date.isoformat(),
            "reference_paths": config.reference_paths,
            "seed": config.seed,
            "underliers": plan["underliers"],
            "artifacts": {
                key: value.get("artifact_id") for key, value in runtime.items()
            },
        }
    )


def _load_report(
    path: Path, campaign_id: str, plan: Mapping[str, Any]
) -> dict[str, Any]:
    if path.exists():
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if loaded.get("campaign_id") != campaign_id:
            raise RuntimeError("stored campaign report identity does not match")
        return loaded
    return {
        "schema_version": CAMPAIGN_VERSION,
        "campaign_id": campaign_id,
        "status": "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "plan": dict(plan),
        "markets": {},
        "results": {"recorded": 0, "already_recorded": 0, "failed": 0},
        "errors": [],
    }


def _frozen_market(
    underlier: CampaignUnderlier,
    *,
    config: CampaignConfig,
    report: dict[str, Any],
    market_service: Any,
) -> tuple[EquityMarketTermStructure, str]:
    stored = (report.get("markets") or {}).get(underlier.symbol) or {}
    snapshot_id = stored.get("snapshot_id")
    if snapshot_id:
        snapshot = get_research_market_snapshot(snapshot_id, db_path=config.snapshot_db)
        if snapshot is None:
            raise RuntimeError("campaign snapshot is missing from the snapshot store")
        return _market_from_payload(snapshot["market_term_structure"]), str(snapshot_id)
    if config.campaign_date != datetime.now(timezone.utc).date():
        raise RuntimeError(
            "a past campaign date can only resume from an already frozen snapshot"
        )
    built = market_service.build_term_structure(
        symbol=underlier.symbol,
        underlier_type=underlier.underlier_type,
        maturity_years=config.calibration_maturity_years,
    )
    quality = (built.calibration.get("quality") or {}).get("status")
    if config.require_research_ready and quality != "research_ready":
        raise RuntimeError(
            "research market calibration did not pass its quality checks"
        )
    metadata = save_research_market_snapshot(
        market=built.market.to_dict(),
        calibration=built.calibration,
        db_path=config.snapshot_db,
    )
    report["markets"][underlier.symbol] = {
        "snapshot_id": metadata["snapshot_id"],
        "term_structure_id": metadata["term_structure_id"],
        "market_data_time": metadata["market_data_time"],
        "quality": quality,
    }
    return built.market, str(metadata["snapshot_id"])


def run_campaign(
    config: CampaignConfig,
    *,
    market_service: Any | None = None,
) -> dict[str, Any]:
    plan = campaign_plan(config)
    campaign_id = _campaign_identity(config, plan)
    report_path = config.output_dir / (
        f"{config.campaign_date.isoformat()}-{campaign_id.removeprefix('sha256:')[:12]}.json"
    )
    report = _load_report(report_path, campaign_id, plan)
    service = market_service or get_research_market_data_service()
    started = time.perf_counter()
    for underlier in config.underliers:
        try:
            market, snapshot_id = _frozen_market(
                underlier, config=config, report=report, market_service=service
            )
            _atomic_json(report_path, report)
        except Exception as exc:
            report["results"]["failed"] += 1
            report["errors"].append(
                {"symbol": underlier.symbol, "stage": "market", "message": str(exc)}
            )
            report["updated_at"] = datetime.now(timezone.utc).isoformat()
            _atomic_json(report_path, report)
            continue
        for case in build_campaign_cases(market):
            case_id = _digest(
                {
                    "campaign_id": campaign_id,
                    "snapshot_id": snapshot_id,
                    "product_key": case.product_key,
                    "template_id": case.template_id,
                    "contract_id": case.contract.contract_id,
                }
            )
            if expanded_shadow_case_exists(case_id, db_path=config.monitoring_db):
                report["results"]["already_recorded"] += 1
                continue
            case_seed = _seed(config, underlier.symbol, case)
            try:
                reference = _price_case(
                    market,
                    case,
                    reference_paths=config.reference_paths,
                    seed=case_seed,
                )
                shadow = evaluate_expanded_shadow(
                    product_key=case.product_key,
                    market=market,
                    contract=case.contract,
                    reference_price=reference["price"],
                    reference_standard_error=reference["standard_error"],
                    reference_latency_ms=reference["latency_ms"],
                    force=True,
                )
                recorded = record_expanded_shadow_observation(
                    product_key=case.product_key,
                    market=market,
                    contract=case.contract,
                    reference_price=reference["price"],
                    reference_standard_error=reference["standard_error"],
                    reference_latency_ms=reference["latency_ms"],
                    shadow_result=shadow,
                    observation_source="out_of_time_campaign",
                    campaign_id=campaign_id,
                    case_id=case_id,
                    reference_paths=config.reference_paths,
                    reference_seed=case_seed,
                    market_snapshot_id=snapshot_id,
                    observation_id=case_id.removeprefix("sha256:"),
                    force=True,
                    db_path=config.monitoring_db,
                )
                if recorded:
                    report["results"]["recorded"] += 1
                elif expanded_shadow_case_exists(case_id, db_path=config.monitoring_db):
                    report["results"]["already_recorded"] += 1
                else:
                    raise RuntimeError("campaign observation was not persisted")
            except Exception as exc:
                report["results"]["failed"] += 1
                report["errors"].append(
                    {
                        "symbol": underlier.symbol,
                        "product_key": case.product_key,
                        "template_id": case.template_id,
                        "stage": "pricing",
                        "message": str(exc),
                    }
                )
        report["updated_at"] = datetime.now(timezone.utc).isoformat()
        _atomic_json(report_path, report)
    report["status"] = (
        "completed_with_errors" if report["results"]["failed"] else "completed"
    )
    report["duration_seconds"] = time.perf_counter() - started
    report["monitoring"] = get_expanded_shadow_summary(
        limit=100_000, db_path=config.monitoring_db
    )
    report["readiness"] = get_expanded_shadow_readiness(
        limit=100_000, db_path=config.monitoring_db
    )
    report["updated_at"] = datetime.now(timezone.utc).isoformat()
    _atomic_json(report_path, report)
    return {**report, "report_path": str(report_path)}


def _parse_underliers(raw_values: Sequence[str]) -> tuple[CampaignUnderlier, ...]:
    if not raw_values:
        return DEFAULT_UNDERLIERS
    parsed = []
    for raw in raw_values:
        symbol, separator, underlier_type = raw.partition(":")
        if not separator:
            raise ValueError("underliers must use SYMBOL:equity or SYMBOL:etf")
        parsed.append(CampaignUnderlier(symbol, underlier_type))
    return tuple(parsed)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign-date",
        type=date.fromisoformat,
        default=datetime.now(timezone.utc).date(),
    )
    parser.add_argument(
        "--underlier",
        action="append",
        default=[],
        help="SYMBOL:equity or SYMBOL:etf; repeat to override the default universe",
    )
    parser.add_argument("--reference-paths", type=int, default=MINIMUM_REFERENCE_PATHS)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--monitoring-db", type=Path, default=DEFAULT_MONITORING_DB)
    parser.add_argument(
        "--snapshot-db", type=Path, default=DEFAULT_MARKET_SNAPSHOT_STORE
    )
    parser.add_argument("--allow-review-required-market", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args(argv)
    config = CampaignConfig(
        campaign_date=args.campaign_date,
        underliers=_parse_underliers(args.underlier),
        reference_paths=args.reference_paths,
        seed=args.seed,
        output_dir=args.output_dir,
        monitoring_db=args.monitoring_db,
        snapshot_db=args.snapshot_db,
        require_research_ready=not args.allow_review_required_market,
    )
    if args.plan_only:
        print(json.dumps(campaign_plan(config), indent=2, sort_keys=True))
        return 0
    result = run_campaign(config)
    print(json.dumps(result, indent=2, sort_keys=True))
    return (
        0
        if result["results"]["recorded"] or result["results"]["already_recorded"]
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
