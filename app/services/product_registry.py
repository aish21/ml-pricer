import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.final.market import (
    EQUITY_GBM_FLAT_MODEL_VERSION,
    EQUITY_GBM_PIECEWISE_MODEL_VERSION,
    EQUITY_MARKET_SNAPSHOT_VERSION,
    EQUITY_MARKET_SCENARIO_VERSION,
    EQUITY_MARKET_TERM_STRUCTURE_VERSION,
    EQUITY_RESEARCH_MARKET_VERSION,
    EQUITY_RISK_ANALYTICS_VERSION,
)
from src.final.barrier_reverse_convertible import BarrierReverseConvertiblePayoff
from src.final.inherited_payoffs import ReverseAccumulatorPayoff, StepDownPhoenixPayoff
from src.final.payoffs import (
    AccumulatorPayoff,
    BarrierOptionPayoff,
    DecumulatorPayoff,
    PhoenixPayoff,
)
from src.final.phoenix_contract import (
    PHOENIX_SINGLE_V2_CONTRACT_VERSION,
    PHOENIX_SINGLE_V3_CONTRACT_VERSION,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = REPO_ROOT / "final" / "results"


@dataclass(frozen=True)
class ProductField:
    name: str
    label: str
    field_type: str
    default: Any
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    choices: Tuple[Tuple[str, str], ...] = ()


@dataclass(frozen=True)
class ProductDefinition:
    key: str
    display_name: str
    terminal_label: str
    payoff_class: type
    contract_version: str
    artifact_dir: str
    validated_for_pricing: bool
    reference_pricing_enabled: bool
    legacy_price_route_enabled: bool
    enabled_for_bb: bool
    bb_fields: Tuple[ProductField, ...]
    additional_contract_versions: Tuple[str, ...] = ()
    market_snapshot_pricing_enabled: bool = False
    research_market_pricing_enabled: bool = False
    scenario_analytics_enabled: bool = False
    risk_analytics_enabled: bool = False


PHOENIX_FIELDS: tuple[ProductField, ...] = (
    ProductField(
        "S0",
        "Spot",
        "float",
        100.0,
        min_value=0.000001,
        max_value=1_000_000_000.0,
    ),
    ProductField("sigma", "Vol", "float", 0.2, min_value=0.000001, max_value=5.0),
    ProductField("r", "Rate", "float", 0.03, min_value=-0.25, max_value=1.0),
    ProductField("T", "Mat", "float", 1.0, min_value=0.000001, max_value=30.0),
    ProductField(
        "autocall_barrier_frac",
        "AutoB",
        "float",
        1.05,
        min_value=0.000001,
        max_value=3.0,
    ),
    ProductField(
        "coupon_barrier_frac", "CpnB", "float", 1.0, min_value=0.000001, max_value=3.0
    ),
    ProductField("coupon_rate", "Cpn", "float", 0.02, min_value=0.0, max_value=1.0),
    ProductField(
        "knock_in_frac", "KI", "float", 0.7, min_value=0.000001, max_value=1.0
    ),
    ProductField("obs_count", "Obs", "int", 6, min_value=1, max_value=252),
)

ACCUMULATOR_FIELDS: tuple[ProductField, ...] = (
    ProductField("S0", "Spot", "float", 100.0, min_value=0.000001),
    ProductField("sigma", "Vol", "float", 0.2, min_value=0.000001),
    ProductField("r", "Rate", "float", 0.03, min_value=0.0),
    ProductField("T", "Mat", "float", 1.0, min_value=0.000001),
    ProductField("upper_barrier_frac", "UpB", "float", 1.05, min_value=0.0),
    ProductField("lower_barrier_frac", "LoB", "float", 0.95, min_value=0.0),
    ProductField("participation_rate", "Part", "float", 2.0, min_value=0.0),
    ProductField("obs_frequency", "ObsF", "float", 0.25, min_value=0.000001),
)

BARRIER_FIELDS: tuple[ProductField, ...] = (
    ProductField("S0", "Spot", "float", 100.0, min_value=0.000001),
    ProductField("sigma", "Vol", "float", 0.2, min_value=0.000001),
    ProductField("r", "Rate", "float", 0.03, min_value=0.0),
    ProductField("T", "Mat", "float", 1.0, min_value=0.000001),
    ProductField("K", "Strike", "float", 100.0, min_value=0.0),
    ProductField("barrier_frac", "Barr", "float", 0.8, min_value=0.0),
    ProductField(
        "option_type",
        "Type",
        "choice",
        "1.0",
        choices=(("1.0", "Call"), ("0.0", "Put")),
    ),
)

BARRIER_REVERSE_CONVERTIBLE_FIELDS: tuple[ProductField, ...] = (
    ProductField("S0", "Reference", "float", 100.0, min_value=0.000001),
    ProductField("sigma", "Vol", "float", 0.2, min_value=0.000001, max_value=5.0),
    ProductField("r", "Rate", "float", 0.03, min_value=-0.25, max_value=1.0),
    ProductField("T", "Mat", "float", 1.0, min_value=0.01, max_value=30.0),
    ProductField(
        "coupon_rate_per_period",
        "Cpn",
        "float",
        0.02,
        min_value=0.0,
        max_value=1.0,
    ),
    ProductField("strike_frac", "Strike", "float", 1.0, min_value=0.01, max_value=3.0),
    ProductField(
        "knock_in_frac",
        "KI",
        "float",
        0.7,
        min_value=0.000001,
        max_value=1.0,
    ),
    ProductField("obs_count", "Obs", "int", 4, min_value=1, max_value=252),
)


PRODUCT_DEFINITIONS: tuple[ProductDefinition, ...] = (
    ProductDefinition(
        key="phoenix",
        display_name="Phoenix Autocallable",
        terminal_label="PHOENIX",
        payoff_class=PhoenixPayoff,
        contract_version=PhoenixPayoff.contract_version,
        artifact_dir="phoenix",
        validated_for_pricing=True,
        reference_pricing_enabled=True,
        legacy_price_route_enabled=True,
        enabled_for_bb=True,
        bb_fields=PHOENIX_FIELDS,
        additional_contract_versions=(
            PHOENIX_SINGLE_V2_CONTRACT_VERSION,
            PHOENIX_SINGLE_V3_CONTRACT_VERSION,
        ),
        market_snapshot_pricing_enabled=True,
        research_market_pricing_enabled=True,
        scenario_analytics_enabled=True,
        risk_analytics_enabled=True,
    ),
    ProductDefinition(
        key="barrier_reverse_convertible",
        display_name="Barrier Reverse Convertible",
        terminal_label="BRC",
        payoff_class=BarrierReverseConvertiblePayoff,
        contract_version=BarrierReverseConvertiblePayoff.contract_version,
        artifact_dir="barrier_reverse_convertible",
        validated_for_pricing=True,
        reference_pricing_enabled=True,
        legacy_price_route_enabled=False,
        enabled_for_bb=False,
        bb_fields=BARRIER_REVERSE_CONVERTIBLE_FIELDS,
        research_market_pricing_enabled=True,
    ),
    ProductDefinition(
        key="accumulator",
        display_name="Accumulator",
        terminal_label="ACCUM",
        payoff_class=AccumulatorPayoff,
        contract_version=AccumulatorPayoff.contract_version,
        artifact_dir="accumulator",
        validated_for_pricing=False,
        reference_pricing_enabled=False,
        legacy_price_route_enabled=True,
        enabled_for_bb=False,
        bb_fields=ACCUMULATOR_FIELDS,
    ),
    ProductDefinition(
        key="barrier",
        display_name="Barrier Option",
        terminal_label="BARRIER",
        payoff_class=BarrierOptionPayoff,
        contract_version=BarrierOptionPayoff.contract_version,
        artifact_dir="barrier",
        validated_for_pricing=False,
        reference_pricing_enabled=False,
        legacy_price_route_enabled=True,
        enabled_for_bb=False,
        bb_fields=BARRIER_FIELDS,
    ),
    ProductDefinition(
        key="decumulator",
        display_name="Decumulator",
        terminal_label="DECUM",
        payoff_class=DecumulatorPayoff,
        contract_version=DecumulatorPayoff.contract_version,
        artifact_dir="decumulator",
        validated_for_pricing=False,
        reference_pricing_enabled=False,
        legacy_price_route_enabled=True,
        enabled_for_bb=False,
        bb_fields=ACCUMULATOR_FIELDS,
    ),
    ProductDefinition(
        key="phoenix_stepdown",
        display_name="Step-Down Phoenix",
        terminal_label="STEP-PHX",
        payoff_class=StepDownPhoenixPayoff,
        contract_version=StepDownPhoenixPayoff.contract_version,
        artifact_dir="phoenix_stepdown",
        validated_for_pricing=False,
        reference_pricing_enabled=False,
        legacy_price_route_enabled=False,
        enabled_for_bb=False,
        bb_fields=PHOENIX_FIELDS,
    ),
    ProductDefinition(
        key="reverse_accumulator",
        display_name="Reverse Accumulator",
        terminal_label="REV-ACC",
        payoff_class=ReverseAccumulatorPayoff,
        contract_version=ReverseAccumulatorPayoff.contract_version,
        artifact_dir="reverse_accumulator",
        validated_for_pricing=False,
        reference_pricing_enabled=False,
        legacy_price_route_enabled=False,
        enabled_for_bb=False,
        bb_fields=ACCUMULATOR_FIELDS,
    ),
)


def get_results_dir() -> Path:
    configured = os.getenv("MODEL_RESULTS_DIR")
    return Path(configured) if configured else DEFAULT_RESULTS_DIR


def get_product_definitions() -> List[ProductDefinition]:
    return list(PRODUCT_DEFINITIONS)


def get_product_definition(key: str) -> Optional[ProductDefinition]:
    normalized = key.strip().lower()
    for product in PRODUCT_DEFINITIONS:
        if product.key == normalized:
            return product
    return None


def get_bb_product_definitions() -> List[ProductDefinition]:
    return [product for product in PRODUCT_DEFINITIONS if product.enabled_for_bb]


def is_product_enabled_for_bb(key: str) -> bool:
    product = get_product_definition(key)
    return bool(product and product.enabled_for_bb)


def build_artifact_status(
    product: ProductDefinition, results_dir: Optional[Path] = None
) -> Dict[str, Any]:
    base_dir = Path(results_dir) if results_dir else get_results_dir()
    product_dir = base_dir / product.artifact_dir
    has_model = (product_dir / "model.joblib").exists()
    has_scaler = (product_dir / "scaler.joblib").exists()
    results_path = product_dir / "results.json"
    has_results = results_path.exists()
    metadata_contract_version = None
    metadata_feature_order = None
    metadata_valid = False
    if has_results:
        try:
            metadata = json.loads(results_path.read_text(encoding="utf-8"))
            config = metadata.get("config") or {}
            metadata_contract_version = config.get("contract_version")
            metadata_feature_order = config.get("feature_order")
            metadata_valid = isinstance(config, dict)
        except (AttributeError, OSError, ValueError, TypeError):
            metadata_valid = False

    expected_feature_order = product.payoff_class().get_feature_order()
    artifact_compatible = bool(
        metadata_valid
        and metadata_contract_version == product.contract_version
        and metadata_feature_order == expected_feature_order
    )

    return {
        "artifact_dir": product.artifact_dir,
        "model_available": has_model,
        "scaler_available": has_scaler,
        "training_metadata_available": has_results,
        "metadata_contract_version": metadata_contract_version,
        "expected_contract_version": product.contract_version,
        "artifact_compatible": artifact_compatible,
        "ready_for_surrogate": has_model and has_scaler and artifact_compatible,
    }


def build_product_status(
    product: ProductDefinition, results_dir: Optional[Path] = None
) -> Dict[str, Any]:
    payoff = product.payoff_class()
    return {
        "key": product.key,
        "display_name": product.display_name,
        "terminal_label": product.terminal_label,
        "payoff_class": product.payoff_class.__name__,
        "contract_version": product.contract_version,
        "contract_versions": [
            product.contract_version,
            *product.additional_contract_versions,
        ],
        "parameter_names": payoff.get_parameter_names(),
        "feature_order": payoff.get_feature_order(),
        "legacy_price_route_enabled": product.legacy_price_route_enabled,
        "validated_for_pricing": product.validated_for_pricing,
        "reference_pricing_available": product.reference_pricing_enabled,
        "market_snapshot_versions": (
            [EQUITY_MARKET_SNAPSHOT_VERSION]
            if product.market_snapshot_pricing_enabled
            else []
        ),
        "market_term_structure_versions": (
            [EQUITY_MARKET_TERM_STRUCTURE_VERSION]
            if product.reference_pricing_enabled
            else []
        ),
        "research_market_versions": (
            [EQUITY_RESEARCH_MARKET_VERSION]
            if product.research_market_pricing_enabled
            else []
        ),
        "scenario_versions": (
            [EQUITY_MARKET_SCENARIO_VERSION]
            if product.scenario_analytics_enabled
            else []
        ),
        "risk_analytics_versions": (
            [EQUITY_RISK_ANALYTICS_VERSION] if product.risk_analytics_enabled else []
        ),
        "market_model_versions": (
            (
                ["gbm-flat-v1", EQUITY_GBM_FLAT_MODEL_VERSION]
                if product.market_snapshot_pricing_enabled
                else []
            )
            + [EQUITY_GBM_PIECEWISE_MODEL_VERSION]
            if product.reference_pricing_enabled
            else []
        ),
        "enabled_for_bb": product.enabled_for_bb,
        "bb_fields": [
            {
                "name": field.name,
                "label": field.label,
                "type": field.field_type,
                "default": field.default,
                "min_value": field.min_value,
                "max_value": field.max_value,
                "choices": [
                    {"value": value, "label": label} for value, label in field.choices
                ],
            }
            for field in product.bb_fields
        ],
        "artifacts": build_artifact_status(product, results_dir),
    }


def list_products(results_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    return [
        build_product_status(product, results_dir)
        for product in get_product_definitions()
    ]


def available_product_keys(products: Iterable[Dict[str, Any]]) -> List[str]:
    return [
        product["key"]
        for product in products
        if product["validated_for_pricing"]
        and (
            product["reference_pricing_available"]
            or product["artifacts"]["ready_for_surrogate"]
        )
    ]


def get_model_info(results_dir: Optional[Path] = None) -> Dict[str, Any]:
    products = list_products(results_dir)
    return {
        "service": "ml-pricer",
        "api": "online",
        "model_family": "Monte Carlo reference",
        "market_snapshot_versions": [EQUITY_MARKET_SNAPSHOT_VERSION],
        "market_term_structure_versions": [EQUITY_MARKET_TERM_STRUCTURE_VERSION],
        "research_market_versions": [EQUITY_RESEARCH_MARKET_VERSION],
        "scenario_versions": [EQUITY_MARKET_SCENARIO_VERSION],
        "risk_analytics_versions": [EQUITY_RISK_ANALYTICS_VERSION],
        "market_model_versions": [
            "gbm-flat-v1",
            EQUITY_GBM_FLAT_MODEL_VERSION,
            EQUITY_GBM_PIECEWISE_MODEL_VERSION,
        ],
        "monte_carlo_fallback": "available_via_backend_evaluator",
        "supported_product_keys": [
            product["key"] for product in products if product["validated_for_pricing"]
        ],
        "research_product_keys": [
            product["key"]
            for product in products
            if not product["validated_for_pricing"]
        ],
        "available_product_keys": available_product_keys(products),
        "products": products,
    }
