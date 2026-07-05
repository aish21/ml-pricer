import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.final.inherited_payoffs import ReverseAccumulatorPayoff, StepDownPhoenixPayoff
from src.final.payoffs import (
    AccumulatorPayoff,
    BarrierOptionPayoff,
    DecumulatorPayoff,
    PhoenixPayoff,
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
    choices: Tuple[Tuple[str, str], ...] = ()


@dataclass(frozen=True)
class ProductDefinition:
    key: str
    display_name: str
    terminal_label: str
    payoff_class: type
    artifact_dir: str
    legacy_price_route_enabled: bool
    enabled_for_bb: bool
    bb_fields: Tuple[ProductField, ...]


PHOENIX_FIELDS: tuple[ProductField, ...] = (
    ProductField("S0", "Spot", "float", 100.0, min_value=0.000001),
    ProductField("sigma", "Vol", "float", 0.2, min_value=0.000001),
    ProductField("r", "Rate", "float", 0.03, min_value=0.0),
    ProductField("T", "Mat", "float", 1.0, min_value=0.000001),
    ProductField("autocall_barrier_frac", "AutoB", "float", 1.05, min_value=0.0),
    ProductField("coupon_barrier_frac", "CpnB", "float", 1.0, min_value=0.0),
    ProductField("coupon_rate", "Cpn", "float", 0.02, min_value=0.0),
    ProductField("knock_in_frac", "KI", "float", 0.7, min_value=0.0),
    ProductField("obs_count", "Obs", "int", 6, min_value=1),
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


PRODUCT_DEFINITIONS: tuple[ProductDefinition, ...] = (
    ProductDefinition(
        key="phoenix",
        display_name="Phoenix Autocallable",
        terminal_label="PHOENIX",
        payoff_class=PhoenixPayoff,
        artifact_dir="phoenix",
        legacy_price_route_enabled=True,
        enabled_for_bb=True,
        bb_fields=PHOENIX_FIELDS,
    ),
    ProductDefinition(
        key="accumulator",
        display_name="Accumulator",
        terminal_label="ACCUM",
        payoff_class=AccumulatorPayoff,
        artifact_dir="accumulator",
        legacy_price_route_enabled=True,
        enabled_for_bb=True,
        bb_fields=ACCUMULATOR_FIELDS,
    ),
    ProductDefinition(
        key="barrier",
        display_name="Barrier Option",
        terminal_label="BARRIER",
        payoff_class=BarrierOptionPayoff,
        artifact_dir="barrier",
        legacy_price_route_enabled=True,
        enabled_for_bb=True,
        bb_fields=BARRIER_FIELDS,
    ),
    ProductDefinition(
        key="decumulator",
        display_name="Decumulator",
        terminal_label="DECUM",
        payoff_class=DecumulatorPayoff,
        artifact_dir="decumulator",
        legacy_price_route_enabled=True,
        enabled_for_bb=True,
        bb_fields=ACCUMULATOR_FIELDS,
    ),
    ProductDefinition(
        key="phoenix_stepdown",
        display_name="Step-Down Phoenix",
        terminal_label="STEP-PHX",
        payoff_class=StepDownPhoenixPayoff,
        artifact_dir="phoenix_stepdown",
        legacy_price_route_enabled=False,
        enabled_for_bb=True,
        bb_fields=PHOENIX_FIELDS,
    ),
    ProductDefinition(
        key="reverse_accumulator",
        display_name="Reverse Accumulator",
        terminal_label="REV-ACC",
        payoff_class=ReverseAccumulatorPayoff,
        artifact_dir="reverse_accumulator",
        legacy_price_route_enabled=False,
        enabled_for_bb=True,
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
    has_results = (product_dir / "results.json").exists()

    return {
        "artifact_dir": product.artifact_dir,
        "model_available": has_model,
        "scaler_available": has_scaler,
        "training_metadata_available": has_results,
        "ready_for_surrogate": has_model and has_scaler,
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
        "parameter_names": payoff.get_parameter_names(),
        "feature_order": payoff.get_feature_order(),
        "legacy_price_route_enabled": product.legacy_price_route_enabled,
        "enabled_for_bb": product.enabled_for_bb,
        "bb_fields": [
            {
                "name": field.name,
                "label": field.label,
                "type": field.field_type,
                "default": field.default,
                "min_value": field.min_value,
                "choices": [
                    {"value": value, "label": label}
                    for value, label in field.choices
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
        if product["artifacts"]["ready_for_surrogate"]
    ]


def get_model_info(results_dir: Optional[Path] = None) -> Dict[str, Any]:
    products = list_products(results_dir)
    return {
        "service": "ml-pricer",
        "api": "online",
        "model_family": "LightGBM surrogate",
        "monte_carlo_fallback": "available_via_backend_evaluator",
        "supported_product_keys": [product["key"] for product in products],
        "available_product_keys": available_product_keys(products),
        "products": products,
    }
