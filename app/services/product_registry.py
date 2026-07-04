import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

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
class ProductDefinition:
    key: str
    display_name: str
    payoff_class: type
    artifact_dir: str
    legacy_price_route_enabled: bool


PRODUCT_DEFINITIONS: tuple[ProductDefinition, ...] = (
    ProductDefinition(
        key="phoenix",
        display_name="Phoenix Autocallable",
        payoff_class=PhoenixPayoff,
        artifact_dir="phoenix",
        legacy_price_route_enabled=True,
    ),
    ProductDefinition(
        key="accumulator",
        display_name="Accumulator",
        payoff_class=AccumulatorPayoff,
        artifact_dir="accumulator",
        legacy_price_route_enabled=True,
    ),
    ProductDefinition(
        key="barrier",
        display_name="Barrier Option",
        payoff_class=BarrierOptionPayoff,
        artifact_dir="barrier",
        legacy_price_route_enabled=True,
    ),
    ProductDefinition(
        key="decumulator",
        display_name="Decumulator",
        payoff_class=DecumulatorPayoff,
        artifact_dir="decumulator",
        legacy_price_route_enabled=True,
    ),
    ProductDefinition(
        key="phoenix_stepdown",
        display_name="Step-Down Phoenix",
        payoff_class=StepDownPhoenixPayoff,
        artifact_dir="phoenix_stepdown",
        legacy_price_route_enabled=False,
    ),
    ProductDefinition(
        key="reverse_accumulator",
        display_name="Reverse Accumulator",
        payoff_class=ReverseAccumulatorPayoff,
        artifact_dir="reverse_accumulator",
        legacy_price_route_enabled=False,
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
        "payoff_class": product.payoff_class.__name__,
        "parameter_names": payoff.get_parameter_names(),
        "feature_order": payoff.get_feature_order(),
        "legacy_price_route_enabled": product.legacy_price_route_enabled,
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
