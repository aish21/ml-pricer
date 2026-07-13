"""Public interfaces for the validated pricer and optional research tooling.

Exports are resolved lazily so importing the NumPy reference pricer does not
also import the optional LightGBM/Optuna training stack.
"""

from importlib import import_module
from typing import Dict, Tuple

__all__ = [
    "BasePayoff",
    "PhoenixPayoff",
    "AccumulatorPayoff",
    "BarrierOptionPayoff",
    "DecumulatorPayoff",
    "DataGenerator",
    "ModelTrainer",
    "Evaluator",
    "PricingPipeline",
    "price_reference",
]

_EXPORTS: Dict[str, Tuple[str, str]] = {
    "BasePayoff": (".payoffs", "BasePayoff"),
    "PhoenixPayoff": (".payoffs", "PhoenixPayoff"),
    "AccumulatorPayoff": (".payoffs", "AccumulatorPayoff"),
    "BarrierOptionPayoff": (".payoffs", "BarrierOptionPayoff"),
    "DecumulatorPayoff": (".payoffs", "DecumulatorPayoff"),
    "DataGenerator": (".data_generator", "DataGenerator"),
    "ModelTrainer": (".model_trainer", "ModelTrainer"),
    "Evaluator": (".evaluator", "Evaluator"),
    "PricingPipeline": (".pipeline", "PricingPipeline"),
    "price_reference": (".reference_pricer", "price_reference"),
}


def __getattr__(name: str):
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value
