from fastapi import APIRouter
from fastapi.responses import PlainTextResponse

from app.services.model_cache import get_model_cache_status
from app.services.product_registry import get_model_info


router = APIRouter(prefix="/api/bb", tags=["api-bb"])


def _product_state(product: dict, cache_status: dict[str, bool]) -> tuple[str, str]:
    artifacts = product["artifacts"]
    if not product.get("enabled_for_bb"):
        return "NO-BB", "-"
    if (
        product.get("reference_pricing_available")
        and not artifacts["ready_for_surrogate"]
    ):
        return "REF", "-"
    if not artifacts["ready_for_surrogate"]:
        return "UNAVAIL", "-"
    return "READY", "CACHED" if cache_status.get(product["key"]) else "COLD"


@router.get("/ping", response_class=PlainTextResponse)
def ping():
    return "OK\nSERVICE=ASHBERRY\n"


@router.get("/model-status", response_class=PlainTextResponse)
def model_status():
    info = get_model_info()
    cache_status = get_model_cache_status()
    lines = ["OK"]
    for product in info["products"]:
        if not product.get("enabled_for_bb"):
            continue
        state, cache = _product_state(product, cache_status)
        label = product.get("terminal_label") or product["key"].upper()
        lines.append(f"{label}={state},{cache}")
    return "\n".join(lines) + "\n"


@router.get("/products", response_class=PlainTextResponse)
def products():
    info = get_model_info()
    lines = ["OK"]
    for product in info["products"]:
        if not product.get("enabled_for_bb"):
            continue
        lines.append(
            "|".join(
                [
                    product["key"],
                    product["display_name"],
                    product.get("terminal_label") or product["key"].upper(),
                ]
            )
        )
    return "\n".join(lines) + "\n"
