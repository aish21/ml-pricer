from fastapi import APIRouter

from app.services.product_registry import get_model_info, list_products


router = APIRouter(prefix="/api/v1", tags=["api-v1"])


@router.get("/products")
def products():
    return {
        "status": "success",
        "products": list_products(),
    }


@router.get("/model-info")
def model_info():
    return {
        "status": "success",
        "model_info": get_model_info(),
    }
