"""Package expanded-product research models for the safe shadow runtime.

This command is deliberately a training-side operation.  It reads the trusted
local joblib produced by the experiment, exports only numerical LightGBM tree
data, proves prediction parity, and writes a checksummed runtime package.  The
API never imports joblib or LightGBM.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO_ROOT / "final" / "research_candidates"
OUTPUT_ROOT = REPO_ROOT / "final" / "shadow_artifacts"
ARTIFACT_SCHEMA = "expanded-shadow-artifact-v1"
REGISTRY_SCHEMA = "expanded-shadow-registry-v1"

PRODUCTS = {
    "phoenix_v3": {
        "contract_version": "phoenix-single-v3",
        "experiment_id": "sha256:06ae0c3d3b1f8dc42bd8d441ce9bab5bc90823fb4452710782e86c14d616c114",
        "domain": {
            "spot_to_reference": [0.55, 1.35],
            "risk_free_rate": [0.0, 0.07],
            "dividend_yield": [0.0, 0.04],
            "volatility": [0.10, 0.50],
            "maturity_years": [0.50, 2.0],
            "first_autocall_barrier_frac": [0.98, 1.20],
            "final_autocall_barrier_frac": [0.65, 1.20],
            "coupon_barrier_frac": [0.65, 1.05],
            "coupon_rate": [0.005, 0.04],
            "knock_in_frac": [0.45, 0.85],
            "observation_count": [2.0, 8.0],
            "memory_coupon": [0.0, 1.0],
            "unpaid_coupon_count": [0.0, 3.0],
            "prior_knock_in_breached": [0.0, 1.0],
        },
        "schedule_policy": "even-observations-linear-autocall-stepdown",
    },
    "barrier_reverse_convertible": {
        "contract_version": "barrier-reverse-convertible-v1",
        "experiment_id": "sha256:fdd30cb2852928faa9f88a46c1deecc908ba4c0651a86368d23fc0e284ef0606",
        "domain": {
            "spot_to_reference": [0.55, 1.35],
            "risk_free_rate": [0.0, 0.07],
            "dividend_yield": [0.0, 0.04],
            "volatility": [0.10, 0.50],
            "maturity_years": [0.25, 2.0],
            "coupon_rate_per_period": [0.005, 0.04],
            "strike_frac": [0.90, 1.10],
            "knock_in_frac": [0.45, 0.90],
            "coupon_count": [1.0, 8.0],
            "prior_knock_in_breached": [0.0, 1.0],
        },
        "schedule_policy": "even-fixed-coupon-observations",
    },
}


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256(data: bytes) -> str:
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def _compact_node(node: dict[str, Any]) -> dict[str, Any]:
    if "leaf_value" in node:
        return {"leaf": float(node["leaf_value"])}
    if node.get("decision_type") != "<=":
        raise ValueError("only numerical <= LightGBM splits are supported")
    if node.get("missing_type") not in {"None", "NaN", "Zero"}:
        raise ValueError("unsupported LightGBM missing-value policy")
    return {
        "feature": int(node["split_feature"]),
        "threshold": float(node["threshold"]),
        "default_left": bool(node["default_left"]),
        "left": _compact_node(node["left_child"]),
        "right": _compact_node(node["right_child"]),
    }


def _predict_tree(node: dict[str, Any], row: np.ndarray) -> float:
    while "leaf" not in node:
        value = float(row[node["feature"]])
        go_left = (
            node["default_left"] if np.isnan(value) else value <= node["threshold"]
        )
        node = node["left"] if go_left else node["right"]
    return float(node["leaf"])


def _predict(payload: dict[str, Any], matrix: np.ndarray) -> np.ndarray:
    return np.asarray(
        [sum(_predict_tree(tree, row) for tree in payload["trees"]) for row in matrix],
        dtype=np.float64,
    )


def _source_dir(product_key: str, experiment_id: str) -> Path:
    return SOURCE_ROOT / product_key / experiment_id.removeprefix("sha256:")


def package_product(
    product_key: str, output_root: Path = OUTPUT_ROOT
) -> dict[str, Any]:
    policy = PRODUCTS[product_key]
    source = _source_dir(product_key, policy["experiment_id"])
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("experiment_id") != policy["experiment_id"]:
        raise ValueError(f"{product_key}: source experiment ID does not match the pin")
    if manifest.get("contract_version") != policy["contract_version"]:
        raise ValueError(f"{product_key}: source contract version does not match")
    if not (manifest.get("sealed_audit") or {}).get("passed"):
        raise ValueError(f"{product_key}: sealed audit did not pass")

    source_model = source / "model.joblib"
    model = joblib.load(source_model)
    dump = model.booster_.dump_model()
    feature_order = list(manifest["configuration"]["feature_order"])
    if int(dump["max_feature_idx"]) + 1 != len(feature_order):
        raise ValueError(f"{product_key}: model feature count does not match manifest")
    if dump.get("num_class") != 1 or dump.get("num_tree_per_iteration") != 1:
        raise ValueError(f"{product_key}: only scalar regression models are supported")

    payload = {
        "schema_version": ARTIFACT_SCHEMA,
        "product_key": product_key,
        "contract_version": policy["contract_version"],
        "feature_order": feature_order,
        "trees": [_compact_node(item["tree_structure"]) for item in dump["tree_info"]],
    }
    model_bytes = _canonical_bytes(payload)
    model_checksum = _sha256(model_bytes)

    rng = np.random.default_rng(20260722)
    parity_matrix = rng.normal(size=(128, len(feature_order)))
    source_predictions = np.asarray(model.predict(parity_matrix), dtype=np.float64)
    runtime_predictions = _predict(payload, parity_matrix)
    max_parity_error = float(np.max(np.abs(source_predictions - runtime_predictions)))
    if max_parity_error > 1e-12:
        raise ValueError(f"{product_key}: runtime export failed prediction parity")

    artifact_id = _sha256(
        _canonical_bytes(
            {
                "schema_version": ARTIFACT_SCHEMA,
                "experiment_id": policy["experiment_id"],
                "model_checksum": model_checksum,
                "contract_version": policy["contract_version"],
                "feature_order": feature_order,
            }
        )
    )
    target = output_root / product_key / artifact_id.removeprefix("sha256:")
    target.mkdir(parents=True, exist_ok=True)
    with (target / "model.json.gz").open("wb") as raw_handle:
        with gzip.GzipFile(fileobj=raw_handle, mode="wb", mtime=0) as handle:
            handle.write(model_bytes)

    runtime_manifest = {
        "schema_version": ARTIFACT_SCHEMA,
        "artifact_id": artifact_id,
        "product_key": product_key,
        "contract_version": policy["contract_version"],
        "source_experiment_id": policy["experiment_id"],
        "source_manifest_checksum": _sha256((source / "manifest.json").read_bytes()),
        "model_checksum": model_checksum,
        "feature_order": feature_order,
        "feature_domain": policy["domain"],
        "schedule_policy": policy["schedule_policy"],
        "tree_count": len(payload["trees"]),
        "model_family": manifest["configuration"]["learner"]["class"],
        "selected_candidate": manifest["configuration"]["learner"][
            "selected_candidate"
        ],
        "training_library_version": manifest["configuration"]["learner"][
            "library_version"
        ],
        "parity": {"samples": 128, "maximum_absolute_error": max_parity_error},
        "sealed_audit": manifest["sealed_audit"],
        "shadow_eligible": True,
        "runtime_approved": False,
        "runtime_policy": "shadow-only",
    }
    (target / "manifest.json").write_text(
        json.dumps(runtime_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return runtime_manifest


def package_all(output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    artifacts = {key: package_product(key, output_root) for key in PRODUCTS}
    registry = {
        "schema_version": REGISTRY_SCHEMA,
        "runtime_policy": "shadow-only",
        "automatic_promotion_permitted": False,
        "artifacts": {
            key: {
                "artifact_id": value["artifact_id"],
                "contract_version": value["contract_version"],
            }
            for key, value in artifacts.items()
        },
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "registry.json").write_text(
        json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return registry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args(argv)
    registry = package_all(args.output)
    print(json.dumps(registry, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
