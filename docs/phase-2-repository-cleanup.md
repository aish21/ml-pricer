# Phase 2 Repository Cleanup

This phase makes a clean checkout reproducible without treating generated
research output as application source. It deliberately separates safe runtime
cleanup from destructive artifact migration.

## Completed foundation

- `pyproject.toml` is the dependency and package source of truth.
- API, frontend, surrogate, training, research, and test dependencies are
  separate pinned groups.
- Importing the reference pricer no longer imports LightGBM, Optuna, or
  scikit-learn.
- Production containers install only their required dependency groups.
- CI runs static checks, tests, a package build, Compose validation, and both
  container builds.
- Liveness and readiness endpoints are available for deployment probes.
- New generated data, local histories, and model results are ignored.

The direct pins currently preserve the repository's established dependency
baseline; they are not a 2026 dependency upgrade. Before any public deployment,
upgrade each runtime group in an isolated pull request, regenerate a transitive
lock, run a vulnerability audit, and repeat the numerical golden tests. Monthly
Dependabot checks are enabled to make that work visible.

## Existing generated files

The repository still contains tracked datasets, tuner output, plots, and model
artifacts from earlier experiments. They remain visible to Git even though the
paths are now ignored, because ignore rules do not affect files already
tracked.

Do not remove those files until there is a verified durable copy. In
particular, the models under `final/results/` are already incompatible with
`phoenix-single-v1`, but they may still be useful for research provenance.

## Target ownership

Keep in Git:

- application and library source;
- versioned payoff and market-model specifications;
- small deterministic test fixtures and golden regression cases;
- artifact manifests, schemas, and reproducibility scripts;
- documentation and infrastructure definitions.

Store outside normal Git:

- generated training and evaluation datasets;
- model/scaler binaries;
- Optuna databases and tuner state;
- generated plots, reports, and notebook output;
- local API history and run databases.

An object store is the simplest production target. DVC is also reasonable if
local research workflows need transparent dataset checkout, but the API should
load immutable artifact URIs from a manifest rather than depend on DVC at
runtime.

## Required artifact manifest

Each promoted model should record at least:

- artifact identifier and immutable URI;
- SHA-256 checksums for every file;
- product key and contract version;
- market-model and feature-schema versions;
- exact feature order;
- training-data snapshot identifier;
- source Git commit and build timestamp;
- dependency/runtime version;
- validation metrics and promotion status.

Only an artifact whose manifest matches the active contract and feature schema
may be loaded by a serving process.

## Safe migration sequence

1. Inventory every tracked generated file and classify it as retain, archive,
   or discard.
2. Upload retained files to versioned object storage and generate checksums.
3. Add manifests and a clean-checkout restoration smoke test.
4. Verify a restored artifact byte-for-byte and run its compatibility check.
5. Remove generated paths from the Git index while leaving ignore rules in
   place.
6. Decide separately whether rewriting Git history is worth the disruption.

History rewriting is not required to stop future repository growth. If it is
eventually chosen to reduce clone size, it should be a coordinated maintenance
operation with a backup tag and explicit instructions for every collaborator.

## Subsequent market-data slice

The dated equity-like snapshot described by the original next step is now
implemented as `equity-market-snapshot-v1`, independently of Phoenix terms. See
[`equity-market-snapshot-v1.md`](equity-market-snapshot-v1.md). The remaining
provider step now has a credential-free yfinance research adapter with bounded
caching/retries, freshness checks, and normalized provenance. See
[`live-market-data.md`](live-market-data.md). Server-sourced curves and
volatility surfaces remain future work.
