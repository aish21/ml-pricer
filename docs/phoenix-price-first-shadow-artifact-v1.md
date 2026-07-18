# Phoenix price-first shadow artifact

This phase packages the sealed-audit winner as a pure-NumPy runtime artifact.
The API can now evaluate the model in monitored shadow mode without importing
PyTorch. The Monte Carlo reference remains the returned price.

## Approved artifact

- model version: `phoenix-price-first-multitask-v1`;
- artifact schema: `phoenix-price-first-branched-artifact-v1`;
- artifact ID:
  `sha256:40025cc823b53e3e9a63571732dd54610776292b0c4d7da9d0824cfbc8b0c216`;
- weights checksum:
  `sha256:d39a9ed2a56fea31d3cb2e844572d2310dc6e65f0b9b7e5f99d4371936ea45c7`;
- sealed report checksum:
  `sha256:963d5ea23f58463896ab2493eef13a2d8c18847aadbeee9deef408ab8d1a06d0`;
- deployment status: `shadow_approved`; and
- runtime policy: `shadow-only`.

The artifact directory is generated under
`data/surrogates/phoenix-price-first-v1/artifacts` and remains outside Git.

## Export

```powershell
python -m src.final.surrogate_pipeline package-price-first `
  data/surrogates/phoenix-v4/datasets/b3dd7e8d7760d4aa2a34b96f69252d6a5e6b01557f201e74d77934b05994981f.npz `
  data/surrogates/phoenix-hazard-v1/datasets/af8f68517df91ec9ec56773ead374470f24769104f07735ea0dad803a6f7d5eb.npz `
  data/surrogates/phoenix-price-first-v1/sealed-audit-report.json
```

Packaging is deliberately narrow. It:

1. checks the exact development, observation, audit, policy, report, and source
   commit IDs;
2. refits the frozen model and reproduces its development fingerprint;
3. exports the shared ReLU trunk, direct-price head, payoff head, and event head
   to `weights.npz`;
4. compares raw outputs from PyTorch and NumPy on all 6,144 development cases;
5. binds the complete frozen training configuration and audit acceptance result
   into the artifact identity; and
6. atomically updates `current.json` only after every check passes.

The maximum PyTorch-to-NumPy output difference was `1.2112e-06`, below the
frozen `5e-06` tolerance. The archive is approximately 194 KiB.

## Runtime controls

Shadow inference is disabled by default. Enable the model and its bounded local
telemetry explicitly:

```powershell
$env:PHOENIX_SURROGATE_SHADOW_ENABLED="true"
$env:PHOENIX_SURROGATE_TELEMETRY_ENABLED="true"
python -m uvicorn app.backend:app --reload --host 127.0.0.1 --port 8000
```

`PHOENIX_SURROGATE_DIR` is optional because the service now defaults to the
price-first artifact directory. A different root is still fail-closed: the
runtime requires the exact approved artifact ID, all audit bindings, the
canonical manifest identity, and the weights checksum.

The legacy `PHOENIX_SURROGATE_ALLOW_UNAPPROVED` research override remains
available for legacy v7 artifacts only. It cannot make an unaudited price-first
artifact loadable.

## Safety boundary

The runtime:

- rejects out-of-domain inputs before inference;
- validates price, cashflow, and event outputs;
- records the artifact and model IDs with each optional shadow observation;
- reports drift and error against the same-call Monte Carlo result; and
- always returns `used_for_price: false`.

Artifact availability or inference failure cannot replace or fail the reference
price. Promotion beyond shadow mode requires evidence from real or out-of-time
observations, a frozen promotion policy, and an operational rollback design.
The evidence policy is now frozen in
[`phoenix-shadow-promotion-readiness-v1.md`](phoenix-shadow-promotion-readiness-v1.md).
