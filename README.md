# Neural Network Enhanced Monte Carlo Pricing

## 🛠 Setup & Environment

1. **Create and activate virtual environment**

```powershell
python -m venv .venv
# Activate venv in PowerShell
.\.venv\Scripts\Activate.ps1
```

2. **Install dependencies**

```powershell
pip install -r requirements.txt
```

## 🚀 Running Scripts

### 1. Monte Carlo Baseline + Convergence Plots

```powershell
$env:PYTHONPATH="."; python src/visualization/plot_mc_convergence.py
```

- This will generate:
  - MC convergence plot vs Black–Scholes price.
  - Runtime vs absolute error plot.
- Modify `path_counts` or other parameters in the script to test different scenarios.

### 2. Running Unit Tests

```powershell
$env:PYTHONPATH="."; pytest unittests/
```

## 💡 Notes for Beginners

- Monte Carlo simulations are **stochastic**, so results may slightly vary each run. Use a `seed` for reproducibility.
- Increasing the number of paths improves accuracy but also increases computation time.
- Closed-form Black–Scholes provides a **ground truth** to validate your MC implementation.
