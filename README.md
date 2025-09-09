# Neural Network Enhanced Monte Carlo Pricing

## 🛠 Setup & Environment

### 1. Create and Activate Virtual Environment

```powershell
python -m venv .venv
# Activate venv in PowerShell
.\.venv\Scripts\Activate.ps1
```

### 2. Install Dependencies

```powershell
pip install -r requirements.txt
```

---

## 🚀 Running Scripts

### 1. Monte Carlo Baseline + Convergence Plots

```powershell
$env:PYTHONPATH="."; python src/visualization/plot_mc_convergence.py
```

**Generates:**

- MC convergence plot vs Black–Scholes price.
- Runtime vs absolute error plot.

**Validates:**  
The Monte Carlo engine against the closed-form Black–Scholes formula.

**Key takeaway:**  
Monte Carlo converges slowly but reliably; error shrinks with more paths.

---

### 2. Running Unit Tests

```powershell
$env:PYTHONPATH="."; pytest unittests/
```

**Unit tests check:**

- MC ≈ BS within tolerance.
- Call/put consistency.

_Tolerance is relaxed (`atol=0.15`) due to stochastic sampling noise._

---

### 3. Training Data Generation

```powershell
$env:PYTHONPATH="."; python src/data/generate_training_data.py
```

**Produces:**  
A dataset of prefixes + parameters → discounted payoffs.

**Example:**  
`data/training_data_small.npz`

**Dataset structure:**

- `X = [prefix prices, K, T, r, sigma, opt_flag]`
  - `prefix prices`: simulated stock path prefix
  - `K`: strike
  - `T`: maturity
  - `r`: risk-free rate
  - `sigma`: volatility
  - `opt_flag`: 1 = Call, 0 = Put
- `y = discounted payoff at maturity`

---

### 4. Exploratory Data Analysis (EDA)

```powershell
$env:PYTHONPATH="."; python src/visualization/explore_training_data.py
```

**Saves plots in `./figures/`, including:**

- `payoff_distribution.png` → shape of payoff distribution
- `parameter_histograms.png` → distributions of K, T, r, sigma
- `option_type_counts.png` → call/put ratio
- `sample_prefixes.png` → sample simulated paths
- `payoff_by_option_type.png` → payoff histograms split by calls/puts
- `payoff_by_moneyness.png` → payoff distribution by ITM/ATM/OTM
- `feature_payoff_correlation.png` → correlation heatmap (parameters vs payoff)
- `pairplot_features.png` → pairwise relationships between parameters, color-coded by option type

**Key takeaways:**

- Strikes negatively correlate with payoffs; vol & maturity generally positive.
- Calls and puts cluster differently.
- Payoff distributions are heavy-tailed, motivating NN approximators.

---

## 💡 Notes for Beginners

- Monte Carlo simulations are stochastic; results may vary — fix a seed for reproducibility.
- More paths = higher accuracy but slower runtime (convergence ~ 1/√N).
- Dataset generation captures a variety of strikes, maturities, volatilities, and option types — critical for NN generalization.
- EDA helps verify that the dataset is balanced and contains useful variation before training.
