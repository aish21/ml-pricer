# Neural Pricer

A lightweight project to build neural network alternative for structured notes pricing.

## Setup

1. Create a virtual environment:

   ```bash
   python -m venv venv
   ```

2. Activate the virtual environment:

   - Windows: `.\venv\Scripts\Activate.ps1`
   - Linux/Mac: `source venv/bin/activate`

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure

- `data/` - Generated datasets
- `notebooks/` - Exploratory notebooks
- `src/` - Scripts for data generation, model training, and evaluation
- `requirements.txt` - Python dependencies
- `.gitignore` - Git ignore rules

## Simplified Phoenix (single underlying)

Implements a Phoenix with monthly coupons, memory, European KI, and an autocall from month 6.
Dynamics: Black–Scholes under risk-neutral measure.

### Components

- `src/gbm.py`: GBM path simulator with antithetics

## Running Unittests in PowerShell

To run the unittests from the project root in PowerShell, use:

```powershell
$env:PYTHONPATH="."; pytest unittests/
```

This ensures the current directory is on your Python path so imports like `from src.gbm import ...` work correctly.

Activate the virtual environment before running any scripts or notebooks.
