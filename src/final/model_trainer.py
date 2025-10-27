import time
import math
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import lightgbm as lgb
from lightgbm import LGBMRegressor
import optuna


class ModelTrainer:
    """Train LightGBM models with Optuna tuning."""

    def __init__(
        self,
        use_log_target: bool = True,
        test_size: float = 0.2,
        random_state: int = 42,
        verbose: bool = True,
    ):
        self.use_log_target = use_log_target
        self.test_size = test_size
        self.random_state = random_state
        self.verbose = verbose
        self.scaler = None
        self.model = None
        self.feature_names = None
        self.train_metrics = None

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        n_trials: int = 30,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Train LightGBM model with Optuna tuning."""
        start_time = time.time()
        if self.verbose:
            print(f"[ModelTrainer] Starting training on {len(y)} samples...")

        self.feature_names = feature_names

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        if self.use_log_target:
            y_train_t = np.log1p(y_train)
            y_val_t = np.log1p(y_val)
            if self.verbose:
                print("  Using log1p target transformation")
        else:
            y_train_t = y_train
            y_val_t = y_val

        self.scaler = StandardScaler()
        X_train_s = self.scaler.fit_transform(X_train)
        X_val_s = self.scaler.transform(X_val)

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 200, 3000),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 1e-3, 0.2, log=True
                ),
                "num_leaves": trial.suggest_int("num_leaves", 16, 512),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 200),
                "subsample": trial.suggest_float("subsample", 0.4, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
                "random_state": self.random_state,
                "n_jobs": 1,
            }

            model = LGBMRegressor(**params)
            model.fit(
                X_train_s,
                y_train_t,
                eval_set=[(X_val_s, y_val_t)],
                eval_metric="rmse",
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50),
                    lgb.log_evaluation(period=0),
                ],
            )

            pred_val = model.predict(X_val_s)
            if self.use_log_target:
                pred_val_inv = np.expm1(pred_val)
                y_val_inv = np.expm1(y_val_t)
            else:
                pred_val_inv = pred_val
                y_val_inv = y_val_t

            rmse_val = math.sqrt(np.mean((y_val_inv - pred_val_inv) ** 2))
            return rmse_val

        if self.verbose:
            print(f"  Running Optuna with {n_trials} trials...")

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
        )
        study.optimize(objective, n_trials=n_trials, timeout=timeout)

        if self.verbose:
            print(f"  Best trial: RMSE={study.best_value:.6f}")

        best_params = study.best_params
        self.model = LGBMRegressor(**best_params)

        X_full_s = self.scaler.fit_transform(np.vstack([X_train, X_val]))
        if self.use_log_target:
            y_full_t = np.log1p(np.concatenate([y_train, y_val]))
        else:
            y_full_t = np.concatenate([y_train, y_val])

        self.model.fit(X_full_s, y_full_t, eval_metric="rmse")

        X_train_all, X_test, y_train_all, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state + 1
        )
        X_test_s = self.scaler.transform(X_test)

        pred_test = self.model.predict(X_test_s)
        if self.use_log_target:
            pred_test = np.expm1(pred_test)

        rmse_test = math.sqrt(np.mean((y_test - pred_test) ** 2))
        mae_test = mean_absolute_error(y_test, pred_test)
        r2_test = r2_score(y_test, pred_test)

        self.train_metrics = {
            "rmse": rmse_test,
            "mae": mae_test,
            "r2": r2_test,
        }

        elapsed = time.time() - start_time
        if self.verbose:
            print(f"[ModelTrainer] Training complete in {elapsed:.1f}s")
            print(f"  Test RMSE: {rmse_test:.6f}")
            print(f"  Test MAE: {mae_test:.6f}")
            print(f"  Test R2: {r2_test:.4f}")

        try:
            booster = self.model.booster_
            fi_vals = booster.feature_importance(importance_type="gain")
        except Exception:
            fi_vals = self.model.feature_importances_

        feature_importance = sorted(
            [
                {"feature": feature_names[i], "importance": float(fi_vals[i])}
                for i in range(len(feature_names))
            ],
            key=lambda x: -x["importance"],
        )

        return {
            "model": self.model,
            "scaler": self.scaler,
            "metrics": self.train_metrics,
            "optuna_study": {
                "best_value": study.best_value,
                "best_params": study.best_params,
            },
            "feature_importance": feature_importance,
            "use_log_target": self.use_log_target,
        }

    def save(self, model_path: Path, scaler_path: Path):
        """Save model and scaler."""
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)

        if self.verbose:
            print(f"[ModelTrainer] Saved model to {model_path}")
            print(f"  Saved scaler to {scaler_path}")

    @staticmethod
    def load(model_path: Path, scaler_path: Path):
        """Load model and scaler."""
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
