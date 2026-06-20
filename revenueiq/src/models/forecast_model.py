"""
src/models/forecast_model.py
------------------------------
Revenue forecasting using a Gradient Boosting Regressor trained on
lagged + rolling + calendar features. Supports multi-step recursive
forecasting into the future.

Usage:
    python src/models/forecast_model.py --input data/processed/sales_clean.csv \
        --horizon 30 --model-out models/forecast_model.joblib
"""

import argparse
import logging
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.features.build_features import build_forecasting_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "DayOfWeek", "Month", "Day", "WeekOfYear", "IsWeekend", "IsMonthStart", "IsMonthEnd",
    "Revenue_lag_1", "Revenue_lag_7", "Revenue_lag_14", "Revenue_lag_30",
    "Revenue_roll_mean_7", "Revenue_roll_std_7",
    "Revenue_roll_mean_14", "Revenue_roll_std_14",
    "Revenue_roll_mean_30", "Revenue_roll_std_30",
]
TARGET_COL = "Revenue"


class SalesForecaster:
    """Wraps a GradientBoostingRegressor for daily revenue forecasting."""

    def __init__(self, n_estimators=300, max_depth=3, learning_rate=0.05, random_state=42):
        self.model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
        )
        self.feature_cols = FEATURE_COLS
        self.is_fitted = False

    def fit(self, ts: pd.DataFrame):
        data = ts.dropna(subset=self.feature_cols + [TARGET_COL])
        X, y = data[self.feature_cols], data[TARGET_COL]
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def evaluate(self, ts: pd.DataFrame, n_splits: int = 4) -> dict:
        """Time-series cross-validation to estimate out-of-sample performance."""
        data = ts.dropna(subset=self.feature_cols + [TARGET_COL]).reset_index(drop=True)
        X, y = data[self.feature_cols], data[TARGET_COL]

        tscv = TimeSeriesSplit(n_splits=n_splits)
        maes, mapes, r2s = [], [], []

        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            model = GradientBoostingRegressor(
                n_estimators=self.model.n_estimators,
                max_depth=self.model.max_depth,
                learning_rate=self.model.learning_rate,
                random_state=42,
            )
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            maes.append(mean_absolute_error(y_test, preds))
            # avoid div-by-zero in MAPE
            mask = y_test != 0
            mapes.append(mean_absolute_percentage_error(y_test[mask], preds[mask]) if mask.any() else np.nan)
            r2s.append(r2_score(y_test, preds))

        return {
            "MAE": float(np.mean(maes)),
            "MAPE": float(np.nanmean(mapes)),
            "R2": float(np.mean(r2s)),
            "n_splits": n_splits,
        }

    def forecast_future(self, ts: pd.DataFrame, horizon: int = 30) -> pd.DataFrame:
        """
        Recursive multi-step forecast: predict one day ahead, append it to
        history, recompute lag/rolling features, repeat for `horizon` days.
        """
        from src.features.build_features import add_lag_features, add_rolling_features, add_calendar_features

        history = ts[["Date", TARGET_COL]].copy()
        last_date = history["Date"].max()

        future_rows = []
        for step in range(1, horizon + 1):
            next_date = last_date + pd.Timedelta(days=step)

            extended = pd.concat([
                history,
                pd.DataFrame({"Date": [next_date], TARGET_COL: [np.nan]})
            ], ignore_index=True)

            feat = add_calendar_features(extended)
            feat = add_lag_features(feat, target_col=TARGET_COL)
            feat = add_rolling_features(feat, target_col=TARGET_COL)

            row = feat.iloc[[-1]]
            X_next = row[self.feature_cols].fillna(0)
            pred = max(0.0, float(self.model.predict(X_next)[0]))

            future_rows.append({"Date": next_date, TARGET_COL: pred})
            history = pd.concat([history, pd.DataFrame({"Date": [next_date], TARGET_COL: [pred]})], ignore_index=True)

        return pd.DataFrame(future_rows)

    def feature_importances(self) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Model not fitted yet.")
        return pd.DataFrame({
            "feature": self.feature_cols,
            "importance": self.model.feature_importances_,
        }).sort_values("importance", ascending=False).reset_index(drop=True)

    def save(self, path: str):
        joblib.dump(self, path)
        logger.info(f"Saved forecasting model -> {path}")

    @staticmethod
    def load(path: str) -> "SalesForecaster":
        return joblib.load(path)


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate the sales forecasting model.")
    parser.add_argument("--input", type=str, default="data/processed/sales_clean.csv")
    parser.add_argument("--freq", type=str, default="D", choices=["D", "W", "M"])
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--model-out", type=str, default="models/forecast_model.joblib")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df["Date"] = pd.to_datetime(df["Date"])

    ts = build_forecasting_features(df, freq=args.freq)

    forecaster = SalesForecaster()
    metrics = forecaster.evaluate(ts)
    logger.info(f"Cross-validated performance: {metrics}")

    forecaster.fit(ts)
    forecaster.save(args.model_out)

    future = forecaster.forecast_future(ts, horizon=args.horizon)
    logger.info(f"Forecast (next {args.horizon} periods):")
    print(future)

    importances = forecaster.feature_importances()
    print("\nTop features:")
    print(importances.head(8))


if __name__ == "__main__":
    main()
