"""
src/models/anomaly_model.py
------------------------------
Order-level anomaly detection using Isolation Forest. Flags orders
that look unusual relative to their category's typical price/quantity
behavior — useful for catching data entry errors, fraud, or genuine
outlier bulk orders worth investigating.

Usage:
    python src/models/anomaly_model.py --input data/processed/sales_clean.csv \
        --contamination 0.02 --model-out models/anomaly_model.joblib
"""

import argparse
import logging
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.features.build_features import add_order_level_anomaly_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

FEATURE_COLS = ["Quantity", "Unit Price", "Total Price", "PriceZScore", "DayOfWeek", "Month"]


class AnomalyDetector:
    """Wraps StandardScaler + IsolationForest for order-level anomaly detection."""

    def __init__(self, contamination: float = 0.02, random_state: int = 42):
        self.contamination = contamination
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=200,
        )
        self.feature_cols = FEATURE_COLS
        self.is_fitted = False

    def fit_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        feat_df = add_order_level_anomaly_features(df)
        X = feat_df[self.feature_cols].fillna(0)
        X_scaled = self.scaler.fit_transform(X)

        raw_pred = self.model.fit_predict(X_scaled)   # -1 = anomaly, 1 = normal
        scores = self.model.decision_function(X_scaled)  # higher = more normal
        self.is_fitted = True

        result = feat_df.copy()
        result["AnomalyScore"] = scores
        result["IsAnomaly"] = (raw_pred == -1)

        return result.sort_values("AnomalyScore")

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Model not fitted yet.")
        feat_df = add_order_level_anomaly_features(df)
        X = feat_df[self.feature_cols].fillna(0)
        X_scaled = self.scaler.transform(X)

        raw_pred = self.model.predict(X_scaled)
        scores = self.model.decision_function(X_scaled)

        result = feat_df.copy()
        result["AnomalyScore"] = scores
        result["IsAnomaly"] = (raw_pred == -1)
        return result

    def save(self, path: str):
        joblib.dump(self, path)
        logger.info(f"Saved anomaly detection model -> {path}")

    @staticmethod
    def load(path: str) -> "AnomalyDetector":
        return joblib.load(path)


def main():
    parser = argparse.ArgumentParser(description="Detect anomalous orders with Isolation Forest.")
    parser.add_argument("--input", type=str, default="data/processed/sales_clean.csv")
    parser.add_argument("--contamination", type=float, default=0.02, help="Expected proportion of anomalies")
    parser.add_argument("--model-out", type=str, default="models/anomaly_model.joblib")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df["Date"] = pd.to_datetime(df["Date"])

    detector = AnomalyDetector(contamination=args.contamination)
    result = detector.fit_predict(df)
    detector.save(args.model_out)

    anomalies = result[result["IsAnomaly"]]
    logger.info(f"Flagged {len(anomalies):,} anomalies out of {len(df):,} orders ({len(anomalies)/len(df):.1%})")

    cols = ["Order ID", "Date", "Category", "Product Name", "Quantity", "Unit Price", "Total Price", "AnomalyScore"]
    print("\nTop 10 most anomalous orders:")
    print(anomalies[cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
