"""
src/models/segmentation_model.py
-----------------------------------
K-Means clustering for product/region segmentation using RFM-style
features (Recency, Frequency, Monetary value). Includes automatic
selection of K via silhouette score.

Usage:
    python src/models/segmentation_model.py --input data/processed/sales_clean.csv \
        --level product --model-out models/segmentation_model.joblib
"""

import argparse
import logging
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.features.build_features import build_product_features, build_customer_like_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

PRODUCT_FEATURE_COLS = ["TotalRevenue", "TotalUnits", "OrderCount", "RecencyDays", "AvgRevenuePerOrder", "PurchaseFrequency"]
REGION_FEATURE_COLS = ["TotalRevenue", "TotalUnits", "OrderCount", "AvgOrderValue", "RecencyDays"]

SEGMENT_LABEL_TIERS = {
    2: ["Low Value", "High Value"],
    3: ["Low Value", "Mid Value", "High Value"],
    4: ["Low Value", "Mid Value", "High Value", "Top Performer"],
    5: ["Low Value", "Below Average", "Mid Value", "High Value", "Top Performer"],
    6: ["Low Value", "Below Average", "Mid Value", "Above Average", "High Value", "Top Performer"],
}


def _labels_for_k(k: int) -> list:
    if k in SEGMENT_LABEL_TIERS:
        return SEGMENT_LABEL_TIERS[k]
    # Fallback for any K outside the predefined tiers
    return [f"Segment {i+1}" for i in range(k)]


class SegmentationModel:
    """Wraps StandardScaler + KMeans for RFM-style segmentation."""

    def __init__(self, k: int = None, k_range=(2, 6), random_state: int = 42):
        self.k = k
        self.k_range = k_range
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = None
        self.feature_cols = None
        self.is_fitted = False

    def _select_k(self, X_scaled: np.ndarray) -> int:
        best_k, best_score = self.k_range[0], -1
        for k in range(self.k_range[0], min(self.k_range[1], len(X_scaled) - 1) + 1):
            if k < 2:
                continue
            km = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = km.fit_predict(X_scaled)
            if len(set(labels)) < 2:
                continue
            score = silhouette_score(X_scaled, labels)
            logger.info(f"k={k} -> silhouette={score:.4f}")
            if score > best_score:
                best_k, best_score = k, score
        return best_k

    def fit(self, df_features: pd.DataFrame, feature_cols: list):
        self.feature_cols = feature_cols
        X = df_features[feature_cols].fillna(0)
        X_scaled = self.scaler.fit_transform(X)

        k = self.k or self._select_k(X_scaled)
        self.k = k
        self.model = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
        labels = self.model.fit_predict(X_scaled)
        self.is_fitted = True

        result = df_features.copy()
        result["Cluster"] = labels

        # Rank clusters by mean monetary value so labels are meaningful (0=low ... k-1=high)
        cluster_rank = (
            result.groupby("Cluster")["TotalRevenue"].mean().rank(method="first").astype(int) - 1
        )
        result["SegmentRank"] = result["Cluster"].map(cluster_rank)

        labels = _labels_for_k(k)
        result["Segment"] = result["SegmentRank"].map(lambda r: labels[r])

        return result

    def predict(self, df_features: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model not fitted yet.")
        X = df_features[self.feature_cols].fillna(0)
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def save(self, path: str):
        joblib.dump(self, path)
        logger.info(f"Saved segmentation model -> {path}")

    @staticmethod
    def load(path: str) -> "SegmentationModel":
        return joblib.load(path)


def run_segmentation(df: pd.DataFrame, level: str = "product", k: int = None):
    if level == "product":
        features = build_product_features(df)
        feature_cols = PRODUCT_FEATURE_COLS
    elif level == "region":
        features = build_customer_like_features(df, group_col="Region")
        feature_cols = REGION_FEATURE_COLS
    else:
        raise ValueError("level must be 'product' or 'region'")

    model = SegmentationModel(k=k)
    result = model.fit(features, feature_cols)
    return model, result


def main():
    parser = argparse.ArgumentParser(description="Run RFM-style segmentation (KMeans).")
    parser.add_argument("--input", type=str, default="data/processed/sales_clean.csv")
    parser.add_argument("--level", type=str, default="product", choices=["product", "region"])
    parser.add_argument("--k", type=int, default=None, help="Force a specific K; auto-selected if omitted")
    parser.add_argument("--model-out", type=str, default="models/segmentation_model.joblib")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df["Date"] = pd.to_datetime(df["Date"])

    model, result = run_segmentation(df, level=args.level, k=args.k)
    model.save(args.model_out)

    print(f"\nSelected K = {model.k}")
    print(result.sort_values("TotalRevenue", ascending=False))

    print("\nSegment summary:")
    print(result.groupby("Segment")[["TotalRevenue", "TotalUnits"]].mean())


if __name__ == "__main__":
    main()
