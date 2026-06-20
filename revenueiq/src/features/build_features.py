"""
src/features/build_features.py
--------------------------------
Feature engineering utilities shared across the forecasting,
segmentation, and anomaly detection models.
"""

import pandas as pd
import numpy as np


def build_daily_revenue_series(df: pd.DataFrame, freq: str = "D") -> pd.DataFrame:
    """Aggregate transaction-level data into a daily/weekly/monthly revenue time series."""
    ts = (
        df.set_index("Date")
        .resample(freq)
        .agg(Revenue=("Total Price", "sum"), Units=("Quantity", "sum"), Orders=("Order ID", "nunique"))
        .reset_index()
    )
    ts["Revenue"] = ts["Revenue"].fillna(0)
    ts["Units"] = ts["Units"].fillna(0)
    ts["Orders"] = ts["Orders"].fillna(0)
    return ts


def add_lag_features(ts: pd.DataFrame, target_col: str = "Revenue", lags=(1, 7, 14, 30)) -> pd.DataFrame:
    """Add lagged values of the target as features for supervised forecasting."""
    ts = ts.copy()
    for lag in lags:
        ts[f"{target_col}_lag_{lag}"] = ts[target_col].shift(lag)
    return ts


def add_rolling_features(ts: pd.DataFrame, target_col: str = "Revenue", windows=(7, 14, 30)) -> pd.DataFrame:
    """Add rolling mean/std features as a proxy for recent trend and volatility."""
    ts = ts.copy()
    for w in windows:
        ts[f"{target_col}_roll_mean_{w}"] = ts[target_col].shift(1).rolling(w).mean()
        ts[f"{target_col}_roll_std_{w}"] = ts[target_col].shift(1).rolling(w).std()
    return ts


def add_calendar_features(ts: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    """Add calendar-based seasonality features."""
    ts = ts.copy()
    ts["DayOfWeek"] = ts[date_col].dt.dayofweek
    ts["Month"] = ts[date_col].dt.month
    ts["Day"] = ts[date_col].dt.day
    ts["WeekOfYear"] = ts[date_col].dt.isocalendar().week.astype(int)
    ts["IsWeekend"] = ts["DayOfWeek"].isin([5, 6]).astype(int)
    ts["IsMonthStart"] = ts[date_col].dt.is_month_start.astype(int)
    ts["IsMonthEnd"] = ts[date_col].dt.is_month_end.astype(int)
    return ts


def build_forecasting_features(df: pd.DataFrame, freq: str = "D", target_col: str = "Revenue") -> pd.DataFrame:
    """End-to-end feature pipeline for the forecasting model."""
    ts = build_daily_revenue_series(df, freq=freq)
    ts = add_calendar_features(ts)
    ts = add_lag_features(ts, target_col=target_col)
    ts = add_rolling_features(ts, target_col=target_col)
    return ts


def build_product_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-product features used for segmentation (RFM-style + price/volume behavior)."""
    snapshot_date = df["Date"].max() + pd.Timedelta(days=1)

    agg = df.groupby("Product Name").agg(
        TotalRevenue=("Total Price", "sum"),
        TotalUnits=("Quantity", "sum"),
        AvgUnitPrice=("Unit Price", "mean"),
        OrderCount=("Order ID", "nunique"),
        LastOrderDate=("Date", "max"),
        FirstOrderDate=("Date", "min"),
    ).reset_index()

    agg["RecencyDays"] = (snapshot_date - agg["LastOrderDate"]).dt.days
    agg["ActiveDays"] = (agg["LastOrderDate"] - agg["FirstOrderDate"]).dt.days + 1
    agg["AvgRevenuePerOrder"] = agg["TotalRevenue"] / agg["OrderCount"]
    agg["PurchaseFrequency"] = agg["OrderCount"] / agg["ActiveDays"].clip(lower=1)

    return agg


def build_customer_like_features(df: pd.DataFrame, group_col: str = "Region") -> pd.DataFrame:
    """
    Aggregate features at a chosen grouping level (Region by default, since this
    dataset has no customer ID). Produces RFM-style features usable for clustering.
    """
    snapshot_date = df["Date"].max() + pd.Timedelta(days=1)

    agg = df.groupby(group_col).agg(
        TotalRevenue=("Total Price", "sum"),
        TotalUnits=("Quantity", "sum"),
        OrderCount=("Order ID", "nunique"),
        AvgOrderValue=("Total Price", "mean"),
        LastOrderDate=("Date", "max"),
    ).reset_index()

    agg["RecencyDays"] = (snapshot_date - agg["LastOrderDate"]).dt.days
    return agg


def add_order_level_anomaly_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build per-order features used as input to the anomaly detection model."""
    df = df.copy()
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["Month"] = df["Date"].dt.month

    # Category-level price stats to detect orders priced abnormally vs. their own category
    cat_stats = df.groupby("Category")["Unit Price"].agg(["mean", "std"]).rename(
        columns={"mean": "CategoryAvgPrice", "std": "CategoryStdPrice"}
    )
    df = df.merge(cat_stats, on="Category", how="left")
    df["CategoryStdPrice"] = df["CategoryStdPrice"].fillna(df["CategoryStdPrice"].mean())
    df["PriceZScore"] = (df["Unit Price"] - df["CategoryAvgPrice"]) / df["CategoryStdPrice"].replace(0, np.nan)
    df["PriceZScore"] = df["PriceZScore"].fillna(0)

    return df
