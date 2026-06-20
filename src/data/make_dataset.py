"""
src/data/make_dataset.py
-------------------------
Loads raw sales CSVs, validates schema, cleans data types, and writes
a processed parquet/csv ready for analysis and modeling.

Usage:
    python src/data/make_dataset.py --input data/raw/sales_data.csv --output data/processed/sales_clean.csv
"""

import argparse
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = [
    "Order ID", "Date", "Category", "Product Name",
    "Quantity", "Unit Price", "Total Price", "Region", "Payment Method"
]


def validate_schema(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}. Expected: {REQUIRED_COLUMNS}")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply standard cleaning: dtype coercion, dedup, derived columns, basic outlier flags."""
    df = df.copy()

    # Dtypes
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for col in ["Quantity", "Unit Price", "Total Price"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with unrecoverable nulls in critical fields
    before = len(df)
    df = df.dropna(subset=["Date", "Quantity", "Unit Price", "Order ID"])
    dropped = before - len(df)
    if dropped:
        logger.info(f"Dropped {dropped} rows with missing critical fields")

    # Recompute Total Price if missing or inconsistent beyond rounding tolerance
    expected_total = df["Quantity"] * df["Unit Price"]
    mismatch = (df["Total Price"] - expected_total).abs() > 1.0
    if mismatch.any():
        logger.info(f"Recomputing Total Price for {mismatch.sum()} inconsistent rows")
        df.loc[mismatch, "Total Price"] = expected_total[mismatch]
    df["Total Price"] = df["Total Price"].fillna(expected_total)

    # Drop negative/zero quantity or price (data entry errors)
    before = len(df)
    df = df[(df["Quantity"] > 0) & (df["Unit Price"] > 0)]
    dropped = before - len(df)
    if dropped:
        logger.info(f"Dropped {dropped} rows with non-positive Quantity/Unit Price")

    # Deduplicate exact duplicate orders
    before = len(df)
    df = df.drop_duplicates(subset=["Order ID"])
    dropped = before - len(df)
    if dropped:
        logger.info(f"Dropped {dropped} duplicate Order ID rows")

    # Derived time features (used heavily downstream by ML modules)
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
    df["IsWeekend"] = df["DayOfWeek"].isin([5, 6]).astype(int)

    df = df.sort_values("Date").reset_index(drop=True)
    return df


def load_and_clean(input_path: str) -> pd.DataFrame:
    logger.info(f"Loading raw data from {input_path}")
    df = pd.read_csv(input_path)
    validate_schema(df)
    df = clean_data(df)
    logger.info(f"Cleaned dataset: {len(df):,} rows, {df['Date'].min().date()} to {df['Date'].max().date()}")
    return df


def main():
    parser = argparse.ArgumentParser(description="Clean and process raw sales data.")
    parser.add_argument("--input", type=str, default="data/raw/sales_data.csv")
    parser.add_argument("--output", type=str, default="data/processed/sales_clean.csv")
    args = parser.parse_args()

    df = load_and_clean(args.input)
    df.to_csv(args.output, index=False)
    logger.info(f"Saved processed data -> {args.output}")


if __name__ == "__main__":
    main()
