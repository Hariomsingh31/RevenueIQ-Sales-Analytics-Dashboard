"""
tests/test_make_dataset.py
----------------------------
Unit tests for src/data/make_dataset.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import pytest

from src.data.make_dataset import validate_schema, clean_data, REQUIRED_COLUMNS


def make_raw_df(**overrides):
    base = {
        "Order ID": ["ORD1", "ORD2", "ORD3"],
        "Date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "Category": ["Electronics", "Books", "Electronics"],
        "Product Name": ["Mouse", "Notebook", "Mouse"],
        "Quantity": [2, 5, 1],
        "Unit Price": [500.0, 50.0, 500.0],
        "Total Price": [1000.0, 250.0, 500.0],
        "Region": ["Asia", "Europe", "Asia"],
        "Payment Method": ["UPI", "Credit Card", "UPI"],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def test_validate_schema_passes_with_all_columns():
    df = make_raw_df()
    validate_schema(df)  # should not raise


def test_validate_schema_raises_on_missing_column():
    df = make_raw_df()
    df = df.drop(columns=["Region"])
    with pytest.raises(ValueError, match="Missing required column"):
        validate_schema(df)


def test_clean_data_parses_dates():
    df = make_raw_df()
    cleaned = clean_data(df)
    assert pd.api.types.is_datetime64_any_dtype(cleaned["Date"])


def test_clean_data_drops_non_positive_quantity_or_price():
    df = make_raw_df(Quantity=[2, 0, -1])
    cleaned = clean_data(df)
    assert len(cleaned) == 1


def test_clean_data_recomputes_inconsistent_total_price():
    df = make_raw_df(**{"Total Price": [9999.0, 250.0, 500.0]})
    cleaned = clean_data(df)
    row = cleaned[cleaned["Order ID"] == "ORD1"].iloc[0]
    assert row["Total Price"] == pytest.approx(1000.0)


def test_clean_data_drops_duplicate_order_ids():
    df = make_raw_df(**{"Order ID": ["ORD1", "ORD1", "ORD3"]})
    cleaned = clean_data(df)
    assert cleaned["Order ID"].is_unique


def test_clean_data_adds_calendar_features():
    df = make_raw_df()
    cleaned = clean_data(df)
    for col in ["Year", "Month", "Day", "DayOfWeek", "WeekOfYear", "IsWeekend"]:
        assert col in cleaned.columns
