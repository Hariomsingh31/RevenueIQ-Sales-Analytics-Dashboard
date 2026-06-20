"""
tests/test_app_utils.py
--------------------------
Unit tests for app/utils.py — the robust CSV loading and column
standardization that handles semicolon delimiters, UTF-8 BOM, header
whitespace, and common alternate column names (artifacts from
regional Excel exports or differently-named source systems).
"""

import sys
import os

# Import utils.py directly (not as `app.utils`) — this mirrors how
# app/app.py itself imports it, and avoids relying on `app` being
# resolvable as a package, which is not guaranteed across all
# environments/launch methods.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_APP_DIR = os.path.join(_PROJECT_ROOT, "app")
for path in (_PROJECT_ROOT, _APP_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

import pandas as pd
import pytest

from utils import _read_csv_robust, validate_schema, standardize_columns, REQUIRED_COLUMNS

HEADER = "Order ID,Date,Category,Product Name,Quantity,Unit Price,Total Price,Region,Payment Method"
ROW = "ORD1,2024-01-01,Electronics,Mouse,2,500,1000,Asia,UPI"


def _write(path, text, encoding="utf-8"):
    with open(path, "w", encoding=encoding, newline="") as f:
        f.write(text)


def test_reads_standard_comma_csv(tmp_path):
    p = tmp_path / "normal.csv"
    _write(p, f"{HEADER}\n{ROW}\n")
    df = _read_csv_robust(str(p))
    assert list(df.columns) == REQUIRED_COLUMNS


def test_reads_semicolon_delimited_csv(tmp_path):
    semi_header = HEADER.replace(",", ";")
    semi_row = ROW.replace(",", ";")
    p = tmp_path / "semicolon.csv"
    _write(p, f"{semi_header}\n{semi_row}\n")
    df = _read_csv_robust(str(p))
    assert list(df.columns) == REQUIRED_COLUMNS


def test_reads_csv_with_utf8_bom(tmp_path):
    p = tmp_path / "bom.csv"
    _write(p, f"{HEADER}\n{ROW}\n", encoding="utf-8-sig")
    df = _read_csv_robust(str(p))
    assert list(df.columns) == REQUIRED_COLUMNS
    # No BOM character leaking into the first column name
    assert "\ufeff" not in df.columns[0]


def test_reads_semicolon_csv_with_bom(tmp_path):
    semi_header = HEADER.replace(",", ";")
    semi_row = ROW.replace(",", ";")
    p = tmp_path / "semicolon_bom.csv"
    _write(p, f"{semi_header}\n{semi_row}\n", encoding="utf-8-sig")
    df = _read_csv_robust(str(p))
    assert list(df.columns) == REQUIRED_COLUMNS


def test_validate_schema_passes_after_robust_read(tmp_path):
    semi_header = HEADER.replace(",", ";")
    semi_row = ROW.replace(",", ";")
    p = tmp_path / "semicolon.csv"
    _write(p, f"{semi_header}\n{semi_row}\n")
    df = _read_csv_robust(str(p))
    df.columns = [c.strip() for c in df.columns]
    missing = validate_schema(df)
    assert missing == []


# --- Column standardization / alias mapping -------------------------------

def test_standardize_columns_exact_match_unchanged():
    df = pd.DataFrame(columns=REQUIRED_COLUMNS)
    result = standardize_columns(df)
    assert list(result.columns) == REQUIRED_COLUMNS


def test_standardize_columns_fixes_case_and_spacing():
    df = pd.DataFrame(columns=[" order id ", "DATE", "category", "Product_Name",
                                "quantity", "unit-price", "TOTAL PRICE", "Region", "payment method"])
    result = standardize_columns(df)
    assert validate_schema(result) == []


def test_standardize_columns_maps_common_aliases():
    df = pd.DataFrame(columns=["order_id", "Order Date", "category", "Item",
                                "Qty", "Price", "Total Amount", "Country", "Payment Mode"])
    result = standardize_columns(df)
    assert validate_schema(result) == []


def test_standardize_columns_leaves_unmappable_columns_alone():
    df = pd.DataFrame(columns=["foo", "bar", "baz"])
    result = standardize_columns(df)
    assert list(result.columns) == ["foo", "bar", "baz"]
    assert len(validate_schema(result)) == len(REQUIRED_COLUMNS)