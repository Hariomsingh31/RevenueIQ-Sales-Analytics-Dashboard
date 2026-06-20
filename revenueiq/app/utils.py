"""
app/utils.py
-------------
Shared helpers for the Streamlit app: data loading, caching, and
CSV export utilities used across all pages/tabs.
"""

import re
import streamlit as st
import pandas as pd

REQUIRED_COLUMNS = [
    "Order ID", "Date", "Category", "Product Name",
    "Quantity", "Unit Price", "Total Price", "Region", "Payment Method"
]

# Common alternate names seen in real-world exports, mapped to the
# canonical column name the app expects. Matching is done after
# normalizing both sides (lowercase, strip, collapse non-alphanumerics).
COLUMN_ALIASES = {
    "orderid": "Order ID",
    "order": "Order ID",
    "orderno": "Order ID",
    "ordernumber": "Order ID",
    "invoiceid": "Order ID",
    "transactionid": "Order ID",

    "date": "Date",
    "orderdate": "Date",
    "transactiondate": "Date",
    "saledate": "Date",

    "category": "Category",
    "productcategory": "Category",

    "productname": "Product Name",
    "product": "Product Name",
    "item": "Product Name",
    "itemname": "Product Name",

    "quantity": "Quantity",
    "qty": "Quantity",
    "units": "Quantity",
    "unitssold": "Quantity",

    "unitprice": "Unit Price",
    "price": "Unit Price",
    "priceperunit": "Unit Price",
    "rate": "Unit Price",

    "totalprice": "Total Price",
    "totalamount": "Total Price",
    "total": "Total Price",
    "revenue": "Total Price",
    "amount": "Total Price",
    "sales": "Total Price",

    "region": "Region",
    "location": "Region",
    "country": "Region",
    "area": "Region",
    "state": "Region",
    "province": "Region",
    "city": "Region",
    "territory": "Region",
    "town": "Region",
    "capital": "Region",
    "stateprovince": "Region",
    "state/province": "Region",
    "district": "Region",
    "village": "Region",
    "metro": "Region",
    "metropolitan": "Region",
    "zone": "Region",
    "tier1": "Region",
    "tier2": "Region",
    "tier3": "Region",

    "paymentmethod": "Payment Method",
    "payment": "Payment Method",
    "paymenttype": "Payment Method",
    "paymentmode": "Payment Method",
}


def _normalize_key(name: str) -> str:
    """Lowercase, strip, and remove anything that isn't a letter/digit
    so 'Unit_Price', 'unit-price', ' Unit Price ' all match the same key."""
    return re.sub(r"[^a-z0-9]", "", str(name).strip().lower())


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to canonical names where a confident alias match exists.
    Columns that already match a required name (after normalization) are
    also renamed to fix casing/spacing/punctuation differences."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    required_lookup = {_normalize_key(c): c for c in REQUIRED_COLUMNS}
    rename_map = {}

    for col in df.columns:
        key = _normalize_key(col)
        if key in required_lookup:
            rename_map[col] = required_lookup[key]
        elif key in COLUMN_ALIASES:
            rename_map[col] = COLUMN_ALIASES[key]

    if rename_map:
        df = df.rename(columns=rename_map)

    return df


@st.cache_data
def load_data(file) -> pd.DataFrame:
    df = _read_csv_robust(file)

    # Normalize header whitespace (e.g. " Order ID " -> "Order ID")
    df.columns = [str(c).strip() for c in df.columns]

    # Map common alternate column names to the canonical schema
    df = standardize_columns(df)

    if "Total Price" not in df.columns and "Quantity" in df.columns and "Unit Price" in df.columns:
        df["Total Price"] = df["Quantity"] * df["Unit Price"]

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    return df


def _read_csv_robust(file) -> pd.DataFrame:
    """
    Read a CSV that may use comma, semicolon, or tab delimiters (common
    across different locales/export tools) and may include a UTF-8 BOM.
    Falls back through delimiters if the first attempt looks wrong (a
    single column containing the separator character is a tell-tale sign
    of a delimiter mismatch).
    """
    def _try_read(sep):
        if hasattr(file, "seek"):
            file.seek(0)
        return pd.read_csv(file, sep=sep, encoding="utf-8-sig")

    df = _try_read(",")

    if df.shape[1] == 1:
        first_col = str(df.columns[0])
        if ";" in first_col:
            df = _try_read(";")
        elif "\t" in first_col:
            df = _try_read("\t")

    return df


def validate_schema(df: pd.DataFrame):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    return missing


def describe_columns_for_user(df: pd.DataFrame) -> str:
    """Human-readable summary of the columns actually found, for error messages."""
    cols = ", ".join(f"`{c}`" for c in df.columns)
    return cols if cols else "(no columns detected)"


def kpi_card(label: str, value: str, col):
    col.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)


def to_csv_bytes(dataframe: pd.DataFrame) -> bytes:
    return dataframe.to_csv(index=False).encode("utf-8")


CUSTOM_CSS = """
<style>
.block-container {
    padding-top: 1.2rem;
    padding-bottom: 1rem;
}
div[data-testid="stVerticalBlock"] > div {
    gap: 0.4rem;
}
div[role="radiogroup"] {
    border-bottom: 1px solid #e0e0e0;
    padding-bottom: 6px;
}
div[role="radiogroup"] label {
    font-size: 15px !important;
    font-weight: 500;
    margin-right: 26px !important;
}
div[role="radiogroup"] input[type="radio"] {
    display: none;
}
div[role="radiogroup"] input[type="radio"]:checked + div {
    border-bottom: 2px solid #FF4B4B;
    padding-bottom: 4px;
    color: #FF4B4B;
}

.kpi-card {
    background: linear-gradient(135deg, #ffffff 0%, #f8f9fb 100%);
    border: 1px solid #e8e8e8;
    padding: 18px 16px;
    border-radius: 12px;
    text-align: center;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.kpi-label {
    font-size: 13px;
    color: #888;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.kpi-value {
    font-size: 24px;
    font-weight: 700;
    color: #1a1a1a;
    margin-top: 4px;
}

.section-title {
    font-size: 18px;
    font-weight: 700;
    margin-top: 18px;
    margin-bottom: 6px;
}

.ml-badge {
    display: inline-block;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white;
    font-size: 11px;
    font-weight: 700;
    padding: 3px 10px;
    border-radius: 999px;
    letter-spacing: 0.5px;
    margin-left: 8px;
}
</style>
"""
