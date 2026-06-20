"""
generate_sample_data.py
------------------------
Generates a realistic synthetic retail sales dataset matching the
project's expected schema. Useful for local development, testing,
and demos when a real dataset isn't available.

Usage:
    python src/data/generate_sample_data.py --rows 5000 --out data/raw/sales_data.csv
"""

import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


CATEGORIES = {
    "Electronics": ["Wireless Mouse", "Bluetooth Speaker", "Laptop Stand", "USB-C Hub", "Smartwatch"],
    "Home Appliances": ["Office Chair", "Air Fryer", "Vacuum Cleaner", "Table Lamp", "Microwave Oven"],
    "Clothing": ["Denim Jacket", "Running Shoes", "Cotton T-Shirt", "Wool Sweater", "Formal Shirt"],
    "Books": ["Notebook Set", "Fiction Novel", "Self-Help Guide", "Cookbook", "Children's Storybook"],
    "Beauty Products": ["Face Moisturizer", "Lipstick", "Shampoo", "Sunscreen SPF50", "Perfume"],
    "Sports": ["Yoga Mat", "Dumbbell Set", "Cricket Bat", "Football", "Resistance Bands"],
}

PRICE_RANGES = {
    "Electronics": (799, 15000),
    "Home Appliances": (999, 12000),
    "Clothing": (399, 3500),
    "Books": (150, 999),
    "Beauty Products": (199, 2500),
    "Sports": (299, 6000),
}

REGIONS = ["North America", "Europe", "Asia", "South America", "Africa", "Australia"]
PAYMENT_METHODS = ["Credit Card", "PayPal", "Debit Card", "UPI", "Net Banking"]

# Region-level seasonal/promotional multipliers to make trends learnable
REGION_WEIGHT = {"North America": 1.3, "Europe": 1.15, "Asia": 1.4, "South America": 0.8, "Africa": 0.6, "Australia": 0.9}


def month_seasonality(month: int) -> float:
    """Boost sales around festive/holiday months (Nov-Jan, similar to real retail)."""
    boost = {11: 1.4, 12: 1.6, 1: 1.2, 7: 1.15}
    return boost.get(month, 1.0)


def generate(n_rows: int, start_date: str, end_date: str, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    date_span_days = (end - start).days

    rows = []
    for i in range(1, n_rows + 1):
        # Slight upward trend over time + random day offset
        day_offset = rng.integers(0, date_span_days + 1)
        order_date = start + timedelta(days=int(day_offset))

        category = rng.choice(list(CATEGORIES.keys()))
        product = rng.choice(CATEGORIES[category])
        low, high = PRICE_RANGES[category]
        unit_price = round(rng.uniform(low, high), 2)

        region = rng.choice(REGIONS)
        seasonal = month_seasonality(order_date.month)
        regional = REGION_WEIGHT[region]

        # Quantity influenced by seasonality/region, with noise + occasional bulk orders
        base_qty = rng.poisson(lam=2.2 * seasonal * regional)
        quantity = max(1, int(base_qty))
        if rng.random() < 0.03:  # rare bulk order (useful for anomaly detection)
            quantity += rng.integers(20, 50)

        total_price = round(unit_price * quantity, 2)
        payment_method = rng.choice(PAYMENT_METHODS, p=[0.35, 0.25, 0.2, 0.15, 0.05])

        rows.append({
            "Order ID": f"ORD{10000 + i}",
            "Date": order_date.strftime("%Y-%m-%d"),
            "Category": category,
            "Product Name": product,
            "Quantity": quantity,
            "Unit Price": unit_price,
            "Total Price": total_price,
            "Region": region,
            "Payment Method": payment_method,
        })

    df = pd.DataFrame(rows).sort_values("Date").reset_index(drop=True)
    return df


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic retail sales data.")
    parser.add_argument("--rows", type=int, default=5000, help="Number of rows to generate")
    parser.add_argument("--start", type=str, default="2023-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2024-12-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--out", type=str, default="data/raw/sales_data.csv", help="Output CSV path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    df = generate(args.rows, args.start, args.end, args.seed)
    df.to_csv(args.out, index=False)
    print(f"Generated {len(df):,} rows -> {args.out}")
    print(df.head())


if __name__ == "__main__":
    main()
