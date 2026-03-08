"""Built-in demo dataset for instant platform exploration."""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_demo_dataset() -> pd.DataFrame:
    """Generate a realistic 2,000-row business dataset."""
    np.random.seed(42)
    n = 2000

    dates = pd.date_range("2023-01-01", periods=24, freq="ME")
    date_col = np.random.choice(dates, n)

    categories = ["Technology", "Furniture", "Office Supplies", "Electronics", "Apparel"]
    products = {
        "Technology": ["Laptop", "Monitor", "Keyboard", "Mouse", "Webcam", "Headset", "Tablet"],
        "Furniture": ["Desk", "Chair", "Bookcase", "Table", "Cabinet", "Shelf"],
        "Office Supplies": ["Paper", "Pens", "Binders", "Stapler", "Tape", "Folders"],
        "Electronics": ["Phone", "Speaker", "Charger", "Cable", "Battery", "Adapter"],
        "Apparel": ["T-Shirt", "Jacket", "Hat", "Shoes", "Backpack"],
    }
    regions = ["North", "South", "East", "West"]
    segments = ["Consumer", "Corporate", "Home Office"]
    customers = [f"C{str(i).zfill(3)}" for i in range(1, 51)]

    cat_col = np.random.choice(categories, n, p=[0.30, 0.25, 0.20, 0.15, 0.10])
    product_col = [np.random.choice(products[c]) for c in cat_col]
    region_col = np.random.choice(regions, n, p=[0.30, 0.25, 0.25, 0.20])
    segment_col = np.random.choice(segments, n, p=[0.50, 0.30, 0.20])
    customer_col = np.random.choice(customers, n)

    base_prices = {"Technology": 400, "Furniture": 250, "Office Supplies": 30,
                   "Electronics": 80, "Apparel": 50}
    revenue = np.array([
        np.random.lognormal(np.log(base_prices[c]), 0.6) for c in cat_col
    ]).round(2)
    revenue = np.clip(revenue, 5, 15000)

    units = np.random.randint(1, 15, n)
    profit = (revenue * np.random.uniform(0.05, 0.45, n)).round(2)
    profit[np.random.random(n) < 0.12] *= -1

    df = pd.DataFrame({
        "Date": date_col,
        "Customer_ID": customer_col,
        "Product": product_col,
        "Category": cat_col,
        "Customer_Segment": segment_col,
        "Region": region_col,
        "Revenue": revenue,
        "Profit": profit,
        "Units_Sold": units,
    })

    return df.sort_values("Date").reset_index(drop=True)
