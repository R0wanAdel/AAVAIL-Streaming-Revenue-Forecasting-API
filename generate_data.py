#!/usr/bin/env python3
"""
Generates synthetic e-commerce revenue data in the format expected by ingest.py.
Output: multiple JSON files in data/cs-train/, one per month.
Each record has: country, customer_id, invoice, price, stream_id, times_viewed, year, month, day
"""

import os
import json
import random
import numpy as np
from datetime import date, timedelta

random.seed(42)
np.random.seed(42)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data", "cs-train")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Countries with realistic revenue weights and seasonal patterns
COUNTRIES = {
    "United Kingdom":  {"weight": 0.40, "base": 3500, "growth": 0.0008},
    "Germany":         {"weight": 0.12, "base": 1200, "growth": 0.0006},
    "France":          {"weight": 0.10, "base": 1000, "growth": 0.0005},
    "EIRE":            {"weight": 0.08, "base":  800, "growth": 0.0004},
    "Netherlands":     {"weight": 0.05, "base":  500, "growth": 0.0004},
    "Spain":           {"weight": 0.04, "base":  400, "growth": 0.0003},
    "Belgium":         {"weight": 0.03, "base":  350, "growth": 0.0003},
    "Switzerland":     {"weight": 0.03, "base":  300, "growth": 0.0002},
    "Portugal":        {"weight": 0.02, "base":  250, "growth": 0.0002},
    "Australia":       {"weight": 0.02, "base":  220, "growth": 0.0002},
    "Norway":          {"weight": 0.02, "base":  200, "growth": 0.0002},
    "Sweden":          {"weight": 0.02, "base":  190, "growth": 0.0001},
    "Denmark":         {"weight": 0.02, "base":  180, "growth": 0.0001},
    "Japan":           {"weight": 0.02, "base":  150, "growth": 0.0002},
    "USA":             {"weight": 0.01, "base":  120, "growth": 0.0003},
}

COUNTRY_NAMES = list(COUNTRIES.keys())
COUNTRY_WEIGHTS = [COUNTRIES[c]["weight"] for c in COUNTRY_NAMES]

# Stream IDs (product types)
STREAM_IDS = [f"S{str(i).zfill(4)}" for i in range(1, 201)]

# Price tiers (realistic e-commerce prices)
PRICE_TIERS = [1.25, 2.50, 3.75, 4.99, 6.50, 8.99, 12.50, 15.00, 19.99, 24.99, 29.99, 49.99]


def seasonal_multiplier(d: date) -> float:
    """Return a revenue multiplier based on day of year (captures seasonality)."""
    doy = d.timetuple().tm_yday
    # Christmas peak (Nov-Dec), summer dip (Jul-Aug), spring bump (Mar-Apr)
    base = 1.0
    # Christmas ramp up
    if d.month == 11:
        base += 0.3 * (d.day / 30)
    elif d.month == 12 and d.day <= 20:
        base += 0.6
    elif d.month == 12 and d.day > 20:
        base -= 0.4  # Christmas Eve / Christmas drop
    # Summer dip
    elif d.month in (7, 8):
        base -= 0.15
    # Spring bump
    elif d.month in (3, 4):
        base += 0.1
    # January slump
    elif d.month == 1:
        base -= 0.2
    # Weekend bump
    if d.weekday() >= 5:
        base += 0.08
    return max(base, 0.3)


def growth_multiplier(d: date, start: date, daily_growth: float) -> float:
    days_elapsed = (d - start).days
    return 1.0 + daily_growth * days_elapsed


def generate_day_records(d: date, start_date: date) -> list:
    """Generate transaction records for a single day across all countries."""
    records = []
    season = seasonal_multiplier(d)
    invoice_counter = int(d.strftime("%Y%m%d")) * 1000

    for country in COUNTRY_NAMES:
        cfg = COUNTRIES[country]
        growth = growth_multiplier(d, start_date, cfg["growth"])
        base_txns = cfg["base"] * season * growth

        # Number of transactions: Poisson-distributed around base
        n_txns = max(1, int(np.random.poisson(base_txns / 15)))

        for _ in range(n_txns):
            price = random.choice(PRICE_TIERS)
            times_viewed = max(1, int(np.random.poisson(12)))
            # Add noise: some transactions have low views, some viral
            if random.random() < 0.05:
                times_viewed = random.randint(50, 300)

            records.append({
                "country":      country,
                "customer_id":  f"C{random.randint(10000, 99999)}",
                "invoice":      str(invoice_counter),
                "price":        round(price * (1 + np.random.normal(0, 0.05)), 2),
                "stream_id":    random.choice(STREAM_IDS),
                "times_viewed": times_viewed,
                "year":         d.year,
                "month":        d.month,
                "day":          d.day,
            })
            invoice_counter += 1

    return records


def generate_dataset(start_year=2017, end_year=2019):
    start_date = date(start_year, 1, 1)
    end_date   = date(end_year, 12, 31)

    current = start_date
    month_records = {}

    total_days = (end_date - start_date).days + 1
    print(f"Generating data from {start_date} to {end_date} ({total_days} days)...")

    while current <= end_date:
        # Skip Christmas Day (realistically near-zero)
        if current.month == 12 and current.day == 25:
            current += timedelta(days=1)
            continue

        records = generate_day_records(current, start_date)
        key = (current.year, current.month)
        if key not in month_records:
            month_records[key] = []
        month_records[key].extend(records)
        current += timedelta(days=1)

    # Write one JSON file per month
    files_written = 0
    total_records = 0
    for (year, month), records in sorted(month_records.items()):
        fname = f"data_{year}_{str(month).zfill(2)}.json"
        fpath = os.path.join(OUTPUT_DIR, fname)
        with open(fpath, "w") as f:
            json.dump(records, f)
        files_written += 1
        total_records += len(records)
        print(f"  Wrote {fname}: {len(records):,} records")

    print(f"\nDone: {files_written} files, {total_records:,} total records")
    print(f"Output directory: {OUTPUT_DIR}")
    return total_records


if __name__ == "__main__":
    generate_dataset(start_year=2017, end_year=2019)

    # Quick sanity check via ingest
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    try:
        from ingest import fetch_data
        df = fetch_data()
        print(f"\n--- Sanity Check ---")
        print(f"Shape: {df.shape}")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"Countries: {df['country'].nunique()}")
        print(f"Total revenue: £{df['revenue'].sum():,.0f}")
        print(f"\nTop 5 countries by revenue:")
        top = df.groupby("country")["revenue"].sum().sort_values(ascending=False).head(5)
        for country, rev in top.items():
            print(f"  {country}: £{rev:,.0f}")
        uk = df[df["country"] == "United Kingdom"]
        print(f"\nUK data points: {len(uk)}")
    except Exception as e:
        print(f"Sanity check error: {e}")