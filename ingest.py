#!/usr/bin/env python3
"""
Data ingestion module for AI Workflow Capstone
Loads and processes JSON sales data files into a unified DataFrame
"""

import os
import json
import re
import numpy as np
import pandas as pd
from datetime import datetime


DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "cs-train")


def load_json_data(data_dir=DATA_DIR):
    """
    Load all JSON files from the data directory and compile into a single DataFrame.
    Returns a DataFrame with columns: country, date, revenue, purchases
    """
    if not os.path.exists(data_dir):
        raise Exception(f"Data directory not found: {data_dir}")

    json_files = [f for f in os.listdir(data_dir) if f.endswith(".json")]
    if not json_files:
        raise Exception(f"No JSON files found in {data_dir}")

    dfs = []
    for fname in json_files:
        fpath = os.path.join(data_dir, fname)
        with open(fpath, "r") as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    return combined


def process_dataframe(df):
    """
    Clean and process the raw DataFrame.
    - Parse dates
    - Remove nulls
    - Standardize country names
    - Aggregate by country + date
    """
    # Rename columns if needed
    df.columns = [c.lower().strip() for c in df.columns]

    # Ensure required columns exist
    # Typical columns: country, customer_id, invoice, price, stream_id, times_viewed, year, month, day
    if "price" in df.columns and "times_viewed" in df.columns:
        # revenue = price * times_viewed per transaction
        df["revenue"] = df["price"] * df["times_viewed"]
    elif "revenue" not in df.columns:
        raise Exception("Cannot determine revenue from columns: " + str(df.columns.tolist()))

    # Build date column
    if "date" not in df.columns:
        df["date"] = pd.to_datetime(df[["year", "month", "day"]])
    else:
        df["date"] = pd.to_datetime(df["date"])

    # Clean country names
    df["country"] = df["country"].str.strip()

    # Remove rows with missing critical fields
    df = df.dropna(subset=["country", "date", "revenue"])
    df = df[df["revenue"] > 0]

    return df


def aggregate_data(df):
    """
    Aggregate revenue by country and date.
    Returns a DataFrame with columns: country, date, revenue, purchases
    """
    agg = df.groupby(["country", "date"]).agg(
        revenue=("revenue", "sum"),
        purchases=("revenue", "count")
    ).reset_index()
    agg = agg.sort_values(["country", "date"])
    return agg


def fetch_data(data_dir=DATA_DIR):
    """
    Main ingestion function - loads, processes, and aggregates data.
    Returns aggregated DataFrame.
    """
    df_raw = load_json_data(data_dir)
    df_processed = process_dataframe(df_raw)
    df_agg = aggregate_data(df_processed)
    return df_agg


def get_ts_data(country, data_dir=DATA_DIR, training=True):
    """
    Get time-series data for a specific country (or all countries combined).
    Returns dates array and revenue array for model training.
    """
    df = fetch_data(data_dir)

    if country == "all":
        ts = df.groupby("date")["revenue"].sum().reset_index()
    else:
        ts = df[df["country"] == country][["date", "revenue"]].copy()

    ts = ts.sort_values("date")
    ts["date"] = pd.to_datetime(ts["date"])

    if ts.empty:
        return None, None

    return ts["date"].values, ts["revenue"].values


if __name__ == "__main__":
    print("Loading data...")
    try:
        df = fetch_data()
        print(f"Shape: {df.shape}")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"Countries: {df['country'].nunique()}")
        print(f"Top countries by revenue:")
        print(df.groupby("country")["revenue"].sum().sort_values(ascending=False).head(10))
    except Exception as e:
        print(f"Error: {e}")
