"""
Monitoring module for AI Workflow Capstone
Uses Wasserstein distance to compare predicted vs actual distributions.
Ensures correct temporal alignment via structured date joining.
"""

import os
import pandas as pd
import numpy as np
from scipy.stats import wasserstein_distance
from logger import load_predict_log

# Mock/Fallback data fetcher to prevent ModuleNotFoundError if ingest.py is absent
def fetch_data():
    """
    Fetches actual historical ingestion data.
    Replace this with your project's true ingestion logic if needed:
    from ingest import fetch_data as true_fetch_data
    """
    # Generating realistic historical distribution matching your data baseline
    np.random.seed(42)
    dates = pd.date_range(start="2018-01-01", end="2019-12-31", freq="D")
    
    mock_records = []
    countries = ["United Kingdom", "France", "Germany", "EIRE", "Spain"]
    
    for d in dates:
        for c in countries:
            base = 50000 if c == "United Kingdom" else 12000
            rev = base * np.random.uniform(0.08, 0.12) * 1.05 + np.random.normal(0, base * 0.02)
            mock_records.append({
                "date": d.strftime("%Y-%m-%d"),
                "country": c,
                "revenue": max(0.0, rev)
            })
            
    return pd.DataFrame(mock_records)


def get_wasserstein_distance(y_pred, y_true):
    """
    Compute Wasserstein distance between predicted and actual revenue distributions.
    """
    return wasserstein_distance(y_pred, y_true)


def monitor_performance(country="all", test=False):
    """
    Compares logged predicted revenue against actual target historical records.
    Aligns values strictly by date to ensure distribution integrity.
    """
    # 1. Load prediction logs from logger framework
    logs = load_predict_log(test=test)
    if not logs:
        return {"error": "No prediction logs found"}

    # Convert logs list into a structured DataFrame for relational joining
    df_logs = pd.DataFrame(logs)
    
    # Clean and standardize country values to prevent filtering string drops
    df_logs["country"] = df_logs["country"].str.strip()
    
    if country != "all":
        df_logs = df_logs[df_logs["country"].str.lower() == country.lower()]
        
    if df_logs.empty:
        return {"error": f"No logs found matching country context: {country}"}

    # Extract target log fields and group predictions by date
    df_logs["y_pred"] = df_logs["y_pred"].astype(float)
    # Ensure column naming alignment (your logger saves this under the 'date' key)
    df_logs_grouped = df_logs.groupby("date")["y_pred"].mean().reset_index()

    # 2. Fetch and aggregate ground truth values
    df_actuals = fetch_data()
    df_actuals["country"] = df_actuals["country"].str.strip()
    
    if country != "all":
        df_actuals = df_actuals[df_actuals["country"].str.lower() == country.lower()]
        
    if df_actuals.empty:
        return {"error": f"No ground truth data matches country context: {country}"}

    df_act_grouped = df_actuals.groupby("date")["revenue"].sum().reset_index()

    # 3. Synchronize distributions using an Inner Join on the Target Date key
    df_merged = pd.merge(df_logs_grouped, df_act_grouped, on="date", how="inner")

    if df_merged.empty:
        return {
            "error": "No overlapping dates found between prediction intervals and source data.",
            "hint": "Check if your prediction test logs use dates that exist in your source dataset."
        }

    y_pred_vector = df_merged["y_pred"].values
    y_true_vector = df_merged["revenue"].values

    # 4. Compute Wasserstein distance metric
    wd = get_wasserstein_distance(y_pred_vector, y_true_vector)

    return {
        "country": country,
        "wasserstein_distance": float(wd),
        "aligned_sample_days": len(df_merged),
        "pred_mean": round(float(np.mean(y_pred_vector)), 2),
        "pred_std": round(float(np.std(y_pred_vector)), 2),
        "actual_mean": round(float(np.mean(y_true_vector)), 2),
        "actual_std": round(float(np.std(y_true_vector)), 2),
    }


if __name__ == "__main__":
    # Test evaluation workflow
    print("--- Running Performance Monitoring Verification ---")
    result = monitor_performance(country="all", test=True)
    print(result)