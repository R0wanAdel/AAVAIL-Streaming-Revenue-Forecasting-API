"""
EDA Script for AI Workflow Capstone
Investigates revenue data, generates visualizations, and compares models to baseline.
Run this script to produce EDA plots.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from model import engineer_features, train_model, compare_models

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "eda_plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_eda():
    from ingest import fetch_data
    print("Loading data...")
    df = fetch_data()

    print(f"\nShape: {df.shape}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    days_span = (pd.to_datetime(df['date'].max()) - pd.to_datetime(df['date'].min())).days
    print(f"Total days span: {days_span}")
    print(f"Countries: {df['country'].nunique()}")

    # Top countries by revenue
    top_countries = df.groupby("country")["revenue"].sum().sort_values(ascending=False)
    print("\nTop 10 countries by total revenue:")
    print(top_countries.head(10))

    # ---- PLOT 1: Total revenue over time (all countries) ----
    ts_all = df.groupby("date")["revenue"].sum().reset_index()
    ts_all["date"] = pd.to_datetime(ts_all["date"])

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(ts_all["date"], ts_all["revenue"], color="steelblue", linewidth=1)
    ax.set_title("Total Daily Revenue - All Countries")
    ax.set_xlabel("Date")
    ax.set_ylabel("Revenue")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "total_revenue_over_time.png"))
    plt.close()
    print("\nSaved: total_revenue_over_time.png")

    # ---- PLOT 2: Top 5 countries revenue over time ----
    top5 = top_countries.head(5).index.tolist()
    fig, ax = plt.subplots(figsize=(14, 6))
    for country in top5:
        ts_c = df[df["country"] == country].groupby("date")["revenue"].sum().reset_index()
        ts_c["date"] = pd.to_datetime(ts_c["date"])
        ax.plot(ts_c["date"], ts_c["revenue"], label=country, linewidth=1)
    ax.set_title("Daily Revenue - Top 5 Countries")
    ax.set_xlabel("Date")
    ax.set_ylabel("Revenue")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "top5_countries_revenue.png"))
    plt.close()
    print("Saved: top5_countries_revenue.png")

    # ---- PLOT 3: Revenue by country (bar chart) ----
    fig, ax = plt.subplots(figsize=(12, 6))
    top_countries.head(15).plot(kind="bar", ax=ax, color="coral")
    ax.set_title("Total Revenue by Country (Top 15)")
    ax.set_xlabel("Country")
    ax.set_ylabel("Total Revenue")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "revenue_by_country.png"))
    plt.close()
    print("Saved: revenue_by_country.png")

    # ---- PLOT 4: 30-day rolling average ----
    ts_all_indexed = ts_all.set_index("date")["revenue"]
    rolling_mean = ts_all_indexed.rolling(30).mean()

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(ts_all_indexed.index, ts_all_indexed.values, alpha=0.4, label="Daily", color="steelblue")
    ax.plot(rolling_mean.index, rolling_mean.values, label="30-Day Rolling Mean", color="red", linewidth=2)
    ax.set_title("Daily Revenue with 30-Day Rolling Mean")
    ax.set_xlabel("Date")
    ax.set_ylabel("Revenue")
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "rolling_mean_revenue.png"))
    plt.close()
    print("Saved: rolling_mean_revenue.png")

    # ---- PLOT 5: Model comparison (RF vs GB vs Baseline) ----
    print("\nComparing models... (this may take a moment)")
    try:
        comparison = compare_models(country="all")
        model_names = list(comparison.keys())
        rmse_vals = [comparison[m].get("rmse", 0) for m in model_names]
        mae_vals = [comparison[m].get("mae", 0) for m in model_names]

        x = np.arange(len(model_names))
        width = 0.35

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].bar(x, rmse_vals, width, color=["gray", "steelblue", "coral"])
        axes[0].set_title("Model Comparison - RMSE (lower is better)")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels([m.upper() for m in model_names])
        axes[0].set_ylabel("RMSE")

        axes[1].bar(x, mae_vals, width, color=["gray", "steelblue", "coral"])
        axes[1].set_title("Model Comparison - MAE (lower is better)")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels([m.upper() for m in model_names])
        axes[1].set_ylabel("MAE")

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "model_comparison.png"))
        plt.close()
        print("Saved: model_comparison.png")

        print("\nModel results:")
        for mname, mvals in comparison.items():
            print(f"  {mname.upper()}: RMSE={mvals.get('rmse', 'N/A'):.2f}, MAE={mvals.get('mae', 'N/A'):.2f}")

    except Exception as e:
        print(f"Could not run model comparison (need data): {e}")

    print(f"\nAll EDA plots saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    run_eda()
