#!/usr/bin/env python3
"""
Model module for AI Workflow Capstone
Trains a time-series revenue prediction model using engineered lag features + Random Forest
Also compares against a baseline model.
"""

import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from ingest import fetch_data

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)


def engineer_features(ts_df):
    """
    Engineer lag features from a time-series DataFrame with columns [date, revenue].
    Features: prev_day, prev_week, prev_month, prev_3months
    Target: next 30-day revenue sum
    """
    ts_df = ts_df.sort_values("date").copy()
    ts_df = ts_df.set_index("date")

    # Resample to daily, fill missing with 0
    ts_daily = ts_df["revenue"].resample("D").sum().fillna(0)

    records = []
    dates = ts_daily.index

    for i in range(90, len(dates) - 30):
        target_date = dates[i]
        prev_day = ts_daily.iloc[i - 1]
        prev_week = ts_daily.iloc[i - 7:i].sum()
        prev_month = ts_daily.iloc[i - 30:i].sum()
        prev_3months = ts_daily.iloc[i - 90:i].sum()
        target = ts_daily.iloc[i:i + 30].sum()

        records.append({
            "date": target_date,
            "prev_day": prev_day,
            "prev_week": prev_week,
            "prev_month": prev_month,
            "prev_3months": prev_3months,
            "target": target
        })

    return pd.DataFrame(records)


def get_model_fname(country, training=True):
    tag = "test" if not training else "prod"
    country_clean = country.replace(" ", "_").lower()
    return os.path.join(MODEL_DIR, f"model_{country_clean}_{tag}.pkl")


def train_model(country="all", data_dir=None, test=False, model_type="rf"):
    """
    Train a model for the given country.
    model_type: 'rf' (Random Forest), 'gb' (Gradient Boosting), 'baseline' (Linear Regression)
    Returns trained model and performance metrics.
    """
    from ingest import fetch_data, DATA_DIR
    if data_dir is None:
        data_dir = DATA_DIR

    df = fetch_data(data_dir)

    if country == "all":
        ts = df.groupby("date")["revenue"].sum().reset_index()
    else:
        ts = df[df["country"] == country][["date", "revenue"]].copy()

    ts["date"] = pd.to_datetime(ts["date"])

    if len(ts) < 120:
        return None, {"error": f"Not enough data for country: {country}"}

    features_df = engineer_features(ts)

    if features_df.empty or len(features_df) < 10:
        return None, {"error": "Not enough engineered features"}

    feature_cols = ["prev_day", "prev_week", "prev_month", "prev_3months"]
    X = features_df[feature_cols].values
    y = features_df["target"].values
    dates = features_df["date"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Select model
    if model_type == "rf":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_type == "gb":
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    else:  # baseline
        model = LinearRegression()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = np.mean(np.abs(y_test - y_pred))

    metrics = {
        "model_type": model_type,
        "country": country,
        "rmse": rmse,
        "mae": mae,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    # Save model
    mode = "test" if test else "prod"
    country_clean = country.replace(" ", "_").lower()
    model_fname = os.path.join(MODEL_DIR, f"model_{country_clean}_{mode}.pkl")

    model_data = {
        "model": model,
        "features": feature_cols,
        "metrics": metrics,
        "ts_daily": ts.groupby("date")["revenue"].sum() if "revenue" in ts.columns else None,
        "features_df": features_df,
        "country": country
    }

    with open(model_fname, "wb") as f:
        pickle.dump(model_data, f)

    return model, metrics


def predict(country, date, test=False, data_dir=None):
    """
    Predict 30-day revenue for a given country and date.
    """
    from ingest import fetch_data, DATA_DIR
    if data_dir is None:
        data_dir = DATA_DIR

    mode = "test" if test else "prod"
    country_clean = country.replace(" ", "_").lower()
    model_fname = os.path.join(MODEL_DIR, f"model_{country_clean}_{mode}.pkl")

    if not os.path.exists(model_fname):
        # Try to train on the fly
        model, metrics = train_model(country=country, data_dir=data_dir, test=test)
        if model is None:
            return None

    with open(model_fname, "rb") as f:
        model_data = pickle.load(f)

    model = model_data["model"]

    # Load data to build features for the given date
    df = fetch_data(data_dir)

    if country == "all":
        ts = df.groupby("date")["revenue"].sum().reset_index()
    else:
        ts = df[df["country"] == country][["date", "revenue"]].copy()

    ts["date"] = pd.to_datetime(ts["date"])
    ts = ts.sort_values("date").set_index("date")
    ts_daily = ts["revenue"].resample("D").sum().fillna(0)

    target_date = pd.to_datetime(date)

    # Find position
    if target_date not in ts_daily.index:
        # Use the last available date
        target_date = ts_daily.index[-1]

    idx = ts_daily.index.get_loc(target_date)

    if idx < 90:
        return None

    prev_day = ts_daily.iloc[idx - 1]
    prev_week = ts_daily.iloc[idx - 7:idx].sum()
    prev_month = ts_daily.iloc[idx - 30:idx].sum()
    prev_3months = ts_daily.iloc[idx - 90:idx].sum()

    X = np.array([[prev_day, prev_week, prev_month, prev_3months]])
    prediction = model.predict(X)[0]

    return float(prediction)


def compare_models(country="all", data_dir=None):
    """
    Compare RF, GB, and baseline model performance.
    Returns a dict with metrics for each model type.
    """
    results = {}
    for mtype in ["baseline", "rf", "gb"]:
        _, metrics = train_model(country=country, data_dir=data_dir, test=True, model_type=mtype)
        results[mtype] = metrics
    return results


if __name__ == "__main__":
    print("Training model for all countries...")
    model, metrics = train_model(country="all")
    print(metrics)
    print("\nComparing models...")
    comparison = compare_models(country="all")
    for mtype, m in comparison.items():
        print(f"{mtype}: RMSE={m.get('rmse', 'N/A'):.2f}")
