#!/usr/bin/env python3
"""
Model module for AI Workflow Capstone
Trains time-series revenue prediction models using enriched lag + seasonality features.
Models: Random Forest, Gradient Boosting, XGBoost, LightGBM, Linear Regression (baseline)
"""

import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("Warning: xgboost not installed. Run: pip install xgboost")

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("Warning: lightgbm not installed. Run: pip install lightgbm")

from ingest import fetch_data

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)


def engineer_features(ts_df):
    """
    Engineer enriched lag + seasonality features from a time-series DataFrame.

    Lag features:
        prev_day, prev_week, prev_month, prev_3months

    Rolling statistics (on 7, 30, 90-day windows):
        rolling_mean, rolling_std, rolling_min, rolling_max

    Trend features:
        trend_7d  (slope of last 7 days revenue)

    Seasonality features:
        day_of_week, day_of_month, month, quarter, is_weekend,
        week_of_year

    Target: sum of next 30-day revenue
    """
    ts_df = ts_df.sort_values("date").copy()
    ts_df = ts_df.set_index("date")

    ts_daily = ts_df["revenue"].resample("D").sum().fillna(0)

    records = []
    dates = ts_daily.index

    for i in range(90, len(dates) - 30):
        target_date = dates[i]
        window_7   = ts_daily.iloc[i - 7:i]
        window_30  = ts_daily.iloc[i - 30:i]
        window_90  = ts_daily.iloc[i - 90:i]

        # --- Lag features ---
        prev_day       = ts_daily.iloc[i - 1]
        prev_week      = window_7.sum()
        prev_month     = window_30.sum()
        prev_3months   = window_90.sum()

        # --- Rolling statistics ---
        roll7_mean  = window_7.mean()
        roll7_std   = window_7.std(ddof=0)
        roll30_mean = window_30.mean()
        roll30_std  = window_30.std(ddof=0)
        roll90_mean = window_90.mean()
        roll90_std  = window_90.std(ddof=0)
        roll30_min  = window_30.min()
        roll30_max  = window_30.max()

        # --- 7-day linear trend slope ---
        x_trend = np.arange(7)
        trend_7d = np.polyfit(x_trend, window_7.values, 1)[0]

        # --- Ratio features (avoid div-by-zero) ---
        mom_ratio = (prev_week / (window_7.shift(1).sum() + 1e-6)
                     if hasattr(window_7, 'shift') else prev_week / (prev_month / 4 + 1e-6))
        week_vs_month_mean = prev_week / (roll30_mean * 7 + 1e-6)

        # --- Seasonality features ---
        dow          = target_date.dayofweek        # 0=Mon … 6=Sun
        dom          = target_date.day
        month        = target_date.month
        quarter      = target_date.quarter
        week_of_year = target_date.isocalendar()[1]
        is_weekend   = int(dow >= 5)

        # --- Target ---
        target = ts_daily.iloc[i:i + 30].sum()

        records.append({
            "date":              target_date,
            # lags
            "prev_day":          prev_day,
            "prev_week":         prev_week,
            "prev_month":        prev_month,
            "prev_3months":      prev_3months,
            # rolling stats
            "roll7_mean":        roll7_mean,
            "roll7_std":         roll7_std,
            "roll30_mean":       roll30_mean,
            "roll30_std":        roll30_std,
            "roll90_mean":       roll90_mean,
            "roll90_std":        roll90_std,
            "roll30_min":        roll30_min,
            "roll30_max":        roll30_max,
            # trend
            "trend_7d":          trend_7d,
            # ratio
            "week_vs_month_mean": week_vs_month_mean,
            # seasonality
            "day_of_week":       dow,
            "day_of_month":      dom,
            "month":             month,
            "quarter":           quarter,
            "week_of_year":      week_of_year,
            "is_weekend":        is_weekend,
            # target
            "target":            target,
        })

    return pd.DataFrame(records)


FEATURE_COLS = [
    "prev_day", "prev_week", "prev_month", "prev_3months",
    "roll7_mean", "roll7_std", "roll30_mean", "roll30_std",
    "roll90_mean", "roll90_std", "roll30_min", "roll30_max",
    "trend_7d", "week_vs_month_mean",
    "day_of_week", "day_of_month", "month", "quarter",
    "week_of_year", "is_weekend",
]


def _build_model(model_type):
    """Return an untrained model instance for the given type."""
    if model_type == "rf":
        return RandomForestRegressor(
            n_estimators=300, max_features="sqrt",
            min_samples_leaf=2, random_state=42, n_jobs=-1
        )
    elif model_type == "gb":
        return GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05,
            max_depth=4, subsample=0.8, random_state=42
        )
    elif model_type == "xgb":
        if not HAS_XGB:
            raise ImportError("xgboost is not installed")
        return xgb.XGBRegressor(
            n_estimators=500, learning_rate=0.03,
            max_depth=5, subsample=0.8, colsample_bytree=0.8,
            min_child_weight=3, reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, n_jobs=-1, verbosity=0
        )
    elif model_type == "lgb":
        if not HAS_LGB:
            raise ImportError("lightgbm is not installed")
        return lgb.LGBMRegressor(
            n_estimators=500, learning_rate=0.03,
            num_leaves=31, subsample=0.8, colsample_bytree=0.8,
            min_child_samples=10, reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, n_jobs=-1, verbose=-1
        )
    elif model_type == "baseline":
        return LinearRegression()
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def get_model_fname(country, mode="prod"):
    country_clean = country.replace(" ", "_").lower()
    return os.path.join(MODEL_DIR, f"model_{country_clean}_{mode}.pkl")


def train_model(country="all", data_dir=None, test=False, model_type="xgb"):
    """
    Train a model for the given country.
    model_type: 'rf' | 'gb' | 'xgb' | 'lgb' | 'baseline'
    Returns (trained_model, metrics_dict).
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

    # Use only columns that exist (fallback for old data)
    available_cols = [c for c in FEATURE_COLS if c in features_df.columns]
    X = features_df[available_cols].values
    y = features_df["target"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = _build_model(model_type)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = np.mean(np.abs(y_test - y_pred))
    mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-6))) * 100

    # Use mean 30-day revenue as denominator for RMSE% and MAE%
    # (y values are 30-day revenue sums, so mean(y) is the right scale reference)
    mean_30d = np.mean(y_test) if len(y_test) > 0 else 1.0
    rmse_pct = (rmse / mean_30d) * 100
    mae_pct  = (mae  / mean_30d) * 100

    metrics = {
        "model_type":    model_type,
        "country":       country,
        "rmse_pct":      round(rmse_pct, 2),
        "mae_pct":       round(mae_pct,  2),
        "mape":          round(mape,     2),
        "mean_30d_revenue": round(mean_30d, 2),
        "train_size":    len(X_train),
        "test_size":     len(X_test),
        "n_features":    len(available_cols),
        "trained_at":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    mode = "test" if test else "prod"
    model_fname = get_model_fname(country, mode)

    model_data = {
        "model":        model,
        "features":     available_cols,
        "metrics":      metrics,
        "country":      country,
        "features_df":  features_df,
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
    model_fname = get_model_fname(country, mode)

    if not os.path.exists(model_fname):
        model, metrics = train_model(country=country, data_dir=data_dir, test=test)
        if model is None:
            return None

    with open(model_fname, "rb") as f:
        model_data = pickle.load(f)

    model    = model_data["model"]
    features = model_data["features"]

    df = fetch_data(data_dir)

    if country == "all":
        ts = df.groupby("date")["revenue"].sum().reset_index()
    else:
        ts = df[df["country"] == country][["date", "revenue"]].copy()

    ts["date"] = pd.to_datetime(ts["date"])
    ts = ts.sort_values("date").set_index("date")
    ts_daily = ts["revenue"].resample("D").sum().fillna(0)

    target_date = pd.to_datetime(date)
    if target_date not in ts_daily.index:
        target_date = ts_daily.index[-1]

    idx = ts_daily.index.get_loc(target_date)
    if idx < 90:
        return None

    window_7  = ts_daily.iloc[idx - 7:idx]
    window_30 = ts_daily.iloc[idx - 30:idx]
    window_90 = ts_daily.iloc[idx - 90:idx]

    x_trend = np.arange(7)
    trend_7d = np.polyfit(x_trend, window_7.values, 1)[0]
    roll30_mean = window_30.mean()
    prev_week   = window_7.sum()

    row = {
        "prev_day":           ts_daily.iloc[idx - 1],
        "prev_week":          prev_week,
        "prev_month":         window_30.sum(),
        "prev_3months":       window_90.sum(),
        "roll7_mean":         window_7.mean(),
        "roll7_std":          window_7.std(ddof=0),
        "roll30_mean":        roll30_mean,
        "roll30_std":         window_30.std(ddof=0),
        "roll90_mean":        window_90.mean(),
        "roll90_std":         window_90.std(ddof=0),
        "roll30_min":         window_30.min(),
        "roll30_max":         window_30.max(),
        "trend_7d":           trend_7d,
        "week_vs_month_mean": prev_week / (roll30_mean * 7 + 1e-6),
        "day_of_week":        target_date.dayofweek,
        "day_of_month":       target_date.day,
        "month":              target_date.month,
        "quarter":            target_date.quarter,
        "week_of_year":       target_date.isocalendar()[1],
        "is_weekend":         int(target_date.dayofweek >= 5),
    }

    X = np.array([[row[f] for f in features]])
    return float(model.predict(X)[0])


def compare_models(country="all", data_dir=None):
    """
    Compare all available model types. Returns dict keyed by model type.
    """
    available = ["baseline", "rf", "gb"]
    if HAS_XGB:
        available.append("xgb")
    if HAS_LGB:
        available.append("lgb")

    results = {}
    for mtype in available:
        _, metrics = train_model(country=country, data_dir=data_dir, test=True, model_type=mtype)
        results[mtype] = metrics
    return results


if __name__ == "__main__":
    print("Training XGBoost model for all countries...")
    model, metrics = train_model(country="all", model_type="xgb")
    print(metrics)

    print("\nComparing all models...")
    comparison = compare_models(country="all")
    print(f"\n{'Model':<10} {'RMSE%':>8} {'MAE%':>8} {'MAPE':>8}")
    print("-" * 38)
    for mtype, m in comparison.items():
        rmse_pct = m.get("rmse_pct", float("nan"))
        mae_pct  = m.get("mae_pct",  float("nan"))
        mape     = m.get("mape",     float("nan"))
        print(f"{mtype.upper():<10} {rmse_pct:>7.1f}% {mae_pct:>7.1f}% {mape:>7.1f}%")