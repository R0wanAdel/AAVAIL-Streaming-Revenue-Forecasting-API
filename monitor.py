"""
Monitoring module for AI Workflow Capstone
Uses Wasserstein distance to compare predicted vs actual distributions.
"""

import numpy as np
from scipy.stats import wasserstein_distance
from logger import load_predict_log
from ingest import fetch_data


def get_wasserstein_distance(y_pred, y_true):
    """
    Compute Wasserstein distance between predicted and actual revenue distributions.
    """
    return wasserstein_distance(y_pred, y_true)


def monitor_performance(country="all", test=False):
    """
    Compare predicted revenue from logs against actual revenue from data.
    Returns a dict with Wasserstein distance and summary stats.
    """
    # Load prediction logs
    logs = load_predict_log(test=test)
    if not logs:
        return {"error": "No prediction logs found"}

    y_pred = [float(row["y_pred"]) for row in logs if row.get("country") == country or country == "all"]

    if not y_pred:
        return {"error": f"No predictions found for country: {country}"}

    # Load actual data
    df = fetch_data()
    if country == "all":
        actuals = df.groupby("date")["revenue"].sum().values
    else:
        actuals = df[df["country"] == country]["revenue"].values

    if len(actuals) == 0:
        return {"error": "No actual data found"}

    # Use overlapping sample sizes
    n = min(len(y_pred), len(actuals))
    y_pred_sample = np.array(y_pred[:n])
    y_true_sample = actuals[:n]

    wd = get_wasserstein_distance(y_pred_sample, y_true_sample)

    return {
        "country": country,
        "wasserstein_distance": wd,
        "n_predictions": len(y_pred),
        "pred_mean": np.mean(y_pred),
        "pred_std": np.std(y_pred),
        "actual_mean": np.mean(actuals),
        "actual_std": np.std(actuals),
    }


if __name__ == "__main__":
    result = monitor_performance(country="all", test=True)
    print(result)
