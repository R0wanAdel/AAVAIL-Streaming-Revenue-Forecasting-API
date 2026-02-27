#!/usr/bin/env python3
"""
Logging module for AI Workflow Capstone
Logs predictions with inputs, outputs, and runtime.
Separates test vs production logs.
"""

import os
import csv
import time
import uuid
from datetime import datetime

LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

TRAIN_LOG = os.path.join(LOG_DIR, "train.log")
PREDICT_PROD_LOG = os.path.join(LOG_DIR, "predict-prod.log")
PREDICT_TEST_LOG = os.path.join(LOG_DIR, "predict-test.log")


def _get_predict_log(test=False):
    return PREDICT_TEST_LOG if test else PREDICT_PROD_LOG


def update_train_log(country, date, eval_test, runtime, model_version, test=False):
    """
    Log a training event.
    """
    log_file = TRAIN_LOG
    unique_id = str(uuid.uuid4())[:8]
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    header = ["unique_id", "timestamp", "country", "date", "eval_test", "runtime_seconds",
              "model_version", "test_mode"]
    row = [unique_id, ts, country, str(date), str(eval_test),
           f"{runtime:.4f}", model_version, str(test)]

    file_exists = os.path.exists(log_file)
    with open(log_file, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(row)


def update_predict_log(country, date, y_pred, runtime, model_version, test=False):
    """
    Log a prediction event. Separate logs for test vs prod.
    """
    log_file = _get_predict_log(test)
    unique_id = str(uuid.uuid4())[:8]
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    header = ["unique_id", "timestamp", "country", "date", "y_pred",
              "runtime_seconds", "model_version", "test_mode"]
    row = [unique_id, ts, country, str(date), f"{y_pred:.2f}",
           f"{runtime:.4f}", model_version, str(test)]

    file_exists = os.path.exists(log_file)
    with open(log_file, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(row)


def load_train_log():
    """Load training log as a list of dicts."""
    if not os.path.exists(TRAIN_LOG):
        return []
    with open(TRAIN_LOG, "r") as f:
        reader = csv.DictReader(f)
        return list(reader)


def load_predict_log(test=False):
    """Load prediction log as a list of dicts."""
    log_file = _get_predict_log(test)
    if not os.path.exists(log_file):
        return []
    with open(log_file, "r") as f:
        reader = csv.DictReader(f)
        return list(reader)


if __name__ == "__main__":
    # Test logging
    update_train_log("all", "2019-01-01", {"rmse": 100.0}, 1.23, "1.0", test=True)
    update_predict_log("all", "2019-01-01", 50000.0, 0.05, "1.0", test=True)
    print("Train log:", load_train_log())
    print("Predict log (test):", load_predict_log(test=True))
