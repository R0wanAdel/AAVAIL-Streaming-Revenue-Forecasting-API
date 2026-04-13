# AAVAIL Streaming Revenue Forecasting API

A Flask-based REST API for predicting 30-day streaming revenue by country, using time-series lag features and ensemble machine learning models. Built as part of the AI Workflow Capstone.

---

## Overview

This project ingests transactional sales data, engineers time-series features, trains revenue prediction models (Random Forest, Gradient Boosting, Linear Regression baseline), and exposes predictions through a simple REST API. Prediction and training events are logged separately for test and production environments.

---

## Setup

**Prerequisites:** Python 3.8+

```bash
pip install -r requirements.txt
```

**Data:** Place JSON sales data files in `data/cs-train/`. Each file should contain transactional records with fields including `country`, `price`, `times_viewed`, `year`, `month`, and `day`.

---

## Running the API

```bash
python app.py
```

The server starts on `http://localhost:8080`.

Alternatively, using Docker:

```bash
docker build -t aavail-api .
docker run -p 8080:8080 aavail-api
```

---

## API Endpoints

### `GET /`
Returns a summary of available endpoints.

---

### `POST /train`
Trains a revenue prediction model for a given country.

**Request body:**
```json
{
  "country": "United Kingdom",
  "test": false
}
```

**Response:**
```json
{
  "status": "success",
  "country": "United Kingdom",
  "metrics": {
    "model_type": "rf",
    "rmse": 12345.67,
    "mae": 9876.54,
    "train_size": 200,
    "test_size": 50,
    "trained_at": "2024-01-15 10:30:00"
  },
  "runtime": 1.23
}
```

Use `"country": "all"` to train on aggregated data across all countries. Set `"test": true` to train in test mode (model saved separately from production).

---

### `POST /predict`
Predicts 30-day revenue for a country starting from a given date.

**Request body:**
```json
{
  "country": "France",
  "date": "2019-08-01",
  "test": false
}
```

**Response:**
```json
{
  "status": "success",
  "country": "France",
  "date": "2019-08-01",
  "predicted_revenue_30_days": 48230.50,
  "runtime": 0.05
}
```

If no trained model exists for the requested country, training is attempted automatically.

---

### `GET /logs`
Returns prediction log entries.

**Query parameters:**
- `test=true` — return test logs (default: production logs)

**Response:**
```json
{
  "status": "success",
  "test_mode": false,
  "count": 12,
  "logs": [...]
}
```

---

## Model Details

**Feature engineering** (`model.py → engineer_features`): For each day in the time series, four lag features are computed:

| Feature | Description |
|---|---|
| `prev_day` | Revenue from the previous day |
| `prev_week` | Revenue sum over the previous 7 days |
| `prev_month` | Revenue sum over the previous 30 days |
| `prev_3months` | Revenue sum over the previous 90 days |

**Target:** Total revenue over the next 30 days.

**Models available:**

| Key | Model |
|---|---|
| `rf` | Random Forest Regressor (default, 100 estimators) |
| `gb` | Gradient Boosting Regressor (100 estimators) |
| `baseline` | Linear Regression |

Trained models are saved to `models/` as `.pkl` files, named by country and mode (e.g., `model_united_kingdom_prod.pkl`).

---

## Monitoring

The `monitor.py` module computes the **Wasserstein distance** between the distribution of predicted revenues (from logs) and actual revenues (from ingested data). A higher distance indicates model drift.

```bash
python monitor.py
```

---

## Exploratory Data Analysis

Run the EDA script to generate plots saved to `eda_plots/`:

```bash
python eda.py
```

Plots generated:
- Total daily revenue over time (all countries)
- Daily revenue for the top 5 countries
- Total revenue by country (bar chart, top 15)
- 30-day rolling mean revenue
- Model comparison: RMSE and MAE across RF, GB, and baseline

---

## Testing

Run all unit tests:

```bash
python run_tests.py
```

Or run individual test files:

```bash
python -m unittest test_api.py
python -m unittest test_model.py
python -m unittest test_logger.py
```

Tests cover API endpoint validation, feature engineering correctness, and logging behavior (with isolated temp directories to avoid polluting production logs).

---

## Logging

Logs are written to CSV files in the `logs/` directory:

| File | Contents |
|---|---|
| `predict-prod.log` | Production prediction events |
| `predict-test.log` | Test prediction events |
| `train.log` | Training events (all modes) |

Each log entry captures a unique ID, timestamp, country, date, predicted value, runtime, model version, and test/prod flag.
