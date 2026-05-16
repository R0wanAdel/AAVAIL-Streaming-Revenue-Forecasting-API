# AAVAIL Streaming Revenue Forecasting API

A Flask-based REST API for predicting 30-day streaming revenue by country using ensemble machine learning models with a StandardScaler pipeline. Built as part of the AI Workflow Capstone.

---

## Overview

This project ingests transactional sales data, engineers time-series features, trains revenue prediction models using scikit-learn Pipelines, and exposes predictions through a REST API. Prediction and training events are logged separately for test and production environments. A monitoring module tracks model drift via Wasserstein distance.

---

## Setup

**Prerequisites:** Python 3.8+

```bash
pip install -r requirements.txt
```

**Data:** Place JSON sales data files in `data/cs-train/`. Each file should contain transactional records with fields including `country`, `price`, `times_viewed`, `year`, `month`, and `day`. To generate synthetic data:

```bash
python generate_data.py
```

This produces monthly JSON files in `data/cs-train/` covering 2017–2019.

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
Returns the web UI dashboard.

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
    "model_type": "Random Forest Regressor",
    "mae_pct": 4.21,
    "rmse_pct": 5.87,
    "mape": 0.0423,
    "train_size": 320,
    "n_features": 3,
    "trained_at": "2024-01-15 10:30:00"
  },
  "runtime": 1.23
}
```

Use `"country": "all"` to train on aggregated data across all countries. Set `"test": true` to train in test mode (uses fewer estimators; model saved separately from production).

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

Each log entry includes `timestamp`, `country`, `target_date`, `y_pred`, `runtime`, and `model_version`.

---

## Model Details

**Architecture:** scikit-learn `Pipeline` with `StandardScaler` + `RandomForestRegressor`.

**Features** (3 input features per sample):

| Feature | Description |
|---|---|
| `past_7d_rev` | Revenue over the prior 7 days |
| `past_30d_rev` | Revenue over the prior 30 days |
| `sub_share` | Active subscriber count share |

**Target:** Total revenue over the next 30 days.

**Evaluation metrics returned at training time:**

| Metric | Description |
|---|---|
| `mae_pct` | Mean Absolute Error as % of mean actual revenue |
| `rmse_pct` | Root Mean Square Error as % of mean actual revenue |
| `mape` | Mean Absolute Percentage Error |

**Model variants by mode:**

| Mode | Estimators | Saved as |
|---|---|---|
| Production (`test=false`) | 150 | `<country>_prod.pkl` |
| Test (`test=true`) | 50 | `<country>_test.pkl` |

Trained models are saved to `models/` as `.pkl` files (e.g., `united_kingdom_prod.pkl`).

---

## Monitoring

The `monitor.py` module computes the **Wasserstein distance** between the distribution of predicted revenues (from logs) and actual revenues (from ingested data). A higher distance indicates model drift.

```bash
python monitor.py
```

Returns country, Wasserstein distance, prediction count, and mean/std for both predicted and actual distributions.

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

**Test coverage:**
- `test_api.py` — API endpoint validation (missing fields, bad payloads, logs endpoint)
- `test_model.py` — Feature engineering shape, edge cases (insufficient data), prediction type checks
- `test_logger.py` — Log file creation, content correctness, test/prod separation, multi-entry logging; uses isolated temp directories to avoid polluting production logs

---

## Logging

Logs are written to CSV files in the `logs/` directory:

| File | Contents |
|---|---|
| `predict-prod.log` | Production prediction events |
| `predict-test.log` | Test prediction events |
| `train.log` | Training events (all modes) |

Each log entry captures: `unique_id`, `timestamp`, `country`, `date`, `y_pred` or `eval_test`, `runtime_seconds`, `model_version`, and `test_mode`.

---

## Project Structure

```
.
├── app.py              # Flask API (/, /train, /predict, /logs)
├── model.py            # Training pipeline and prediction logic
├── ingest.py           # Data loading and aggregation
├── logger.py           # CSV logging (train + predict, test + prod)
├── monitor.py          # Wasserstein drift detection
├── eda.py              # Exploratory data analysis and plots
├── generate_data.py    # Synthetic data generator (2017–2019)
├── run_tests.py        # Test runner
├── test_api.py         # API unit tests
├── test_model.py       # Model unit tests
├── test_logger.py      # Logger unit tests
├── requirements.txt
├── Dockerfile
├── data/cs-train/      # JSON sales data files
├── models/             # Saved .pkl model files
├── logs/               # CSV prediction and training logs
├── eda_plots/          # Generated EDA plot images
├── static/
│   ├── app.js          # Frontend dashboard logic
│   └── index.css       # Dashboard styles
└── templates/
    └── index.html      # Web UI template
```