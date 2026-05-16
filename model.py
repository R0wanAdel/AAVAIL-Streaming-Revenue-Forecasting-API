"""
Model Training and Prediction Pipeline for AI Workflow Capstone
Calculates true evaluation metrics dynamically using verification splits.
"""

import os
import pickle
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

def _get_model_path(country, test=False):
    """
    Constructs the file path for storing/loading trained model artifacts.
    """
    model_name = f"{country}_test.pkl" if test else f"{country}_prod.pkl"
    return os.path.join(MODEL_DIR, model_name)

def _generate_mock_data(country, num_samples=300):
    """
    Helper to generate data with realistic distributions based on country context.
    Features: [Prior 7-day revenue, Prior 30-day revenue, Active Subscriber Count Share]
    """
    np.random.seed(42 if country == "all" else hash(country) % 1000)
    
    # Simulate active user base size based on market sizing
    base_users = 50000 if country in ["all", "United Kingdom"] else 12000
    sub_share = np.random.uniform(0.5, 2.5, num_samples)
    
    # Structural features mimicking streaming platform data
    past_30d_rev = sub_share * base_users * np.random.uniform(0.08, 0.12, num_samples)
    past_7d_rev = past_30d_rev * np.random.uniform(0.22, 0.26, num_samples)
    
    X = np.column_stack((past_7d_rev, past_30d_rev, sub_share))
    
    # Target variable: Realistic 30-day forward revenue forecast with noise
    y = past_30d_rev * 1.05 + (past_7d_rev * 0.1) + np.random.normal(0, base_users * 0.02, num_samples)
    y = np.clip(y, a_min=0, a_max=None) # Revenue cannot be negative
    
    return X, y

def _get_feature_row(country, date_str, test=False):
    """
    Simulates fetching a real feature engineering inference vector for a target window.
    """
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        raise ValueError("Date format must be YYYY-MM-DD")

    # High-level 90 day verification rule simulation
    has_90_days_history = True 
    if not has_90_days_history:
        return None

    # Generate a realistic inference data feature row based on the market context
    X_mock, _ = _generate_mock_data(country, num_samples=1)
    return X_mock

def train_model(country="all", test=False):
    """
    Trains a stable model pipeline using StandardScaler.
    Splits data into train/test splits to compute valid MAE, RMSE, and MAPE scores.
    """
    try:
        # 1. Load data distribution based on country context
        # (For production, swap this out with your primary data pipeline loader)
        X, y = _generate_mock_data(country, num_samples=100 if test else 400)

        # 2. Split dataset into Training and Validation sets to calculate true metrics
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # 3. Assemble and train the scikit-learn pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', RandomForestRegressor(n_estimators=50 if test else 150, random_state=42))
        ])
        pipeline.fit(X_train, y_train)

        # 4. Generate validation predictions to compute operational metrics
        y_val_pred = pipeline.predict(X_val)
        
        # Calculate standard mathematical errors
        mae = mean_absolute_error(y_val, y_val_pred)
        rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        
        # Avoid division-by-zero errors in mean baseline computations
        mean_y = np.mean(y_val) if np.mean(y_val) != 0 else 1.0
        
        # Transform standard errors to proportional percentage scales for the dashboard
        mae_pct = (mae / mean_y) * 100
        rmse_pct = (rmse / mean_y) * 100
        mape = np.mean(np.abs((y_val - y_val_pred) / np.clip(y_val, a_min=1.0, a_max=None)))

        trained_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        metrics = {
            "trained_at": trained_at,
            "mae_pct": round(float(mae_pct), 2),
            "rmse_pct": round(float(rmse_pct), 2),
            "mape": round(float(mape), 4),
            "model_type": "Random Forest Regressor",
            "train_size": len(X_train),
            "n_features": X_train.shape[1]
        }

        # 5. Persist the compiled pipeline artifact to disk
        model_path = _get_model_path(country, test)
        with open(model_path, "wb") as f:
            pickle.dump(pipeline, f)

        return pipeline, metrics

    except Exception as e:
        return None, {"error": str(e)}

def predict(country, date, test=False):
    """Generates a contextualized 30-day forward revenue prediction vector."""
    features = _get_feature_row(country, date, test)
    if features is None:
        return None

    model_path = _get_model_path(country, test)
    if not os.path.exists(model_path):
        pipeline, metrics = train_model(country=country, test=test)
        if pipeline is None:
            raise RuntimeError(f"Fallback training sequence failed for country: {country}")
    else:
        with open(model_path, "rb") as f:
            pipeline = pickle.load(f)

    predictions = pipeline.predict(features)
    return float(predictions[0])