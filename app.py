"""
Flask API for AI Workflow Capstone
Endpoints: /, /train, /predict, /logs
"""

import time
import os
from flask import Flask, jsonify, request, render_template

app = Flask(__name__)

MODEL_VERSION = "1.1"  # Updated version to reflect model changes

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/train", methods=["POST"])
def train():
    """Train the model for a given country."""
    if not request.json:
        return jsonify({"error": "Request must be JSON"}), 400

    country = request.json.get("country", "all")
    test_mode = request.json.get("test", False)

    if not country:
        return jsonify({"error": "Missing required field: country"}), 400

    try:
        from model import train_model
        from logger import update_train_log

        start = time.time()
        # The updated model uses a Pipeline with StandardScaler for stability
        model, metrics = train_model(country=country, test=test_mode)
        runtime = time.time() - start

        if model is None:
            return jsonify({"error": metrics.get("error", "Training failed")}), 500

        # Pass target date placeholder or training timestamp to logger
        update_train_log(
            country=country,
            date=metrics.get("trained_at", ""),
            eval_test=metrics,
            runtime=runtime,
            model_version=MODEL_VERSION,
            test=test_mode
        )

        return jsonify({
            "status": "success",
            "country": country,
            "metrics": metrics,
            "runtime": runtime
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict", methods=["POST"])
def predict():
    """Predict 30-day revenue for a country and date."""
    if not request.json:
        return jsonify({"error": "Request must be JSON"}), 400

    country = request.json.get("country")
    date = request.json.get("date")
    test_mode = request.json.get("test", False)

    if not country:
        return jsonify({"error": "Missing required field: country"}), 400
    if not date:
        return jsonify({"error": "Missing required field: date"}), 400

    try:
        from model import predict as model_predict
        from logger import update_predict_log

        start = time.time()
        y_pred = model_predict(country=country, date=date, test=test_mode)
        runtime = time.time() - start

        if y_pred is None:
            return jsonify({"error": f"Could not generate prediction for {country}. Ensure 90 days of history exist before {date}."}), 500

        # Log the prediction using the exact key layout expected by logger.py
        update_predict_log(
            country=country,
            date=date,
            y_pred=y_pred,
            runtime=runtime,
            model_version=MODEL_VERSION,
            test=test_mode
        )

        # Keys synchronized to match app.js properties
        return jsonify({
            "status": "success",
            "country": country,
            "date": date,
            "predicted_revenue_30_days": round(y_pred, 2), 
            "runtime": round(runtime, 4)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/logs", methods=["GET"])
def logs():
    """Return prediction logs formatted for the UI grid."""
    test_mode = request.args.get("test", "false").lower() == "true"
    try:
        from logger import load_predict_log
        log_data = load_predict_log(test=test_mode)
        
        # UI maps over 'target_date' from log dict elements
        formatted_logs = []
        for entry in log_data:
            formatted_logs.append({
                "timestamp": entry.get("timestamp"),
                "country": entry.get("country"),
                "target_date": entry.get("date"), # Maps 'date' column to 'target_date' field for app.js
                "y_pred": entry.get("y_pred"),
                "runtime": entry.get("runtime_seconds"),
                "model_version": entry.get("model_version")
            })

        return jsonify({
            "status": "success",
            "test_mode": test_mode,
            "count": len(formatted_logs),
            "logs": formatted_logs
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)