#!/usr/bin/env python3
"""
Flask API for AI Workflow Capstone
Endpoints: /, /train, /predict, /logs
"""

import time
import os
from flask import Flask, jsonify, request, render_template_string

app = Flask(__name__)

MODEL_VERSION = "1.0"

HOME_HTML = """
<!DOCTYPE html>
<html>
<head><title>AAVAIL Revenue Prediction API</title></head>
<body>
<h1>AAVAIL Revenue Prediction API</h1>
<p>Available endpoints:</p>
<ul>
  <li><b>GET /</b> - This page</li>
  <li><b>POST /train</b> - Train the model. JSON body: {"country": "France", "test": false}</li>
  <li><b>POST /predict</b> - Predict revenue. JSON body: {"country": "France", "date": "2019-08-01", "test": false}</li>
  <li><b>GET /logs</b> - View prediction logs. Query params: ?test=true</li>
</ul>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HOME_HTML)


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
        model, metrics = train_model(country=country, test=test_mode)
        runtime = time.time() - start

        if model is None:
            return jsonify({"error": metrics.get("error", "Training failed")}), 500

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

    # Input validation
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
            return jsonify({"error": f"Could not generate prediction for country: {country}"}), 500

        update_predict_log(
            country=country,
            date=date,
            y_pred=y_pred,
            runtime=runtime,
            model_version=MODEL_VERSION,
            test=test_mode
        )

        return jsonify({
            "status": "success",
            "country": country,
            "date": date,
            "predicted_revenue_30_days": round(y_pred, 2),
            "runtime": runtime
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/logs", methods=["GET"])
def logs():
    """Return prediction logs."""
    test_mode = request.args.get("test", "false").lower() == "true"
    try:
        from logger import load_predict_log
        log_data = load_predict_log(test=test_mode)
        return jsonify({
            "status": "success",
            "test_mode": test_mode,
            "count": len(log_data),
            "logs": log_data
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)
