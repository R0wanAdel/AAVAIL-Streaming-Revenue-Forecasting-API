#!/usr/bin/env python3
"""
Unit tests for the Flask API
"""

import unittest
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app import app


class TestAPI(unittest.TestCase):

    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_index(self):
        """Test that the home page loads"""
        response = self.app.get("/")
        self.assertEqual(response.status_code, 200)

    def test_predict_missing_country(self):
        """Test that predict returns error when country is missing"""
        response = self.app.post(
            "/predict",
            data=json.dumps({"date": "2019-01-01", "test": True}),
            content_type="application/json"
        )
        data = json.loads(response.data)
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", data)

    def test_predict_missing_date(self):
        """Test that predict returns error when date is missing"""
        response = self.app.post(
            "/predict",
            data=json.dumps({"country": "United Kingdom", "test": True}),
            content_type="application/json"
        )
        data = json.loads(response.data)
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", data)

    def test_predict_no_json(self):
        """Test that predict returns error when no JSON is provided"""
        response = self.app.post("/predict")
        self.assertIn(response.status_code, [400, 415])

    def test_train_missing_body(self):
        """Test that train returns error when no JSON is provided"""
        response = self.app.post("/train")
        self.assertIn(response.status_code, [400, 415])

    def test_logs_endpoint(self):
        """Test that logs endpoint returns valid response"""
        response = self.app.get("/logs?test=true")
        data = json.loads(response.data)
        self.assertEqual(response.status_code, 200)
        self.assertIn("logs", data)
        self.assertIn("count", data)


if __name__ == "__main__":
    unittest.main()
