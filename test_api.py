import unittest
import json
from app import app

class TestFlaskAPI(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        self.client.testing = True

    def test_train_endpoint(self):
        payload = {"country": "United Kingdom", "test": True}
        response = self.client.post('/train', 
                                    data=json.dumps(payload), 
                                    content_type='application/json')
        data = json.loads(response.data.decode('utf-8'))
        
        self.assertEqual(response.status_code, 200)
        self.assertEqual(data["status"], "success")
        self.assertIn("metrics", data)

    def test_predict_endpoint(self):
        payload = {"country": "United Kingdom", "date": "2026-05-16", "test": True}
        response = self.client.post('/predict', 
                                    data=json.dumps(payload), 
                                    content_type='application/json')
        data = json.loads(response.data.decode('utf-8'))
        
        self.assertEqual(response.status_code, 200)
        self.assertEqual(data["status"], "success")
        # Confirms sync with app.js expectations
        self.assertIn("predicted_revenue_30_days", data)