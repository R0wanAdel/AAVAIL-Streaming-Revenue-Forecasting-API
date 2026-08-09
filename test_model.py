import unittest
from model import train_model, predict

class TestModelPipeline(unittest.TestCase):
    def test_pipeline_training(self):
        pipeline, metrics = train_model(country="all", test=True)
        self.assertIsNotNone(pipeline)
        self.assertEqual(metrics["model_type"], "Random Forest Regressor")
        self.assertIn("mae_pct", metrics)
        self.assertGreater(metrics["mae_pct"], 0.0) # Confirms realistic metrics

    def test_pipeline_inference(self):
        y_pred = predict(country="France", date="2026-06-01", test=True)
        self.assertIsNotNone(y_pred)
        self.assertIsInstance(y_pred, float)