import unittest
from monitor import monitor_performance

class TestMonitoringLayer(unittest.TestCase):
    def test_wasserstein_evaluation(self):
        # Run performance script
        result = monitor_performance(country="all", test=True)
        
        # If no logs exist yet during test compilation, pass gracefully or asset hint
        if "error" in result:
            self.assertIn("No prediction logs found", result["error"])
        else:
            self.assertIn("wasserstein_distance", result)
            self.assertIn("aligned_sample_days", result)