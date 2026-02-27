#!/usr/bin/env python3
"""
Unit tests for the logging module
"""

import unittest
import os
import sys
import csv
import tempfile
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


class TestLogger(unittest.TestCase):

    def setUp(self):
        """Create a temp directory for logs during testing"""
        self.test_log_dir = tempfile.mkdtemp()
        import logger
        self.original_log_dir = logger.LOG_DIR
        self.original_prod_log = logger.PREDICT_PROD_LOG
        self.original_test_log = logger.PREDICT_TEST_LOG
        self.original_train_log = logger.TRAIN_LOG

        # Redirect logs to temp dir
        logger.LOG_DIR = self.test_log_dir
        logger.PREDICT_PROD_LOG = os.path.join(self.test_log_dir, "predict-prod.log")
        logger.PREDICT_TEST_LOG = os.path.join(self.test_log_dir, "predict-test.log")
        logger.TRAIN_LOG = os.path.join(self.test_log_dir, "train.log")

    def tearDown(self):
        """Remove temp log dir and restore original paths"""
        shutil.rmtree(self.test_log_dir)
        import logger
        logger.LOG_DIR = self.original_log_dir
        logger.PREDICT_PROD_LOG = self.original_prod_log
        logger.PREDICT_TEST_LOG = self.original_test_log
        logger.TRAIN_LOG = self.original_train_log

    def test_update_predict_log_creates_file(self):
        """Test that logging a prediction creates the log file"""
        from logger import update_predict_log, PREDICT_TEST_LOG
        update_predict_log("United Kingdom", "2019-01-01", 50000.0, 0.05, "1.0", test=True)
        self.assertTrue(os.path.exists(PREDICT_TEST_LOG))

    def test_update_predict_log_content(self):
        """Test that prediction log contains expected data"""
        from logger import update_predict_log, load_predict_log
        update_predict_log("France", "2019-06-01", 12345.67, 0.10, "1.0", test=True)
        logs = load_predict_log(test=True)
        self.assertEqual(len(logs), 1)
        self.assertEqual(logs[0]["country"], "France")
        self.assertEqual(logs[0]["date"], "2019-06-01")

    def test_update_train_log_creates_file(self):
        """Test that logging a training event creates the log file"""
        from logger import update_train_log, TRAIN_LOG
        update_train_log("all", "2019-01-01", {"rmse": 100.0}, 2.5, "1.0", test=True)
        self.assertTrue(os.path.exists(TRAIN_LOG))

    def test_test_prod_logs_are_separate(self):
        """Test that test and production logs are written to different files"""
        from logger import update_predict_log, PREDICT_PROD_LOG, PREDICT_TEST_LOG
        update_predict_log("Germany", "2019-01-01", 30000.0, 0.05, "1.0", test=False)
        update_predict_log("Germany", "2019-01-01", 30000.0, 0.05, "1.0", test=True)
        self.assertTrue(os.path.exists(PREDICT_PROD_LOG))
        self.assertTrue(os.path.exists(PREDICT_TEST_LOG))
        self.assertNotEqual(PREDICT_PROD_LOG, PREDICT_TEST_LOG)

    def test_load_empty_log_returns_empty_list(self):
        """Test that loading a non-existent log returns empty list"""
        from logger import load_predict_log
        result = load_predict_log(test=True)
        self.assertEqual(result, [])

    def test_multiple_log_entries(self):
        """Test that multiple entries can be logged and retrieved"""
        from logger import update_predict_log, load_predict_log
        update_predict_log("UK", "2019-01-01", 1000.0, 0.1, "1.0", test=True)
        update_predict_log("UK", "2019-02-01", 2000.0, 0.1, "1.0", test=True)
        update_predict_log("UK", "2019-03-01", 3000.0, 0.1, "1.0", test=True)
        logs = load_predict_log(test=True)
        self.assertEqual(len(logs), 3)


if __name__ == "__main__":
    unittest.main()
