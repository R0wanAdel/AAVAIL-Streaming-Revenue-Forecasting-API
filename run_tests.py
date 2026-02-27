#!/usr/bin/env python3
"""
Run all unit tests with a single script.
Usage: python run_tests.py
"""

import unittest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

loader = unittest.TestLoader()
suite = unittest.TestSuite()

# Discover and add all test files
test_dir = os.path.dirname(__file__)
discovered = loader.discover(test_dir, pattern="test_*.py")
suite.addTests(discovered)

runner = unittest.TextTestRunner(verbosity=2)
result = runner.run(suite)

# Exit with non-zero code if tests fail (for CI/CD)
sys.exit(0 if result.wasSuccessful() else 1)
