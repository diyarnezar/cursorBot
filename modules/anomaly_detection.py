import numpy as np
from sklearn.ensemble import IsolationForest
import logging

class AnomalyDetector:
    """
    Real-time anomaly detection for trading. Supports rolling Z-score and Isolation Forest.
    Can pause trading if anomaly is detected.
    """
    def __init__(self, window=100, z_thresh=3.0):
        self.window = window
        self.z_thresh = z_thresh
        self.history = []
        self.isolation_forest = IsolationForest(contamination=0.01)
    def rolling_zscore(self, value):
        self.history.append(value)
        if len(self.history) > self.window:
            self.history.pop(0)
        if len(self.history) < self.window:
            return False  # Not enough data
        mean = np.mean(self.history)
        std = np.std(self.history)
        z = (value - mean) / (std + 1e-9)
        return abs(z) > self.z_thresh
    def isolation_forest_anomaly(self, X):
        if len(X) < self.window:
            return False
        preds = self.isolation_forest.fit_predict(X)
        return np.any(preds == -1)
    def pause_trading_if_anomaly(self, value, X=None):
        if self.rolling_zscore(value):
            logging.warning("[Anomaly] Rolling Z-score anomaly detected. Pausing trading.")
            return True
        if X is not None and self.isolation_forest_anomaly(X):
            logging.warning("[Anomaly] Isolation Forest anomaly detected. Pausing trading.")
            return True
        return False

"""
How to use:
- Call pause_trading_if_anomaly(value, X) before each trade. If True, skip trading cycle.
- Use only free/open-source tools (sklearn, numpy).
""" 