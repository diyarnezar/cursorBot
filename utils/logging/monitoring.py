"""
Training monitoring utilities for Project Hyperion
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import pandas as pd

class TrainingMonitor:
    """Monitor training progress and performance"""
    
    def __init__(self, training_mode: str):
        """
        Initialize training monitor
        
        Args:
            training_mode: Name of the training mode
        """
        self.training_mode = training_mode
        self.logger = logging.getLogger(f"hyperion.monitor.{training_mode}")
        
        # Training state
        self.start_time = None
        self.end_time = None
        self.current_step = None
        self.progress = 0.0
        
        # Performance metrics
        self.metrics = {}
        self.errors = []
        
        # Resource usage
        self.memory_usage = []
        self.cpu_usage = []
    
    def start_training(self):
        """Start monitoring training session"""
        self.start_time = datetime.now()
        self.logger.info(f"Training monitor started for {self.training_mode}")
    
    def end_training(self):
        """End monitoring training session"""
        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds()
        self.logger.info(f"Training completed in {duration:.2f} seconds")
    
    def update_progress(self, step: str, progress: float):
        """Update training progress"""
        self.current_step = step
        self.progress = progress
        self.logger.info(f"Progress: {progress:.1f}% - {step}")
    
    def log_metric(self, name: str, value: float):
        """Log a performance metric"""
        self.metrics[name] = value
        self.logger.info(f"Metric {name}: {value:.4f}")
    
    def log_error(self, error: Exception, context: str = ""):
        """Log an error"""
        error_info = {
            'timestamp': datetime.now(),
            'error': str(error),
            'error_type': type(error).__name__,
            'context': context
        }
        self.errors.append(error_info)
        self.logger.error(f"Error in {context}: {error}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get training summary"""
        duration = None
        if self.start_time and self.end_time:
            duration = (self.end_time - self.start_time).total_seconds()
        
        return {
            'training_mode': self.training_mode,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration': duration,
            'current_step': self.current_step,
            'progress': self.progress,
            'metrics': self.metrics,
            'error_count': len(self.errors),
            'success': len(self.errors) == 0
        }
    
    def export_metrics(self, filepath: str):
        """Export metrics to file"""
        summary = self.get_summary()
        
        # Convert to DataFrame for easy export
        metrics_df = pd.DataFrame([summary])
        metrics_df.to_csv(filepath, index=False)
        
        self.logger.info(f"Metrics exported to {filepath}")
    
    def is_training_successful(self) -> bool:
        """Check if training was successful"""
        return len(self.errors) == 0 and self.progress >= 100.0 