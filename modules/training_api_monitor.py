#!/usr/bin/env python3
"""
Training API Monitor
Ensures training process never contributes to exceeding 1K/second API limit
"""

import logging
from typing import Dict, Any
from modules.global_api_monitor import global_api_monitor, monitor_api_call

class TrainingAPIMonitor:
    """Monitors API calls during training process"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.training_sources = [
            'training_data_collection',
            'training_feature_engineering', 
            'training_model_training',
            'training_validation',
            'training_background_collection'
        ]
        
        self.logger.info("🎯 Training API Monitor initialized")
    
    @monitor_api_call('training_data_collection')
    def collect_training_data(self, pair: str, days: float):
        """Monitor data collection during training"""
        self.logger.info(f"📊 Training data collection for {pair}")
        # This would be called before any data collection
    
    @monitor_api_call('training_feature_engineering')
    def engineer_features(self, data):
        """Monitor feature engineering during training"""
        self.logger.info("🔧 Training feature engineering")
        # This would be called before feature engineering
    
    @monitor_api_call('training_model_training')
    def train_model(self, model_type: str):
        """Monitor model training"""
        self.logger.info(f"🧠 Training {model_type} model")
        # This would be called before model training
    
    @monitor_api_call('training_validation')
    def validate_model(self, model_type: str):
        """Monitor model validation"""
        self.logger.info(f"✅ Validating {model_type} model")
        # This would be called before model validation
    
    @monitor_api_call('training_background_collection')
    def background_collection(self):
        """Monitor background data collection"""
        self.logger.info("🔄 Background data collection during training")
        # This would be called before background collection
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training-specific API statistics"""
        global_stats = global_api_monitor.get_global_stats()
        source_stats = global_api_monitor.get_source_stats()
        
        # Filter training-related sources
        training_stats = {}
        for source in self.training_sources:
            if source in source_stats:
                training_stats[source] = source_stats[source]
        
        return {
            'global_stats': global_stats,
            'training_sources': training_stats,
            'training_total_requests': sum(stats.get('total_requests', 0) for stats in training_stats.values()),
            'training_requests_last_second': sum(stats.get('requests_last_second', 0) for stats in training_stats.values())
        }
    
    def check_training_safety(self) -> Dict[str, Any]:
        """Check if training is safe to proceed"""
        global_stats = global_api_monitor.get_global_stats()
        
        # Calculate training contribution to global limit
        training_stats = self.get_training_stats()
        training_requests = training_stats['training_requests_last_second']
        
        safety_status = {
            'global_usage': global_stats['usage_percent'],
            'training_contribution': training_requests,
            'training_contribution_percent': (training_requests / global_stats['global_safety_limit']) * 100,
            'safe_to_proceed': global_stats['usage_percent'] < 80,  # Safe if under 80%
            'recommendation': 'PROCEED' if global_stats['usage_percent'] < 80 else 'WAIT' if global_stats['usage_percent'] < 95 else 'STOP'
        }
        
        if safety_status['recommendation'] != 'PROCEED':
            self.logger.warning(f"⚠️ Training safety check: {safety_status['recommendation']} - Global usage: {safety_status['global_usage']:.1f}%")
        
        return safety_status

# Global training monitor instance
training_monitor = TrainingAPIMonitor() 