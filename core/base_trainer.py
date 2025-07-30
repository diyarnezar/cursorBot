"""
Base Trainer for Project Hyperion
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from config.training_config import training_config


class BaseTrainer:
    """
    Base Trainer class for all training modes
    """
    
    def __init__(self, mode: str, symbols: Optional[List[str]] = None):
        """Initialize base trainer"""
        self.mode = mode
        self.symbols = symbols or ["ETHFDUSD"]
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        
        # Get mode configuration
        self.config = training_config.get_mode_config(mode)
        self.weight_estimate = self.config.get('weight', 0)
        
        # Parallel processing settings
        self.max_workers = 4  # Conservative default
        
    def is_training_safe(self) -> bool:
        """Check if training mode is safe for rate limits"""
        return self.weight_estimate <= 100  # Conservative threshold
        
    def collect_data_parallel(self, days: int, interval: str = '1m') -> pd.DataFrame:
        """Collect data using parallel processing (placeholder)"""
        self.logger.info(f"Collecting {days} days of data using parallel processing...")
        # Placeholder implementation
        return pd.DataFrame()
        
    def collect_data(self, symbol: str, limit: int = None) -> pd.DataFrame:
        """Collect training data with proper amount for training mode"""
        try:
            from data.collectors.binance_collector import BinanceDataCollector, BinanceConfig
            
            # Initialize collector
            config = BinanceConfig()
            collector = BinanceDataCollector(config)
            
            # Calculate proper data amount based on training mode if limit not provided
            if limit is None:
                if self.mode == 'quick':
                    days = 1
                    limit = 1440  # 1 day of minute data
                elif self.mode == 'month':
                    days = 30
                    limit = 43200  # 30 days of minute data
                elif self.mode == 'quarter':
                    days = 90
                    limit = 129600  # 90 days of minute data
                elif self.mode == 'half_year':
                    days = 180
                    limit = 259200  # 180 days of minute data
                elif self.mode == 'year':
                    days = 365
                    limit = 525600  # 365 days of minute data
                elif self.mode == 'two_year':
                    days = 730
                    limit = 1051200  # 730 days of minute data
                else:
                    days = 30
                    limit = 43200  # Default to 30 days
            else:
                # Calculate days from limit (assuming 1-minute intervals)
                days = limit / (24 * 60)  # Convert minutes to days
            
            self.logger.info(f"📊 Collecting {days:.1f} days of data for {symbol} (limit: {limit})")
            
            # Use fetch_historical_data for large datasets (more than 1000 minutes)
            if limit > 1000:
                data = collector.fetch_historical_data(symbol, days, '1m')
            else:
                # Use get_klines for smaller datasets
                data = collector.get_klines(symbol, '1m', limit=limit)
            
            if data is not None and not data.empty:
                self.logger.info(f"✅ Collected {len(data)} data points for {symbol}")
                return data
            else:
                self.logger.warning(f"⚠️ No data collected for {symbol}")
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"❌ Error collecting data for {symbol}: {e}")
            return pd.DataFrame()
        
    def prepare_features(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
        """Prepare features (to be implemented by subclasses)"""
        raise NotImplementedError
        
    def train_models(self, features: pd.DataFrame, targets: Dict[str, pd.Series]) -> Dict[str, Any]:
        """Train models (to be implemented by subclasses)"""
        raise NotImplementedError
        
    def evaluate_models(self, models: Dict[str, Any], features: pd.DataFrame, targets: Dict[str, pd.Series]) -> Dict[str, float]:
        """Evaluate models (to be implemented by subclasses)"""
        raise NotImplementedError 