"""
Real-time Pipeline for Live Data Processing
Handles streaming data, real-time feature engineering, and live predictions
"""

import logging
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from queue import Queue, Empty
from threading import Thread, Lock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

# Import functions from data_ingestion instead of class
from .data_ingestion import fetch_klines, fetch_ticker_24hr, fetch_order_book
from .feature_engineering import EnhancedFeatureEngineer

@dataclass
class StreamConfig:
    """Configuration for real-time data streams"""
    symbol: str
    interval: str
    buffer_size: int = 1000
    processing_delay_ms: int = 100
    enable_websocket: bool = False  # Disabled for now
    enable_rest_polling: bool = True
    max_retries: int = 3
    retry_delay: float = 1.0

class RealTimePipeline:
    """
    Real-time data processing pipeline for live trading
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.config = config
        self.feature_engineer = EnhancedFeatureEngineer(
            use_microstructure=True,
            use_alternative_data=True,
            use_advanced_indicators=True,
            use_adaptive_features=True,
            use_normalization=True,
            use_crypto_features=True
        )
        
        # Stream configuration
        self.stream_config = StreamConfig(
            symbol=config['trading_parameters']['trading_pair'],
            interval=config['trading_parameters']['timeframe'],
            buffer_size=1000,
            processing_delay_ms=100
        )
        
        # Data buffers
        self.data_buffer = Queue(maxsize=self.stream_config.buffer_size)
        self.feature_buffer = Queue(maxsize=self.stream_config.buffer_size)
        self.prediction_buffer = Queue(maxsize=100)
        
        # Processing state
        self.is_running = False
        self.processing_thread = None
        self.lock = Lock()
        
        # Performance tracking
        self.processing_times = []
        self.latency_metrics = {
            'data_reception': [],
            'feature_engineering': [],
            'prediction': [],
            'execution': []
        }
        
        # Callbacks
        self.on_new_data: Optional[Callable] = None
        self.on_prediction: Optional[Callable] = None
        self.on_trade_signal: Optional[Callable] = None
        
        self.logger.info("🚀 Real-time pipeline initialized")

    def start(self):
        """Start the real-time pipeline"""
        with self.lock:
            if self.is_running:
                self.logger.warning("Pipeline already running")
                return
                
            self.is_running = True
            self.logger.info("🚀 Starting real-time pipeline")
            
            # Start processing thread
            self.processing_thread = Thread(target=self._processing_loop, daemon=True)
            self.processing_thread.start()
            
            # Start data collection
            if self.stream_config.enable_rest_polling:
                self._start_rest_polling()

    def stop(self):
        """Stop the real-time pipeline"""
        with self.lock:
            if not self.is_running:
                return
                
            self.is_running = False
            self.logger.info("🛑 Stopping real-time pipeline")
            
            # Clear buffers
            self._clear_buffers()
            
            # Wait for threads to finish
            if self.processing_thread:
                self.processing_thread.join(timeout=5)

    def _processing_loop(self):
        """Main processing loop for real-time data"""
        while self.is_running:
            try:
                # Process data from buffer
                if not self.data_buffer.empty():
                    data = self.data_buffer.get_nowait()
                    self._process_data(data)
                
                # Process features
                if not self.feature_buffer.empty():
                    features = self.feature_buffer.get_nowait()
                    self._process_features(features)
                
                # Process predictions
                if not self.prediction_buffer.empty():
                    prediction = self.prediction_buffer.get_nowait()
                    self._process_prediction(prediction)
                
                # Sleep to prevent CPU overload
                time.sleep(self.stream_config.processing_delay_ms / 1000.0)
                
            except Empty:
                continue
            except Exception as e:
                self.logger.error(f"❌ Error in processing loop: {e}")
                time.sleep(1)

    def _process_data(self, data: Dict[str, Any]):
        """Process incoming market data"""
        start_time = time.time()
        
        try:
            # Convert to DataFrame
            df = pd.DataFrame([data])
            
            # Generate features
            features = self.feature_engineer.add_enhanced_features(df)
            
            # Add to feature buffer
            if not self.feature_buffer.full():
                self.feature_buffer.put(features)
            
            # Calculate latency
            latency = (time.time() - start_time) * 1000
            self.latency_metrics['data_reception'].append(latency)
            
            # Call callback
            if self.on_new_data:
                self.on_new_data(data)
                
        except Exception as e:
            self.logger.error(f"❌ Error processing data: {e}")

    def _process_features(self, features: pd.DataFrame):
        """Process engineered features"""
        start_time = time.time()
        
        try:
            # For now, just create a simple prediction
            # In a full implementation, this would use the prediction engine
            predictions = {
                'direction': 'neutral',
                'confidence': 0.5,
                'price_prediction': features['close'].iloc[-1] if 'close' in features.columns else 0
            }
            
            # Add to prediction buffer
            if not self.prediction_buffer.full():
                self.prediction_buffer.put(predictions)
            
            # Calculate latency
            latency = (time.time() - start_time) * 1000
            self.latency_metrics['feature_engineering'].append(latency)
            
        except Exception as e:
            self.logger.error(f"❌ Error processing features: {e}")

    def _process_prediction(self, predictions: Dict[str, Any]):
        """Process model predictions and generate trading signals"""
        start_time = time.time()
        
        try:
            # For now, just log the prediction
            self.logger.info(f"📊 Prediction: {predictions}")
            
            # Calculate latency
            latency = (time.time() - start_time) * 1000
            self.latency_metrics['prediction'].append(latency)
            
            # Call callback
            if self.on_prediction:
                self.on_prediction(predictions)
                
        except Exception as e:
            self.logger.error(f"❌ Error processing prediction: {e}")

    def _start_rest_polling(self):
        """Start REST API polling for data"""
        def polling_loop():
            while self.is_running:
                try:
                    # Get latest kline data using imported function
                    end_time = datetime.now()
                    start_time = end_time - timedelta(minutes=5)
                    klines = fetch_klines(
                        symbol=self.stream_config.symbol,
                        interval=self.stream_config.interval,
                        start_time=start_time,
                        end_time=end_time
                    )
                    
                    if klines and len(klines) > 0:
                        kline = klines[-1]  # Get the latest kline
                        market_data = {
                            'timestamp': kline[0],
                            'open': float(kline[1]),
                            'high': float(kline[2]),
                            'low': float(kline[3]),
                            'close': float(kline[4]),
                            'volume': float(kline[5]),
                            'is_closed': True
                        }
                        
                        # Add to buffer
                        if not self.data_buffer.full():
                            self.data_buffer.put(market_data)
                    
                    # Wait before next poll
                    time.sleep(1)
                    
                except Exception as e:
                    self.logger.error(f"❌ Error in REST polling: {e}")
                    time.sleep(self.stream_config.retry_delay)
        
        # Start polling thread
        polling_thread = Thread(target=polling_loop, daemon=True)
        polling_thread.start()
        self.logger.info("📡 REST polling started")

    def _clear_buffers(self):
        """Clear all data buffers"""
        while not self.data_buffer.empty():
            self.data_buffer.get()
        while not self.feature_buffer.empty():
            self.feature_buffer.get()
        while not self.prediction_buffer.empty():
            self.prediction_buffer.get()

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the pipeline"""
        return {
            'is_running': self.is_running,
            'buffer_sizes': {
                'data': self.data_buffer.qsize(),
                'features': self.feature_buffer.qsize(),
                'predictions': self.prediction_buffer.qsize()
            },
            'latency_metrics': {
                'data_reception': np.mean(self.latency_metrics['data_reception']) if self.latency_metrics['data_reception'] else 0,
                'feature_engineering': np.mean(self.latency_metrics['feature_engineering']) if self.latency_metrics['feature_engineering'] else 0,
                'prediction': np.mean(self.latency_metrics['prediction']) if self.latency_metrics['prediction'] else 0
            },
            'processing_times': {
                'mean': np.mean(self.processing_times) if self.processing_times else 0,
                'max': np.max(self.processing_times) if self.processing_times else 0,
                'min': np.min(self.processing_times) if self.processing_times else 0
            }
        }

    def set_callbacks(self, on_new_data=None, on_prediction=None, on_trade_signal=None):
        """Set callback functions for pipeline events"""
        self.on_new_data = on_new_data
        self.on_prediction = on_prediction
        self.on_trade_signal = on_trade_signal 