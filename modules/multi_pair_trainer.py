#!/usr/bin/env python3
"""
MULTI-PAIR TRAINER - ETH/FDUSD LEVEL FOR ALL PAIRS
==================================================

Advanced training system that applies ETH/FDUSD level sophistication
to all 26 pairs with 10X intelligence features and multi-timeframe models.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import joblib
import os
import sys
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.multi_pair_data_collector import MultiPairDataCollector
from modules.feature_engineering import EnhancedFeatureEngineer
from ultra_train_enhanced import UltraEnhancedTrainer

logger = logging.getLogger(__name__)

class MultiPairTrainer:
    """Advanced trainer for all 26 pairs at ETH/FDUSD level"""
    
    def __init__(self):
        self.data_collector = MultiPairDataCollector()
        self.feature_engineer = EnhancedFeatureEngineer()
        
        # All 26 pairs
        self.all_pairs = [
            # Bedrock (6 pairs)
            'BTC', 'ETH', 'BNB', 'SOL', 'XRP', 'DOGE',
            # Infrastructure (8 pairs)
            'AVAX', 'DOT', 'LINK', 'ARB', 'OP', 'ADA', 'MATIC', 'ATOM',
            # DeFi Bluechips (4 pairs)
            'UNI', 'AAVE', 'JUP', 'PENDLE',
            # Volatility Engine (5 pairs)
            'PEPE', 'SHIB', 'BONK', 'WIF', 'BOME',
            # AI & Data (3 pairs)
            'FET', 'RNDR', 'WLD'
        ]
        
        # Training results
        self.training_results = {}
        self.models = {}
        
        logger.info(f"Multi-Pair Trainer initialized for {len(self.all_pairs)} pairs")
    
    def train_single_pair(self, pair: str, days: float = 15.0) -> bool:
        """Train advanced models for a single pair"""
        try:
            logger.info(f"Training {pair} with ETH/FDUSD level sophistication...")
            
            # 1. Collect advanced data
            data = self.data_collector.collect_advanced_data_for_pair(pair, days)
            
            if data is None or data.empty:
                logger.error(f"No data available for {pair}")
                return False
            
            # 2. Create ETH/FDUSD level trainer
            trainer = UltraEnhancedTrainer()
            
            # 3. Add 10X intelligence features
            data = trainer.add_10x_intelligence_features(data)
            
            # 4. Add maker order features
            data = trainer.add_maker_order_features(data)
            
            # 5. Prepare features and targets
            X, y_1m, y_5m, y_15m, y_30m, y_1h, y_4h, y_1d = trainer.prepare_features(data)
            
            if X.empty:
                logger.error(f"Feature preparation failed for {pair}")
                return False
            
            # 6. Train 10X intelligence models
            trainer.train_10x_intelligence_models(X, y_1m, y_5m, y_15m, y_30m, y_1h, y_4h, y_1d)
            
            # 7. Save models for this pair
            pair_models = {}
            for model_name, model in trainer.models.items():
                pair_models[model_name] = model
            
            self.models[pair] = pair_models
            
            # 8. Save performance metrics
            self.training_results[pair] = {
                'data_points': len(data),
                'features': len(X.columns),
                'models_trained': len(pair_models),
                'training_time': datetime.now(),
                'performance': trainer.model_performance
            }
            
            logger.info(f"{pair} training complete: {len(pair_models)} models trained")
            
            return True
            
        except Exception as e:
            logger.error(f"Training failed for {pair}: {e}")
            return False
    
    def train_all_pairs(self, days: float = 15.0) -> Dict[str, bool]:
        """Train advanced models for all 26 pairs"""
        logger.info(f"Starting advanced training for all {len(self.all_pairs)} pairs...")
        
        results = {}
        
        for pair in self.all_pairs:
            logger.info(f"--- Training {pair} ---")
            success = self.train_single_pair(pair, days)
            results[pair] = success
            
            if success:
                logger.info(f"{pair}: Training successful")
            else:
                logger.error(f"{pair}: Training failed")
        
        # Summary
        successful = sum(results.values())
        total = len(results)
        
        logger.info(f"Training Summary:")
        logger.info(f"   Successful: {successful}/{total}")
        logger.info(f"   Failed: {total - successful}/{total}")
        
        return results
    
    def save_all_models(self):
        """Save all trained models"""
        logger.info("Saving all trained models...")
        
        models_dir = "models/multi_pair"
        os.makedirs(models_dir, exist_ok=True)
        
        for pair, pair_models in self.models.items():
            pair_dir = f"{models_dir}/{pair}"
            os.makedirs(pair_dir, exist_ok=True)
            
            for model_name, model in pair_models.items():
                model_path = f"{pair_dir}/{model_name}.joblib"
                try:
                    joblib.dump(model, model_path)
                    logger.info(f"   {pair}/{model_name} saved")
                except Exception as e:
                    logger.error(f"   Failed to save {pair}/{model_name}: {e}")
        
        # Save training results
        results_path = f"{models_dir}/training_results.json"
        try:
            with open(results_path, 'w') as f:
                json.dump(self.training_results, f, default=str, indent=2)
            logger.info(f"Training results saved to {results_path}")
        except Exception as e:
            logger.error(f"Failed to save training results: {e}")
    
    def load_all_models(self):
        """Load all trained models"""
        logger.info("Loading all trained models...")
        
        models_dir = "models/multi_pair"
        
        if not os.path.exists(models_dir):
            logger.warning(f"Models directory not found: {models_dir}")
            return
        
        for pair in self.all_pairs:
            pair_dir = f"{models_dir}/{pair}"
            
            if not os.path.exists(pair_dir):
                logger.warning(f"No models found for {pair}")
                continue
            
            pair_models = {}
            
            for model_file in os.listdir(pair_dir):
                if model_file.endswith('.joblib'):
                    model_name = model_file.replace('.joblib', '')
                    model_path = f"{pair_dir}/{model_file}"
                    
                    try:
                        model = joblib.load(model_path)
                        pair_models[model_name] = model
                        logger.info(f"   {pair}/{model_name} loaded")
                    except Exception as e:
                        logger.error(f"   Failed to load {pair}/{model_name}: {e}")
            
            if pair_models:
                self.models[pair] = pair_models
        
        logger.info(f"Loaded models for {len(self.models)} pairs")

# Global instance
multi_pair_trainer = MultiPairTrainer()
