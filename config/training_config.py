"""
Enhanced Training Configuration for Project Hyperion
Incorporates all advanced features from the legacy system
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

class TrainingConfig:
    """Enhanced training configuration with all legacy system features"""
    
    # Correct symbol list from gemini_plan_new.md - All FDUSD pairs
    ALL_FDUSD_PAIRS = [
        # Cluster 1: The Bedrock (Core Large Caps - 6 Assets)
        'BTCFDUSD', 'ETHFDUSD', 'BNBFDUSD', 'SOLFDUSD', 'XRPFDUSD', 'DOGEFDUSD',
        # Cluster 2: The Infrastructure (Major L1s & L2s - 5 Assets)
        'AVAXFDUSD', 'DOTFDUSD', 'LINKFDUSD', 'ARBFDUSD', 'OPFDUSD',
        # Cluster 3: The DeFi Blue Chips (5 Assets)
        'UNIFDUSD', 'AAVEFDUSD', 'JUPFDUSD', 'PENDLEFDUSD', 'ENAFDUSD',
        # Cluster 4: The Volatility Engine (Memecoins & High Beta - 5 Assets)
        'PEPEFDUSD', 'SHIBFDUSD', 'BONKFDUSD', 'WIFFDUSD', 'BOMEFDUSD',
        # Cluster 5: The AI & Data Sector (Emerging Tech - 5 Assets)
        'FETFDUSD', 'RNDRFDUSD', 'WLDFDUSD', 'TAOFDUSD', 'GRTFDUSD'
    ]
    
    # Default symbols for single-pair training
    DEFAULT_SYMBOLS = ['ETHFDUSD']
    
    # Advanced training modes with enhanced features
    TRAINING_MODES = {
        'test': {
            'name': 'Test Training',
            'description': 'Fast Test (15 minutes)',
            'days': 0.01,  # 15 minutes
            'minutes': 15,
            'estimated_time': '2-3 minutes',
            'weight': 'Very Low',
            'recommended_for': 'Quick testing and validation',
            'trainer_class': None,
            'rate_limit_safe': True,
            'max_symbols_per_batch': 5,
            'batch_delay_seconds': 30,
            'features_enabled': ['basic', 'technical', 'price', 'volume'],
            'models_enabled': ['lightgbm', 'xgboost', 'random_forest'],
            'advanced_features': False
        },
        'ultra_short': {
            'name': 'Ultra Short Training',
            'description': 'Ultra-Short Test (30 minutes)',
            'days': 0.02,  # 30 minutes
            'minutes': 30,
            'estimated_time': '3-5 minutes',
            'weight': 'Very Low',
            'recommended_for': 'Fast validation',
            'trainer_class': None,
            'rate_limit_safe': True,
            'max_symbols_per_batch': 10,
            'batch_delay_seconds': 45,
            'features_enabled': ['basic', 'technical', 'price', 'volume', 'volatility'],
            'models_enabled': ['lightgbm', 'xgboost', 'catboost', 'random_forest'],
            'advanced_features': False
        },
        'ultra_fast': {
            'name': 'Ultra Fast Training',
            'description': 'Ultra-Fast Testing (2 hours)',
            'days': 0.083,  # 2 hours
            'minutes': 120,
            'estimated_time': '5-8 minutes',
            'weight': 'Low',
            'recommended_for': 'Quick model testing',
            'trainer_class': None,
            'rate_limit_safe': True,
            'max_symbols_per_batch': 15,
            'batch_delay_seconds': 60,
            'features_enabled': ['basic', 'technical', 'price', 'volume', 'volatility', 'momentum'],
            'models_enabled': ['lightgbm', 'xgboost', 'catboost', 'random_forest', 'gradient_boosting'],
            'advanced_features': False
        },
        'quick': {
            'name': 'Quick Training',
            'description': 'Advanced Quick Training (1 day) - Maximum Intelligence',
            'days': 1.0,
            'minutes': 1440,
            'estimated_time': '15-25 minutes',
            'weight': 'Medium',
            'recommended_for': 'Daily model updates with full intelligence',
            'trainer_class': 'QuickTrainer',
            'rate_limit_safe': True,
            'max_symbols_per_batch': 20,
            'batch_delay_seconds': 60,
            'features_enabled': ['basic', 'technical', 'price', 'volume', 'volatility', 'momentum', 'microstructure', 'psychology', 'patterns', 'regime_detection', 'external_alpha'],
            'models_enabled': ['lightgbm', 'xgboost', 'catboost', 'random_forest', 'gradient_boosting', 'decision_tree', 'lstm', 'gru', 'transformer', 'ensemble'],
            'advanced_features': True,
            'use_reinforcement_learning': True,
            'use_self_improvement': True,
            'use_hyperparameter_optimization': True,
            'use_meta_learning': True
        },
        'month': {
            'name': 'Month Training',
            'description': '30-Day Training (1 month)',
            'days': 30.0,
            'minutes': 43200,
            'estimated_time': '30-45 minutes',
            'weight': 'Medium',
            'recommended_for': 'Monthly model retraining',
            'trainer_class': 'MonthTrainer',
            'rate_limit_safe': True,
            'max_symbols_per_batch': 25,
            'batch_delay_seconds': 90,
            'features_enabled': ['basic', 'technical', 'price', 'volume', 'volatility', 'momentum', 'microstructure', 'psychology'],
            'models_enabled': ['lightgbm', 'xgboost', 'catboost', 'random_forest', 'gradient_boosting', 'decision_tree', 'lstm', 'gru'],
            'advanced_features': True
        },
        'quarter': {
            'name': 'Quarter Training',
            'description': '3-Month Training (1 quarter)',
            'days': 90.0,
            'minutes': 129600,
            'estimated_time': '1-2 hours',
            'weight': 'Medium',
            'recommended_for': 'Quarterly model updates',
            'trainer_class': 'QuarterTrainer',
            'rate_limit_safe': True,
            'max_symbols_per_batch': 26,
            'batch_delay_seconds': 120,
            'features_enabled': ['basic', 'technical', 'price', 'volume', 'volatility', 'momentum', 'microstructure', 'psychology', 'patterns'],
            'models_enabled': ['lightgbm', 'xgboost', 'catboost', 'random_forest', 'gradient_boosting', 'decision_tree', 'lstm', 'gru', 'transformer'],
            'advanced_features': True
        },
        'half_year': {
            'name': 'Half Year Training',
            'description': '6-Month Training (half year)',
            'days': 180.0,
            'minutes': 259200,
            'estimated_time': '2-3 hours',
            'weight': 'High',
            'recommended_for': 'Semi-annual model retraining',
            'trainer_class': 'HalfYearTrainer',
            'rate_limit_safe': True,
            'max_symbols_per_batch': 26,
            'batch_delay_seconds': 150,
            'features_enabled': ['basic', 'technical', 'price', 'volume', 'volatility', 'momentum', 'microstructure', 'psychology', 'patterns', 'regime_detection'],
            'models_enabled': ['lightgbm', 'xgboost', 'catboost', 'random_forest', 'gradient_boosting', 'decision_tree', 'lstm', 'gru', 'transformer', 'conv1d'],
            'advanced_features': True
        },
        'year': {
            'name': 'Year Training',
            'description': '1-Year Training (full year)',
            'days': 365.0,
            'minutes': 525600,
            'estimated_time': '4-6 hours',
            'weight': 'High',
            'recommended_for': 'Annual model retraining',
            'trainer_class': 'YearTrainer',
            'rate_limit_safe': True,
            'max_symbols_per_batch': 26,
            'batch_delay_seconds': 180,
            'features_enabled': ['basic', 'technical', 'price', 'volume', 'volatility', 'momentum', 'microstructure', 'psychology', 'patterns', 'regime_detection', 'external_alpha'],
            'models_enabled': ['lightgbm', 'xgboost', 'catboost', 'random_forest', 'gradient_boosting', 'decision_tree', 'lstm', 'gru', 'transformer', 'conv1d', 'ensemble'],
            'advanced_features': True
        },
        'two_year': {
            'name': 'Two Year Training',
            'description': '2-Year Training (extended)',
            'days': 730.0,
            'minutes': 1051200,
            'estimated_time': '8-12 hours',
            'weight': 'Very High',
            'recommended_for': 'Long-term model training',
            'trainer_class': 'TwoYearTrainer',
            'rate_limit_safe': True,
            'max_symbols_per_batch': 26,
            'batch_delay_seconds': 240,
            'features_enabled': ['basic', 'technical', 'price', 'volume', 'volatility', 'momentum', 'microstructure', 'psychology', 'patterns', 'regime_detection', 'external_alpha', 'quantum', 'ai_enhanced'],
            'models_enabled': ['lightgbm', 'xgboost', 'catboost', 'random_forest', 'gradient_boosting', 'decision_tree', 'lstm', 'gru', 'transformer', 'conv1d', 'ensemble', 'meta_learning'],
            'advanced_features': True
        }
    }
    
    # Advanced feature configurations
    FEATURE_CONFIGS = {
        'basic': {
            'enabled': True,
            'price_features': True,
            'volume_features': True,
            'time_features': True
        },
        'technical': {
            'enabled': True,
            'rsi': True,
            'macd': True,
            'bollinger_bands': True,
            'atr': True,
            'stochastic': True,
            'adx': True
        },
        'price': {
            'enabled': True,
            'returns': True,
            'log_returns': True,
            'price_momentum': True,
            'price_acceleration': True
        },
        'volume': {
            'enabled': True,
            'volume_momentum': True,
            'volume_ratio': True,
            'obv': True,
            'vwap': True
        },
        'volatility': {
            'enabled': True,
            'rolling_volatility': True,
            'garch': True,
            'realized_volatility': True
        },
        'momentum': {
            'enabled': True,
            'price_momentum': True,
            'volume_momentum': True,
            'momentum_divergence': True
        },
        'microstructure': {
            'enabled': True,
            'bid_ask_spread': True,
            'order_flow': True,
            'market_impact': True
        },
        'psychology': {
            'enabled': True,
            'fear_greed': True,
            'sentiment': True,
            'market_regime': True
        },
        'patterns': {
            'enabled': True,
            'candlestick_patterns': True,
            'chart_patterns': True,
            'support_resistance': True
        },
        'regime_detection': {
            'enabled': True,
            'volatility_regime': True,
            'trend_regime': True,
            'correlation_regime': True
        },
        'external_alpha': {
            'enabled': True,
            'alternative_data': True,
            'news_sentiment': True,
            'social_media': True
        },
        'quantum': {
            'enabled': True,
            'quantum_features': True,
            'quantum_entanglement': True
        },
        'ai_enhanced': {
            'enabled': True,
            'ai_features': True,
            'neural_features': True
        }
    }
    
    # Model configurations
    MODEL_CONFIGS = {
        'lightgbm': {
            'enabled': True,
            'params': {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1
            }
        },
        'xgboost': {
            'enabled': True,
            'params': {
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'n_estimators': 100
            }
        },
        'catboost': {
            'enabled': True,
            'params': {
                'objective': 'RMSE',
                'eval_metric': 'RMSE',
                'depth': 6,
                'learning_rate': 0.1,
                'iterations': 100,
                'verbose': False
            }
        },
        'random_forest': {
            'enabled': True,
            'params': {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'random_state': 42
            }
        },
        'gradient_boosting': {
            'enabled': True,
            'params': {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 6,
                'subsample': 0.8,
                'random_state': 42
            }
        },
        'decision_tree': {
            'enabled': True,
            'params': {
                'max_depth': 10,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'random_state': 42
            }
        },
        'lstm': {
            'enabled': True,
            'params': {
                'units': 50,
                'dropout': 0.2,
                'recurrent_dropout': 0.2,
                'return_sequences': False
            }
        },
        'gru': {
            'enabled': True,
            'params': {
                'units': 50,
                'dropout': 0.2,
                'recurrent_dropout': 0.2,
                'return_sequences': False
            }
        },
        'transformer': {
            'enabled': True,
            'params': {
                'num_heads': 8,
                'd_model': 64,
                'num_layers': 4,
                'dropout': 0.1
            }
        },
        'conv1d': {
            'enabled': True,
            'params': {
                'filters': 64,
                'kernel_size': 3,
                'activation': 'relu',
                'dropout': 0.2
            }
        },
        'ensemble': {
            'enabled': True,
            'methods': ['stacking', 'blending', 'voting'],
            'meta_learner': 'lightgbm'
        },
        'meta_learning': {
            'enabled': True,
            'base_models': ['lightgbm', 'xgboost', 'catboost'],
            'meta_model': 'lightgbm'
        }
    }
    
    def __init__(self, config_path: str = 'config.json'):
        self.config_path = config_path
        self.config = self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load config file: {e}")
            return {}
    
    def get_mode_config(self, mode: str) -> Dict[str, Any]:
        """Get configuration for a specific training mode"""
        return self.TRAINING_MODES.get(mode, {})
    
    def get_all_modes(self) -> List[str]:
        """Get all available training modes"""
        return list(self.TRAINING_MODES.keys())
    
    def get_feature_config(self, feature_type: str) -> Dict[str, Any]:
        """Get configuration for a specific feature type"""
        return self.FEATURE_CONFIGS.get(feature_type, {})
    
    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Get configuration for a specific model"""
        return self.MODEL_CONFIGS.get(model_name, {})
    
    def get_all_pairs(self) -> List[str]:
        """Get all available FDUSD pairs"""
        return self.ALL_FDUSD_PAIRS.copy()
    
    def get_default_symbols(self) -> List[str]:
        """Get default symbols for training"""
        return self.DEFAULT_SYMBOLS.copy()
    
    def validate_mode(self, mode: str) -> bool:
        """Validate if a training mode exists"""
        return mode in self.TRAINING_MODES
    
    def get_enabled_features(self, mode: str) -> List[str]:
        """Get enabled features for a specific mode"""
        config = self.get_mode_config(mode)
        return config.get('features_enabled', [])
    
    def get_enabled_models(self, mode: str) -> List[str]:
        """Get enabled models for a specific mode"""
        config = self.get_mode_config(mode)
        return config.get('models_enabled', [])
    
    def is_advanced_features_enabled(self, mode: str) -> bool:
        """Check if advanced features are enabled for a mode"""
        config = self.get_mode_config(mode)
        return config.get('advanced_features', False)
    
    def get_rate_limit_config(self, mode: str) -> Dict[str, Any]:
        """Get rate limiting configuration for a mode"""
        config = self.get_mode_config(mode)
        return {
            'max_symbols_per_batch': config.get('max_symbols_per_batch', 20),
            'batch_delay_seconds': config.get('batch_delay_seconds', 60),
            'rate_limit_safe': config.get('rate_limit_safe', True)
        }

# Global training configuration instance
training_config = TrainingConfig() 