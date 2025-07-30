"""
Multi-Timeframe Trainer for Advanced Cryptocurrency Trading
Part of Project Hyperion - Ultimate Autonomous Trading Bot
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from core.base_trainer import BaseTrainer
from modules.feature_engineering import EnhancedFeatureEngineer
from models.enhanced_model_trainer import EnhancedModelTrainer
from models.ensemble_trainer import EnsembleTrainer
from utils.logging.logger import setup_logger
from utils.metrics import calculate_advanced_metrics
from utils.optimization import HyperparameterOptimizer
from utils.reinforcement_learning import RLAgent
from utils.self_improvement import SelfImprovementEngine

logger = logging.getLogger(__name__)


class MultiTimeframeTrainer(BaseTrainer):
    """
    Advanced Multi-Timeframe Trainer for Cryptocurrency Trading
    
    Features:
    - Multiple Timeframe Support (1m, 5m, 15m, 1h, 4h, 1d)
    - Timeframe Alignment and Synchronization
    - Hierarchical Feature Engineering
    - Cross-Timeframe Validation
    - Ensemble Predictions
    - Performance Aggregation
    - Parallel Processing
    """
    
    def __init__(self, symbols: Optional[List[str]] = None):
        """Initialize multi-timeframe trainer"""
        super().__init__('multi_timeframe', symbols)
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        
        # Advanced components for maximum intelligence
        self.feature_engineer = EnhancedFeatureEngineer()
        self.model_trainer = EnhancedModelTrainer()
        self.ensemble_trainer = EnsembleTrainer()
        self.hyperopt = HyperparameterOptimizer()
        self.rl_agent = RLAgent()
        self.self_improvement = SelfImprovementEngine()
        
        # Advanced configuration for multi-timeframe mode
        self.config.update({
            'use_deep_learning': True,
            'use_ensemble': True,
            'use_meta_learning': True,
            'use_reinforcement_learning': True,
            'use_self_improvement': True,
            'use_hyperparameter_optimization': True,
            'use_advanced_features': True,
            'use_alternative_data': True,
            'use_crypto_specific_features': True,
            'max_optimization_iterations': 80,
            'rl_episodes': 160,
            'self_improvement_cycles': 8
        })
        
        self.logger.info("🚀 Advanced MultiTimeframeTrainer initialized with maximum intelligence")
        
        # Multi-timeframe configuration
        self.timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']
        self.base_timeframe = '1m'
        self.target_timeframe = '15m'
        self.lookback_periods = {
            '1m': 60, '5m': 12, '15m': 4, '1h': 1, '4h': 1, '1d': 1
        }
        self.feature_aggregation_methods = ['mean', 'std', 'min', 'max', 'last']
        
        # Model storage
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.performance_metrics = {}
        
        # Timeframe data storage
        self.timeframe_data = {}
        self.aligned_data = {}
        
        # Training results
        self.training_results = {}
        self.ensemble_weights = {}
        
        self.logger.info("Multi-Timeframe Trainer initialized with parallel processing")
    
    def collect_data(self) -> pd.DataFrame:
        """Collect data for multi-timeframe training using parallel processing"""
        self.logger.info("Collecting data for multi-timeframe training using parallel processing...")
        return self.collect_data_parallel(days=30, interval='1m')  # Use 30 days for multi-timeframe
    
    def prepare_features(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
        """Prepare features for multi-timeframe training"""
        self.logger.info("Preparing features for multi-timeframe training...")
        
        if data.empty:
            self.logger.error("No data available for feature preparation")
            return pd.DataFrame(), {}
        
        try:
            # Generate base features using parallel processing
            features = self.feature_engineer.enhance_features(data)
            
            # Create multi-timeframe features
            features = self.create_hierarchical_features(features)
            
            # Create targets for different timeframes
            targets = {}
            
            # 1-minute target
            targets['1m'] = data['close'].pct_change(1).shift(-1)
            
            # 5-minute target
            targets['5m'] = data['close'].pct_change(5).shift(-5)
            
            # 15-minute target
            targets['15m'] = data['close'].pct_change(15).shift(-15)
            
            # 1-hour target
            targets['1h'] = data['close'].pct_change(60).shift(-60)
            
            # 4-hour target
            targets['4h'] = data['close'].pct_change(240).shift(-240)
            
            # 1-day target
            targets['1d'] = data['close'].pct_change(1440).shift(-1440)
            
            # Binary targets (up/down)
            for timeframe in ['1m', '5m', '15m', '1h', '4h', '1d']:
                targets[f'{timeframe}_binary'] = (targets[timeframe] > 0).astype(int)
            
            # Remove NaN values
            features = features.dropna()
            for target_name, target_series in targets.items():
                targets[target_name] = target_series.dropna()
            
            # Align features and targets
            common_index = features.index.intersection(
                pd.concat(targets.values(), axis=1).dropna().index
            )
            
            features = features.loc[common_index]
            for target_name in targets:
                targets[target_name] = targets[target_name].loc[common_index]
            
            self.logger.info(f"✅ Prepared features: {features.shape}")
            self.logger.info(f"✅ Prepared targets: {len(targets)} timeframes")
            
            return features, targets
            
        except Exception as e:
            self.logger.error(f"Error preparing features: {e}")
            return pd.DataFrame(), {}
    
    def train_models(self, features: pd.DataFrame, targets: Dict[str, pd.Series]) -> Dict[str, Any]:
        """Train models for multi-timeframe prediction using parallel processing"""
        self.logger.info("Training models for multi-timeframe prediction...")
        
        if features.empty or not targets:
            self.logger.error("No features or targets available for training")
            return {}
        
        try:
            # Train models using parallel processing
            models = self.model_trainer.train_models(features, targets)
            
            # Add multi-timeframe specific models
            models.update(self.train_timeframe_specific_models(features, targets))
            
            self.logger.info(f"✅ Trained {len(models)} models")
            return models
            
        except Exception as e:
            self.logger.error(f"Error training models: {e}")
            return {}
    
    def evaluate_models(self, models: Dict[str, Any], features: pd.DataFrame, targets: Dict[str, pd.Series]) -> Dict[str, float]:
        """Evaluate trained models for multi-timeframe prediction"""
        self.logger.info("Evaluating models for multi-timeframe prediction...")
        
        if not models:
            self.logger.error("No models available for evaluation")
            return {}
        
        try:
            # Evaluate models
            metrics = self.model_trainer.evaluate_models(models, features, targets)
            
            # Add multi-timeframe specific metrics
            metrics.update(self.evaluate_timeframe_specific_models(models, features, targets))
            
            self.logger.info(f"✅ Evaluated {len(metrics)} metrics")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error evaluating models: {e}")
            return {}
    
    def train_timeframe_specific_models(self, features: pd.DataFrame, targets: Dict[str, pd.Series]) -> Dict[str, Any]:
        """Train models specific to each timeframe"""
        timeframe_models = {}
        
        for timeframe in self.timeframes:
            if f'{timeframe}_binary' in targets:
                # Train binary classification model for this timeframe
                target = targets[f'{timeframe}_binary']
                
                # Prepare data
                common_index = features.index.intersection(target.dropna().index)
                X = features.loc[common_index]
                y = target.loc[common_index]
                
                if len(X) > 100:  # Ensure sufficient data
                    # Train model (simplified for now)
                    from sklearn.ensemble import RandomForestClassifier
                    model = RandomForestClassifier(n_estimators=100, n_jobs=self.max_workers)
                    model.fit(X, y)
                    
                    timeframe_models[f'timeframe_{timeframe}_binary'] = {
                        'model': model,
                        'timeframe': timeframe,
                        'type': 'binary_classification'
                    }
        
        return timeframe_models
    
    def evaluate_timeframe_specific_models(self, models: Dict[str, Any], features: pd.DataFrame, targets: Dict[str, pd.Series]) -> Dict[str, float]:
        """Evaluate timeframe-specific models"""
        metrics = {}
        
        for model_name, model_info in models.items():
            if model_name.startswith('timeframe_'):
                timeframe = model_info['timeframe']
                target_name = f'{timeframe}_binary'
                
                if target_name in targets:
                    target = targets[target_name]
                    common_index = features.index.intersection(target.dropna().index)
                    X = features.loc[common_index]
                    y = target.loc[common_index]
                    
                    if len(X) > 0:
                        y_pred = model_info['model'].predict(X)
                        from sklearn.metrics import accuracy_score
                        accuracy = accuracy_score(y, y_pred)
                        metrics[f'{model_name}_accuracy'] = accuracy
        
        return metrics

    def train(self) -> Dict[str, Any]:
        """Execute advanced multi-timeframe training with full automation"""
        import time
        start_time = time.time()
        self.logger.info("🎯 Starting ADVANCED Multi-Timeframe Training with maximum intelligence")

        try:
            # Phase 1: Advanced Data Collection & Processing
            self.logger.info("📊 Phase 1: Advanced Data Collection & Processing")
            data = self.collect_data()

            # Phase 2: Maximum Feature Engineering
            self.logger.info("🔧 Phase 2: Maximum Feature Engineering")
            features, targets = self.prepare_features(data)

            # Phase 3: Advanced Model Training
            self.logger.info("🤖 Phase 3: Advanced Model Training")
            models = self._train_advanced_models(features, targets)

            # Phase 4: Reinforcement Learning Optimization
            self.logger.info("🧠 Phase 4: Reinforcement Learning Optimization")
            rl_models = self._apply_reinforcement_learning(features, targets, models)

            # Phase 5: Self-Improvement & Enhancement
            self.logger.info("⚡ Phase 5: Self-Improvement & Enhancement")
            enhanced_models = self._apply_self_improvement(features, targets, rl_models)

            # Phase 6: Ensemble & Meta-Learning
            self.logger.info("🎯 Phase 6: Ensemble & Meta-Learning")
            final_models = self._create_advanced_ensemble(features, targets, enhanced_models)

            # Phase 7: Performance Validation & Optimization
            self.logger.info("📈 Phase 7: Performance Validation & Optimization")
            results = self._validate_and_optimize(features, targets, final_models)

            training_time = time.time() - start_time
            self.logger.info(f"✅ Advanced Multi-Timeframe Training completed in {training_time:.2f} seconds")

            return results

        except Exception as e:
            self.logger.error(f"❌ Advanced Multi-Timeframe Training failed: {str(e)}")
            raise

    def _train_advanced_models(self, features: pd.DataFrame, targets: Dict[str, pd.Series]) -> Dict[str, Any]:
        """Train all advanced models with maximum intelligence"""
        self.logger.info("🤖 Training advanced models with maximum intelligence")

        models = {}

        # Train tree-based models
        tree_models = self.model_trainer.train_tree_models(features, targets)
        models.update(tree_models)

        # Train neural networks
        nn_models = self.model_trainer.train_neural_networks(features, targets)
        models.update(nn_models)

        # Train time series models
        ts_models = self.model_trainer.train_time_series_models(features, targets)
        models.update(ts_models)

        # Train deep learning models
        dl_models = self.model_trainer.train_deep_learning_models(features, targets)
        models.update(dl_models)

        # Train transformer models
        transformer_models = self.model_trainer.train_transformer_models(features, targets)
        models.update(transformer_models)

        # Train timeframe-specific models
        timeframe_models = self.train_timeframe_specific_models(features, targets)
        models.update(timeframe_models)

        self.logger.info(f"🤖 Trained {len(models)} advanced models")
        return models

    def _apply_reinforcement_learning(self, features: pd.DataFrame, targets: Dict[str, pd.Series], models: Dict[str, Any]) -> Dict[str, Any]:
        """Apply reinforcement learning for model optimization"""
        self.logger.info("🧠 Applying reinforcement learning optimization")

        # Initialize RL agent with models
        self.rl_agent.initialize_models(models)

        # Use the primary target (1m) for RL optimization
        primary_target = targets.get('1m', targets.get('1m_binary', pd.Series()))
        if primary_target.empty:
            self.logger.warning("No valid target found for RL optimization, skipping")
            return models
        
        # Run RL episodes for optimization
        optimized_models = self.rl_agent.optimize_models(
            features=features,
            target=primary_target,
            episodes=self.config['rl_episodes'],
            learning_rate=0.001
        )

        self.logger.info("🧠 Reinforcement learning optimization completed")
        return optimized_models

    def _apply_self_improvement(self, features: pd.DataFrame, targets: Dict[str, pd.Series], models: Dict[str, Any]) -> Dict[str, Any]:
        """Apply self-improvement and enhancement"""
        self.logger.info("⚡ Applying self-improvement and enhancement")

        # Initialize self-improvement engine
        self.self_improvement.initialize(features, targets, models)

        # Run self-improvement cycles
        enhanced_models = self.self_improvement.improve_models(
            cycles=self.config['self_improvement_cycles'],
            improvement_rate=0.1
        )

        self.logger.info("⚡ Self-improvement and enhancement completed")
        return enhanced_models

    def _create_advanced_ensemble(self, features: pd.DataFrame, targets: Dict[str, pd.Series], models: Dict[str, Any]) -> Dict[str, Any]:
        """Create advanced ensemble with meta-learning"""
        self.logger.info("🎯 Creating advanced ensemble with meta-learning")

        # Create ensemble with all models
        ensemble = self.ensemble_trainer.create_ensemble(
            models=models,
            features=features,
            targets=targets,
            use_meta_learning=True,
            use_dynamic_weighting=True
        )

        # Add meta-learning layer
        meta_ensemble = self.ensemble_trainer.add_meta_learning_layer(
            ensemble=ensemble,
            features=features,
            targets=targets
        )

        self.logger.info("🎯 Advanced ensemble with meta-learning created")
        return meta_ensemble

    def _validate_and_optimize(self, features: pd.DataFrame, targets: Dict[str, pd.Series], models: Dict[str, Any]) -> Dict[str, Any]:
        """Validate performance and optimize hyperparameters"""
        self.logger.info("📈 Validating performance and optimizing hyperparameters")

        # Calculate advanced metrics
        metrics = calculate_advanced_metrics(features, targets, models)

        # Optimize hyperparameters
        optimized_models = self.hyperopt.optimize_models(
            models=models,
            features=features,
            targets=targets,
            max_iterations=self.config['max_optimization_iterations']
        )

        # Final validation
        final_metrics = calculate_advanced_metrics(features, targets, optimized_models)

        results = {
            'models': optimized_models,
            'metrics': final_metrics,
            'training_time': time.time(),
            'mode': 'multi_timeframe',
            'symbols': self.symbols,
            'config': self.config
        }

        self.logger.info("📈 Performance validation and optimization completed")
        return results

    def create_hierarchical_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Create hierarchical features for multi-timeframe analysis"""
        self.logger.info("🔧 Creating hierarchical features for multi-timeframe analysis")
        
        # This is a placeholder for hierarchical feature creation
        # In a full implementation, this would create features that capture
        # relationships between different timeframes
        
        return features 