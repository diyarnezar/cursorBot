"""
🚀 PROJECT HYPERION - AUTOMATED STRATEGY DISCOVERY
================================================

Automated research mode that discovers new strategies and auto-promotes winners.
Periodically backtests new feature combinations and model architectures.

Author: Project Hyperion Team
Date: 2025
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import json
from pathlib import Path
import itertools
import random
import asyncio
import time
from scipy import stats

from core.high_fidelity_backtester import HighFidelityBacktester
from core.historical_data_warehouse import HistoricalDataWarehouse
from core.asset_cluster_manager import AssetClusterManager
# Import models
from models.tree_based.tree_models import TreeBasedModels
from models.time_series.time_series_models import TimeSeriesModels
from models.neural.lstm_models import LSTMModels
from models.neural.transformer_models import TransformerModels
from models.neural.conv1d_models import Conv1DModels
from models.neural.gru_models import GRUModels
from features.psychology.psychology_features import PsychologyFeatures
from features.external_alpha.external_alpha_features import ExternalAlphaFeatures
from features.microstructure.microstructure_features import MicrostructureFeatures
from features.patterns.pattern_features import PatternFeatures
from features.regime_detection.regime_detection_features import RegimeDetectionFeatures
from features.volatility_momentum.volatility_momentum_features import VolatilityMomentumFeatures
from features.adaptive_risk.adaptive_risk_features import AdaptiveRiskFeatures
from features.profitability.profitability_features import ProfitabilityFeatures
from features.meta_learning.meta_learning_features import MetaLearningFeatures
from features.ai_enhanced.ai_features import AIEnhancedFeatures
from features.quantum.quantum_features import QuantumFeatures


class AutomatedStrategyDiscovery:
    """
    Automated strategy discovery system
    Discovers and validates new trading strategies automatically
    """
    
    def __init__(self, config_path: str = "config.json"):
        """Initialize the Automated Strategy Discovery"""
        self.logger = logging.getLogger(__name__)
        self.config_path = config_path
        
        # Load configuration
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}")
            self.config = {}
        
        # Initialize components
        self.backtester = HighFidelityBacktester(config_path)
        self.data_warehouse = HistoricalDataWarehouse(config_path)
        self.asset_cluster_manager = AssetClusterManager(config_path)
        
        # Feature generators
        self.feature_generators = {
            'psychology': PsychologyFeatures(config=self.config),
            'external_alpha': ExternalAlphaFeatures(),
            'microstructure': MicrostructureFeatures(config=self.config),
            'patterns': PatternFeatures(),
            'regime_detection': RegimeDetectionFeatures(),
            'volatility_momentum': VolatilityMomentumFeatures(),
            'adaptive_risk': AdaptiveRiskFeatures(),
            'profitability': ProfitabilityFeatures(),
            'meta_learning': MetaLearningFeatures(),
            'ai_enhanced': AIEnhancedFeatures(config=self.config),
            'quantum': QuantumFeatures(config=self.config)
        }
        
        # Model trainers
        self.tree_models = TreeBasedModels()
        self.time_series_models = TimeSeriesModels()
        self.neural_models = LSTMModels(config=self.config) # Changed from NeuralModels to LSTMModels
        
        # Discovery settings
        self.research_interval_hours = 24  # Run research every 24 hours
        self.max_strategies_per_research = 10  # Test max 10 strategies per research cycle
        self.min_improvement_threshold = 0.05  # 5% improvement required
        self.statistical_significance_level = 0.05  # 5% significance level
        self.min_backtest_days = 30  # Minimum 30 days for backtest
        self.max_backtest_days = 180  # Maximum 180 days for backtest
        
        # Strategy components
        self.feature_combinations = self._generate_feature_combinations()
        self.model_architectures = self._generate_model_architectures()
        self.strategy_parameters = self._generate_strategy_parameters()
        
        # Research state
        self.current_champion_strategies = {}
        self.research_history = []
        self.discovered_strategies = []
        self.promoted_strategies = []
        
        # Performance tracking
        self.research_stats = {
            'total_research_cycles': 0,
            'strategies_tested': 0,
            'strategies_promoted': 0,
            'best_improvement': 0.0,
            'last_research_time': None
        }
        
        self.logger.info("🚀 Automated Strategy Discovery initialized")
    
    def _generate_feature_combinations(self) -> List[List[str]]:
        """Generate feature combinations for testing"""
        try:
            feature_types = list(self.feature_generators.keys())
            
            # Generate combinations of different sizes
            combinations = []
            
            # Single features
            combinations.extend([[feature] for feature in feature_types])
            
            # Two-feature combinations
            combinations.extend(list(itertools.combinations(feature_types, 2)))
            
            # Three-feature combinations (most common)
            combinations.extend(list(itertools.combinations(feature_types, 3)))
            
            # Four-feature combinations (advanced)
            combinations.extend(list(itertools.combinations(feature_types, 4)))
            
            # Convert tuples to lists
            combinations = [list(combo) for combo in combinations]
            
            self.logger.info(f"🧬 Generated {len(combinations)} feature combinations")
            return combinations
            
        except Exception as e:
            self.logger.error(f"❌ Error generating feature combinations: {e}")
            return []
    
    def _generate_model_architectures(self) -> List[Dict[str, Any]]:
        """Generate model architectures for testing"""
        try:
            architectures = []
            
            # Tree-based models
            tree_configs = [
                {'type': 'lightgbm', 'params': {'n_estimators': 100, 'learning_rate': 0.1}},
                {'type': 'lightgbm', 'params': {'n_estimators': 200, 'learning_rate': 0.05}},
                {'type': 'xgboost', 'params': {'n_estimators': 100, 'max_depth': 6}},
                {'type': 'xgboost', 'params': {'n_estimators': 200, 'max_depth': 8}},
                {'type': 'catboost', 'params': {'iterations': 100, 'depth': 6}},
                {'type': 'random_forest', 'params': {'n_estimators': 100, 'max_depth': 10}}
            ]
            
            # Time series models
            ts_configs = [
                {'type': 'lstm', 'params': {'units': 50, 'layers': 2}},
                {'type': 'lstm', 'params': {'units': 100, 'layers': 3}},
                {'type': 'gru', 'params': {'units': 50, 'layers': 2}},
                {'type': 'transformer', 'params': {'heads': 8, 'layers': 4}},
                {'type': 'conv1d', 'params': {'filters': 64, 'kernel_size': 3}}
            ]
            
            # Neural network models
            nn_configs = [
                {'type': 'neural_network', 'params': {'layers': [64, 32], 'dropout': 0.2}},
                {'type': 'neural_network', 'params': {'layers': [128, 64, 32], 'dropout': 0.3}},
                {'type': 'neural_network', 'params': {'layers': [256, 128, 64], 'dropout': 0.4}}
            ]
            
            architectures.extend(tree_configs)
            architectures.extend(ts_configs)
            architectures.extend(nn_configs)
            
            self.logger.info(f"🧠 Generated {len(architectures)} model architectures")
            return architectures
            
        except Exception as e:
            self.logger.error(f"❌ Error generating model architectures: {e}")
            return []
    
    def _generate_strategy_parameters(self) -> List[Dict[str, Any]]:
        """Generate strategy parameters for testing"""
        try:
            parameters = []
            
            # Momentum strategies
            momentum_params = [
                {'strategy_type': 'momentum', 'lookback_period': 5, 'threshold': 0.001},
                {'strategy_type': 'momentum', 'lookback_period': 10, 'threshold': 0.002},
                {'strategy_type': 'momentum', 'lookback_period': 20, 'threshold': 0.003}
            ]
            
            # Mean reversion strategies
            mean_reversion_params = [
                {'strategy_type': 'mean_reversion', 'window': 20, 'std_threshold': 1.5},
                {'strategy_type': 'mean_reversion', 'window': 50, 'std_threshold': 2.0},
                {'strategy_type': 'mean_reversion', 'window': 100, 'std_threshold': 2.5}
            ]
            
            # Volatility strategies
            volatility_params = [
                {'strategy_type': 'volatility', 'vol_window': 20, 'vol_threshold': 0.02},
                {'strategy_type': 'volatility', 'vol_window': 50, 'vol_threshold': 0.03}
            ]
            
            # Multi-timeframe strategies
            multi_tf_params = [
                {'strategy_type': 'multi_timeframe', 'short_period': 5, 'long_period': 20},
                {'strategy_type': 'multi_timeframe', 'short_period': 10, 'long_period': 50}
            ]
            
            parameters.extend(momentum_params)
            parameters.extend(mean_reversion_params)
            parameters.extend(volatility_params)
            parameters.extend(multi_tf_params)
            
            self.logger.info(f"⚙️ Generated {len(parameters)} strategy parameter sets")
            return parameters
            
        except Exception as e:
            self.logger.error(f"❌ Error generating strategy parameters: {e}")
            return []
    
    async def start_research_mode(self, symbols: List[str]):
        """Start continuous research mode"""
        try:
            self.logger.info("🚀 Starting automated strategy discovery research mode")
            
            while True:
                try:
                    # Run research cycle
                    await self.run_research_cycle(symbols)
                    
                    # Wait for next research cycle
                    await asyncio.sleep(self.research_interval_hours * 3600)
                    
                except Exception as e:
                    self.logger.error(f"❌ Error in research cycle: {e}")
                    await asyncio.sleep(3600)  # Wait 1 hour on error
                    
        except Exception as e:
            self.logger.error(f"❌ Error in research mode: {e}")
    
    async def run_research_cycle(self, symbols: List[str]):
        """Run a single research cycle"""
        try:
            self.logger.info("🔬 Starting research cycle")
            
            start_time = datetime.now()
            self.research_stats['total_research_cycles'] += 1
            
            # Generate candidate strategies
            candidate_strategies = self._generate_candidate_strategies()
            
            # Test strategies
            test_results = []
            strategies_tested = 0
            
            for strategy in candidate_strategies[:self.max_strategies_per_research]:
                try:
                    # Test strategy
                    result = await self._test_strategy(strategy, symbols)
                    
                    if result and result['success']:
                        test_results.append(result)
                        strategies_tested += 1
                        
                        self.logger.info(f"🧪 Tested strategy {strategies_tested}: "
                                       f"Sharpe={result['sharpe_ratio']:.3f}, "
                                       f"Return={result['total_return']:.2%}")
                    
                except Exception as e:
                    self.logger.error(f"❌ Error testing strategy: {e}")
                    continue
            
            # Analyze results and promote winners
            promoted_strategies = self._analyze_and_promote_strategies(test_results)
            
            # Update research statistics
            self.research_stats['strategies_tested'] += strategies_tested
            self.research_stats['strategies_promoted'] += len(promoted_strategies)
            self.research_stats['last_research_time'] = datetime.now()
            
            # Record research cycle
            research_cycle = {
                'timestamp': datetime.now(),
                'strategies_tested': strategies_tested,
                'strategies_promoted': len(promoted_strategies),
                'test_results': test_results,
                'promoted_strategies': promoted_strategies,
                'duration': (datetime.now() - start_time).total_seconds()
            }
            
            self.research_history.append(research_cycle)
            
            # Save research results
            self._save_research_results(research_cycle)
            
            self.logger.info(f"✅ Research cycle completed: {strategies_tested} tested, "
                           f"{len(promoted_strategies)} promoted")
            
        except Exception as e:
            self.logger.error(f"❌ Error in research cycle: {e}")
    
    def _generate_candidate_strategies(self) -> List[Dict[str, Any]]:
        """Generate candidate strategies for testing"""
        try:
            strategies = []
            
            # Randomly sample feature combinations, model architectures, and parameters
            for _ in range(self.max_strategies_per_research):
                strategy = {
                    'id': f"strategy_{int(time.time() * 1000)}",
                    'features': random.choice(self.feature_combinations),
                    'model': random.choice(self.model_architectures),
                    'parameters': random.choice(self.strategy_parameters),
                    'created_at': datetime.now()
                }
                
                strategies.append(strategy)
            
            return strategies
            
        except Exception as e:
            self.logger.error(f"❌ Error generating candidate strategies: {e}")
            return []
    
    async def _test_strategy(self, strategy: Dict[str, Any], symbols: List[str]) -> Optional[Dict[str, Any]]:
        """Test a single strategy"""
        try:
            # Get historical data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.min_backtest_days)
            
            # Load data for testing
            test_data = self.data_warehouse.merge_data_by_timestamp(
                ['price_data', 'sentiment_data'], symbols[0], start_date, end_date
            )
            
            if test_data is None or test_data.empty:
                return None
            
            # Generate features based on strategy
            features = self._generate_strategy_features(test_data, strategy['features'])
            
            if features is None or features.empty:
                return None
            
            # Train model based on strategy
            model = self._train_strategy_model(features, strategy['model'])
            
            if model is None:
                return None
            
            # Create strategy configuration
            strategy_config = {
                'strategy_type': strategy['parameters']['strategy_type'],
                'model': model,
                'features': strategy['features'],
                'parameters': strategy['parameters']
            }
            
            # Run backtest
            backtest_result = self.backtester.run_backtest(
                strategy_config, start_date, end_date, symbols[:1]  # Test on single symbol
            )
            
            if backtest_result and 'error' not in backtest_result:
                # Add strategy metadata
                backtest_result['strategy_id'] = strategy['id']
                backtest_result['strategy_config'] = strategy
                backtest_result['success'] = True
                
                return backtest_result
            
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Error testing strategy: {e}")
            return None
    
    def _generate_strategy_features(self, data: pd.DataFrame, feature_types: List[str]) -> Optional[pd.DataFrame]:
        """Generate features for strategy testing"""
        try:
            features = data.copy()
            
            for feature_type in feature_types:
                if feature_type in self.feature_generators:
                    try:
                        feature_generator = self.feature_generators[feature_type]
                        
                        if hasattr(feature_generator, 'generate_features'):
                            new_features = feature_generator.generate_features(data)
                        elif hasattr(feature_generator, 'get_all_features'):
                            new_features = feature_generator.get_all_features('TESTFDUSD')
                        else:
                            continue
                        
                        if new_features is not None and not new_features.empty:
                            # Add new features
                            for col in new_features.columns:
                                if col not in features.columns:
                                    features[col] = new_features[col]
                    
                    except Exception as e:
                        self.logger.warning(f"⚠️ Error generating {feature_type} features: {e}")
                        continue
            
            # Clean features
            features = features.dropna()
            
            if len(features) < 100:  # Need minimum data points
                return None
            
            return features
            
        except Exception as e:
            self.logger.error(f"❌ Error generating strategy features: {e}")
            return None
    
    def _train_strategy_model(self, features: pd.DataFrame, model_config: Dict[str, Any]) -> Any:
        """Train model for strategy testing"""
        try:
            # Create target variable (next period return)
            features['target'] = features['close'].pct_change().shift(-1)
            features = features.dropna()
            
            if len(features) < 50:  # Need minimum data points
                return None
            
            # Prepare features and target
            feature_cols = [col for col in features.columns if col not in ['timestamp', 'symbol', 'target']]
            X = features[feature_cols].values
            y = features['target'].values
            
            # Train model based on type
            model_type = model_config['type']
            params = model_config['params']
            
            if model_type in ['lightgbm', 'xgboost', 'catboost', 'random_forest']:
                model = self.tree_models.train_model(model_type, features, target_col='target')
            elif model_type in ['lstm', 'gru', 'transformer', 'conv1d']:
                model = self.time_series_models.train_model(model_type, features, target_col='target')
            elif model_type == 'neural_network':
                model = self.neural_models.train_model(model_type, features, target_col='target')
            else:
                return None
            
            return model
            
        except Exception as e:
            self.logger.error(f"❌ Error training strategy model: {e}")
            return None
    
    def _analyze_and_promote_strategies(self, test_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze test results and promote winning strategies"""
        try:
            promoted_strategies = []
            
            if not test_results:
                return promoted_strategies
            
            # Get current champion performance
            champion_performance = self._get_champion_performance()
            
            for result in test_results:
                try:
                    # Check if strategy meets improvement threshold
                    improvement = self._calculate_improvement(result, champion_performance)
                    
                    if improvement > self.min_improvement_threshold:
                        # Check statistical significance
                        if self._is_statistically_significant(result, champion_performance):
                            # Promote strategy
                            promoted_strategy = self._promote_strategy(result)
                            promoted_strategies.append(promoted_strategy)
                            
                            self.logger.info(f"🏆 Promoted strategy {result['strategy_id']}: "
                                           f"{improvement:.2%} improvement")
                
                except Exception as e:
                    self.logger.error(f"❌ Error analyzing strategy result: {e}")
                    continue
            
            # Update best improvement
            if promoted_strategies:
                best_improvement = max([s['improvement'] for s in promoted_strategies])
                self.research_stats['best_improvement'] = max(
                    self.research_stats['best_improvement'], best_improvement
                )
            
            return promoted_strategies
            
        except Exception as e:
            self.logger.error(f"❌ Error analyzing and promoting strategies: {e}")
            return []
    
    def _get_champion_performance(self) -> Dict[str, float]:
        """Get current champion strategy performance"""
        try:
            # For now, use baseline performance
            # In production, this would get the current best strategy performance
            return {
                'sharpe_ratio': 1.0,
                'total_return': 0.15,  # 15% annual return
                'max_drawdown': 0.10,  # 10% max drawdown
                'win_rate': 0.55      # 55% win rate
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error getting champion performance: {e}")
            return {}
    
    def _calculate_improvement(self, result: Dict[str, Any], champion: Dict[str, Any]) -> float:
        """Calculate improvement over champion strategy"""
        try:
            # Calculate improvement score based on multiple metrics
            sharpe_improvement = (result['sharpe_ratio'] - champion['sharpe_ratio']) / max(champion['sharpe_ratio'], 0.1)
            return_improvement = (result['total_return'] - champion['total_return']) / max(champion['total_return'], 0.01)
            drawdown_improvement = (champion['max_drawdown'] - result['max_drawdown']) / max(champion['max_drawdown'], 0.01)
            winrate_improvement = (result['win_rate'] - champion['win_rate']) / max(champion['win_rate'], 0.1)
            
            # Weighted improvement score
            improvement = (
                sharpe_improvement * 0.4 +
                return_improvement * 0.3 +
                drawdown_improvement * 0.2 +
                winrate_improvement * 0.1
            )
            
            return improvement
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating improvement: {e}")
            return 0.0
    
    def _is_statistically_significant(self, result: Dict[str, Any], champion: Dict[str, Any]) -> bool:
        """Check if improvement is statistically significant"""
        try:
            # Simple statistical significance test
            # In production, this would use proper statistical tests
            
            # For now, use a simple threshold-based approach
            improvement = self._calculate_improvement(result, champion)
            
            # Require both improvement and good absolute performance
            return (improvement > self.min_improvement_threshold and 
                   result['sharpe_ratio'] > 1.0 and
                   result['total_return'] > 0.05)
            
        except Exception as e:
            self.logger.error(f"❌ Error checking statistical significance: {e}")
            return False
    
    def _promote_strategy(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Promote a winning strategy"""
        try:
            promoted_strategy = {
                'strategy_id': result['strategy_id'],
                'promotion_time': datetime.now(),
                'performance': {
                    'sharpe_ratio': result['sharpe_ratio'],
                    'total_return': result['total_return'],
                    'max_drawdown': result['max_drawdown'],
                    'win_rate': result['win_rate'],
                    'profit_factor': result['profit_factor']
                },
                'strategy_config': result['strategy_config'],
                'improvement': self._calculate_improvement(result, self._get_champion_performance())
            }
            
            # Add to promoted strategies
            self.promoted_strategies.append(promoted_strategy)
            
            # Update champion strategies
            cluster_name = self.asset_cluster_manager.get_cluster_for_asset('ETHFDUSD')
            if cluster_name:
                self.current_champion_strategies[cluster_name] = promoted_strategy
            
            # Save promoted strategy
            self._save_promoted_strategy(promoted_strategy)
            
            return promoted_strategy
            
        except Exception as e:
            self.logger.error(f"❌ Error promoting strategy: {e}")
            return {}
    
    def _save_research_results(self, research_cycle: Dict[str, Any]):
        """Save research cycle results"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = f"research/research_cycle_{timestamp}.json"
            
            # Create directory if it doesn't exist
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            with open(filepath, 'w') as f:
                json.dump(research_cycle, f, indent=2, default=str)
            
            self.logger.info(f"💾 Research results saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"❌ Error saving research results: {e}")
    
    def _save_promoted_strategy(self, strategy: Dict[str, Any]):
        """Save promoted strategy"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = f"strategies/promoted_strategy_{strategy['strategy_id']}_{timestamp}.json"
            
            # Create directory if it doesn't exist
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            with open(filepath, 'w') as f:
                json.dump(strategy, f, indent=2, default=str)
            
            self.logger.info(f"💾 Promoted strategy saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"❌ Error saving promoted strategy: {e}")
    
    def get_research_stats(self) -> Dict[str, Any]:
        """Get research statistics"""
        return self.research_stats.copy()
    
    def get_promoted_strategies(self) -> List[Dict[str, Any]]:
        """Get all promoted strategies"""
        return self.promoted_strategies.copy()
    
    def get_champion_strategies(self) -> Dict[str, Any]:
        """Get current champion strategies by cluster"""
        return self.current_champion_strategies.copy()
    
    def export_research_report(self, filepath: str = None):
        """Export comprehensive research report"""
        try:
            if filepath is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filepath = f"reports/research_report_{timestamp}.json"
            
            # Create directory if it doesn't exist
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            report = {
                'timestamp': datetime.now().isoformat(),
                'research_stats': self.research_stats,
                'promoted_strategies': self.promoted_strategies,
                'champion_strategies': self.current_champion_strategies,
                'research_history': self.research_history[-10:]  # Last 10 cycles
            }
            
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            self.logger.info(f"💾 Research report exported to {filepath}")
            
        except Exception as e:
            self.logger.error(f"❌ Error exporting research report: {e}") 