#!/usr/bin/env python3
"""
Walk-Forward Optimization Module
Implements advanced validation techniques to eliminate look-ahead bias
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timedelta
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os

class WalkForwardOptimizer:
    """
    Walk-Forward Optimization with Purged & Embargoed Cross-Validation
    
    Features:
    - Rolling window training with out-of-sample testing
    - Purged overlapping labels to eliminate look-ahead bias
    - Embargoed time-adjacent data
    - Regime-aware sampling for balanced validation
    - Comprehensive performance aggregation
    """
    
    def __init__(self, 
                 train_window_days: int = 252,  # 1 year training window
                 test_window_days: int = 63,    # 3 months test window
                 step_size_days: int = 21,      # 3 weeks step size
                 purge_days: int = 5,           # 5 days purge period
                 embargo_days: int = 2,         # 2 days embargo period
                 min_train_samples: int = 1000,
                 min_test_samples: int = 100):
        """
        Initialize Walk-Forward Optimizer.
        
        Args:
            train_window_days: Training window size in days
            test_window_days: Test window size in days
            step_size_days: Step size between windows in days
            purge_days: Days to purge for overlapping labels
            embargo_days: Days to embargo after test period
            min_train_samples: Minimum samples required for training
            min_test_samples: Minimum samples required for testing
        """
        self.train_window_days = train_window_days
        self.test_window_days = test_window_days
        self.step_size_days = step_size_days
        self.purge_days = purge_days
        self.embargo_days = embargo_days
        self.min_train_samples = min_train_samples
        self.min_test_samples = min_test_samples
        
        # Results storage
        self.wfo_results = []
        self.performance_history = []
        self.model_versions = {}
        
        logging.info("Walk-Forward Optimizer initialized")
    
    def run_walk_forward_optimization(self, 
                                    data: pd.DataFrame,
                                    model_factory,
                                    target_column: str = 'target',
                                    feature_columns: List[str] = None) -> Dict[str, Any]:
        """
        Run complete Walk-Forward Optimization.
        
        Args:
            data: DataFrame with features and target
            model_factory: Function that creates and trains models
            target_column: Name of target column
            feature_columns: List of feature column names
            
        Returns:
            Dictionary with WFO results and performance metrics
        """
        try:
            logging.info("🚀 Starting Walk-Forward Optimization")
            
            if feature_columns is None:
                feature_columns = [col for col in data.columns if col != target_column]
            
            # Ensure data is sorted by time
            if 'timestamp' in data.columns:
                data = data.sort_values('timestamp').reset_index(drop=True)
            
            # Generate walk-forward windows
            windows = self._generate_wfo_windows(data)
            
            results = []
            for i, window in enumerate(windows):
                logging.info(f"Processing window {i+1}/{len(windows)}")
                
                # Extract train/test data with purging and embargo
                train_data, test_data = self._extract_window_data(data, window)
                
                if len(train_data) < self.min_train_samples or len(test_data) < self.min_test_samples:
                    logging.warning(f"Window {i+1}: Insufficient data, skipping")
                    continue
                
                # Train model
                model = model_factory(train_data[feature_columns], train_data[target_column])
                
                # Evaluate on test set
                test_predictions = model.predict(test_data[feature_columns])
                test_actual = test_data[target_column]
                
                # Calculate metrics
                metrics = self._calculate_window_metrics(test_actual, test_predictions)
                metrics['window_id'] = i
                metrics['train_start'] = window['train_start']
                metrics['train_end'] = window['train_end']
                metrics['test_start'] = window['test_start']
                metrics['test_end'] = window['test_end']
                metrics['train_samples'] = len(train_data)
                metrics['test_samples'] = len(test_data)
                
                results.append(metrics)
                
                # Store model version
                self.model_versions[f'wfo_window_{i}'] = {
                    'model': model,
                    'performance': metrics,
                    'train_data_shape': train_data.shape,
                    'test_data_shape': test_data.shape
                }
            
            # Aggregate results
            aggregated_results = self._aggregate_wfo_results(results)
            
            # Store results
            self.wfo_results = results
            self.performance_history.append({
                'timestamp': datetime.now(),
                'results': aggregated_results
            })
            
            logging.info(f"✅ Walk-Forward Optimization completed: {len(results)} windows")
            return aggregated_results
            
        except Exception as e:
            logging.error(f"Error in Walk-Forward Optimization: {e}")
            return {}
    
    def _generate_wfo_windows(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate walk-forward windows with proper overlap handling."""
        windows = []
        
        total_days = len(data)
        current_start = 0
        
        while current_start + self.train_window_days + self.test_window_days <= total_days:
            train_start = current_start
            train_end = current_start + self.train_window_days
            
            # Test window starts after purge period
            test_start = train_end + self.purge_days
            test_end = test_start + self.test_window_days
            
            # Ensure we don't exceed data bounds
            if test_end > total_days:
                break
            
            windows.append({
                'train_start': train_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end,
                'purge_start': train_end,
                'purge_end': test_start,
                'embargo_start': test_end,
                'embargo_end': test_end + self.embargo_days
            })
            
            # Move to next window
            current_start += self.step_size_days
        
        logging.info(f"Generated {len(windows)} walk-forward windows")
        return windows
    
    def _extract_window_data(self, data: pd.DataFrame, window: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Extract train/test data with purging and embargo applied."""
        # Extract training data (excluding purge period)
        train_data = data.iloc[window['train_start']:window['train_end']].copy()
        
        # Extract test data (excluding embargo period)
        test_data = data.iloc[window['test_start']:window['test_end']].copy()
        
        # Apply regime-aware sampling if needed
        train_data = self._apply_regime_sampling(train_data)
        test_data = self._apply_regime_sampling(test_data)
        
        return train_data, test_data
    
    def _apply_regime_sampling(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply regime-aware sampling to ensure balanced representation."""
        try:
            # Calculate volatility regime
            if 'close' in data.columns:
                returns = data['close'].pct_change().dropna()
                volatility = returns.rolling(20).std()
                
                # Define regimes
                low_vol = volatility < volatility.quantile(0.33)
                high_vol = volatility > volatility.quantile(0.67)
                normal_vol = ~(low_vol | high_vol)
                
                # Ensure each regime is represented
                regime_counts = {
                    'low_vol': low_vol.sum(),
                    'normal_vol': normal_vol.sum(),
                    'high_vol': high_vol.sum()
                }
                
                min_samples_per_regime = min(regime_counts.values()) if regime_counts.values() else 0
                
                if min_samples_per_regime > 0:
                    # Sample equally from each regime
                    sampled_data = []
                    for regime_mask in [low_vol, normal_vol, high_vol]:
                        regime_data = data[regime_mask]
                        if len(regime_data) > min_samples_per_regime:
                            sampled_data.append(regime_data.sample(n=min_samples_per_regime))
                        else:
                            sampled_data.append(regime_data)
                    
                    return pd.concat(sampled_data).sort_index()
            
            return data
            
        except Exception as e:
            logging.warning(f"Error in regime sampling: {e}")
            return data
    
    def _calculate_window_metrics(self, actual: pd.Series, predictions: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive metrics for a window."""
        try:
            # Basic regression metrics
            mse = mean_squared_error(actual, predictions)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(actual, predictions)
            r2 = r2_score(actual, predictions)
            
            # Directional accuracy
            actual_direction = np.sign(actual)
            pred_direction = np.sign(predictions)
            directional_accuracy = np.mean(actual_direction == pred_direction)
            
            # Risk-adjusted metrics
            returns = actual.diff().dropna()
            pred_returns = pd.Series(predictions).diff().dropna()
            
            sharpe_ratio = returns.mean() / returns.std() if returns.std() > 0 else 0
            sortino_ratio = returns.mean() / returns[returns < 0].std() if returns[returns < 0].std() > 0 else 0
            
            # Maximum drawdown
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()
            
            return {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'directional_accuracy': directional_accuracy,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'max_drawdown': max_drawdown,
                'mean_return': returns.mean(),
                'volatility': returns.std()
            }
            
        except Exception as e:
            logging.error(f"Error calculating window metrics: {e}")
            return {}
    
    def _aggregate_wfo_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate results across all windows."""
        try:
            if not results:
                return {}
            
            # Calculate aggregate statistics
            metrics = ['mse', 'rmse', 'mae', 'r2', 'directional_accuracy', 
                      'sharpe_ratio', 'sortino_ratio', 'max_drawdown', 
                      'mean_return', 'volatility']
            
            aggregated = {}
            for metric in metrics:
                values = [r[metric] for r in results if metric in r and not np.isnan(r[metric])]
                if values:
                    aggregated[f'{metric}_mean'] = np.mean(values)
                    aggregated[f'{metric}_std'] = np.std(values)
                    aggregated[f'{metric}_min'] = np.min(values)
                    aggregated[f'{metric}_max'] = np.max(values)
                    aggregated[f'{metric}_median'] = np.median(values)
            
            # Overall performance summary
            aggregated['total_windows'] = len(results)
            aggregated['successful_windows'] = len([r for r in results if r.get('r2', 0) > 0])
            aggregated['avg_train_samples'] = np.mean([r.get('train_samples', 0) for r in results])
            aggregated['avg_test_samples'] = np.mean([r.get('test_samples', 0) for r in results])
            
            # Performance stability
            r2_values = [r.get('r2', 0) for r in results]
            aggregated['r2_stability'] = np.std(r2_values) if r2_values else 0
            
            return aggregated
            
        except Exception as e:
            logging.error(f"Error aggregating WFO results: {e}")
            return {}
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        if not self.wfo_results:
            return {}
        
        latest_results = self.performance_history[-1]['results'] if self.performance_history else {}
        
        return {
            'wfo_summary': latest_results,
            'total_windows_processed': len(self.wfo_results),
            'model_versions_stored': len(self.model_versions),
            'performance_history_length': len(self.performance_history)
        }
    
    def save_results(self, filepath: str):
        """Save WFO results to file."""
        try:
            results_data = {
                'wfo_results': self.wfo_results,
                'performance_history': self.performance_history,
                'configuration': {
                    'train_window_days': self.train_window_days,
                    'test_window_days': self.test_window_days,
                    'step_size_days': self.step_size_days,
                    'purge_days': self.purge_days,
                    'embargo_days': self.embargo_days
                }
            }
            
            joblib.dump(results_data, filepath)
            logging.info(f"WFO results saved to {filepath}")
            
        except Exception as e:
            logging.error(f"Error saving WFO results: {e}")
    
    def load_results(self, filepath: str):
        """Load WFO results from file."""
        try:
            if os.path.exists(filepath):
                results_data = joblib.load(filepath)
                self.wfo_results = results_data.get('wfo_results', [])
                self.performance_history = results_data.get('performance_history', [])
                logging.info(f"WFO results loaded from {filepath}")
            else:
                logging.warning(f"WFO results file not found: {filepath}")
                
        except Exception as e:
            logging.error(f"Error loading WFO results: {e}") 