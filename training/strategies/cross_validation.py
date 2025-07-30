"""
Cross-Validation Strategies for Time Series Data
Part of Project Hyperion - Ultimate Autonomous Trading Bot
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional, Generator
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class TimeSeriesCrossValidation:
    """
    Advanced Cross-Validation Strategies for Time Series Data
    
    Features:
    - Purged Group Time Series Split
    - Combinatorial Purged Cross-Validation
    - Expanding Window Cross-Validation
    - Rolling Window Cross-Validation
    - Blocked Time Series Split
    - Statistical Significance Testing
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.n_splits = config.get('n_splits', 5)
        self.test_size = config.get('test_size', 0.2)
        self.purge_period = config.get('purge_period', 0.1)  # Fraction of data to purge
        self.embargo_period = config.get('embargo_period', 0.05)  # Fraction of data for embargo
        self.min_train_size = config.get('min_train_size', 0.3)  # Minimum training size
        self.expanding_window = config.get('expanding_window', True)
        self.block_size = config.get('block_size', 0.1)  # Size of blocks for blocked CV
        
        # Results storage
        self.cv_results = []
        self.model_performances = {}
        self.statistical_tests = {}
        
        logger.info("Time Series Cross-Validation initialized")

    def purged_group_time_series_split(self, data: pd.DataFrame, 
                                     groups: Optional[np.ndarray] = None) -> List[Dict[str, Any]]:
        """Purged Group Time Series Split with proper purging and embargo periods"""
        splits = []
        n_samples = len(data)
        
        # Calculate sizes
        test_size_samples = int(n_samples * self.test_size)
        purge_size_samples = int(n_samples * self.purge_period)
        embargo_size_samples = int(n_samples * self.embargo_period)
        min_train_size_samples = int(n_samples * self.min_train_size)
        
        if groups is None:
            # Create groups based on time periods
            group_size = max(1, n_samples // 20)  # 20 groups by default
            groups = np.arange(n_samples) // group_size
        
        unique_groups = np.unique(groups)
        
        for i in range(self.n_splits):
            if self.expanding_window:
                # Expanding window approach
                split_point = min_train_size_samples + i * test_size_samples
                
                if split_point + test_size_samples > n_samples:
                    break
                
                # Training data (expanding)
                train_start = 0
                train_end = split_point
                
                # Test data
                test_start = split_point
                test_end = min(test_start + test_size_samples, n_samples)
                
            else:
                # Rolling window approach
                window_size = int(n_samples * (1 - self.test_size))
                train_start = i * test_size_samples
                train_end = train_start + window_size
                test_start = train_end
                test_end = min(test_start + test_size_samples, n_samples)
                
                if test_end > n_samples:
                    break
            
            # Apply purging and embargo
            purge_start = train_end
            purge_end = min(purge_start + purge_size_samples, n_samples)
            
            embargo_start = test_end
            embargo_end = min(embargo_start + embargo_size_samples, n_samples)
            
            # Get group indices
            train_groups = groups[train_start:train_end]
            test_groups = groups[test_start:test_end]
            purge_groups = groups[purge_start:purge_end]
            embargo_groups = groups[embargo_start:embargo_end]
            
            # Remove overlapping groups
            train_groups = np.setdiff1d(train_groups, purge_groups)
            train_groups = np.setdiff1d(train_groups, embargo_groups)
            test_groups = np.setdiff1d(test_groups, purge_groups)
            test_groups = np.setdiff1d(test_groups, embargo_groups)
            
            # Get indices for each group
            train_indices = np.where(np.isin(groups, train_groups))[0]
            test_indices = np.where(np.isin(groups, test_groups))[0]
            
            # Ensure temporal order
            train_indices = train_indices[train_indices < test_indices.min()] if len(test_indices) > 0 else train_indices
            test_indices = test_indices[test_indices > train_indices.max()] if len(train_indices) > 0 else test_indices
            
            split = {
                'fold': i + 1,
                'train_indices': train_indices,
                'test_indices': test_indices,
                'train_groups': train_groups,
                'test_groups': test_groups,
                'purge_groups': purge_groups,
                'embargo_groups': embargo_groups,
                'train_size': len(train_indices),
                'test_size': len(test_indices)
            }
            
            splits.append(split)
        
        logger.info(f"Created {len(splits)} purged group time series splits")
        return splits

    def combinatorial_purged_cv(self, data: pd.DataFrame, 
                              n_combinations: int = 10) -> List[List[Dict[str, Any]]]:
        """Combinatorial Purged Cross-Validation"""
        all_splits = []
        
        for combination in range(n_combinations):
            # Vary parameters slightly for each combination
            test_size_var = self.test_size * (0.8 + 0.4 * np.random.random())
            purge_period_var = self.purge_period * (0.8 + 0.4 * np.random.random())
            embargo_period_var = self.embargo_period * (0.8 + 0.4 * np.random.random())
            
            # Create splits with varied parameters
            config_var = self.config.copy()
            config_var['test_size'] = test_size_var
            config_var['purge_period'] = purge_period_var
            config_var['embargo_period'] = embargo_period_var
            
            cv_var = TimeSeriesCrossValidation(config_var)
            splits = cv_var.purged_group_time_series_split(data)
            
            all_splits.append(splits)
        
        logger.info(f"Created {len(all_splits)} combinatorial CV combinations")
        return all_splits

    def blocked_time_series_split(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Blocked Time Series Split to prevent data leakage"""
        splits = []
        n_samples = len(data)
        
        # Calculate block size
        block_size_samples = int(n_samples * self.block_size)
        n_blocks = n_samples // block_size_samples
        
        for i in range(self.n_splits):
            if i >= n_blocks - 1:
                break
            
            # Training blocks
            train_blocks = list(range(i))
            train_indices = []
            for block in train_blocks:
                start_idx = block * block_size_samples
                end_idx = (block + 1) * block_size_samples
                train_indices.extend(range(start_idx, end_idx))
            
            # Test block
            test_start = i * block_size_samples
            test_end = (i + 1) * block_size_samples
            test_indices = list(range(test_start, test_end))
            
            # Validation block (next block)
            if i + 1 < n_blocks:
                val_start = (i + 1) * block_size_samples
                val_end = (i + 2) * block_size_samples
                val_indices = list(range(val_start, val_end))
            else:
                val_indices = []
            
            split = {
                'fold': i + 1,
                'train_indices': np.array(train_indices),
                'test_indices': np.array(test_indices),
                'val_indices': np.array(val_indices),
                'train_size': len(train_indices),
                'test_size': len(test_indices),
                'val_size': len(val_indices)
            }
            
            splits.append(split)
        
        logger.info(f"Created {len(splits)} blocked time series splits")
        return splits

    def expanding_window_cv(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Expanding Window Cross-Validation"""
        splits = []
        n_samples = len(data)
        
        # Calculate sizes
        test_size_samples = int(n_samples * self.test_size)
        min_train_size_samples = int(n_samples * self.min_train_size)
        
        for i in range(self.n_splits):
            # Calculate split points
            split_point = min_train_size_samples + i * test_size_samples
            
            if split_point + test_size_samples > n_samples:
                break
            
            # Training data (expanding)
            train_indices = np.arange(split_point)
            test_indices = np.arange(split_point, split_point + test_size_samples)
            
            split = {
                'fold': i + 1,
                'train_indices': train_indices,
                'test_indices': test_indices,
                'train_size': len(train_indices),
                'test_size': len(test_indices)
            }
            
            splits.append(split)
        
        logger.info(f"Created {len(splits)} expanding window splits")
        return splits

    def rolling_window_cv(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Rolling Window Cross-Validation"""
        splits = []
        n_samples = len(data)
        
        # Calculate sizes
        test_size_samples = int(n_samples * self.test_size)
        window_size = int(n_samples * (1 - self.test_size))
        
        for i in range(self.n_splits):
            # Calculate split points
            train_start = i * test_size_samples
            train_end = train_start + window_size
            test_start = train_end
            test_end = test_start + test_size_samples
            
            if test_end > n_samples:
                break
            
            train_indices = np.arange(train_start, train_end)
            test_indices = np.arange(test_start, test_end)
            
            split = {
                'fold': i + 1,
                'train_indices': train_indices,
                'test_indices': test_indices,
                'train_size': len(train_indices),
                'test_size': len(test_indices)
            }
            
            splits.append(split)
        
        logger.info(f"Created {len(splits)} rolling window splits")
        return splits

    def evaluate_model_cv(self, model: Any, data: pd.DataFrame, target_column: str,
                         feature_columns: Optional[List[str]] = None,
                         cv_method: str = 'purged_group',
                         **cv_params) -> Dict[str, Any]:
        """Evaluate model using cross-validation"""
        
        if feature_columns is None:
            feature_columns = [col for col in data.columns if col != target_column]
        
        # Prepare data
        X = data[feature_columns].values
        y = data[target_column].values
        
        # Create splits based on method
        if cv_method == 'purged_group':
            splits = self.purged_group_time_series_split(data, **cv_params)
        elif cv_method == 'blocked':
            splits = self.blocked_time_series_split(data)
        elif cv_method == 'expanding':
            splits = self.expanding_window_cv(data)
        elif cv_method == 'rolling':
            splits = self.rolling_window_cv(data)
        else:
            raise ValueError(f"Unknown CV method: {cv_method}")
        
        # Evaluate on each fold
        fold_results = []
        all_predictions = []
        all_targets = []
        
        for split in splits:
            # Get train/test indices
            train_indices = split['train_indices']
            test_indices = split['test_indices']
            
            if len(train_indices) == 0 or len(test_indices) == 0:
                logger.warning(f"Skipping fold {split['fold']} due to empty train/test sets")
                continue
            
            # Split data
            X_train, X_test = X[train_indices], X[test_indices]
            y_train, y_test = y[train_indices], y[test_indices]
            
            # Train model
            if hasattr(model, 'fit'):
                model.fit(X_train, y_train)
            
            # Make predictions
            if hasattr(model, 'predict'):
                y_pred = model.predict(X_test)
            else:
                y_pred = model.forecast(steps=len(y_test))
            
            # Calculate metrics
            fold_metrics = {
                'fold': split['fold'],
                'mse': mean_squared_error(y_test, y_pred),
                'mae': mean_absolute_error(y_test, y_pred),
                'r2': r2_score(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'train_size': len(train_indices),
                'test_size': len(test_indices)
            }
            
            # Calculate directional accuracy
            if len(y_test) > 1:
                directional_accuracy = np.mean(np.sign(np.diff(y_test)) == np.sign(np.diff(y_pred)))
                fold_metrics['directional_accuracy'] = directional_accuracy
            
            fold_results.append(fold_metrics)
            all_predictions.extend(y_pred)
            all_targets.extend(y_test)
        
        # Aggregate results
        cv_summary = self.aggregate_cv_results(fold_results)
        
        # Store results
        self.cv_results = fold_results
        self.model_performances = cv_summary
        
        return {
            'fold_results': fold_results,
            'cv_summary': cv_summary,
            'all_predictions': all_predictions,
            'all_targets': all_targets
        }

    def aggregate_cv_results(self, fold_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate cross-validation results"""
        if not fold_results:
            return {}
        
        # Convert to DataFrame for easier analysis
        results_df = pd.DataFrame(fold_results)
        
        # Calculate summary statistics
        summary = {}
        for metric in ['mse', 'mae', 'r2', 'rmse', 'directional_accuracy']:
            if metric in results_df.columns:
                summary[f'{metric}_mean'] = results_df[metric].mean()
                summary[f'{metric}_std'] = results_df[metric].std()
                summary[f'{metric}_min'] = results_df[metric].min()
                summary[f'{metric}_max'] = results_df[metric].max()
        
        # Add overall statistics
        summary['num_folds'] = len(fold_results)
        summary['total_train_samples'] = results_df['train_size'].sum()
        summary['total_test_samples'] = results_df['test_size'].sum()
        
        return summary

    def perform_statistical_tests(self, cv_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform statistical significance tests on CV results"""
        from scipy import stats
        
        if not cv_results:
            return {}
        
        results_df = pd.DataFrame(cv_results)
        
        # Test for normality
        normality_tests = {}
        for metric in ['mse', 'mae', 'r2', 'rmse']:
            if metric in results_df.columns:
                statistic, p_value = stats.shapiro(results_df[metric])
                normality_tests[metric] = {
                    'statistic': statistic,
                    'p_value': p_value,
                    'is_normal': p_value > 0.05
                }
        
        # Confidence intervals
        confidence_intervals = {}
        for metric in ['mse', 'mae', 'r2', 'rmse']:
            if metric in results_df.columns:
                mean_val = results_df[metric].mean()
                std_val = results_df[metric].std()
                n_samples = len(results_df)
                
                # 95% confidence interval
                ci_lower, ci_upper = stats.t.interval(0.95, n_samples - 1, 
                                                     loc=mean_val, scale=std_val / np.sqrt(n_samples))
                
                confidence_intervals[metric] = {
                    'mean': mean_val,
                    'std': std_val,
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper
                }
        
        # Trend analysis
        trend_tests = {}
        for metric in ['mse', 'mae', 'r2', 'rmse']:
            if metric in results_df.columns:
                # Test for trend across folds
                x = np.arange(len(results_df))
                y = results_df[metric].values
                
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                
                trend_tests[metric] = {
                    'slope': slope,
                    'intercept': intercept,
                    'r_value': r_value,
                    'p_value': p_value,
                    'std_err': std_err,
                    'has_trend': p_value < 0.05
                }
        
        statistical_tests = {
            'normality_tests': normality_tests,
            'confidence_intervals': confidence_intervals,
            'trend_tests': trend_tests
        }
        
        self.statistical_tests = statistical_tests
        return statistical_tests

    def compare_models_cv(self, models: Dict[str, Any], data: pd.DataFrame, 
                         target_column: str, feature_columns: Optional[List[str]] = None,
                         cv_method: str = 'purged_group') -> Dict[str, Any]:
        """Compare multiple models using cross-validation"""
        
        comparison_results = {}
        
        for model_name, model in models.items():
            logger.info(f"Evaluating {model_name} with cross-validation")
            
            result = self.evaluate_model_cv(
                model, data, target_column, feature_columns, cv_method
            )
            
            comparison_results[model_name] = result['cv_summary']
        
        # Perform pairwise statistical tests
        pairwise_tests = self.perform_pairwise_tests(comparison_results)
        
        return {
            'model_comparisons': comparison_results,
            'pairwise_tests': pairwise_tests
        }

    def perform_pairwise_tests(self, comparison_results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform pairwise statistical tests between models"""
        from scipy import stats
        
        model_names = list(comparison_results.keys())
        pairwise_tests = {}
        
        for i, model1 in enumerate(model_names):
            for model2 in model_names[i+1:]:
                # Get R² scores for comparison
                r2_1 = [result['r2_mean'] for result in self.cv_results 
                       if result.get('model_name') == model1]
                r2_2 = [result['r2_mean'] for result in self.cv_results 
                       if result.get('model_name') == model2]
                
                if r2_1 and r2_2:
                    # Paired t-test
                    t_stat, p_value = stats.ttest_rel(r2_1, r2_2)
                    
                    test_key = f"{model1}_vs_{model2}"
                    pairwise_tests[test_key] = {
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'significant': p_value < 0.05,
                        'model1_mean_r2': np.mean(r2_1),
                        'model2_mean_r2': np.mean(r2_2)
                    }
        
        return pairwise_tests

    def plot_cv_results(self, save_path: Optional[str] = None):
        """Plot cross-validation results"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            if not self.cv_results:
                logger.warning("No CV results to plot")
                return
            
            results_df = pd.DataFrame(self.cv_results)
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # R² across folds
            axes[0, 0].plot(results_df['fold'], results_df['r2'], marker='o')
            axes[0, 0].set_title('R² Across Folds')
            axes[0, 0].set_xlabel('Fold')
            axes[0, 0].set_ylabel('R²')
            axes[0, 0].grid(True)
            
            # MSE across folds
            axes[0, 1].plot(results_df['fold'], results_df['mse'], marker='o')
            axes[0, 1].set_title('MSE Across Folds')
            axes[0, 1].set_xlabel('Fold')
            axes[0, 1].set_ylabel('MSE')
            axes[0, 1].grid(True)
            
            # Box plot of metrics
            metrics_to_plot = ['r2', 'mse', 'mae', 'rmse']
            plot_data = results_df[metrics_to_plot].melt()
            sns.boxplot(data=plot_data, x='variable', y='value', ax=axes[1, 0])
            axes[1, 0].set_title('Distribution of Metrics')
            axes[1, 0].set_xlabel('Metric')
            axes[1, 0].set_ylabel('Value')
            
            # Sample sizes
            axes[1, 1].bar(results_df['fold'], results_df['test_size'])
            axes[1, 1].set_title('Test Set Sizes')
            axes[1, 1].set_xlabel('Fold')
            axes[1, 1].set_ylabel('Number of Samples')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"CV results plot saved to {save_path}")
            
            plt.show()
            
        except ImportError:
            logger.warning("Matplotlib/Seaborn not available for plotting")

    def save_cv_results(self, filepath: str):
        """Save cross-validation results"""
        import joblib
        
        results = {
            'cv_results': self.cv_results,
            'model_performances': self.model_performances,
            'statistical_tests': self.statistical_tests,
            'config': self.config
        }
        
        joblib.dump(results, filepath)
        logger.info(f"CV results saved to {filepath}")

    def load_cv_results(self, filepath: str):
        """Load cross-validation results"""
        import joblib
        
        results = joblib.load(filepath)
        
        self.cv_results = results['cv_results']
        self.model_performances = results['model_performances']
        self.statistical_tests = results['statistical_tests']
        
        logger.info(f"CV results loaded from {filepath}")

    def get_cv_summary(self) -> pd.DataFrame:
        """Get cross-validation summary as DataFrame"""
        if not self.cv_results:
            return pd.DataFrame()
        
        return pd.DataFrame(self.cv_results) 