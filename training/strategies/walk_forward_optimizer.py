"""
Walk-Forward Optimizer for Robust Time Series Model Training
Part of Project Hyperion - Ultimate Autonomous Trading Bot
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional, Callable
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class WalkForwardOptimizer:
    """
    Advanced Walk-Forward Optimization for Time Series Models
    
    Features:
    - Purge and Embargo Periods
    - Expanding Window Strategy
    - Multiple Evaluation Metrics
    - Model Performance Tracking
    - Out-of-Sample Validation
    - Statistical Significance Testing
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.n_splits = config.get('n_splits', 5)
        self.test_size = config.get('test_size', 0.2)
        self.purge_period = config.get('purge_period', 0.1)  # Fraction of data to purge
        self.embargo_period = config.get('embargo_period', 0.05)  # Fraction of data for embargo
        self.expanding_window = config.get('expanding_window', True)
        self.min_train_size = config.get('min_train_size', 0.3)  # Minimum training size
        self.evaluation_metrics = config.get('evaluation_metrics', ['mse', 'mae', 'r2', 'directional_accuracy'])
        
        # Results storage
        self.fold_results = []
        self.model_performances = {}
        self.feature_importance_history = []
        self.parameter_history = []
        
        # Performance tracking
        self.best_models = {}
        self.performance_summary = {}
        
        logger.info("Walk-Forward Optimizer initialized")

    def create_walk_forward_splits(self, data: pd.DataFrame, 
                                 target_column: str = 'target') -> List[Dict[str, Any]]:
        """Create walk-forward splits with purge and embargo periods"""
        splits = []
        n_samples = len(data)
        
        # Calculate sizes
        test_size_samples = int(n_samples * self.test_size)
        purge_size_samples = int(n_samples * self.purge_period)
        embargo_size_samples = int(n_samples * self.embargo_period)
        min_train_size_samples = int(n_samples * self.min_train_size)
        
        if self.expanding_window:
            # Expanding window approach
            for i in range(self.n_splits):
                # Calculate split points
                split_point = min_train_size_samples + i * test_size_samples
                
                if split_point + test_size_samples > n_samples:
                    break
                
                # Training data (expanding)
                train_start = 0
                train_end = split_point
                
                # Purge period
                purge_start = train_end
                purge_end = min(purge_start + purge_size_samples, n_samples)
                
                # Test data
                test_start = purge_end
                test_end = min(test_start + test_size_samples, n_samples)
                
                # Embargo period
                embargo_start = test_end
                embargo_end = min(embargo_start + embargo_size_samples, n_samples)
                
                # Create split
                split = {
                    'fold': i + 1,
                        'train_start': train_start,
                        'train_end': train_end,
                    'purge_start': purge_start,
                    'purge_end': purge_end,
                        'test_start': test_start,
                        'test_end': test_end,
                        'embargo_start': embargo_start,
                        'embargo_end': embargo_end,
                    'train_data': data.iloc[train_start:train_end],
                    'test_data': data.iloc[test_start:test_end],
                    'train_size': train_end - train_start,
                    'test_size': test_end - test_start
                }
                
                splits.append(split)
        else:
            # Rolling window approach
            window_size = int(n_samples * (1 - self.test_size))
            
            for i in range(self.n_splits):
                # Calculate split points
                train_start = i * test_size_samples
                train_end = train_start + window_size
                
                if train_end + test_size_samples > n_samples:
                    break
                
                # Purge period
                purge_start = train_end
                purge_end = min(purge_start + purge_size_samples, n_samples)
                
                # Test data
                test_start = purge_end
                test_end = min(test_start + test_size_samples, n_samples)
                
                # Embargo period
                embargo_start = test_end
                embargo_end = min(embargo_start + embargo_size_samples, n_samples)
                
                # Create split
                split = {
                    'fold': i + 1,
                    'train_start': train_start,
                    'train_end': train_end,
                    'purge_start': purge_start,
                    'purge_end': purge_end,
                    'test_start': test_start,
                    'test_end': test_end,
                    'embargo_start': embargo_start,
                    'embargo_end': embargo_end,
                    'train_data': data.iloc[train_start:train_end],
                    'test_data': data.iloc[test_start:test_end],
                    'train_size': train_end - train_start,
                    'test_size': test_end - test_start
                }
                
                splits.append(split)
        
        logger.info(f"Created {len(splits)} walk-forward splits")
        return splits

    def evaluate_model(self, model: Any, X_train: np.ndarray, y_train: np.ndarray,
                      X_test: np.ndarray, y_test: np.ndarray, 
                      model_name: str, fold: int) -> Dict[str, float]:
        """Evaluate model performance on a fold"""
        try:
            # Handle different model types
            if model_name == 'neural_network':
                # For neural networks, use the pre-trained model directly
                if hasattr(model, 'predict'):
                    y_pred = model.predict(X_test, verbose=0).flatten()
                else:
                    y_pred = np.zeros(len(y_test))
            else:
                # For other models, train and predict
                if hasattr(model, 'fit'):
                    model.fit(X_train, y_train)
                
                if hasattr(model, 'predict'):
                    y_pred = model.predict(X_test)
                else:
                    y_pred = model.forecast(steps=len(y_test))
        except Exception as e:
            logger.warning(f"Error evaluating {model_name} on fold {fold}: {e}")
            y_pred = np.zeros(len(y_test))
            
            # Calculate metrics
        metrics = {}
        
        if 'mse' in self.evaluation_metrics:
            metrics['mse'] = mean_squared_error(y_test, y_pred)
        
        if 'mae' in self.evaluation_metrics:
            metrics['mae'] = mean_absolute_error(y_test, y_pred)
        
        if 'r2' in self.evaluation_metrics:
            metrics['r2'] = r2_score(y_test, y_pred)
        
        if 'rmse' in self.evaluation_metrics:
            metrics['rmse'] = np.sqrt(metrics.get('mse', mean_squared_error(y_test, y_pred)))
        
        if 'directional_accuracy' in self.evaluation_metrics:
            directional_accuracy = np.mean(np.sign(np.diff(y_test)) == np.sign(np.diff(y_pred)))
            metrics['directional_accuracy'] = directional_accuracy
        
        # Add fold information
        metrics['fold'] = fold
        metrics['model_name'] = model_name
        metrics['train_size'] = len(X_train)
        metrics['test_size'] = len(X_test)
        
        return metrics
    
    def optimize_hyperparameters(self, model_class: Callable, 
                               param_grid: Dict[str, List[Any]],
                               X_train: np.ndarray, y_train: np.ndarray,
                               X_val: np.ndarray, y_val: np.ndarray,
                               optimization_method: str = 'grid') -> Dict[str, Any]:
        """Optimize hyperparameters for a model"""
        from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
        
        # Create base model
        base_model = model_class()
        
        if optimization_method == 'grid':
            optimizer = GridSearchCV(
                base_model, param_grid, cv=3, scoring='neg_mean_squared_error',
                n_jobs=-1, verbose=0
            )
        else:
            optimizer = RandomizedSearchCV(
                base_model, param_grid, cv=3, scoring='neg_mean_squared_error',
                n_iter=20, n_jobs=-1, verbose=0
            )
        
        # Optimize
        optimizer.fit(X_train, y_train)
        
        # Evaluate on validation set
        best_model = optimizer.best_estimator_
        val_score = -optimizer.score(X_val, y_val)
        
        return {
            'best_params': optimizer.best_params_,
            'best_score': optimizer.best_score_,
            'val_score': val_score,
            'best_model': best_model
        }

    def run_walk_forward_optimization(self, data: pd.DataFrame, 
                                    models: Dict[str, Any],
                                    target_column: str = 'target',
                                    feature_columns: Optional[List[str]] = None,
                                    optimize_hyperparams: bool = False,
                                    param_grids: Optional[Dict[str, Dict[str, List[Any]]]] = None) -> Dict[str, Any]:
        """Run complete walk-forward optimization"""
        
        # Prepare data
        if feature_columns is None:
            feature_columns = [col for col in data.columns if col != target_column]
        
        # Create splits
        splits = self.create_walk_forward_splits(data, target_column)
        
        # Results storage
        all_results = []
        model_performances = {model_name: [] for model_name in models.keys()}
        
        logger.info(f"Starting walk-forward optimization with {len(splits)} folds")
        
        for split in splits:
            fold = split['fold']
            logger.info(f"Processing fold {fold}/{len(splits)}")
            
            # Prepare data for this fold
            train_data = split['train_data']
            test_data = split['test_data']
            
            X_train = train_data[feature_columns].values
            y_train = train_data[target_column].values
            X_test = test_data[feature_columns].values
            y_test = test_data[target_column].values
            
            fold_results = []
            
            for model_name, model in models.items():
                logger.info(f"  Training {model_name} on fold {fold}")
                
                # Hyperparameter optimization if requested
                if optimize_hyperparams and param_grids and model_name in param_grids:
                    # Use a portion of training data for validation
                    val_split = int(0.8 * len(X_train))
                    X_train_opt = X_train[:val_split]
                    y_train_opt = y_train[:val_split]
                    X_val_opt = X_train[val_split:]
                    y_val_opt = y_train[val_split:]
                    
                    optimization_result = self.optimize_hyperparameters(
                        type(model), param_grids[model_name],
                        X_train_opt, y_train_opt, X_val_opt, y_val_opt
                    )
                    
                    # Use best model for evaluation
                    model = optimization_result['best_model']
                    
                    # Store optimization results
                    self.parameter_history.append({
                        'fold': fold,
                        'model_name': model_name,
                        'best_params': optimization_result['best_params'],
                        'val_score': optimization_result['val_score']
                    })
                
                # Evaluate model
                metrics = self.evaluate_model(
                    model, X_train, y_train, X_test, y_test, model_name, fold
                )
                
                fold_results.append(metrics)
                model_performances[model_name].append(metrics)
                
                # Store feature importance if available
                if hasattr(model, 'feature_importances_'):
                    self.feature_importance_history.append({
                        'fold': fold,
                        'model_name': model_name,
                        'feature_importance': model.feature_importances_.tolist(),
                        'feature_names': feature_columns
                    })
            
            # Store fold results
            all_results.extend(fold_results)
            self.fold_results.append({
                'fold': fold,
                'results': fold_results,
                'train_size': len(X_train),
                'test_size': len(X_test)
            })
        
        # Analyze results
        self.analyze_results(all_results, model_performances)
        
        return {
            'all_results': all_results,
            'model_performances': model_performances,
            'performance_summary': self.performance_summary,
            'best_models': self.best_models
        }

    def analyze_results(self, all_results: List[Dict[str, float]], 
                       model_performances: Dict[str, List[Dict[str, float]]]):
        """Analyze walk-forward optimization results"""
        
        # Convert to DataFrame for easier analysis
        results_df = pd.DataFrame(all_results)
        
        # Performance summary by model
        for model_name in model_performances.keys():
            model_results = results_df[results_df['model_name'] == model_name]
            
            if len(model_results) > 0:
                summary = {}
                for metric in self.evaluation_metrics:
                    if metric in model_results.columns:
                        summary[f'{metric}_mean'] = model_results[metric].mean()
                        summary[f'{metric}_std'] = model_results[metric].std()
                        summary[f'{metric}_min'] = model_results[metric].min()
                        summary[f'{metric}_max'] = model_results[metric].max()
                
                self.performance_summary[model_name] = summary
        
        # Find best models for each metric
        for metric in self.evaluation_metrics:
            if metric in results_df.columns:
                if metric in ['mse', 'mae', 'rmse']:
                    # Lower is better
                    best_model = results_df.groupby('model_name')[metric].mean().idxmin()
                else:
                    # Higher is better
                    best_model = results_df.groupby('model_name')[metric].mean().idxmax()
                
                self.best_models[metric] = best_model
        
        # Statistical significance testing
        self.perform_statistical_tests(results_df)
        
        logger.info("Results analysis completed")

    def perform_statistical_tests(self, results_df: pd.DataFrame):
        """Perform statistical significance tests between models"""
        from scipy import stats
        
        model_names = results_df['model_name'].unique()
        
        if len(model_names) < 2:
            return
        
        # Pairwise t-tests for each metric
        significance_tests = {}
        
        for metric in self.evaluation_metrics:
            if metric not in results_df.columns:
                continue
            
            metric_tests = {}
            
            for i, model1 in enumerate(model_names):
                for model2 in model_names[i+1:]:
                    model1_scores = results_df[results_df['model_name'] == model1][metric].values
                    model2_scores = results_df[results_df['model_name'] == model2][metric].values
                    
                    # Paired t-test
                    t_stat, p_value = stats.ttest_rel(model1_scores, model2_scores)
                    
                    test_key = f"{model1}_vs_{model2}"
                    metric_tests[test_key] = {
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'significant': p_value < 0.05
                    }
            
            significance_tests[metric] = metric_tests
        
        self.performance_summary['statistical_tests'] = significance_tests

    def get_performance_summary(self) -> pd.DataFrame:
        """Get performance summary as DataFrame"""
        if not self.performance_summary:
            return pd.DataFrame()
        
        summary_data = []
        for model_name, metrics in self.performance_summary.items():
            if model_name == 'statistical_tests':
                continue
            
            row = {'model': model_name}
            row.update(metrics)
            summary_data.append(row)
        
        return pd.DataFrame(summary_data)

    def get_feature_importance_summary(self) -> pd.DataFrame:
        """Get feature importance summary across all folds"""
        if not self.feature_importance_history:
            return pd.DataFrame()
        
        # Aggregate feature importance across folds
        feature_importance_df = pd.DataFrame(self.feature_importance_history)
        
        # Calculate mean importance for each feature across folds
        importance_summary = {}
        
        for model_name in feature_importance_df['model_name'].unique():
            model_data = feature_importance_df[feature_importance_df['model_name'] == model_name]
            
            # Get feature names (should be the same for all folds)
            feature_names = model_data.iloc[0]['feature_names']
            
            # Calculate mean importance
            mean_importance = np.mean([fold_data['feature_importance'] 
                                     for fold_data in model_data['feature_importance']], axis=0)
            
            importance_summary[model_name] = dict(zip(feature_names, mean_importance))
        
        return pd.DataFrame(importance_summary).T

    def plot_performance_trends(self, save_path: Optional[str] = None):
        """Plot performance trends across folds"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            if not self.fold_results:
                logger.warning("No results to plot")
                return
            
            # Prepare data for plotting
            plot_data = []
            for fold_result in self.fold_results:
                fold = fold_result['fold']
                for result in fold_result['results']:
                    plot_data.append({
                        'fold': fold,
                        'model': result['model_name'],
                        'mse': result.get('mse', 0),
                        'r2': result.get('r2', 0),
                        'directional_accuracy': result.get('directional_accuracy', 0)
                    })
            
            plot_df = pd.DataFrame(plot_data)
            
            # Create plots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # MSE trend
            for model in plot_df['model'].unique():
                model_data = plot_df[plot_df['model'] == model]
                axes[0, 0].plot(model_data['fold'], model_data['mse'], 
                               marker='o', label=model)
            axes[0, 0].set_title('MSE Trend Across Folds')
            axes[0, 0].set_xlabel('Fold')
            axes[0, 0].set_ylabel('MSE')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
            
            # R² trend
            for model in plot_df['model'].unique():
                model_data = plot_df[plot_df['model'] == model]
                axes[0, 1].plot(model_data['fold'], model_data['r2'], 
                               marker='o', label=model)
            axes[0, 1].set_title('R² Trend Across Folds')
            axes[0, 1].set_xlabel('Fold')
            axes[0, 1].set_ylabel('R²')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
            
            # Directional accuracy trend
            for model in plot_df['model'].unique():
                model_data = plot_df[plot_df['model'] == model]
                axes[1, 0].plot(model_data['fold'], model_data['directional_accuracy'], 
                               marker='o', label=model)
            axes[1, 0].set_title('Directional Accuracy Trend Across Folds')
            axes[1, 0].set_xlabel('Fold')
            axes[1, 0].set_ylabel('Directional Accuracy')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
            
            # Box plot of MSE distribution
            sns.boxplot(data=plot_df, x='model', y='mse', ax=axes[1, 1])
            axes[1, 1].set_title('MSE Distribution by Model')
            axes[1, 1].set_xlabel('Model')
            axes[1, 1].set_ylabel('MSE')
            axes[1, 1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Performance plots saved to {save_path}")
            
            plt.show()
            
        except ImportError:
            logger.warning("Matplotlib/Seaborn not available for plotting")

    def save_results(self, filepath: str):
        """Save walk-forward optimization results"""
        import joblib
        
        results = {
            'fold_results': self.fold_results,
            'model_performances': self.model_performances,
            'performance_summary': self.performance_summary,
            'best_models': self.best_models,
            'feature_importance_history': self.feature_importance_history,
            'parameter_history': self.parameter_history,
            'config': self.config
        }
        
        joblib.dump(results, filepath)
        logger.info(f"Walk-forward optimization results saved to {filepath}")

    def load_results(self, filepath: str):
        """Load walk-forward optimization results"""
        import joblib
        
        results = joblib.load(filepath)
        
        self.fold_results = results['fold_results']
        self.model_performances = results['model_performances']
        self.performance_summary = results['performance_summary']
        self.best_models = results['best_models']
        self.feature_importance_history = results['feature_importance_history']
        self.parameter_history = results['parameter_history']
        
        logger.info(f"Walk-forward optimization results loaded from {filepath}") 