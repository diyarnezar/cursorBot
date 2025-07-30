"""
Advanced Metrics for Project Hyperion
Comprehensive model evaluation and performance metrics
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, log_loss
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_advanced_metrics(features: pd.DataFrame, models: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate comprehensive advanced metrics for all models
    """
    logger = logging.getLogger(__name__)
    logger.info("📊 Calculating advanced metrics for all models")
    
    all_metrics = {}
    
    for model_name, model in models.items():
        try:
            model_metrics = _calculate_model_metrics(features, model, model_name)
            all_metrics[model_name] = model_metrics
            logger.info(f"📊 {model_name}: R² = {model_metrics['r2_score']:.4f}")
        except Exception as e:
            logger.warning(f"📊 Failed to calculate metrics for {model_name}: {str(e)}")
            all_metrics[model_name] = {'error': str(e)}
    
    # Calculate ensemble metrics
    ensemble_metrics = _calculate_ensemble_metrics(features, models)
    all_metrics['ensemble'] = ensemble_metrics
    
    # Calculate overall system metrics
    system_metrics = _calculate_system_metrics(all_metrics)
    all_metrics['system'] = system_metrics
    
    logger.info("📊 Advanced metrics calculation completed")
    return all_metrics


def _calculate_model_metrics(features: pd.DataFrame, model, model_name: str) -> Dict[str, Any]:
    """Calculate comprehensive metrics for a single model"""
    try:
        # Prepare data
        X = features.drop('target', axis=1, errors='ignore')
        y = features['target']
        
        # Split data for evaluation
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Train model if not already trained
        if not hasattr(model, 'fitted_') or not model.fitted_:
            model.fit(X_train, y_train)
            model.fitted_ = True
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate regression metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Calculate additional metrics
        mape = _calculate_mape(y_test, y_pred)
        directional_accuracy = _calculate_directional_accuracy(y_test, y_pred)
        sharpe_ratio = _calculate_sharpe_ratio(y_test, y_pred)
        max_drawdown = _calculate_max_drawdown(y_test, y_pred)
        
        # Calculate feature importance if available
        feature_importance = _extract_feature_importance(model, X.columns)
        
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2_score': r2,
            'mape': mape,
            'directional_accuracy': directional_accuracy,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'feature_importance': feature_importance,
            'predictions': y_pred,
            'actual': y_test.values,
            'model_type': type(model).__name__,
            'timestamp': datetime.now()
        }
        
        return metrics
        
    except Exception as e:
        return {'error': str(e)}


def _calculate_ensemble_metrics(features: pd.DataFrame, models: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate ensemble-level metrics"""
    try:
        # Get predictions from all models
        all_predictions = []
        valid_models = []
        
        for model_name, model in models.items():
            try:
                X = features.drop('target', axis=1, errors='ignore')
                y = features['target']
                
                split_idx = int(len(X) * 0.8)
                X_test = X[split_idx:]
                y_test = y[split_idx:]
                
                if hasattr(model, 'predict'):
                    pred = model.predict(X_test)
                    all_predictions.append(pred)
                    valid_models.append(model_name)
            except:
                continue
        
        if not all_predictions:
            return {'error': 'No valid predictions available'}
        
        # Calculate ensemble prediction (simple average)
        ensemble_pred = np.mean(all_predictions, axis=0)
        y_test = features['target'].iloc[int(len(features) * 0.8):].values
        
        # Calculate ensemble metrics
        mse = mean_squared_error(y_test, ensemble_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, ensemble_pred)
        r2 = r2_score(y_test, ensemble_pred)
        mape = _calculate_mape(y_test, ensemble_pred)
        directional_accuracy = _calculate_directional_accuracy(y_test, ensemble_pred)
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2_score': r2,
            'mape': mape,
            'directional_accuracy': directional_accuracy,
            'ensemble_prediction': ensemble_pred,
            'actual': y_test,
            'model_count': len(valid_models),
            'models_used': valid_models,
            'timestamp': datetime.now()
        }
        
    except Exception as e:
        return {'error': str(e)}


def _calculate_system_metrics(all_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate overall system performance metrics"""
    try:
        # Extract R² scores from all models
        r2_scores = []
        model_names = []
        
        for model_name, metrics in all_metrics.items():
            if isinstance(metrics, dict) and 'r2_score' in metrics:
                r2_scores.append(metrics['r2_score'])
                model_names.append(model_name)
        
        if not r2_scores:
            return {'error': 'No valid metrics available'}
        
        # Calculate system statistics
        avg_r2 = np.mean(r2_scores)
        std_r2 = np.std(r2_scores)
        best_model = model_names[np.argmax(r2_scores)]
        worst_model = model_names[np.argmin(r2_scores)]
        
        # Calculate diversity metrics
        diversity_score = _calculate_model_diversity(all_metrics)
        
        # Calculate stability metrics
        stability_score = _calculate_stability_score(all_metrics)
        
        return {
            'average_r2': avg_r2,
            'std_r2': std_r2,
            'best_model': best_model,
            'worst_model': worst_model,
            'best_r2': max(r2_scores),
            'worst_r2': min(r2_scores),
            'model_count': len(r2_scores),
            'diversity_score': diversity_score,
            'stability_score': stability_score,
            'overall_performance': avg_r2,
            'timestamp': datetime.now()
        }
        
    except Exception as e:
        return {'error': str(e)}


def _calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Absolute Percentage Error"""
    try:
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    except:
        return np.nan


def _calculate_directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate directional accuracy"""
    try:
        # Calculate direction changes
        true_direction = np.diff(y_true) > 0
        pred_direction = np.diff(y_pred) > 0
        
        # Calculate accuracy
        accuracy = np.mean(true_direction == pred_direction)
        return accuracy
    except:
        return np.nan


def _calculate_sharpe_ratio(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Sharpe ratio of predictions"""
    try:
        # Calculate returns
        returns = np.diff(y_pred)
        
        if len(returns) == 0:
            return np.nan
        
        # Calculate Sharpe ratio
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return np.nan
        
        sharpe = mean_return / std_return
        return sharpe
    except:
        return np.nan


def _calculate_max_drawdown(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate maximum drawdown"""
    try:
        # Calculate cumulative returns
        returns = np.diff(y_pred)
        cumulative = np.cumsum(returns)
        
        # Calculate running maximum
        running_max = np.maximum.accumulate(cumulative)
        
        # Calculate drawdown
        drawdown = cumulative - running_max
        
        # Get maximum drawdown
        max_drawdown = np.min(drawdown)
        return max_drawdown
    except:
        return np.nan


def _extract_feature_importance(model, feature_names: List[str]) -> Dict[str, float]:
    """Extract feature importance from model"""
    try:
        if hasattr(model, 'feature_importances_'):
            importance_dict = dict(zip(feature_names, model.feature_importances_))
            return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        elif hasattr(model, 'coef_'):
            importance_dict = dict(zip(feature_names, np.abs(model.coef_)))
            return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        else:
            return {}
    except:
        return {}


def _calculate_model_diversity(all_metrics: Dict[str, Any]) -> float:
    """Calculate diversity among models"""
    try:
        predictions = []
        
        for model_name, metrics in all_metrics.items():
            if isinstance(metrics, dict) and 'predictions' in metrics:
                predictions.append(metrics['predictions'])
        
        if len(predictions) < 2:
            return 0.0
        
        # Calculate pairwise correlations
        correlations = []
        for i in range(len(predictions)):
            for j in range(i + 1, len(predictions)):
                corr = np.corrcoef(predictions[i], predictions[j])[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)
        
        if not correlations:
            return 0.0
        
        # Diversity is 1 - average correlation
        diversity = 1 - np.mean(correlations)
        return max(0.0, diversity)
        
    except:
        return 0.0


def _calculate_stability_score(all_metrics: Dict[str, Any]) -> float:
    """Calculate stability score across models"""
    try:
        r2_scores = []
        
        for model_name, metrics in all_metrics.items():
            if isinstance(metrics, dict) and 'r2_score' in metrics:
                r2_scores.append(metrics['r2_score'])
        
        if len(r2_scores) < 2:
            return 0.0
        
        # Stability is inverse of standard deviation
        std_r2 = np.std(r2_scores)
        stability = 1 / (1 + std_r2)
        return stability
        
    except:
        return 0.0


def generate_metrics_report(all_metrics: Dict[str, Any], save_path: str = None) -> str:
    """Generate comprehensive metrics report"""
    logger = logging.getLogger(__name__)
    logger.info("📊 Generating comprehensive metrics report")
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("PROJECT HYPERION - ADVANCED METRICS REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # System overview
    if 'system' in all_metrics and 'error' not in all_metrics['system']:
        system = all_metrics['system']
        report_lines.append("SYSTEM OVERVIEW:")
        report_lines.append(f"  Average R² Score: {system['average_r2']:.4f}")
        report_lines.append(f"  Best Model: {system['best_model']} (R² = {system['best_r2']:.4f})")
        report_lines.append(f"  Model Count: {system['model_count']}")
        report_lines.append(f"  Diversity Score: {system['diversity_score']:.4f}")
        report_lines.append(f"  Stability Score: {system['stability_score']:.4f}")
        report_lines.append("")
    
    # Individual model metrics
    report_lines.append("INDIVIDUAL MODEL METRICS:")
    report_lines.append("-" * 50)
    
    for model_name, metrics in all_metrics.items():
        if model_name in ['system', 'ensemble']:
            continue
            
        if isinstance(metrics, dict) and 'error' not in metrics:
            report_lines.append(f"\n{model_name.upper()}:")
            report_lines.append(f"  R² Score: {metrics['r2_score']:.4f}")
            report_lines.append(f"  RMSE: {metrics['rmse']:.4f}")
            report_lines.append(f"  MAE: {metrics['mae']:.4f}")
            report_lines.append(f"  MAPE: {metrics['mape']:.2f}%")
            report_lines.append(f"  Directional Accuracy: {metrics['directional_accuracy']:.4f}")
            report_lines.append(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
            report_lines.append(f"  Max Drawdown: {metrics['max_drawdown']:.4f}")
        elif isinstance(metrics, dict) and 'error' in metrics:
            report_lines.append(f"\n{model_name.upper()}: ERROR - {metrics['error']}")
    
    # Ensemble metrics
    if 'ensemble' in all_metrics and 'error' not in all_metrics['ensemble']:
        ensemble = all_metrics['ensemble']
        report_lines.append(f"\nENSEMBLE PERFORMANCE:")
        report_lines.append(f"  R² Score: {ensemble['r2_score']:.4f}")
        report_lines.append(f"  RMSE: {ensemble['rmse']:.4f}")
        report_lines.append(f"  MAE: {ensemble['mae']:.4f}")
        report_lines.append(f"  MAPE: {ensemble['mape']:.2f}%")
        report_lines.append(f"  Directional Accuracy: {ensemble['directional_accuracy']:.4f}")
        report_lines.append(f"  Models Used: {ensemble['model_count']}")
    
    report_lines.append("\n" + "=" * 80)
    
    report = "\n".join(report_lines)
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report)
        logger.info(f"📊 Metrics report saved to {save_path}")
    
    return report 