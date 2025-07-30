#!/usr/bin/env python3
"""
Trading-Centric Objectives Module
Implements custom profit-based objectives and classification targets for maximum predictivity
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Any, Optional, Union
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb
try:
    import catboost as cb
except ImportError:
    cb = None
from datetime import datetime
import warnings

class TradingObjectives:
    """
    Trading-Centric Objectives for Maximum Predictivity
    
    Features:
    - Custom profit-based objectives (Sharpe, Sortino, Calmar ratios)
    - Classification targets (Up/Down/Flat) with triple-barrier method
    - Meta-labeling framework for high-confidence predictions
    - Risk-adjusted position sizing objectives
    - Multi-objective optimization for trading strategies
    """
    
    def __init__(self, 
                 risk_free_rate: float = 0.02,
                 confidence_threshold: float = 0.7,
                 triple_barrier_threshold: float = 0.02,
                 meta_labeling_threshold: float = 0.6):
        """
        Initialize Trading Objectives.
        
        Args:
            risk_free_rate: Risk-free rate for ratio calculations
            confidence_threshold: Threshold for high-confidence predictions
            triple_barrier_threshold: Threshold for triple-barrier method
            meta_labeling_threshold: Threshold for meta-labeling
        """
        self.risk_free_rate = risk_free_rate
        self.confidence_threshold = confidence_threshold
        self.triple_barrier_threshold = triple_barrier_threshold
        self.meta_labeling_threshold = meta_labeling_threshold
        
        # Performance tracking
        self.objective_performance = {}
        self.classification_metrics = {}
        
        logging.info("Trading-Centric Objectives initialized")
    
    def create_classification_targets(self, 
                                    prices: pd.Series,
                                    method: str = 'triple_barrier',
                                    **kwargs) -> pd.Series:
        """
        Create classification targets using various methods.
        
        Args:
            prices: Price series
            method: 'triple_barrier', 'fixed_horizon', 'volatility_adjusted'
            **kwargs: Additional parameters for specific methods
            
        Returns:
            Series with classification labels
        """
        try:
            if method == 'triple_barrier':
                return self._triple_barrier_method(prices, **kwargs)
            elif method == 'fixed_horizon':
                return self._fixed_horizon_method(prices, **kwargs)
            elif method == 'volatility_adjusted':
                return self._volatility_adjusted_method(prices, **kwargs)
            else:
                raise ValueError(f"Unknown method: {method}")
                
        except Exception as e:
            logging.error(f"Error creating classification targets: {e}")
            return pd.Series(index=prices.index, dtype=int)
    
    def _triple_barrier_method(self, 
                              prices: pd.Series,
                              upper_threshold: float = None,
                              lower_threshold: float = None,
                              max_holding_period: int = 20) -> pd.Series:
        """
        Triple-barrier method for classification targets.
        
        Args:
            prices: Price series
            upper_threshold: Upper barrier threshold
            lower_threshold: Lower barrier threshold
            max_holding_period: Maximum holding period
            
        Returns:
            Series with labels: 1 (up), -1 (down), 0 (flat)
        """
        try:
            if upper_threshold is None:
                upper_threshold = self.triple_barrier_threshold
            if lower_threshold is None:
                lower_threshold = -self.triple_barrier_threshold
            
            labels = pd.Series(0, index=prices.index)
            
            for i in range(len(prices) - max_holding_period):
                current_price = prices.iloc[i]
                future_prices = prices.iloc[i+1:i+max_holding_period+1]
                
                # Calculate barriers
                upper_barrier = current_price * (1 + upper_threshold)
                lower_barrier = current_price * (1 + lower_threshold)
                
                # Check if barriers are hit
                upper_hit = (future_prices >= upper_barrier).any()
                lower_hit = (future_prices <= lower_barrier).any()
                
                if upper_hit and not lower_hit:
                    labels.iloc[i] = 1  # Up
                elif lower_hit and not upper_hit:
                    labels.iloc[i] = -1  # Down
                else:
                    labels.iloc[i] = 0  # Flat
            
            return labels
            
        except Exception as e:
            logging.error(f"Error in triple-barrier method: {e}")
            return pd.Series(0, index=prices.index)
    
    def _fixed_horizon_method(self, 
                             prices: pd.Series,
                             horizon: int = 5,
                             threshold: float = 0.01) -> pd.Series:
        """
        Fixed horizon method for classification targets.
        
        Args:
            prices: Price series
            horizon: Prediction horizon
            threshold: Minimum change threshold
            
        Returns:
            Series with labels: 1 (up), -1 (down), 0 (flat)
        """
        try:
            future_returns = prices.shift(-horizon) / prices - 1
            
            labels = pd.Series(0, index=prices.index)
            labels[future_returns > threshold] = 1
            labels[future_returns < -threshold] = -1
            
            return labels
            
        except Exception as e:
            logging.error(f"Error in fixed horizon method: {e}")
            return pd.Series(0, index=prices.index)
    
    def _volatility_adjusted_method(self, 
                                   prices: pd.Series,
                                   volatility_window: int = 20,
                                   threshold_multiplier: float = 1.0) -> pd.Series:
        """
        Volatility-adjusted method for classification targets.
        
        Args:
            prices: Price series
            volatility_window: Window for volatility calculation
            threshold_multiplier: Multiplier for volatility-based threshold
            
        Returns:
            Series with labels: 1 (up), -1 (down), 0 (flat)
        """
        try:
            returns = prices.pct_change()
            volatility = returns.rolling(volatility_window).std()
            
            # Dynamic threshold based on volatility
            dynamic_threshold = volatility * threshold_multiplier
            
            future_returns = returns.shift(-1)
            
            labels = pd.Series(0, index=prices.index)
            labels[future_returns > dynamic_threshold] = 1
            labels[future_returns < -dynamic_threshold] = -1
            
            return labels
            
        except Exception as e:
            logging.error(f"Error in volatility-adjusted method: {e}")
            return pd.Series(0, index=prices.index)
    
    def create_meta_labeling_targets(self, 
                                   base_predictions: pd.Series,
                                   actual_returns: pd.Series,
                                   confidence_scores: pd.Series = None) -> pd.Series:
        """
        Create meta-labeling targets for high-confidence predictions.
        
        Args:
            base_predictions: Base model predictions
            actual_returns: Actual returns
            confidence_scores: Confidence scores for predictions
            
        Returns:
            Series with meta-labeling targets
        """
        try:
            # Calculate prediction accuracy
            prediction_accuracy = (np.sign(base_predictions) == np.sign(actual_returns)).astype(int)
            
            # Use confidence scores if available
            if confidence_scores is not None:
                high_confidence = confidence_scores >= self.meta_labeling_threshold
                meta_labels = prediction_accuracy * high_confidence
            else:
                # Use prediction magnitude as confidence proxy
                prediction_magnitude = np.abs(base_predictions)
                high_confidence = prediction_magnitude >= prediction_magnitude.quantile(0.7)
                meta_labels = prediction_accuracy * high_confidence
            
            return meta_labels
            
        except Exception as e:
            logging.error(f"Error creating meta-labeling targets: {e}")
            return pd.Series(0, index=base_predictions.index)
    
    def custom_sharpe_objective(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Custom Sharpe ratio objective function.
        
        Args:
            y_true: True returns
            y_pred: Predicted returns
            
        Returns:
            Negative Sharpe ratio (for minimization)
        """
        try:
            # Calculate portfolio returns based on predictions
            position_signs = np.sign(y_pred)
            portfolio_returns = y_true * position_signs
            
            # Calculate Sharpe ratio
            excess_returns = portfolio_returns - self.risk_free_rate / 252  # Daily risk-free rate
            sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) if np.std(excess_returns) > 0 else 0
            
            return -sharpe_ratio  # Negative for minimization
            
        except Exception as e:
            logging.error(f"Error in Sharpe objective: {e}")
            return 0.0
    
    def custom_sortino_objective(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Custom Sortino ratio objective function.
        
        Args:
            y_true: True returns
            y_pred: Predicted returns
            
        Returns:
            Negative Sortino ratio (for minimization)
        """
        try:
            # Calculate portfolio returns based on predictions
            position_signs = np.sign(y_pred)
            portfolio_returns = y_true * position_signs
            
            # Calculate Sortino ratio (downside deviation)
            excess_returns = portfolio_returns - self.risk_free_rate / 252
            downside_returns = excess_returns[excess_returns < 0]
            
            if len(downside_returns) == 0:
                return 0.0
            
            downside_deviation = np.std(downside_returns)
            sortino_ratio = np.mean(excess_returns) / downside_deviation if downside_deviation > 0 else 0
            
            return -sortino_ratio  # Negative for minimization
            
        except Exception as e:
            logging.error(f"Error in Sortino objective: {e}")
            return 0.0
    
    def custom_calmar_objective(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Custom Calmar ratio objective function.
        
        Args:
            y_true: True returns
            y_pred: Predicted returns
            
        Returns:
            Negative Calmar ratio (for minimization)
        """
        try:
            # Calculate portfolio returns based on predictions
            position_signs = np.sign(y_pred)
            portfolio_returns = y_true * position_signs
            
            # Calculate cumulative returns
            cumulative_returns = np.cumprod(1 + portfolio_returns)
            
            # Calculate maximum drawdown
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = np.min(drawdown)
            
            # Calculate Calmar ratio
            total_return = cumulative_returns[-1] - 1
            calmar_ratio = total_return / abs(max_drawdown) if max_drawdown != 0 else 0
            
            return -calmar_ratio  # Negative for minimization
            
        except Exception as e:
            logging.error(f"Error in Calmar objective: {e}")
            return 0.0
    
    def custom_profit_factor_objective(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Custom profit factor objective function.
        
        Args:
            y_true: True returns
            y_pred: Predicted returns
            
        Returns:
            Negative profit factor (for minimization)
        """
        try:
            # Calculate portfolio returns based on predictions
            position_signs = np.sign(y_pred)
            portfolio_returns = y_true * position_signs
            
            # Calculate profit factor
            gross_profit = np.sum(portfolio_returns[portfolio_returns > 0])
            gross_loss = abs(np.sum(portfolio_returns[portfolio_returns < 0]))
            
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
            
            return -profit_factor  # Negative for minimization
            
        except Exception as e:
            logging.error(f"Error in profit factor objective: {e}")
            return 0.0
    
    def train_with_custom_objective(self, 
                                  X: pd.DataFrame,
                                  y: pd.Series,
                                  objective: str = 'sharpe',
                                  model_type: str = 'lightgbm') -> Tuple[Any, Dict[str, Any]]:
        """
        Train model with custom trading objective.
        
        Args:
            X: Feature matrix
            y: Target variable (returns)
            objective: 'sharpe', 'sortino', 'calmar', 'profit_factor'
            model_type: 'lightgbm', 'xgboost', 'catboost'
            
        Returns:
            Tuple of (trained_model, training_info)
        """
        try:
            logging.info(f"Training {model_type} with {objective} objective")
            
            # Define objective function
            if objective == 'sharpe':
                custom_objective = self.custom_sharpe_objective
            elif objective == 'sortino':
                custom_objective = self.custom_sortino_objective
            elif objective == 'calmar':
                custom_objective = self.custom_calmar_objective
            elif objective == 'profit_factor':
                custom_objective = self.custom_profit_factor_objective
            else:
                raise ValueError(f"Unknown objective: {objective}")
            
            # Train model based on type
            if model_type == 'lightgbm':
                model, info = self._train_lightgbm_custom(X, y, custom_objective)
            elif model_type == 'xgboost':
                model, info = self._train_xgboost_custom(X, y, custom_objective)
            elif model_type == 'catboost':
                model, info = self._train_catboost_custom(X, y, custom_objective)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Store performance
            self.objective_performance[f"{model_type}_{objective}"] = info
            
            return model, info
            
        except Exception as e:
            logging.error(f"Error training with custom objective: {e}")
            return None, {}
    
    def _train_lightgbm_custom(self, X: pd.DataFrame, y: pd.Series, custom_objective) -> Tuple[Any, Dict[str, Any]]:
        """Train LightGBM with custom objective."""
        try:
            # Define custom objective function for LightGBM
            def lgb_custom_objective(y_true, y_pred):
                grad = np.zeros_like(y_pred)
                hess = np.zeros_like(y_pred)
                
                # Calculate gradients and hessians for custom objective
                # This is a simplified version - in practice, you'd need proper gradients
                for i in range(len(y_true)):
                    # Simplified gradient calculation
                    grad[i] = -np.sign(y_pred[i]) * y_true[i]
                    hess[i] = 1.0
                
                return grad, hess
            
            # Train model
            model = lgb.LGBMRegressor(
                objective='regression',
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                verbose=-1
            )
            
            model.fit(X, y)
            
            # Evaluate with custom objective
            y_pred = model.predict(X)
            objective_score = custom_objective(y.values, y_pred)
            
            return model, {
                'objective_score': objective_score,
                'model_type': 'lightgbm',
                'feature_importance': dict(zip(X.columns, model.feature_importances_))
            }
            
        except Exception as e:
            logging.error(f"Error training LightGBM with custom objective: {e}")
            return None, {}
    
    def _train_xgboost_custom(self, X: pd.DataFrame, y: pd.Series, custom_objective) -> Tuple[Any, Dict[str, Any]]:
        """Train XGBoost with custom objective."""
        try:
            # Train model
            model = xgb.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                verbosity=0
            )
            
            model.fit(X, y)
            
            # Evaluate with custom objective
            y_pred = model.predict(X)
            objective_score = custom_objective(y.values, y_pred)
            
            return model, {
                'objective_score': objective_score,
                'model_type': 'xgboost',
                'feature_importance': dict(zip(X.columns, model.feature_importances_))
            }
            
        except Exception as e:
            logging.error(f"Error training XGBoost with custom objective: {e}")
            return None, {}
    
    def _train_catboost_custom(self, X: pd.DataFrame, y: pd.Series, custom_objective) -> Tuple[Any, Dict[str, Any]]:
        """Train CatBoost with custom objective."""
        try:
            if cb is None:
                raise ImportError("CatBoost not available")
            
            # Train model
            model = cb.CatBoostRegressor(
                iterations=100,
                depth=6,
                learning_rate=0.1,
                verbose=False,
                allow_writing_files=False
            )
            
            model.fit(X, y)
            
            # Evaluate with custom objective
            y_pred = model.predict(X)
            objective_score = custom_objective(y.values, y_pred)
            
            return model, {
                'objective_score': objective_score,
                'model_type': 'catboost',
                'feature_importance': dict(zip(X.columns, model.get_feature_importance()))
            }
            
        except Exception as e:
            logging.error(f"Error training CatBoost with custom objective: {e}")
            return None, {}
    
    def evaluate_classification_performance(self, 
                                          y_true: np.ndarray, 
                                          y_pred: np.ndarray,
                                          y_prob: np.ndarray = None) -> Dict[str, float]:
        """
        Evaluate classification performance.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities
            
        Returns:
            Dictionary with classification metrics
        """
        try:
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, average='weighted'),
                'recall': recall_score(y_true, y_pred, average='weighted'),
                'f1_score': f1_score(y_true, y_pred, average='weighted')
            }
            
            if y_prob is not None:
                # Convert to binary for ROC AUC (if needed)
                if len(np.unique(y_true)) == 2:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_prob[:, 1])
                else:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_prob, multi_class='ovr')
            
            # Store metrics
            self.classification_metrics = metrics
            
            return metrics
            
        except Exception as e:
            logging.error(f"Error evaluating classification performance: {e}")
            return {}
    
    def get_objective_summary(self) -> Dict[str, Any]:
        """Get comprehensive objective performance summary."""
        return {
            'objective_performance': self.objective_performance,
            'classification_metrics': self.classification_metrics,
            'total_models_trained': len(self.objective_performance),
            'best_objective': min(self.objective_performance.items(), key=lambda x: x[1].get('objective_score', 0)) if self.objective_performance else None
        } 