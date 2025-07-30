#!/usr/bin/env python3
"""
Advanced Overfitting Prevention Module
Implements comprehensive overfitting detection and prevention strategies
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Any, Optional, Callable
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import xgboost as xgb
try:
    import catboost as cb
except ImportError:
    cb = None
import joblib
import os
from datetime import datetime
import warnings

class OverfittingPrevention:
    """
    Advanced Overfitting Prevention System
    
    Features:
    - CatBoost overfitting detector with od_type, od_wait, od_pval
    - Feature stability selection across multiple folds
    - Comprehensive regularization strategies
    - Model complexity monitoring
    - Performance degradation detection
    - Adaptive hyperparameter adjustment
    """
    
    def __init__(self, 
                 cv_folds: int = 5,
                 stability_threshold: float = 0.7,
                 overfitting_threshold: float = 0.1,
                 max_feature_importance_std: float = 0.3):
        """
        Initialize Overfitting Prevention System.
        
        Args:
            cv_folds: Number of cross-validation folds
            stability_threshold: Minimum feature stability score
            overfitting_threshold: Threshold for overfitting detection
            max_feature_importance_std: Maximum standard deviation for feature importance
        """
        self.cv_folds = cv_folds
        self.stability_threshold = stability_threshold
        self.overfitting_threshold = overfitting_threshold
        self.max_feature_importance_std = max_feature_importance_std
        
        # Tracking
        self.feature_stability_scores = {}
        self.model_complexity_history = []
        self.overfitting_alerts = []
        
        logging.info("Advanced Overfitting Prevention System initialized")
    
    def train_catboost_with_overfitting_detection(self, 
                                                X: pd.DataFrame, 
                                                y: pd.Series,
                                                task_type: str = 'CPU') -> Tuple[Any, Dict[str, Any]]:
        """
        Train CatBoost with comprehensive overfitting detection.
        
        Args:
            X: Feature matrix
            y: Target variable
            task_type: 'CPU' or 'GPU'
            
        Returns:
            Tuple of (trained_model, training_info)
        """
        try:
            if cb is None:
                raise ImportError("CatBoost not available")
            
            logging.info("Training CatBoost with overfitting detection")
            
            # Enhanced CatBoost parameters with overfitting detection
            params = {
                'iterations': 1000,
                'depth': 6,
                'learning_rate': 0.1,
                'l2_leaf_reg': 3.0,
                'border_count': 128,
                'bagging_temperature': 0.8,
                'random_strength': 0.8,
                'verbose': False,
                'allow_writing_files': False,
                'task_type': task_type,
                
                # Overfitting detection parameters
                'od_type': 'Iter',  # Iteration-based overfitting detection
                'od_wait': 50,      # Wait 50 iterations before checking
                'od_pval': 0.001,   # P-value threshold for overfitting detection
                'od_metric': 'RMSE' # Metric to monitor for overfitting
            }
            
            # Split data for validation
            split_idx = int(0.8 * len(X))
            X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
            
            # Train with overfitting detection
            model = cb.CatBoostRegressor(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=100,
                verbose=False
            )
            
            # Analyze training results
            training_info = self._analyze_catboost_training(model, X_train, y_train, X_val, y_val)
            
            # Check for overfitting
            if training_info['overfitting_detected']:
                logging.warning("⚠️ Overfitting detected in CatBoost training")
                self.overfitting_alerts.append({
                    'model': 'CatBoost',
                    'timestamp': datetime.now(),
                    'severity': 'warning',
                    'details': training_info
                })
            
            return model, training_info
            
        except Exception as e:
            logging.error(f"Error in CatBoost overfitting detection: {e}")
            return None, {}
    
    def _analyze_catboost_training(self, model, X_train, y_train, X_val, y_val) -> Dict[str, Any]:
        """Analyze CatBoost training for overfitting indicators."""
        try:
            # Get training history
            train_metrics = model.get_evals_result()
            
            # Calculate metrics
            train_pred = model.predict(X_train)
            val_pred = model.predict(X_val)
            
            train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
            val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
            
            # Overfitting indicators
            train_val_gap = train_rmse - val_rmse
            overfitting_ratio = train_rmse / val_rmse if val_rmse > 0 else float('inf')
            
            # Feature importance analysis
            feature_importance = model.get_feature_importance()
            importance_std = np.std(feature_importance)
            
            return {
                'train_rmse': train_rmse,
                'val_rmse': val_rmse,
                'train_val_gap': train_val_gap,
                'overfitting_ratio': overfitting_ratio,
                'feature_importance_std': importance_std,
                'overfitting_detected': overfitting_ratio < 0.8 or importance_std > self.max_feature_importance_std,
                'best_iteration': model.get_best_iteration(),
                'training_metrics': train_metrics
            }
            
        except Exception as e:
            logging.error(f"Error analyzing CatBoost training: {e}")
            return {}
    
    def perform_feature_stability_selection(self, 
                                          X: pd.DataFrame, 
                                          y: pd.Series,
                                          n_features: int = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Perform feature stability selection across multiple folds.
        
        Args:
            X: Feature matrix
            y: Target variable
            n_features: Number of features to select (None for auto)
            
        Returns:
            Tuple of (selected_features, selection_info)
        """
        try:
            logging.info("Performing feature stability selection")
            
            if n_features is None:
                n_features = min(50, X.shape[1] // 2)
            
            # Multiple feature selection methods
            selection_methods = {
                'mutual_info': SelectKBest(score_func=f_regression, k=n_features),
                'rfe': RFE(estimator=RandomForestRegressor(n_estimators=50), n_features_to_select=n_features),
                'lgb_importance': self._get_lightgbm_importance(X, y, n_features),
                'xgb_importance': self._get_xgboost_importance(X, y, n_features)
            }
            
            # Track feature selection across folds
            feature_selection_counts = {col: 0 for col in X.columns}
            stability_scores = {}
            
            # Cross-validation for stability
            cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
            
            for fold, (train_idx, val_idx) in enumerate(cv.split(X, pd.qcut(y, q=5, labels=False, duplicates='drop'))):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # Apply each selection method
                for method_name, selector in selection_methods.items():
                    if method_name in ['lgb_importance', 'xgb_importance']:
                        selected_features = selector
                    else:
                        selector.fit(X_train, y_train)
                        selected_features = X_train.columns[selector.get_support()].tolist()
                    
                    # Count feature selections
                    for feature in selected_features:
                        if feature in feature_selection_counts:
                            feature_selection_counts[feature] += 1
            
            # Calculate stability scores
            total_selections = self.cv_folds * len(selection_methods)
            for feature, count in feature_selection_counts.items():
                stability_scores[feature] = count / total_selections
            
            # Select stable features
            stable_features = [
                feature for feature, score in stability_scores.items()
                if score >= self.stability_threshold
            ]
            
            # Sort by stability score
            stable_features.sort(key=lambda x: stability_scores[x], reverse=True)
            
            # Limit to requested number of features
            if len(stable_features) > n_features:
                stable_features = stable_features[:n_features]
            
            # Create selected feature matrix
            X_selected = X[stable_features]
            
            # Store stability information
            self.feature_stability_scores = stability_scores
            
            selection_info = {
                'selected_features': stable_features,
                'stability_scores': {f: stability_scores[f] for f in stable_features},
                'avg_stability': np.mean([stability_scores[f] for f in stable_features]),
                'total_features_original': X.shape[1],
                'total_features_selected': len(stable_features),
                'selection_ratio': len(stable_features) / X.shape[1]
            }
            
            logging.info(f"Feature stability selection completed: {len(stable_features)} features selected")
            return X_selected, selection_info
            
        except Exception as e:
            logging.error(f"Error in feature stability selection: {e}")
            return X, {}
    
    def _get_lightgbm_importance(self, X: pd.DataFrame, y: pd.Series, n_features: int) -> List[str]:
        """Get feature importance from LightGBM."""
        try:
            model = lgb.LGBMRegressor(n_estimators=100, verbose=-1, random_state=42)
            model.fit(X, y)
            
            importance = model.feature_importances_
            feature_importance = list(zip(X.columns, importance))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            
            return [feature for feature, _ in feature_importance[:n_features]]
            
        except Exception as e:
            logging.warning(f"Error getting LightGBM importance: {e}")
            return X.columns[:n_features].tolist()
    
    def _get_xgboost_importance(self, X: pd.DataFrame, y: pd.Series, n_features: int) -> List[str]:
        """Get feature importance from XGBoost."""
        try:
            model = xgb.XGBRegressor(n_estimators=100, verbosity=0, random_state=42)
            model.fit(X, y)
            
            importance = model.feature_importances_
            feature_importance = list(zip(X.columns, importance))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            
            return [feature for feature, _ in feature_importance[:n_features]]
            
        except Exception as e:
            logging.warning(f"Error getting XGBoost importance: {e}")
            return X.columns[:n_features].tolist()
    
    def detect_model_overfitting(self, 
                               model, 
                               X_train: pd.DataFrame, 
                               y_train: pd.Series,
                               X_val: pd.DataFrame, 
                               y_val: pd.Series) -> Dict[str, Any]:
        """
        Comprehensive overfitting detection for any model.
        
        Args:
            model: Trained model
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            
        Returns:
            Dictionary with overfitting analysis
        """
        try:
            # Get predictions
            train_pred = model.predict(X_train)
            val_pred = model.predict(X_val)
            
            # Calculate metrics
            train_mse = mean_squared_error(y_train, train_pred)
            val_mse = mean_squared_error(y_val, val_pred)
            train_rmse = np.sqrt(train_mse)
            val_rmse = np.sqrt(val_mse)
            
            # Overfitting indicators
            train_val_gap = train_rmse - val_rmse
            overfitting_ratio = train_rmse / val_rmse if val_rmse > 0 else float('inf')
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_train, y_train, cv=self.cv_folds, scoring='neg_mean_squared_error')
            cv_rmse = np.sqrt(-cv_scores.mean())
            
            # Model complexity estimation
            complexity_score = self._estimate_model_complexity(model)
            
            # Overfitting detection
            overfitting_detected = (
                overfitting_ratio < 0.8 or  # Train performance much better than validation
                train_val_gap < -self.overfitting_threshold or  # Large gap
                cv_rmse > val_rmse * 1.2 or  # CV performance worse than validation
                complexity_score > 0.8  # High complexity
            )
            
            analysis = {
                'train_rmse': train_rmse,
                'val_rmse': val_rmse,
                'cv_rmse': cv_rmse,
                'train_val_gap': train_val_gap,
                'overfitting_ratio': overfitting_ratio,
                'complexity_score': complexity_score,
                'overfitting_detected': overfitting_detected,
                'severity': self._calculate_overfitting_severity(overfitting_ratio, complexity_score)
            }
            
            if overfitting_detected:
                logging.warning(f"⚠️ Overfitting detected: {analysis['severity']}")
                self.overfitting_alerts.append({
                    'timestamp': datetime.now(),
                    'severity': analysis['severity'],
                    'details': analysis
                })
            
            return analysis
            
        except Exception as e:
            logging.error(f"Error in overfitting detection: {e}")
            return {}
    
    def _estimate_model_complexity(self, model) -> float:
        """Estimate model complexity score (0-1, higher = more complex)."""
        try:
            complexity_score = 0.0
            
            # Tree-based models
            if hasattr(model, 'n_estimators'):
                complexity_score += min(model.n_estimators / 1000, 0.3)
            
            if hasattr(model, 'max_depth'):
                complexity_score += min(model.max_depth / 20, 0.2)
            
            # Neural networks
            if hasattr(model, 'count_params'):
                param_count = model.count_params()
                complexity_score += min(param_count / 1000000, 0.4)
            
            # Feature count
            if hasattr(model, 'n_features_in_'):
                complexity_score += min(model.n_features_in_ / 1000, 0.2)
            
            return min(complexity_score, 1.0)
            
        except Exception as e:
            logging.warning(f"Error estimating model complexity: {e}")
            return 0.5
    
    def _calculate_overfitting_severity(self, overfitting_ratio: float, complexity_score: float) -> str:
        """Calculate overfitting severity level."""
        if overfitting_ratio < 0.5 or complexity_score > 0.9:
            return 'critical'
        elif overfitting_ratio < 0.7 or complexity_score > 0.7:
            return 'high'
        elif overfitting_ratio < 0.8 or complexity_score > 0.5:
            return 'medium'
        else:
            return 'low'
    
    def get_regularization_recommendations(self, overfitting_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Get regularization recommendations based on overfitting analysis."""
        try:
            recommendations = {}
            
            if overfitting_analysis.get('overfitting_detected', False):
                severity = overfitting_analysis.get('severity', 'medium')
                
                if severity in ['critical', 'high']:
                    recommendations.update({
                        'increase_l2_regularization': True,
                        'reduce_model_complexity': True,
                        'increase_dropout': True,
                        'reduce_learning_rate': True,
                        'increase_early_stopping_patience': True
                    })
                elif severity == 'medium':
                    recommendations.update({
                        'moderate_l2_regularization': True,
                        'slight_complexity_reduction': True,
                        'increase_cross_validation_folds': True
                    })
                
                # Specific recommendations based on model type
                if overfitting_analysis.get('complexity_score', 0) > 0.7:
                    recommendations['reduce_max_depth'] = True
                    recommendations['reduce_n_estimators'] = True
                
                if overfitting_analysis.get('overfitting_ratio', 1) < 0.7:
                    recommendations['increase_bagging'] = True
                    recommendations['reduce_feature_fraction'] = True
            
            return recommendations
            
        except Exception as e:
            logging.error(f"Error generating regularization recommendations: {e}")
            return {}
    
    def get_overfitting_summary(self) -> Dict[str, Any]:
        """Get comprehensive overfitting prevention summary."""
        return {
            'total_alerts': len(self.overfitting_alerts),
            'critical_alerts': len([a for a in self.overfitting_alerts if a.get('severity') == 'critical']),
            'high_alerts': len([a for a in self.overfitting_alerts if a.get('severity') == 'high']),
            'feature_stability_avg': np.mean(list(self.feature_stability_scores.values())) if self.feature_stability_scores else 0,
            'stable_features_count': len([f for f, s in self.feature_stability_scores.items() if s >= self.stability_threshold]),
            'recent_alerts': self.overfitting_alerts[-10:] if self.overfitting_alerts else []
        } 