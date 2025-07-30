import numpy as np
import pandas as pd
import logging
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
import matplotlib.pyplot as plt
import seaborn as sns

# SHAP for explainability
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("SHAP not available. Explainability will be limited.")

# LIME for local explainability
try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    logging.warning("LIME not available. Local explainability will be limited.")

class MarketRegimeDetector:
    """
    Advanced market regime detection using multiple methodologies.
    """
    
    def __init__(self, 
                 n_regimes: int = 4,
                 lookback_window: int = 100,
                 volatility_threshold: float = 0.02):
        """
        Initialize the market regime detector.
        
        Args:
            n_regimes: Number of market regimes to detect
            lookback_window: Window for regime analysis
            volatility_threshold: Threshold for volatility-based regime changes
        """
        self.n_regimes = n_regimes
        self.lookback_window = lookback_window
        self.volatility_threshold = volatility_threshold
        
        # Regime characteristics
        self.regime_centers = None
        self.regime_labels = None
        self.regime_transitions = []
        self.current_regime = None
        
        # Models
        self.kmeans = KMeans(n_clusters=n_regimes, random_state=42)
        self.scaler = StandardScaler()
        
        logging.info("🎯 Market Regime Detector initialized")
    
    def detect_regimes(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect market regimes from price data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary with regime information
        """
        try:
            # Calculate regime features
            features = self._calculate_regime_features(data)
            
            if len(features) < self.lookback_window:
                logging.warning("Insufficient data for regime detection")
                return {}
            
            # Scale features
            features_scaled = self.scaler.fit_transform(features)
            
            # Detect regimes using K-means
            regime_labels = self.kmeans.fit_predict(features_scaled)
            self.regime_centers = self.kmeans.cluster_centers_
            self.regime_labels = regime_labels
            
            # Analyze regime characteristics
            regime_analysis = self._analyze_regimes(features, regime_labels)
            
            # Detect regime transitions
            transitions = self._detect_transitions(regime_labels)
            
            # Determine current regime
            current_regime = self._determine_current_regime(features_scaled[-1])
            
            return {
                'regime_labels': regime_labels.tolist(),
                'regime_centers': self.regime_centers.tolist(),
                'regime_analysis': regime_analysis,
                'transitions': transitions,
                'current_regime': current_regime,
                'regime_probabilities': self._calculate_regime_probabilities(features_scaled[-1])
            }
            
        except Exception as e:
            logging.error(f"Error detecting regimes: {e}")
            return {}
    
    def _calculate_regime_features(self, data: pd.DataFrame) -> np.ndarray:
        """Calculate features for regime detection."""
        try:
            features = []
            
            for i in range(self.lookback_window, len(data)):
                window_data = data.iloc[i-self.lookback_window:i]
                
                # Price-based features
                returns = window_data['close'].pct_change().dropna()
                volatility = returns.std()
                mean_return = returns.mean()
                skewness = returns.skew()
                kurtosis = returns.kurtosis()
                
                # Volume-based features
                volume_mean = window_data['volume'].mean()
                volume_std = window_data['volume'].std()
                volume_trend = np.polyfit(range(len(window_data)), window_data['volume'], 1)[0]
                
                # Technical features
                sma_20 = window_data['close'].rolling(20).mean().iloc[-1]
                sma_50 = window_data['close'].rolling(50).mean().iloc[-1]
                price_vs_sma20 = (window_data['close'].iloc[-1] - sma_20) / sma_20
                price_vs_sma50 = (window_data['close'].iloc[-1] - sma_50) / sma_50
                
                # Momentum features
                momentum_5 = (window_data['close'].iloc[-1] - window_data['close'].iloc[-6]) / window_data['close'].iloc[-6]
                momentum_20 = (window_data['close'].iloc[-1] - window_data['close'].iloc[-21]) / window_data['close'].iloc[-21]
                
                # Volatility features
                high_low_volatility = (window_data['high'] - window_data['low']).mean() / window_data['close'].mean()
                
                feature_vector = [
                    volatility, mean_return, skewness, kurtosis,
                    volume_mean, volume_std, volume_trend,
                    price_vs_sma20, price_vs_sma50,
                    momentum_5, momentum_20, high_low_volatility
                ]
                
                features.append(feature_vector)
            
            return np.array(features)
            
        except Exception as e:
            logging.error(f"Error calculating regime features: {e}")
            return np.array([])
    
    def _analyze_regimes(self, features: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        """Analyze characteristics of each regime."""
        try:
            analysis = {}
            
            for regime_id in range(self.n_regimes):
                regime_mask = labels == regime_id
                regime_features = features[regime_mask]
                
                if len(regime_features) == 0:
                    continue
                
                # Calculate regime statistics
                regime_stats = {
                    'count': len(regime_features),
                    'percentage': len(regime_features) / len(features) * 100,
                    'volatility': np.mean(regime_features[:, 0]),
                    'mean_return': np.mean(regime_features[:, 1]),
                    'skewness': np.mean(regime_features[:, 2]),
                    'kurtosis': np.mean(regime_features[:, 3]),
                    'volume_trend': np.mean(regime_features[:, 6]),
                    'momentum': np.mean(regime_features[:, 10])
                }
                
                # Determine regime type
                regime_type = self._classify_regime(regime_stats)
                regime_stats['type'] = regime_type
                
                analysis[f'regime_{regime_id}'] = regime_stats
            
            return analysis
            
        except Exception as e:
            logging.error(f"Error analyzing regimes: {e}")
            return {}
    
    def _classify_regime(self, stats: Dict[str, float]) -> str:
        """Classify regime based on characteristics."""
        try:
            volatility = stats['volatility']
            mean_return = stats['mean_return']
            momentum = stats['momentum']
            
            if volatility > 0.03:  # High volatility
                if mean_return > 0:
                    return 'bull_volatile'
                else:
                    return 'bear_volatile'
            else:  # Low volatility
                if mean_return > 0:
                    return 'bull_stable'
                else:
                    return 'bear_stable'
                    
        except Exception as e:
            logging.error(f"Error classifying regime: {e}")
            return 'unknown'
    
    def _detect_transitions(self, labels: np.ndarray) -> List[Dict[str, Any]]:
        """Detect regime transitions."""
        try:
            transitions = []
            
            for i in range(1, len(labels)):
                if labels[i] != labels[i-1]:
                    transition = {
                        'index': i,
                        'from_regime': int(labels[i-1]),
                        'to_regime': int(labels[i]),
                        'timestamp': datetime.now() - timedelta(days=len(labels)-i)
                    }
                    transitions.append(transition)
            
            return transitions
            
        except Exception as e:
            logging.error(f"Error detecting transitions: {e}")
            return []
    
    def _determine_current_regime(self, current_features: np.ndarray) -> int:
        """Determine the current market regime."""
        try:
            # Find closest regime center
            distances = np.linalg.norm(self.regime_centers - current_features, axis=1)
            current_regime = np.argmin(distances)
            
            return int(current_regime)
            
        except Exception as e:
            logging.error(f"Error determining current regime: {e}")
            return 0
    
    def _calculate_regime_probabilities(self, current_features: np.ndarray) -> List[float]:
        """Calculate probabilities for each regime."""
        try:
            # Calculate distances to regime centers
            distances = np.linalg.norm(self.regime_centers - current_features, axis=1)
            
            # Convert distances to probabilities (inverse relationship)
            probabilities = 1 / (1 + distances)
            probabilities = probabilities / np.sum(probabilities)
            
            return probabilities.tolist()
            
        except Exception as e:
            logging.error(f"Error calculating regime probabilities: {e}")
            return [1.0/self.n_regimes] * self.n_regimes
    
    def adapt_strategy_to_regime(self, strategy_params: Dict[str, Any], regime_id: int) -> Dict[str, Any]:
        """
        Adapt trading strategy parameters based on current regime.
        
        Args:
            strategy_params: Base strategy parameters
            regime_id: Current regime ID
            
        Returns:
            Adapted strategy parameters
        """
        try:
            adapted_params = strategy_params.copy()
            
            # Get regime analysis
            if hasattr(self, 'regime_analysis') and f'regime_{regime_id}' in self.regime_analysis:
                regime_stats = self.regime_analysis[f'regime_{regime_id}']
                regime_type = regime_stats.get('type', 'unknown')
                
                # Adapt parameters based on regime type
                if regime_type == 'bull_volatile':
                    adapted_params.update({
                        'position_size_multiplier': 0.7,  # Reduce position size
                        'stop_loss_multiplier': 1.5,      # Wider stops
                        'take_profit_multiplier': 2.0,    # Higher targets
                        'confidence_threshold': 0.7        # Higher confidence required
                    })
                
                elif regime_type == 'bear_volatile':
                    adapted_params.update({
                        'position_size_multiplier': 0.5,  # Much smaller positions
                        'stop_loss_multiplier': 1.2,      # Tighter stops
                        'take_profit_multiplier': 1.5,    # Lower targets
                        'confidence_threshold': 0.8        # Very high confidence required
                    })
                
                elif regime_type == 'bull_stable':
                    adapted_params.update({
                        'position_size_multiplier': 1.0,  # Normal position size
                        'stop_loss_multiplier': 1.0,      # Normal stops
                        'take_profit_multiplier': 1.5,    # Normal targets
                        'confidence_threshold': 0.6        # Normal confidence
                    })
                
                elif regime_type == 'bear_stable':
                    adapted_params.update({
                        'position_size_multiplier': 0.8,  # Slightly smaller positions
                        'stop_loss_multiplier': 0.8,      # Tighter stops
                        'take_profit_multiplier': 1.2,    # Lower targets
                        'confidence_threshold': 0.7        # Higher confidence
                    })
            
            return adapted_params
            
        except Exception as e:
            logging.error(f"Error adapting strategy to regime: {e}")
            return strategy_params

class AdvancedExplainability:
    """
    Advanced explainability system using SHAP and LIME.
    """
    
    def __init__(self, 
                 model = None,
                 feature_names: List[str] = None):
        """
        Initialize the explainability system.
        
        Args:
            model: Trained model to explain
            feature_names: List of feature names
        """
        self.model = model
        self.feature_names = feature_names or []
        self.shap_explainer = None
        self.lime_explainer = None
        
        # Initialize explainers
        self._initialize_explainers()
        
        logging.info("🔍 Advanced Explainability System initialized")
    
    def _initialize_explainers(self) -> None:
        """Initialize SHAP and LIME explainers."""
        try:
            if SHAP_AVAILABLE and self.model is not None:
                # Initialize SHAP explainer based on model type
                if hasattr(self.model, 'predict_proba'):
                    self.shap_explainer = shap.TreeExplainer(self.model)
                else:
                    # For other model types, use KernelExplainer
                    self.shap_explainer = shap.KernelExplainer(self.model.predict_proba, shap.sample(X, 100))
            
            if LIME_AVAILABLE:
                # Initialize LIME explainer
                self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                    training_data=np.zeros((100, len(self.feature_names))),
                    feature_names=self.feature_names,
                    class_names=['0', '1'],
                    mode='classification'
                )
                
        except Exception as e:
            logging.error(f"Error initializing explainers: {e}")
    
    def explain_prediction(self, 
                          X: np.ndarray, 
                          instance_idx: int = 0,
                          use_shap: bool = True,
                          use_lime: bool = True) -> Dict[str, Any]:
        """
        Explain a specific prediction.
        
        Args:
            X: Feature matrix
            instance_idx: Index of instance to explain
            use_shap: Whether to use SHAP
            use_lime: Whether to use LIME
            
        Returns:
            Explanation results
        """
        try:
            explanation = {
                'instance_idx': instance_idx,
                'prediction': None,
                'shap_values': None,
                'lime_explanation': None,
                'feature_importance': {}
            }
            
            # Get prediction
            if self.model is not None:
                if hasattr(self.model, 'predict_proba'):
                    prediction = self.model.predict_proba(X[instance_idx:instance_idx+1])[0]
                    explanation['prediction'] = prediction.tolist()
                else:
                    prediction = self.model.predict(X[instance_idx:instance_idx+1])[0]
                    explanation['prediction'] = prediction
            
            # SHAP explanation
            if use_shap and self.shap_explainer is not None:
                try:
                    shap_values = self.shap_explainer.shap_values(X[instance_idx:instance_idx+1])
                    
                    if isinstance(shap_values, list):
                        shap_values = shap_values[1]  # For binary classification
                    
                    explanation['shap_values'] = shap_values.tolist()
                    
                    # Create feature importance dictionary
                    for i, (feature, value) in enumerate(zip(self.feature_names, shap_values[0])):
                        explanation['feature_importance'][feature] = {
                            'shap_value': float(value),
                            'abs_shap_value': abs(float(value))
                        }
                        
                except Exception as e:
                    logging.error(f"Error in SHAP explanation: {e}")
            
            # LIME explanation
            if use_lime and self.lime_explainer is not None:
                try:
                    lime_exp = self.lime_explainer.explain_instance(
                        X[instance_idx],
                        self.model.predict_proba,
                        num_features=len(self.feature_names)
                    )
                    
                    explanation['lime_explanation'] = {
                        'feature_weights': dict(lime_exp.as_list()),
                        'score': lime_exp.score
                    }
                    
                    # Update feature importance with LIME values
                    for feature, weight in lime_exp.as_list():
                        if feature in explanation['feature_importance']:
                            explanation['feature_importance'][feature]['lime_weight'] = weight
                        else:
                            explanation['feature_importance'][feature] = {'lime_weight': weight}
                            
                except Exception as e:
                    logging.error(f"Error in LIME explanation: {e}")
            
            return explanation
            
        except Exception as e:
            logging.error(f"Error explaining prediction: {e}")
            return {}
    
    def explain_model_globally(self, X: np.ndarray, sample_size: int = 1000) -> Dict[str, Any]:
        """
        Provide global model explanation.
        
        Args:
            X: Feature matrix
            sample_size: Number of samples to use for explanation
            
        Returns:
            Global explanation results
        """
        try:
            # Sample data if too large
            if len(X) > sample_size:
                indices = np.random.choice(len(X), sample_size, replace=False)
                X_sample = X[indices]
            else:
                X_sample = X
            
            explanation = {
                'global_feature_importance': {},
                'feature_interactions': {},
                'model_summary': {}
            }
            
            # SHAP global explanation
            if self.shap_explainer is not None:
                try:
                    shap_values = self.shap_explainer.shap_values(X_sample)
                    
                    if isinstance(shap_values, list):
                        shap_values = shap_values[1]  # For binary classification
                    
                    # Calculate mean absolute SHAP values
                    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
                    
                    for i, feature in enumerate(self.feature_names):
                        explanation['global_feature_importance'][feature] = {
                            'mean_abs_shap': float(mean_abs_shap[i]),
                            'rank': 0  # Will be updated below
                        }
                    
                    # Rank features by importance
                    feature_ranks = sorted(
                        explanation['global_feature_importance'].items(),
                        key=lambda x: x[1]['mean_abs_shap'],
                        reverse=True
                    )
                    
                    for rank, (feature, _) in enumerate(feature_ranks):
                        explanation['global_feature_importance'][feature]['rank'] = rank + 1
                        
                except Exception as e:
                    logging.error(f"Error in SHAP global explanation: {e}")
            
            # Model summary statistics
            if self.model is not None:
                explanation['model_summary'] = {
                    'model_type': type(self.model).__name__,
                    'n_features': len(self.feature_names),
                    'n_samples': len(X_sample)
                }
            
            return explanation
            
        except Exception as e:
            logging.error(f"Error in global explanation: {e}")
            return {}
    
    def generate_explanation_report(self, 
                                   X: np.ndarray, 
                                   y: np.ndarray = None,
                                   output_path: str = "reports/explanation_report.html") -> str:
        """
        Generate a comprehensive explanation report.
        
        Args:
            X: Feature matrix
            y: Target variable (optional)
            output_path: Path to save the report
            
        Returns:
            Path to the generated report
        """
        try:
            # Create reports directory
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Get global explanation
            global_exp = self.explain_model_globally(X)
            
            # Generate HTML report
            html_content = self._generate_html_report(global_exp, X, y)
            
            # Save report
            with open(output_path, 'w') as f:
                f.write(html_content)
            
            logging.info(f"📊 Explanation report saved to {output_path}")
            return output_path
            
        except Exception as e:
            logging.error(f"Error generating explanation report: {e}")
            return ""
    
    def _generate_html_report(self, 
                             global_exp: Dict[str, Any], 
                             X: np.ndarray, 
                             y: np.ndarray = None) -> str:
        """Generate HTML report content."""
        try:
            html = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Model Explanation Report</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    .section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; }
                    .feature-importance { margin: 10px 0; }
                    .feature-item { padding: 5px; margin: 2px 0; background: #f9f9f9; }
                </style>
            </head>
            <body>
                <h1>Model Explanation Report</h1>
                <div class="section">
                    <h2>Model Summary</h2>
                    <p><strong>Model Type:</strong> {model_type}</p>
                    <p><strong>Number of Features:</strong> {n_features}</p>
                    <p><strong>Number of Samples:</strong> {n_samples}</p>
                </div>
                <div class="section">
                    <h2>Feature Importance (SHAP)</h2>
                    {feature_importance_html}
                </div>
            </body>
            </html>
            """
            
            # Fill in template
            model_summary = global_exp.get('model_summary', {})
            feature_importance = global_exp.get('global_feature_importance', {})
            
            # Generate feature importance HTML
            feature_html = ""
            for feature, info in sorted(feature_importance.items(), 
                                      key=lambda x: x[1]['rank']):
                feature_html += f"""
                <div class="feature-item">
                    <strong>{feature}</strong> (Rank: {info['rank']}) - 
                    Mean |SHAP|: {info['mean_abs_shap']:.4f}
                </div>
                """
            
            html = html.format(
                model_type=model_summary.get('model_type', 'Unknown'),
                n_features=model_summary.get('n_features', 0),
                n_samples=model_summary.get('n_samples', 0),
                feature_importance_html=feature_html
            )
            
            return html
            
        except Exception as e:
            logging.error(f"Error generating HTML report: {e}")
            return "<html><body><h1>Error generating report</h1></body></html>"

class AnomalyDetector:
    """
    Advanced anomaly detection system for market data.
    """
    
    def __init__(self, 
                 contamination: float = 0.1,
                 random_state: int = 42):
        """
        Initialize the anomaly detector.
        
        Args:
            contamination: Expected proportion of anomalies
            random_state: Random state for reproducibility
        """
        self.contamination = contamination
        self.random_state = random_state
        
        # Initialize detection models
        self.isolation_forest = IsolationForest(
            contamination=contamination,
            random_state=random_state
        )
        self.elliptic_envelope = EllipticEnvelope(
            contamination=contamination,
            random_state=random_state
        )
        self.dbscan = DBSCAN(eps=0.5, min_samples=5)
        
        # Anomaly history
        self.anomaly_history = []
        self.anomaly_thresholds = {}
        
        logging.info("🚨 Anomaly Detector initialized")
    
    def detect_anomalies(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect anomalies in market data.
        
        Args:
            data: DataFrame with market data
            
        Returns:
            Anomaly detection results
        """
        try:
            # Calculate anomaly features
            features = self._calculate_anomaly_features(data)
            
            if len(features) == 0:
                return {}
            
            # Scale features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Detect anomalies using multiple methods
            results = {}
            
            # Isolation Forest
            if_anomalies = self.isolation_forest.fit_predict(features_scaled)
            results['isolation_forest'] = {
                'anomalies': (if_anomalies == -1).tolist(),
                'scores': self.isolation_forest.decision_function(features_scaled).tolist()
            }
            
            # Elliptic Envelope
            try:
                ee_anomalies = self.elliptic_envelope.fit_predict(features_scaled)
                results['elliptic_envelope'] = {
                    'anomalies': (ee_anomalies == -1).tolist(),
                    'scores': self.elliptic_envelope.decision_function(features_scaled).tolist()
                }
            except Exception as e:
                logging.warning(f"Elliptic Envelope failed: {e}")
            
            # DBSCAN
            dbscan_labels = self.dbscan.fit_predict(features_scaled)
            results['dbscan'] = {
                'anomalies': (dbscan_labels == -1).tolist(),
                'clusters': dbscan_labels.tolist()
            }
            
            # Combine results
            combined_results = self._combine_anomaly_results(results, len(features))
            
            # Update anomaly history
            self._update_anomaly_history(combined_results, data)
            
            return {
                'individual_results': results,
                'combined_results': combined_results,
                'anomaly_summary': self._generate_anomaly_summary(combined_results)
            }
            
        except Exception as e:
            logging.error(f"Error detecting anomalies: {e}")
            return {}
    
    def _calculate_anomaly_features(self, data: pd.DataFrame) -> np.ndarray:
        """Calculate features for anomaly detection."""
        try:
            features = []
            
            for i in range(20, len(data)):  # Minimum window
                window_data = data.iloc[i-20:i]
                
                # Price-based features
                returns = window_data['close'].pct_change().dropna()
                price_volatility = returns.std()
                price_skewness = returns.skew()
                price_kurtosis = returns.kurtosis()
                
                # Volume-based features
                volume_mean = window_data['volume'].mean()
                volume_std = window_data['volume'].std()
                volume_ratio = window_data['volume'].iloc[-1] / volume_mean if volume_mean > 0 else 1
                
                # Price movement features
                price_change = (window_data['close'].iloc[-1] - window_data['close'].iloc[0]) / window_data['close'].iloc[0]
                high_low_ratio = (window_data['high'].max() - window_data['low'].min()) / window_data['close'].mean()
                
                # Technical indicators
                sma_10 = window_data['close'].rolling(10).mean().iloc[-1]
                sma_20 = window_data['close'].rolling(20).mean().iloc[-1]
                sma_ratio = sma_10 / sma_20 if sma_20 > 0 else 1
                
                # Momentum features
                momentum_5 = (window_data['close'].iloc[-1] - window_data['close'].iloc[-6]) / window_data['close'].iloc[-6]
                momentum_10 = (window_data['close'].iloc[-1] - window_data['close'].iloc[-11]) / window_data['close'].iloc[-11]
                
                feature_vector = [
                    price_volatility, price_skewness, price_kurtosis,
                    volume_ratio, volume_std / volume_mean if volume_mean > 0 else 0,
                    price_change, high_low_ratio, sma_ratio,
                    momentum_5, momentum_10
                ]
                
                features.append(feature_vector)
            
            return np.array(features)
            
        except Exception as e:
            logging.error(f"Error calculating anomaly features: {e}")
            return np.array([])
    
    def _combine_anomaly_results(self, results: Dict[str, Any], n_samples: int) -> Dict[str, Any]:
        """Combine results from multiple anomaly detection methods."""
        try:
            combined = {
                'anomaly_scores': np.zeros(n_samples),
                'anomaly_flags': np.zeros(n_samples, dtype=bool),
                'method_agreement': np.zeros(n_samples)
            }
            
            # Count methods that detected each point as anomalous
            method_count = 0
            for method, result in results.items():
                if 'anomalies' in result:
                    anomalies = np.array(result['anomalies'])
                    combined['anomaly_flags'] += anomalies
                    method_count += 1
                    
                    # Add scores if available
                    if 'scores' in result:
                        scores = np.array(result['scores'])
                        combined['anomaly_scores'] += scores
            
            # Normalize scores
            if method_count > 0:
                combined['anomaly_scores'] /= method_count
                combined['method_agreement'] = combined['anomaly_flags'] / method_count
            
            # Determine final anomaly flags
            # Point is anomalous if majority of methods agree
            combined['final_anomalies'] = combined['method_agreement'] > 0.5
            
            return {
                'anomaly_scores': combined['anomaly_scores'].tolist(),
                'anomaly_flags': combined['anomaly_flags'].tolist(),
                'method_agreement': combined['method_agreement'].tolist(),
                'final_anomalies': combined['final_anomalies'].tolist()
            }
            
        except Exception as e:
            logging.error(f"Error combining anomaly results: {e}")
            return {}
    
    def _update_anomaly_history(self, results: Dict[str, Any], data: pd.DataFrame) -> None:
        """Update anomaly detection history."""
        try:
            timestamp = datetime.now()
            
            anomaly_count = sum(results.get('final_anomalies', []))
            total_points = len(results.get('final_anomalies', []))
            
            history_entry = {
                'timestamp': timestamp,
                'anomaly_count': anomaly_count,
                'total_points': total_points,
                'anomaly_rate': anomaly_count / total_points if total_points > 0 else 0,
                'avg_anomaly_score': np.mean(results.get('anomaly_scores', [0]))
            }
            
            self.anomaly_history.append(history_entry)
            
            # Keep only last 1000 entries
            if len(self.anomaly_history) > 1000:
                self.anomaly_history = self.anomaly_history[-1000:]
                
        except Exception as e:
            logging.error(f"Error updating anomaly history: {e}")
    
    def _generate_anomaly_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of anomaly detection results."""
        try:
            final_anomalies = results.get('final_anomalies', [])
            anomaly_scores = results.get('anomaly_scores', [])
            
            if not final_anomalies:
                return {}
            
            anomaly_count = sum(final_anomalies)
            total_points = len(final_anomalies)
            
            return {
                'total_anomalies': anomaly_count,
                'anomaly_rate': anomaly_count / total_points,
                'avg_anomaly_score': np.mean(anomaly_scores),
                'max_anomaly_score': np.max(anomaly_scores),
                'anomaly_severity': 'high' if anomaly_count / total_points > 0.1 else 'medium' if anomaly_count / total_points > 0.05 else 'low'
            }
            
        except Exception as e:
            logging.error(f"Error generating anomaly summary: {e}")
            return {}
    
    def should_trade_during_anomalies(self, 
                                     anomaly_summary: Dict[str, Any],
                                     risk_tolerance: str = 'medium') -> bool:
        """
        Determine if trading should continue during detected anomalies.
        
        Args:
            anomaly_summary: Summary of anomaly detection results
            risk_tolerance: Risk tolerance level ('low', 'medium', 'high')
            
        Returns:
            Whether trading should continue
        """
        try:
            anomaly_rate = anomaly_summary.get('anomaly_rate', 0)
            severity = anomaly_summary.get('anomaly_severity', 'low')
            
            # Risk tolerance thresholds
            thresholds = {
                'low': 0.02,      # Stop trading if >2% anomalies
                'medium': 0.05,   # Stop trading if >5% anomalies
                'high': 0.10      # Stop trading if >10% anomalies
            }
            
            threshold = thresholds.get(risk_tolerance, 0.05)
            
            # Additional checks
            if severity == 'high' and risk_tolerance != 'high':
                return False
            
            if anomaly_rate > threshold:
                return False
            
            return True
            
        except Exception as e:
            logging.error(f"Error determining trading during anomalies: {e}")
            return False  # Conservative approach

class IntelligenceEnhancer:
    """
    Main intelligence enhancer that coordinates all components.
    """
    
    def __init__(self, 
                 model = None,
                 feature_names: List[str] = None):
        """
        Initialize the intelligence enhancer.
        
        Args:
            model: Trained model
            feature_names: List of feature names
        """
        self.model = model
        self.feature_names = feature_names or []
        
        # Initialize components
        self.regime_detector = MarketRegimeDetector()
        self.explainability = AdvancedExplainability(model, feature_names)
        self.anomaly_detector = AnomalyDetector()
        
        # Dynamic feature importance
        self.feature_importance_history = []
        self.adaptation_history = []
        
        logging.info("🧠 Intelligence Enhancer initialized")
    
    def enhance_intelligence(self, 
                           data: pd.DataFrame,
                           X: np.ndarray = None,
                           y: np.ndarray = None) -> Dict[str, Any]:
        """
        Perform comprehensive intelligence enhancement.
        
        Args:
            data: Market data
            X: Feature matrix (optional)
            y: Target variable (optional)
            
        Returns:
            Enhanced intelligence results
        """
        try:
            results = {}
            
            # Market regime detection
            logging.info("🔍 Detecting market regimes")
            regime_results = self.regime_detector.detect_regimes(data)
            results['market_regimes'] = regime_results
            
            # Anomaly detection
            logging.info("🚨 Detecting anomalies")
            anomaly_results = self.anomaly_detector.detect_anomalies(data)
            results['anomalies'] = anomaly_results
            
            # Model explainability
            if X is not None and self.model is not None:
                logging.info("📊 Generating model explanations")
                explanation_results = self.explainability.explain_model_globally(X)
                results['explanations'] = explanation_results
                
                # Update feature importance
                self._update_feature_importance(explanation_results)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(results)
            results['recommendations'] = recommendations
            
            # Update adaptation history
            self._update_adaptation_history(results)
            
            logging.info("✅ Intelligence enhancement completed")
            return results
            
        except Exception as e:
            logging.error(f"Error in intelligence enhancement: {e}")
            return {}
    
    def _update_feature_importance(self, explanation_results: Dict[str, Any]) -> None:
        """Update feature importance history."""
        try:
            global_importance = explanation_results.get('global_feature_importance', {})
            
            if global_importance:
                timestamp = datetime.now()
                
                importance_entry = {
                    'timestamp': timestamp,
                    'feature_importance': global_importance.copy()
                }
                
                self.feature_importance_history.append(importance_entry)
                
                # Keep only last 100 entries
                if len(self.feature_importance_history) > 100:
                    self.feature_importance_history = self.feature_importance_history[-100:]
                    
        except Exception as e:
            logging.error(f"Error updating feature importance: {e}")
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trading recommendations based on intelligence analysis."""
        try:
            recommendations = {
                'trading_decision': 'hold',
                'confidence': 0.5,
                'risk_level': 'medium',
                'position_size': 1.0,
                'stop_loss': 0.02,
                'take_profit': 0.04,
                'reasoning': []
            }
            
            # Market regime recommendations
            if 'market_regimes' in results:
                regime_results = results['market_regimes']
                current_regime = regime_results.get('current_regime', 0)
                regime_analysis = regime_results.get('regime_analysis', {})
                
                if f'regime_{current_regime}' in regime_analysis:
                    regime_stats = regime_analysis[f'regime_{current_regime}']
                    regime_type = regime_stats.get('type', 'unknown')
                    
                    if regime_type == 'bull_stable':
                        recommendations.update({
                            'trading_decision': 'buy',
                            'confidence': 0.7,
                            'position_size': 1.0
                        })
                        recommendations['reasoning'].append(f"Bullish stable regime detected")
                    
                    elif regime_type == 'bear_stable':
                        recommendations.update({
                            'trading_decision': 'sell',
                            'confidence': 0.6,
                            'position_size': 0.8
                        })
                        recommendations['reasoning'].append(f"Bearish stable regime detected")
                    
                    elif 'volatile' in regime_type:
                        recommendations.update({
                            'position_size': 0.5,
                            'stop_loss': 0.015,
                            'take_profit': 0.03
                        })
                        recommendations['reasoning'].append(f"Volatile regime - reduced position size")
            
            # Anomaly recommendations
            if 'anomalies' in results:
                anomaly_summary = results['anomalies'].get('anomaly_summary', {})
                anomaly_rate = anomaly_summary.get('anomaly_rate', 0)
                
                if anomaly_rate > 0.05:
                    recommendations.update({
                        'trading_decision': 'hold',
                        'confidence': 0.3,
                        'position_size': 0.3
                    })
                    recommendations['reasoning'].append(f"High anomaly rate ({anomaly_rate:.2%}) - trading suspended")
            
            # Feature importance recommendations
            if self.feature_importance_history:
                latest_importance = self.feature_importance_history[-1]['feature_importance']
                
                # Check if important features are changing rapidly
                if len(self.feature_importance_history) > 1:
                    prev_importance = self.feature_importance_history[-2]['feature_importance']
                    
                    # Calculate feature importance stability
                    stability_score = self._calculate_feature_stability(latest_importance, prev_importance)
                    
                    if stability_score < 0.7:
                        recommendations['confidence'] *= 0.8
                        recommendations['reasoning'].append("Unstable feature importance - reduced confidence")
            
            return recommendations
            
        except Exception as e:
            logging.error(f"Error generating recommendations: {e}")
            return {'trading_decision': 'hold', 'confidence': 0.0, 'reasoning': ['Error in recommendation generation']}
    
    def _calculate_feature_stability(self, 
                                   current_importance: Dict[str, Any],
                                   previous_importance: Dict[str, Any]) -> float:
        """Calculate stability of feature importance over time."""
        try:
            common_features = set(current_importance.keys()) & set(previous_importance.keys())
            
            if not common_features:
                return 0.0
            
            stability_scores = []
            for feature in common_features:
                current_rank = current_importance[feature].get('rank', 0)
                previous_rank = previous_importance[feature].get('rank', 0)
                
                # Calculate rank stability
                rank_diff = abs(current_rank - previous_rank)
                max_rank = max(current_rank, previous_rank)
                
                if max_rank > 0:
                    stability = 1 - (rank_diff / max_rank)
                    stability_scores.append(stability)
            
            return np.mean(stability_scores) if stability_scores else 0.0
            
        except Exception as e:
            logging.error(f"Error calculating feature stability: {e}")
            return 0.0
    
    def _update_adaptation_history(self, results: Dict[str, Any]) -> None:
        """Update adaptation history."""
        try:
            timestamp = datetime.now()
            
            adaptation_entry = {
                'timestamp': timestamp,
                'market_regime': results.get('market_regimes', {}).get('current_regime', 0),
                'anomaly_rate': results.get('anomalies', {}).get('anomaly_summary', {}).get('anomaly_rate', 0),
                'recommendations': results.get('recommendations', {})
            }
            
            self.adaptation_history.append(adaptation_entry)
            
            # Keep only last 1000 entries
            if len(self.adaptation_history) > 1000:
                self.adaptation_history = self.adaptation_history[-1000:]
                
        except Exception as e:
            logging.error(f"Error updating adaptation history: {e}")
    
    def get_intelligence_summary(self) -> Dict[str, Any]:
        """Get summary of intelligence enhancement results."""
        try:
            summary = {
                'timestamp': datetime.now().isoformat(),
                'feature_importance_updates': len(self.feature_importance_history),
                'adaptation_events': len(self.adaptation_history),
                'current_market_regime': None,
                'recent_anomaly_rate': 0.0,
                'feature_stability': 0.0
            }
            
            # Get current market regime
            if self.adaptation_history:
                summary['current_market_regime'] = self.adaptation_history[-1]['market_regime']
                summary['recent_anomaly_rate'] = self.adaptation_history[-1]['anomaly_rate']
            
            # Calculate feature stability
            if len(self.feature_importance_history) > 1:
                latest = self.feature_importance_history[-1]['feature_importance']
                previous = self.feature_importance_history[-2]['feature_importance']
                summary['feature_stability'] = self._calculate_feature_stability(latest, previous)
            
            return summary
            
        except Exception as e:
            logging.error(f"Error getting intelligence summary: {e}")
            return {} 