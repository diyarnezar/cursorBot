"""
Performance Anomaly Detection System
Part of Project Hyperion - Ultimate Autonomous Trading Bot

Detects anomalies in:
- Trading performance metrics
- Model prediction accuracy
- Portfolio returns
- Risk metrics
- System behavior patterns
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Anomaly detection libraries
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats

logger = logging.getLogger(__name__)


class PerformanceAnomalyDetector:
    """
    Performance Anomaly Detection System
    
    Features:
    - Statistical anomaly detection
    - Machine learning-based detection
    - Real-time monitoring
    - Multi-metric analysis
    - Trend analysis
    - Alert generation
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.anomaly_history = []
        self.performance_history = []
        self.detection_models = {}
        self.alert_thresholds = {}
        
        # Detection parameters
        self.contamination = config.get('contamination', 0.1)
        self.window_size = config.get('window_size', 100)
        self.confidence_level = config.get('confidence_level', 0.95)
        self.alert_cooldown = config.get('alert_cooldown', 300)  # 5 minutes
        
        # Initialize detection models
        self._initialize_detection_models()
        
        logger.info("Performance Anomaly Detector initialized")

    def _initialize_detection_models(self):
        """Initialize anomaly detection models"""
        try:
            # Isolation Forest for general anomaly detection
            self.detection_models['isolation_forest'] = IsolationForest(
                contamination=self.contamination,
                random_state=42
            )
            
            # Local Outlier Factor for density-based detection
            self.detection_models['lof'] = LocalOutlierFactor(
                contamination=self.contamination,
                n_neighbors=20
            )
            
            # Elliptic Envelope for Gaussian assumption
            self.detection_models['elliptic_envelope'] = EllipticEnvelope(
                contamination=self.contamination,
                random_state=42
            )
            
            # Standard scaler for normalization
            self.detection_models['scaler'] = StandardScaler()
            
            logger.info("Anomaly detection models initialized")
            
        except Exception as e:
            logger.error(f"Error initializing detection models: {e}")

    def detect_performance_anomalies(self, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect anomalies in performance metrics"""
        try:
            logger.info("Detecting performance anomalies")
            
            # Extract performance metrics
            metrics = self._extract_performance_metrics(performance_data)
            
            if not metrics:
                return {'anomalies_detected': False, 'error': 'No metrics available'}
            
            # Convert to DataFrame
            metrics_df = pd.DataFrame([metrics])
            
            # Detect anomalies using multiple methods
            anomaly_results = {}
            
            # Statistical detection
            statistical_anomalies = self._detect_statistical_anomalies(metrics_df)
            anomaly_results['statistical'] = statistical_anomalies
            
            # Machine learning detection
            ml_anomalies = self._detect_ml_anomalies(metrics_df)
            anomaly_results['machine_learning'] = ml_anomalies
            
            # Trend-based detection
            trend_anomalies = self._detect_trend_anomalies(metrics)
            anomaly_results['trend'] = trend_anomalies
            
            # Threshold-based detection
            threshold_anomalies = self._detect_threshold_anomalies(metrics)
            anomaly_results['threshold'] = threshold_anomalies
            
            # Combine results
            combined_anomalies = self._combine_anomaly_results(anomaly_results)
            
            # Generate alerts if anomalies detected
            alerts = self._generate_anomaly_alerts(combined_anomalies, metrics)
            
            # Store results
            result = {
                'timestamp': datetime.now().isoformat(),
                'anomalies_detected': len(combined_anomalies) > 0,
                'anomaly_results': anomaly_results,
                'combined_anomalies': combined_anomalies,
                'alerts': alerts,
                'performance_metrics': metrics
            }
            
            self.anomaly_history.append(result)
            
            logger.info(f"Performance anomaly detection completed. Anomalies: {len(combined_anomalies)}")
            return result
            
        except Exception as e:
            logger.error(f"Error in performance anomaly detection: {e}")
            return {'anomalies_detected': False, 'error': str(e)}

    def _extract_performance_metrics(self, performance_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract relevant performance metrics for anomaly detection"""
        try:
            metrics = {}
            
            # Trading performance metrics
            if 'trading_performance' in performance_data:
                trading = performance_data['trading_performance']
                metrics.update({
                    'total_return': trading.get('total_return', 0.0),
                    'sharpe_ratio': trading.get('sharpe_ratio', 0.0),
                    'max_drawdown': trading.get('max_drawdown', 0.0),
                    'win_rate': trading.get('win_rate', 0.0),
                    'profit_factor': trading.get('profit_factor', 0.0),
                    'avg_trade': trading.get('avg_trade', 0.0),
                    'trade_count': trading.get('trade_count', 0)
                })
            
            # Model performance metrics
            if 'model_performance' in performance_data:
                model = performance_data['model_performance']
                metrics.update({
                    'model_accuracy': model.get('accuracy', 0.0),
                    'model_precision': model.get('precision', 0.0),
                    'model_recall': model.get('recall', 0.0),
                    'model_f1_score': model.get('f1_score', 0.0),
                    'prediction_error': model.get('prediction_error', 0.0)
                })
            
            # Risk metrics
            if 'risk_metrics' in performance_data:
                risk = performance_data['risk_metrics']
                metrics.update({
                    'portfolio_volatility': risk.get('portfolio_volatility', 0.0),
                    'var_95': risk.get('var_95', 0.0),
                    'cvar_95': risk.get('cvar_95', 0.0),
                    'beta': risk.get('beta', 0.0),
                    'correlation': risk.get('correlation', 0.0)
                })
            
            # System metrics
            if 'system_metrics' in performance_data:
                system = performance_data['system_metrics']
                metrics.update({
                    'cpu_usage': system.get('cpu_usage', 0.0),
                    'memory_usage': system.get('memory_usage', 0.0),
                    'latency': system.get('latency', 0.0),
                    'error_rate': system.get('error_rate', 0.0)
                })
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error extracting performance metrics: {e}")
            return {}

    def _detect_statistical_anomalies(self, metrics_df: pd.DataFrame) -> Dict[str, Any]:
        """Detect anomalies using statistical methods"""
        try:
            anomalies = {}
            
            for column in metrics_df.columns:
                values = metrics_df[column].values
                
                if len(values) == 0:
                    continue
                
                # Z-score method
                z_scores = np.abs(stats.zscore(values))
                z_score_anomalies = z_scores > 3  # 3 standard deviations
                
                # IQR method
                Q1 = np.percentile(values, 25)
                Q3 = np.percentile(values, 75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                iqr_anomalies = (values < lower_bound) | (values > upper_bound)
                
                # Modified Z-score method (robust to outliers)
                median = np.median(values)
                mad = np.median(np.abs(values - median))
                modified_z_scores = 0.6745 * (values - median) / mad
                modified_z_anomalies = np.abs(modified_z_scores) > 3.5
                
                anomalies[column] = {
                    'z_score_anomalies': z_score_anomalies.tolist(),
                    'iqr_anomalies': iqr_anomalies.tolist(),
                    'modified_z_anomalies': modified_z_anomalies.tolist(),
                    'z_scores': z_scores.tolist(),
                    'modified_z_scores': modified_z_scores.tolist(),
                    'iqr_bounds': [lower_bound, upper_bound]
                }
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Error in statistical anomaly detection: {e}")
            return {}

    def _detect_ml_anomalies(self, metrics_df: pd.DataFrame) -> Dict[str, Any]:
        """Detect anomalies using machine learning methods"""
        try:
            anomalies = {}
            
            # Prepare data
            X = metrics_df.values
            
            if X.shape[0] == 0 or X.shape[1] == 0:
                return anomalies
            
            # Scale the data
            X_scaled = self.detection_models['scaler'].fit_transform(X)
            
            # Isolation Forest
            try:
                iso_forest = self.detection_models['isolation_forest']
                iso_forest.fit(X_scaled)
                iso_predictions = iso_forest.predict(X_scaled)
                iso_anomalies = iso_predictions == -1
                anomalies['isolation_forest'] = {
                    'anomalies': iso_anomalies.tolist(),
                    'scores': iso_forest.decision_function(X_scaled).tolist()
                }
            except Exception as e:
                logger.warning(f"Isolation Forest failed: {e}")
            
            # Local Outlier Factor
            try:
                lof = self.detection_models['lof']
                lof.fit(X_scaled)
                lof_predictions = lof.predict(X_scaled)
                lof_anomalies = lof_predictions == -1
                anomalies['local_outlier_factor'] = {
                    'anomalies': lof_anomalies.tolist(),
                    'scores': lof.negative_outlier_factor_.tolist()
                }
            except Exception as e:
                logger.warning(f"Local Outlier Factor failed: {e}")
            
            # Elliptic Envelope
            try:
                elliptic = self.detection_models['elliptic_envelope']
                elliptic.fit(X_scaled)
                elliptic_predictions = elliptic.predict(X_scaled)
                elliptic_anomalies = elliptic_predictions == -1
                anomalies['elliptic_envelope'] = {
                    'anomalies': elliptic_anomalies.tolist(),
                    'scores': elliptic.decision_function(X_scaled).tolist()
                }
            except Exception as e:
                logger.warning(f"Elliptic Envelope failed: {e}")
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Error in ML anomaly detection: {e}")
            return {}

    def _detect_trend_anomalies(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Detect anomalies based on trend analysis"""
        try:
            anomalies = {}
            
            # Add current metrics to history
            self.performance_history.append({
                'timestamp': datetime.now(),
                'metrics': metrics
            })
            
            # Keep only recent history
            cutoff_time = datetime.now() - timedelta(hours=24)
            self.performance_history = [
                entry for entry in self.performance_history
                if entry['timestamp'] >= cutoff_time
            ]
            
            if len(self.performance_history) < 10:
                return anomalies
            
            # Analyze trends for each metric
            for metric_name, current_value in metrics.items():
                try:
                    # Get historical values
                    historical_values = [
                        entry['metrics'].get(metric_name, 0.0)
                        for entry in self.performance_history
                    ]
                    
                    if len(historical_values) < 5:
                        continue
                    
                    # Calculate trend
                    x = np.arange(len(historical_values))
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x, historical_values)
                    
                    # Calculate expected value based on trend
                    expected_value = slope * (len(historical_values) - 1) + intercept
                    
                    # Calculate deviation from trend
                    deviation = abs(current_value - expected_value)
                    
                    # Calculate trend strength
                    trend_strength = abs(slope) / np.std(historical_values) if np.std(historical_values) > 0 else 0
                    
                    # Detect trend anomalies
                    trend_anomaly = False
                    if trend_strength > 0.1:  # Significant trend
                        # Check if current value deviates significantly from trend
                        if deviation > 2 * np.std(historical_values):
                            trend_anomaly = True
                    
                    anomalies[metric_name] = {
                        'trend_anomaly': trend_anomaly,
                        'slope': slope,
                        'trend_strength': trend_strength,
                        'deviation': deviation,
                        'expected_value': expected_value,
                        'current_value': current_value,
                        'r_squared': r_value ** 2
                    }
                    
                except Exception as e:
                    logger.warning(f"Error analyzing trend for {metric_name}: {e}")
                    continue
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Error in trend anomaly detection: {e}")
            return {}

    def _detect_threshold_anomalies(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Detect anomalies based on predefined thresholds"""
        try:
            anomalies = {}
            
            # Define thresholds for different metrics
            thresholds = {
                'total_return': {'min': -0.1, 'max': 0.2},  # -10% to +20%
                'sharpe_ratio': {'min': -2.0, 'max': 5.0},
                'max_drawdown': {'min': 0.0, 'max': 0.3},  # 0% to 30%
                'win_rate': {'min': 0.3, 'max': 0.9},  # 30% to 90%
                'profit_factor': {'min': 0.5, 'max': 5.0},
                'model_accuracy': {'min': 0.4, 'max': 0.95},  # 40% to 95%
                'portfolio_volatility': {'min': 0.05, 'max': 0.5},  # 5% to 50%
                'var_95': {'min': -0.05, 'max': -0.001},  # -5% to -0.1%
                'cpu_usage': {'min': 0.0, 'max': 0.9},  # 0% to 90%
                'memory_usage': {'min': 0.0, 'max': 0.9},  # 0% to 90%
                'latency': {'min': 0.0, 'max': 1000},  # 0 to 1000ms
                'error_rate': {'min': 0.0, 'max': 0.1}  # 0% to 10%
            }
            
            for metric_name, current_value in metrics.items():
                if metric_name in thresholds:
                    threshold = thresholds[metric_name]
                    
                    # Check if value is outside thresholds
                    below_min = current_value < threshold['min']
                    above_max = current_value > threshold['max']
                    
                    if below_min or above_max:
                        anomalies[metric_name] = {
                            'threshold_anomaly': True,
                            'current_value': current_value,
                            'threshold_min': threshold['min'],
                            'threshold_max': threshold['max'],
                            'below_min': below_min,
                            'above_max': above_max,
                            'severity': 'high' if (below_min and current_value < threshold['min'] * 0.5) or 
                                       (above_max and current_value > threshold['max'] * 1.5) else 'medium'
                        }
                    else:
                        anomalies[metric_name] = {
                            'threshold_anomaly': False,
                            'current_value': current_value,
                            'threshold_min': threshold['min'],
                            'threshold_max': threshold['max']
                        }
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Error in threshold anomaly detection: {e}")
            return {}

    def _combine_anomaly_results(self, anomaly_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Combine results from different anomaly detection methods"""
        try:
            combined_anomalies = []
            
            # Get all unique metrics
            all_metrics = set()
            for method_results in anomaly_results.values():
                if isinstance(method_results, dict):
                    all_metrics.update(method_results.keys())
            
            # Combine results for each metric
            for metric in all_metrics:
                anomaly_info = {
                    'metric': metric,
                    'detection_methods': {},
                    'overall_anomaly_score': 0.0,
                    'is_anomaly': False
                }
                
                # Collect results from each method
                for method_name, method_results in anomaly_results.items():
                    if metric in method_results:
                        anomaly_info['detection_methods'][method_name] = method_results[metric]
                        
                        # Calculate anomaly score for this method
                        method_score = self._calculate_method_anomaly_score(method_results[metric])
                        anomaly_info['overall_anomaly_score'] += method_score
                
                # Determine if overall anomaly
                if anomaly_info['overall_anomaly_score'] >= 0.5:  # Threshold for overall anomaly
                    anomaly_info['is_anomaly'] = True
                    combined_anomalies.append(anomaly_info)
            
            return combined_anomalies
            
        except Exception as e:
            logger.error(f"Error combining anomaly results: {e}")
            return []

    def _calculate_method_anomaly_score(self, method_result: Dict[str, Any]) -> float:
        """Calculate anomaly score for a specific detection method"""
        try:
            score = 0.0
            
            # Statistical methods
            if 'z_score_anomalies' in method_result:
                if any(method_result['z_score_anomalies']):
                    score += 0.3
            
            if 'iqr_anomalies' in method_result:
                if any(method_result['iqr_anomalies']):
                    score += 0.3
            
            if 'modified_z_anomalies' in method_result:
                if any(method_result['modified_z_anomalies']):
                    score += 0.3
            
            # ML methods
            if 'anomalies' in method_result:
                if any(method_result['anomalies']):
                    score += 0.4
            
            # Trend methods
            if 'trend_anomaly' in method_result:
                if method_result['trend_anomaly']:
                    score += 0.3
            
            # Threshold methods
            if 'threshold_anomaly' in method_result:
                if method_result['threshold_anomaly']:
                    score += 0.4
                    if method_result.get('severity') == 'high':
                        score += 0.2
            
            return min(score, 1.0)  # Cap at 1.0
            
        except Exception as e:
            logger.error(f"Error calculating method anomaly score: {e}")
            return 0.0

    def _generate_anomaly_alerts(self, anomalies: List[Dict[str, Any]], 
                               metrics: Dict[str, float]) -> List[Dict[str, Any]]:
        """Generate alerts for detected anomalies"""
        try:
            alerts = []
            
            for anomaly in anomalies:
                if anomaly['is_anomaly']:
                    alert = {
                        'timestamp': datetime.now().isoformat(),
                        'alert_type': 'performance_anomaly',
                        'metric': anomaly['metric'],
                        'current_value': metrics.get(anomaly['metric'], 0.0),
                        'anomaly_score': anomaly['overall_anomaly_score'],
                        'detection_methods': list(anomaly['detection_methods'].keys()),
                        'severity': 'high' if anomaly['overall_anomaly_score'] > 0.7 else 'medium',
                        'message': f"Anomaly detected in {anomaly['metric']} with score {anomaly['overall_anomaly_score']:.3f}",
                        'recommendations': self._generate_recommendations(anomaly)
                    }
                    
                    alerts.append(alert)
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error generating anomaly alerts: {e}")
            return []

    def _generate_recommendations(self, anomaly: Dict[str, Any]) -> List[str]:
        """Generate recommendations for anomaly resolution"""
        try:
            recommendations = []
            metric = anomaly['metric']
            
            # General recommendations
            recommendations.append(f"Investigate the root cause of the anomaly in {metric}")
            recommendations.append("Review recent system changes or market conditions")
            
            # Metric-specific recommendations
            if 'return' in metric.lower():
                recommendations.append("Review trading strategy and risk management")
                recommendations.append("Check for unusual market conditions")
            
            elif 'model' in metric.lower():
                recommendations.append("Retrain or update the model")
                recommendations.append("Check data quality and feature engineering")
            
            elif 'risk' in metric.lower():
                recommendations.append("Review position sizing and risk limits")
                recommendations.append("Check correlation and diversification")
            
            elif 'system' in metric.lower() or 'cpu' in metric.lower() or 'memory' in metric.lower():
                recommendations.append("Check system resources and performance")
                recommendations.append("Review system logs for errors")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return ["Investigate the anomaly manually"]

    def get_anomaly_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get summary of recent anomalies"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            recent_anomalies = [
                entry for entry in self.anomaly_history
                if datetime.fromisoformat(entry['timestamp']) >= cutoff_time
            ]
            
            if not recent_anomalies:
                return {'total_anomalies': 0}
            
            # Calculate summary statistics
            total_anomalies = sum(1 for entry in recent_anomalies if entry['anomalies_detected'])
            
            # Most common anomalous metrics
            metric_counts = {}
            for entry in recent_anomalies:
                if entry['anomalies_detected']:
                    for anomaly in entry['combined_anomalies']:
                        metric = anomaly['metric']
                        metric_counts[metric] = metric_counts.get(metric, 0) + 1
            
            most_common_metrics = sorted(metric_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            
            # Average anomaly scores
            anomaly_scores = [
                anomaly['overall_anomaly_score']
                for entry in recent_anomalies
                for anomaly in entry.get('combined_anomalies', [])
            ]
            
            return {
                'total_anomalies': total_anomalies,
                'total_checks': len(recent_anomalies),
                'anomaly_rate': total_anomalies / len(recent_anomalies) if recent_anomalies else 0,
                'most_common_metrics': most_common_metrics,
                'average_anomaly_score': np.mean(anomaly_scores) if anomaly_scores else 0,
                'max_anomaly_score': np.max(anomaly_scores) if anomaly_scores else 0,
                'last_anomaly': recent_anomalies[-1]['timestamp'] if recent_anomalies else None
            }
            
        except Exception as e:
            logger.error(f"Error getting anomaly summary: {e}")
            return {'total_anomalies': 0}


# Example usage
if __name__ == "__main__":
    config = {
        'contamination': 0.1,
        'window_size': 100,
        'confidence_level': 0.95,
        'alert_cooldown': 300
    }
    
    detector = PerformanceAnomalyDetector(config)
    
    # Sample performance data
    performance_data = {
        'trading_performance': {
            'total_return': 0.15,
            'sharpe_ratio': 2.1,
            'max_drawdown': 0.08,
            'win_rate': 0.65,
            'profit_factor': 1.8,
            'avg_trade': 0.02,
            'trade_count': 50
        },
        'model_performance': {
            'accuracy': 0.72,
            'precision': 0.68,
            'recall': 0.75,
            'f1_score': 0.71,
            'prediction_error': 0.15
        },
        'risk_metrics': {
            'portfolio_volatility': 0.12,
            'var_95': -0.03,
            'cvar_95': -0.05,
            'beta': 0.85,
            'correlation': 0.45
        },
        'system_metrics': {
            'cpu_usage': 0.45,
            'memory_usage': 0.60,
            'latency': 150,
            'error_rate': 0.02
        }
    }
    
    # Detect anomalies
    anomalies = detector.detect_performance_anomalies(performance_data)
    
    print("Anomalies detected:", anomalies['anomalies_detected'])
    print("Number of anomalies:", len(anomalies['combined_anomalies']))
    print("Alerts generated:", len(anomalies['alerts'])) 