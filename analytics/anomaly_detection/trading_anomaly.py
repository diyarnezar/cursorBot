"""
Trading Anomaly Detection System
Part of Project Hyperion - Ultimate Autonomous Trading Bot

Detects anomalies in:
- Trading patterns and behavior
- Order flow and execution
- Market microstructure
- Price movements
- Volume patterns
- Execution quality
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
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats

logger = logging.getLogger(__name__)


class TradingAnomalyDetector:
    """
    Trading Anomaly Detection System
    
    Features:
    - Trading pattern analysis
    - Order flow monitoring
    - Market microstructure analysis
    - Execution quality assessment
    - Real-time anomaly detection
    - Multi-timeframe analysis
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.trading_history = []
        self.anomaly_history = []
        self.detection_models = {}
        self.baseline_metrics = {}
        
        # Detection parameters
        self.contamination = config.get('contamination', 0.05)
        self.lookback_window = config.get('lookback_window', 1000)
        self.confidence_level = config.get('confidence_level', 0.99)
        self.alert_threshold = config.get('alert_threshold', 0.7)
        
        # Trading-specific parameters
        self.min_trade_size = config.get('min_trade_size', 0.001)
        self.max_slippage = config.get('max_slippage', 0.002)
        self.max_spread = config.get('max_spread', 0.001)
        
        # Initialize detection models
        self._initialize_detection_models()
        
        logger.info("Trading Anomaly Detector initialized")

    def _initialize_detection_models(self):
        """Initialize anomaly detection models"""
        try:
            # Isolation Forest for general trading anomalies
            self.detection_models['isolation_forest'] = IsolationForest(
                contamination=self.contamination,
                random_state=42
            )
            
            # Local Outlier Factor for density-based detection
            self.detection_models['lof'] = LocalOutlierFactor(
                contamination=self.contamination,
                n_neighbors=20
            )
            
            # Standard scaler for normalization
            self.detection_models['scaler'] = StandardScaler()
            
            logger.info("Trading anomaly detection models initialized")
            
        except Exception as e:
            logger.error(f"Error initializing trading detection models: {e}")

    def detect_trading_anomalies(self, trading_data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect anomalies in trading data"""
        try:
            logger.info("Detecting trading anomalies")
            
            # Extract trading metrics
            metrics = self._extract_trading_metrics(trading_data)
            
            if not metrics:
                return {'anomalies_detected': False, 'error': 'No trading data available'}
            
            # Add to trading history
            self.trading_history.append({
                'timestamp': datetime.now(),
                'metrics': metrics
            })
            
            # Keep only recent history
            self._clean_trading_history()
            
            # Detect different types of anomalies
            anomaly_results = {}
            
            # Pattern anomalies
            pattern_anomalies = self._detect_pattern_anomalies(metrics)
            anomaly_results['pattern'] = pattern_anomalies
            
            # Volume anomalies
            volume_anomalies = self._detect_volume_anomalies(metrics)
            anomaly_results['volume'] = volume_anomalies
            
            # Price anomalies
            price_anomalies = self._detect_price_anomalies(metrics)
            anomaly_results['price'] = price_anomalies
            
            # Execution anomalies
            execution_anomalies = self._detect_execution_anomalies(metrics)
            anomaly_results['execution'] = execution_anomalies
            
            # Microstructure anomalies
            microstructure_anomalies = self._detect_microstructure_anomalies(metrics)
            anomaly_results['microstructure'] = microstructure_anomalies
            
            # ML-based anomalies
            ml_anomalies = self._detect_ml_trading_anomalies(metrics)
            anomaly_results['machine_learning'] = ml_anomalies
            
            # Combine results
            combined_anomalies = self._combine_trading_anomalies(anomaly_results)
            
            # Generate alerts
            alerts = self._generate_trading_alerts(combined_anomalies, metrics)
            
            # Store results
            result = {
                'timestamp': datetime.now().isoformat(),
                'anomalies_detected': len(combined_anomalies) > 0,
                'anomaly_results': anomaly_results,
                'combined_anomalies': combined_anomalies,
                'alerts': alerts,
                'trading_metrics': metrics
            }
            
            self.anomaly_history.append(result)
            
            logger.info(f"Trading anomaly detection completed. Anomalies: {len(combined_anomalies)}")
            return result
            
        except Exception as e:
            logger.error(f"Error in trading anomaly detection: {e}")
            return {'anomalies_detected': False, 'error': str(e)}

    def _extract_trading_metrics(self, trading_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract relevant trading metrics for anomaly detection"""
        try:
            metrics = {}
            
            # Trade execution metrics
            if 'trade_execution' in trading_data:
                execution = trading_data['trade_execution']
                metrics.update({
                    'trade_size': execution.get('trade_size', 0.0),
                    'execution_price': execution.get('execution_price', 0.0),
                    'slippage': execution.get('slippage', 0.0),
                    'spread': execution.get('spread', 0.0),
                    'execution_time': execution.get('execution_time', 0.0),
                    'fill_ratio': execution.get('fill_ratio', 0.0),
                    'order_type': execution.get('order_type', 'market')
                })
            
            # Market data metrics
            if 'market_data' in trading_data:
                market = trading_data['market_data']
                metrics.update({
                    'price': market.get('price', 0.0),
                    'volume': market.get('volume', 0.0),
                    'bid': market.get('bid', 0.0),
                    'ask': market.get('ask', 0.0),
                    'bid_size': market.get('bid_size', 0.0),
                    'ask_size': market.get('ask_size', 0.0),
                    'price_change': market.get('price_change', 0.0),
                    'volume_change': market.get('volume_change', 0.0)
                })
            
            # Order book metrics
            if 'order_book' in trading_data:
                orderbook = trading_data['order_book']
                metrics.update({
                    'order_book_imbalance': orderbook.get('imbalance', 0.0),
                    'order_book_depth': orderbook.get('depth', 0.0),
                    'order_book_spread': orderbook.get('spread', 0.0),
                    'order_book_volume': orderbook.get('total_volume', 0.0)
                })
            
            # Trading pattern metrics
            if 'trading_patterns' in trading_data:
                patterns = trading_data['trading_patterns']
                metrics.update({
                    'trade_frequency': patterns.get('trade_frequency', 0.0),
                    'trade_direction': patterns.get('trade_direction', 0.0),
                    'trade_clustering': patterns.get('trade_clustering', 0.0),
                    'momentum': patterns.get('momentum', 0.0),
                    'mean_reversion': patterns.get('mean_reversion', 0.0)
                })
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error extracting trading metrics: {e}")
            return {}

    def _detect_pattern_anomalies(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Detect anomalies in trading patterns"""
        try:
            anomalies = {}
            
            # Trade frequency anomalies
            if 'trade_frequency' in metrics:
                frequency = metrics['trade_frequency']
                
                # Check for unusual trade frequency
                if frequency > 100:  # Very high frequency
                    anomalies['high_frequency'] = {
                        'anomaly': True,
                        'value': frequency,
                        'threshold': 100,
                        'severity': 'high'
                    }
                elif frequency < 0.1:  # Very low frequency
                    anomalies['low_frequency'] = {
                        'anomaly': True,
                        'value': frequency,
                        'threshold': 0.1,
                        'severity': 'medium'
                    }
            
            # Trade direction anomalies
            if 'trade_direction' in metrics:
                direction = metrics['trade_direction']
                
                # Check for extreme directional bias
                if abs(direction) > 0.9:  # Very strong directional bias
                    anomalies['directional_bias'] = {
                        'anomaly': True,
                        'value': direction,
                        'threshold': 0.9,
                        'severity': 'medium'
                    }
            
            # Trade clustering anomalies
            if 'trade_clustering' in metrics:
                clustering = metrics['trade_clustering']
                
                # Check for unusual trade clustering
                if clustering > 0.8:  # Very high clustering
                    anomalies['trade_clustering'] = {
                        'anomaly': True,
                        'value': clustering,
                        'threshold': 0.8,
                        'severity': 'medium'
                    }
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Error in pattern anomaly detection: {e}")
            return {}

    def _detect_volume_anomalies(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Detect anomalies in trading volume"""
        try:
            anomalies = {}
            
            # Volume anomalies
            if 'volume' in metrics:
                volume = metrics['volume']
                
                # Calculate volume statistics from history
                if len(self.trading_history) > 10:
                    historical_volumes = [
                        entry['metrics'].get('volume', 0.0)
                        for entry in self.trading_history[-100:]  # Last 100 entries
                    ]
                    
                    volume_mean = np.mean(historical_volumes)
                    volume_std = np.std(historical_volumes)
                    
                    if volume_std > 0:
                        volume_z_score = abs(volume - volume_mean) / volume_std
                        
                        if volume_z_score > 3:  # 3 standard deviations
                            anomalies['volume_spike'] = {
                                'anomaly': True,
                                'value': volume,
                                'z_score': volume_z_score,
                                'mean': volume_mean,
                                'std': volume_std,
                                'severity': 'high' if volume_z_score > 5 else 'medium'
                            }
            
            # Volume change anomalies
            if 'volume_change' in metrics:
                volume_change = metrics['volume_change']
                
                if abs(volume_change) > 2.0:  # 200% change
                    anomalies['volume_change'] = {
                        'anomaly': True,
                        'value': volume_change,
                        'threshold': 2.0,
                        'severity': 'high' if abs(volume_change) > 5.0 else 'medium'
                    }
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Error in volume anomaly detection: {e}")
            return {}

    def _detect_price_anomalies(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Detect anomalies in price movements"""
        try:
            anomalies = {}
            
            # Price change anomalies
            if 'price_change' in metrics:
                price_change = metrics['price_change']
                
                # Check for extreme price movements
                if abs(price_change) > 0.1:  # 10% price change
                    anomalies['price_movement'] = {
                        'anomaly': True,
                        'value': price_change,
                        'threshold': 0.1,
                        'severity': 'high' if abs(price_change) > 0.2 else 'medium'
                    }
            
            # Price level anomalies
            if 'price' in metrics and len(self.trading_history) > 10:
                current_price = metrics['price']
                historical_prices = [
                    entry['metrics'].get('price', 0.0)
                    for entry in self.trading_history[-100:]
                ]
                
                if historical_prices:
                    price_mean = np.mean(historical_prices)
                    price_std = np.std(historical_prices)
                    
                    if price_std > 0:
                        price_z_score = abs(current_price - price_mean) / price_std
                        
                        if price_z_score > 3:
                            anomalies['price_level'] = {
                                'anomaly': True,
                                'value': current_price,
                                'z_score': price_z_score,
                                'mean': price_mean,
                                'std': price_std,
                                'severity': 'high' if price_z_score > 5 else 'medium'
                            }
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Error in price anomaly detection: {e}")
            return {}

    def _detect_execution_anomalies(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Detect anomalies in trade execution"""
        try:
            anomalies = {}
            
            # Slippage anomalies
            if 'slippage' in metrics:
                slippage = abs(metrics['slippage'])
                
                if slippage > self.max_slippage:
                    anomalies['high_slippage'] = {
                        'anomaly': True,
                        'value': slippage,
                        'threshold': self.max_slippage,
                        'severity': 'high' if slippage > self.max_slippage * 2 else 'medium'
                    }
            
            # Spread anomalies
            if 'spread' in metrics:
                spread = metrics['spread']
                
                if spread > self.max_spread:
                    anomalies['wide_spread'] = {
                        'anomaly': True,
                        'value': spread,
                        'threshold': self.max_spread,
                        'severity': 'high' if spread > self.max_spread * 2 else 'medium'
                    }
            
            # Fill ratio anomalies
            if 'fill_ratio' in metrics:
                fill_ratio = metrics['fill_ratio']
                
                if fill_ratio < 0.5:  # Less than 50% fill
                    anomalies['low_fill_ratio'] = {
                        'anomaly': True,
                        'value': fill_ratio,
                        'threshold': 0.5,
                        'severity': 'medium'
                    }
            
            # Execution time anomalies
            if 'execution_time' in metrics:
                execution_time = metrics['execution_time']
                
                if execution_time > 1000:  # More than 1 second
                    anomalies['slow_execution'] = {
                        'anomaly': True,
                        'value': execution_time,
                        'threshold': 1000,
                        'severity': 'medium'
                    }
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Error in execution anomaly detection: {e}")
            return {}

    def _detect_microstructure_anomalies(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Detect anomalies in market microstructure"""
        try:
            anomalies = {}
            
            # Order book imbalance anomalies
            if 'order_book_imbalance' in metrics:
                imbalance = abs(metrics['order_book_imbalance'])
                
                if imbalance > 0.8:  # Very high imbalance
                    anomalies['order_book_imbalance'] = {
                        'anomaly': True,
                        'value': imbalance,
                        'threshold': 0.8,
                        'severity': 'medium'
                    }
            
            # Order book depth anomalies
            if 'order_book_depth' in metrics:
                depth = metrics['order_book_depth']
                
                if depth < 1000:  # Low depth
                    anomalies['low_depth'] = {
                        'anomaly': True,
                        'value': depth,
                        'threshold': 1000,
                        'severity': 'medium'
                    }
            
            # Bid-ask spread anomalies
            if 'order_book_spread' in metrics:
                spread = metrics['order_book_spread']
                
                if spread > 0.002:  # Wide spread
                    anomalies['wide_order_book_spread'] = {
                        'anomaly': True,
                        'value': spread,
                        'threshold': 0.002,
                        'severity': 'medium'
                    }
            
            # Bid/ask size anomalies
            if 'bid_size' in metrics and 'ask_size' in metrics:
                bid_size = metrics['bid_size']
                ask_size = metrics['ask_size']
                
                if bid_size > 0 and ask_size > 0:
                    size_ratio = bid_size / ask_size
                    
                    if size_ratio > 5 or size_ratio < 0.2:  # Extreme size imbalance
                        anomalies['size_imbalance'] = {
                            'anomaly': True,
                            'value': size_ratio,
                            'threshold': [0.2, 5.0],
                            'severity': 'medium'
                        }
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Error in microstructure anomaly detection: {e}")
            return {}

    def _detect_ml_trading_anomalies(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Detect anomalies using machine learning methods"""
        try:
            anomalies = {}
            
            if len(self.trading_history) < 10:
                return anomalies
            
            # Prepare feature vector
            feature_vector = []
            feature_names = []
            
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    feature_vector.append(value)
                    feature_names.append(key)
            
            if not feature_vector:
                return anomalies
            
            # Convert to numpy array
            X = np.array([feature_vector])
            
            # Scale features
            try:
                X_scaled = self.detection_models['scaler'].fit_transform(X)
            except:
                # If scaler fails, use original data
                X_scaled = X
            
            # Isolation Forest
            try:
                iso_forest = self.detection_models['isolation_forest']
                iso_forest.fit(X_scaled)
                iso_score = iso_forest.decision_function(X_scaled)[0]
                iso_anomaly = iso_score < -0.5  # Threshold for anomaly
                
                if iso_anomaly:
                    anomalies['isolation_forest'] = {
                        'anomaly': True,
                        'score': iso_score,
                        'severity': 'high' if iso_score < -1.0 else 'medium'
                    }
            except Exception as e:
                logger.warning(f"Isolation Forest failed: {e}")
            
            # Local Outlier Factor
            try:
                lof = self.detection_models['lof']
                lof.fit(X_scaled)
                lof_score = lof.negative_outlier_factor_[0]
                lof_anomaly = lof_score < -0.5
                
                if lof_anomaly:
                    anomalies['local_outlier_factor'] = {
                        'anomaly': True,
                        'score': lof_score,
                        'severity': 'high' if lof_score < -1.0 else 'medium'
                    }
            except Exception as e:
                logger.warning(f"Local Outlier Factor failed: {e}")
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Error in ML trading anomaly detection: {e}")
            return {}

    def _combine_trading_anomalies(self, anomaly_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Combine results from different trading anomaly detection methods"""
        try:
            combined_anomalies = []
            
            # Get all unique anomaly types
            all_anomaly_types = set()
            for method_results in anomaly_results.values():
                if isinstance(method_results, dict):
                    all_anomaly_types.update(method_results.keys())
            
            # Combine results for each anomaly type
            for anomaly_type in all_anomaly_types:
                anomaly_info = {
                    'anomaly_type': anomaly_type,
                    'detection_methods': {},
                    'overall_severity': 'low',
                    'is_anomaly': False
                }
                
                # Collect results from each method
                for method_name, method_results in anomaly_results.items():
                    if anomaly_type in method_results:
                        anomaly_info['detection_methods'][method_name] = method_results[anomaly_type]
                        
                        # Update overall severity
                        method_severity = method_results[anomaly_type].get('severity', 'low')
                        if method_severity == 'high':
                            anomaly_info['overall_severity'] = 'high'
                        elif method_severity == 'medium' and anomaly_info['overall_severity'] != 'high':
                            anomaly_info['overall_severity'] = 'medium'
                
                # Determine if overall anomaly
                if len(anomaly_info['detection_methods']) > 0:
                    anomaly_info['is_anomaly'] = True
                    combined_anomalies.append(anomaly_info)
            
            return combined_anomalies
            
        except Exception as e:
            logger.error(f"Error combining trading anomalies: {e}")
            return []

    def _generate_trading_alerts(self, anomalies: List[Dict[str, Any]], 
                               metrics: Dict[str, float]) -> List[Dict[str, Any]]:
        """Generate alerts for detected trading anomalies"""
        try:
            alerts = []
            
            for anomaly in anomalies:
                if anomaly['is_anomaly']:
                    alert = {
                        'timestamp': datetime.now().isoformat(),
                        'alert_type': 'trading_anomaly',
                        'anomaly_type': anomaly['anomaly_type'],
                        'severity': anomaly['overall_severity'],
                        'detection_methods': list(anomaly['detection_methods'].keys()),
                        'message': f"Trading anomaly detected: {anomaly['anomaly_type']}",
                        'recommendations': self._generate_trading_recommendations(anomaly)
                    }
                    
                    alerts.append(alert)
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error generating trading alerts: {e}")
            return []

    def _generate_trading_recommendations(self, anomaly: Dict[str, Any]) -> List[str]:
        """Generate recommendations for trading anomaly resolution"""
        try:
            recommendations = []
            anomaly_type = anomaly['anomaly_type']
            
            # General recommendations
            recommendations.append(f"Investigate the root cause of {anomaly_type}")
            recommendations.append("Review recent market conditions and news")
            
            # Specific recommendations based on anomaly type
            if 'slippage' in anomaly_type.lower():
                recommendations.append("Check market liquidity and order book depth")
                recommendations.append("Consider using limit orders instead of market orders")
                recommendations.append("Review position sizing")
            
            elif 'volume' in anomaly_type.lower():
                recommendations.append("Check for news events or market announcements")
                recommendations.append("Review trading strategy for volume sensitivity")
                recommendations.append("Monitor for potential market manipulation")
            
            elif 'price' in anomaly_type.lower():
                recommendations.append("Check for fundamental news or events")
                recommendations.append("Review technical analysis indicators")
                recommendations.append("Consider adjusting stop-loss levels")
            
            elif 'execution' in anomaly_type.lower():
                recommendations.append("Check system connectivity and latency")
                recommendations.append("Review order routing and execution venue")
                recommendations.append("Monitor for exchange issues")
            
            elif 'microstructure' in anomaly_type.lower():
                recommendations.append("Check order book for unusual patterns")
                recommendations.append("Review market maker behavior")
                recommendations.append("Monitor for potential market manipulation")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating trading recommendations: {e}")
            return ["Investigate the trading anomaly manually"]

    def _clean_trading_history(self):
        """Clean trading history to keep only recent data"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=24)
            self.trading_history = [
                entry for entry in self.trading_history
                if entry['timestamp'] >= cutoff_time
            ]
        except Exception as e:
            logger.error(f"Error cleaning trading history: {e}")

    def get_trading_anomaly_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get summary of recent trading anomalies"""
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
            
            # Most common anomaly types
            anomaly_type_counts = {}
            for entry in recent_anomalies:
                if entry['anomalies_detected']:
                    for anomaly in entry['combined_anomalies']:
                        anomaly_type = anomaly['anomaly_type']
                        anomaly_type_counts[anomaly_type] = anomaly_type_counts.get(anomaly_type, 0) + 1
            
            most_common_types = sorted(anomaly_type_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            
            # Severity distribution
            severity_counts = {'low': 0, 'medium': 0, 'high': 0}
            for entry in recent_anomalies:
                if entry['anomalies_detected']:
                    for anomaly in entry['combined_anomalies']:
                        severity = anomaly.get('overall_severity', 'low')
                        severity_counts[severity] += 1
            
            return {
                'total_anomalies': total_anomalies,
                'total_checks': len(recent_anomalies),
                'anomaly_rate': total_anomalies / len(recent_anomalies) if recent_anomalies else 0,
                'most_common_types': most_common_types,
                'severity_distribution': severity_counts,
                'last_anomaly': recent_anomalies[-1]['timestamp'] if recent_anomalies else None
            }
            
        except Exception as e:
            logger.error(f"Error getting trading anomaly summary: {e}")
            return {'total_anomalies': 0}


# Example usage
if __name__ == "__main__":
    config = {
        'contamination': 0.05,
        'lookback_window': 1000,
        'confidence_level': 0.99,
        'alert_threshold': 0.7,
        'min_trade_size': 0.001,
        'max_slippage': 0.002,
        'max_spread': 0.001
    }
    
    detector = TradingAnomalyDetector(config)
    
    # Sample trading data
    trading_data = {
        'trade_execution': {
            'trade_size': 0.1,
            'execution_price': 50000.0,
            'slippage': 0.001,
            'spread': 0.0005,
            'execution_time': 150,
            'fill_ratio': 0.95,
            'order_type': 'market'
        },
        'market_data': {
            'price': 50000.0,
            'volume': 1000.0,
            'bid': 49999.0,
            'ask': 50001.0,
            'bid_size': 50.0,
            'ask_size': 45.0,
            'price_change': 0.02,
            'volume_change': 1.5
        },
        'order_book': {
            'imbalance': 0.1,
            'depth': 5000.0,
            'spread': 0.0005,
            'total_volume': 10000.0
        },
        'trading_patterns': {
            'trade_frequency': 10.0,
            'trade_direction': 0.6,
            'trade_clustering': 0.3,
            'momentum': 0.1,
            'mean_reversion': 0.05
        }
    }
    
    # Detect anomalies
    anomalies = detector.detect_trading_anomalies(trading_data)
    
    print("Trading anomalies detected:", anomalies['anomalies_detected'])
    print("Number of anomalies:", len(anomalies['combined_anomalies']))
    print("Alerts generated:", len(anomalies['alerts'])) 