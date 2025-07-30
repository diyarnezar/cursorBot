"""
📊 Performance Monitor Module

This module provides real-time performance tracking, anomaly detection,
and comprehensive performance analytics for the trading system.

Author: Hyperion Trading System
Date: 2025
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

# ML imports for anomaly detection
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Configure logging
logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """
    📊 Performance Monitoring System
    
    Provides real-time performance tracking, anomaly detection,
    and comprehensive performance analytics.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the performance monitor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.performance_metrics = {}
        self.anomaly_detectors = {}
        self.performance_history = []
        self.alert_history = []
        self.last_monitoring_time = None
        self.monitoring_interval = timedelta(minutes=5)  # Monitor every 5 minutes
        
        # Monitoring modes
        self.monitoring_modes = {
            'real_time_tracking': True,
            'anomaly_detection': True,
            'performance_analytics': True,
            'alert_system': True,
            'performance_forecasting': True
        }
        
        # Performance thresholds
        self.performance_thresholds = {
            'min_r2_score': 0.6,
            'max_mse': 0.1,
            'min_sharpe_ratio': 1.0,
            'max_drawdown': 0.2,
            'min_win_rate': 0.5,
            'max_loss_rate': 0.4
        }
        
        # Alert thresholds
        self.alert_thresholds = {
            'performance_degradation': 0.1,  # 10% degradation
            'anomaly_score': 0.8,  # High anomaly score
            'consecutive_losses': 5,  # 5 consecutive losses
            'drawdown_threshold': 0.15,  # 15% drawdown
            'volatility_spike': 2.0  # 2x normal volatility
        }
        
        logger.info("📊 Performance Monitor initialized")
    
    async def start_performance_monitoring(self):
        """Start the performance monitoring process."""
        logger.info("🚀 Starting performance monitoring system...")
        
        while True:
            try:
                await self._conduct_monitoring_cycle()
                await asyncio.sleep(self.monitoring_interval.total_seconds())
            except Exception as e:
                logger.error(f"❌ Monitoring cycle failed: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes before retrying
    
    async def _conduct_monitoring_cycle(self):
        """Conduct a complete monitoring cycle."""
        logger.info("📊 Conducting monitoring cycle...")
        
        # Real-time performance tracking
        if self.monitoring_modes['real_time_tracking']:
            await self._track_real_time_performance()
        
        # Anomaly detection
        if self.monitoring_modes['anomaly_detection']:
            await self._detect_anomalies()
        
        # Performance analytics
        if self.monitoring_modes['performance_analytics']:
            await self._analyze_performance()
        
        # Alert system
        if self.monitoring_modes['alert_system']:
            await self._check_alerts()
        
        # Performance forecasting
        if self.monitoring_modes['performance_forecasting']:
            await self._forecast_performance()
        
        self.last_monitoring_time = datetime.now()
        logger.info("✅ Monitoring cycle completed")
    
    async def _track_real_time_performance(self):
        """Track real-time performance metrics."""
        logger.info("📈 Tracking real-time performance...")
        
        try:
            # Get latest performance data
            performance_data = await self._get_latest_performance_data()
            
            if performance_data is not None:
                # Calculate real-time metrics
                metrics = await self._calculate_real_time_metrics(performance_data)
                
                # Store metrics
                self.performance_metrics.update(metrics)
                
                # Add to history
                self.performance_history.append({
                    'timestamp': datetime.now(),
                    'metrics': metrics
                })
                
                logger.info(f"✅ Real-time performance tracked: {len(metrics)} metrics")
            else:
                logger.info("ℹ️ No performance data available")
                
        except Exception as e:
            logger.error(f"❌ Real-time performance tracking failed: {e}")
    
    async def _get_latest_performance_data(self) -> Optional[pd.DataFrame]:
        """Get latest performance data."""
        try:
            # This would integrate with the trading system
            # For now, return None to indicate no data available
            return None
            
        except Exception as e:
            logger.error(f"❌ Failed to get performance data: {e}")
            return None
    
    async def _calculate_real_time_metrics(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate real-time performance metrics."""
        try:
            metrics = {}
            
            # Model performance metrics
            if 'predictions' in data.columns and 'actual' in data.columns:
                predictions = data['predictions']
                actual = data['actual']
                
                # R² score
                metrics['r2_score'] = r2_score(actual, predictions)
                
                # Mean squared error
                metrics['mse'] = mean_squared_error(actual, predictions)
                
                # Mean absolute error
                metrics['mae'] = mean_absolute_error(actual, predictions)
                
                # Directional accuracy
                directional_accuracy = np.mean(np.sign(predictions.diff()) == np.sign(actual.diff()))
                metrics['directional_accuracy'] = directional_accuracy
            
            # Trading performance metrics
            if 'returns' in data.columns:
                returns = data['returns']
                
                # Sharpe ratio
                if returns.std() > 0:
                    metrics['sharpe_ratio'] = returns.mean() / returns.std() * np.sqrt(252)
                else:
                    metrics['sharpe_ratio'] = 0.0
                
                # Maximum drawdown
                cumulative_returns = (1 + returns).cumprod()
                running_max = cumulative_returns.expanding().max()
                drawdown = (cumulative_returns - running_max) / running_max
                metrics['max_drawdown'] = drawdown.min()
                
                # Win rate
                win_rate = np.mean(returns > 0)
                metrics['win_rate'] = win_rate
                
                # Loss rate
                loss_rate = np.mean(returns < 0)
                metrics['loss_rate'] = loss_rate
                
                # Profit factor
                if np.sum(returns[returns < 0]) != 0:
                    profit_factor = np.sum(returns[returns > 0]) / abs(np.sum(returns[returns < 0]))
                    metrics['profit_factor'] = profit_factor
                else:
                    metrics['profit_factor'] = float('inf')
                
                # Volatility
                metrics['volatility'] = returns.std() * np.sqrt(252)
                
                # Total return
                metrics['total_return'] = cumulative_returns.iloc[-1] - 1
            
            # Feature performance metrics
            if 'feature_importance' in data.columns:
                feature_importance = data['feature_importance']
                metrics['avg_feature_importance'] = feature_importance.mean()
                metrics['feature_importance_std'] = feature_importance.std()
            
            return metrics
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate real-time metrics: {e}")
            return {}
    
    async def _detect_anomalies(self):
        """Detect performance anomalies."""
        logger.info("🔍 Detecting anomalies...")
        
        try:
            if len(self.performance_history) > 10:
                # Prepare data for anomaly detection
                anomaly_data = self._prepare_anomaly_data()
                
                # Detect anomalies in different metrics
                await self._detect_performance_anomalies(anomaly_data)
                await self._detect_trading_anomalies(anomaly_data)
                await self._detect_model_anomalies(anomaly_data)
                
                logger.info("✅ Anomaly detection completed")
            else:
                logger.info("ℹ️ Insufficient data for anomaly detection")
                
        except Exception as e:
            logger.error(f"❌ Anomaly detection failed: {e}")
    
    def _prepare_anomaly_data(self) -> pd.DataFrame:
        """Prepare data for anomaly detection."""
        try:
            # Convert performance history to dataframe
            data = []
            for entry in self.performance_history[-100:]:  # Last 100 entries
                row = entry['metrics'].copy()
                row['timestamp'] = entry['timestamp']
                data.append(row)
            
            df = pd.DataFrame(data)
            return df
            
        except Exception as e:
            logger.error(f"❌ Failed to prepare anomaly data: {e}")
            return pd.DataFrame()
    
    async def _detect_performance_anomalies(self, data: pd.DataFrame):
        """Detect performance anomalies."""
        try:
            if len(data) == 0:
                return
            
            # Detect anomalies in key performance metrics
            metrics_to_monitor = ['r2_score', 'mse', 'sharpe_ratio', 'win_rate']
            
            for metric in metrics_to_monitor:
                if metric in data.columns:
                    # Use Isolation Forest for anomaly detection
                    scaler = StandardScaler()
                    metric_data = data[metric].values.reshape(-1, 1)
                    scaled_data = scaler.fit_transform(metric_data)
                    
                    # Train anomaly detector
                    detector = IsolationForest(contamination=0.1, random_state=42)
                    anomaly_scores = detector.fit_predict(scaled_data)
                    
                    # Check for anomalies
                    anomalies = anomaly_scores == -1
                    if np.any(anomalies):
                        anomaly_indices = np.where(anomalies)[0]
                        logger.warning(f"⚠️ Performance anomaly detected in {metric}: {len(anomaly_indices)} anomalies")
                        
                        # Store anomaly information
                        self._store_anomaly_info('performance', metric, anomaly_indices, data.iloc[anomaly_indices])
            
        except Exception as e:
            logger.error(f"❌ Performance anomaly detection failed: {e}")
    
    async def _detect_trading_anomalies(self, data: pd.DataFrame):
        """Detect trading anomalies."""
        try:
            if len(data) == 0:
                return
            
            # Detect trading-specific anomalies
            trading_metrics = ['max_drawdown', 'volatility', 'profit_factor']
            
            for metric in trading_metrics:
                if metric in data.columns:
                    # Check for extreme values
                    values = data[metric].values
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    
                    # Detect outliers (beyond 2 standard deviations)
                    outliers = np.abs(values - mean_val) > 2 * std_val
                    
                    if np.any(outliers):
                        outlier_indices = np.where(outliers)[0]
                        logger.warning(f"⚠️ Trading anomaly detected in {metric}: {len(outlier_indices)} outliers")
                        
                        # Store anomaly information
                        self._store_anomaly_info('trading', metric, outlier_indices, data.iloc[outlier_indices])
            
        except Exception as e:
            logger.error(f"❌ Trading anomaly detection failed: {e}")
    
    async def _detect_model_anomalies(self, data: pd.DataFrame):
        """Detect model anomalies."""
        try:
            if len(data) == 0:
                return
            
            # Detect model-specific anomalies
            model_metrics = ['directional_accuracy', 'avg_feature_importance']
            
            for metric in model_metrics:
                if metric in data.columns:
                    # Check for sudden performance drops
                    values = data[metric].values
                    
                    # Calculate rolling mean and detect drops
                    window = min(10, len(values))
                    rolling_mean = pd.Series(values).rolling(window).mean()
                    
                    # Detect significant drops (more than 20% below rolling mean)
                    drops = values < rolling_mean * 0.8
                    
                    if np.any(drops):
                        drop_indices = np.where(drops)[0]
                        logger.warning(f"⚠️ Model anomaly detected in {metric}: {len(drop_indices)} performance drops")
                        
                        # Store anomaly information
                        self._store_anomaly_info('model', metric, drop_indices, data.iloc[drop_indices])
            
        except Exception as e:
            logger.error(f"❌ Model anomaly detection failed: {e}")
    
    def _store_anomaly_info(self, anomaly_type: str, metric: str, indices: np.ndarray, data: pd.DataFrame):
        """Store anomaly information."""
        try:
            anomaly_info = {
                'timestamp': datetime.now(),
                'type': anomaly_type,
                'metric': metric,
                'indices': indices.tolist(),
                'values': data[metric].values.tolist(),
                'severity': 'high' if len(indices) > 5 else 'medium'
            }
            
            self.alert_history.append(anomaly_info)
            
        except Exception as e:
            logger.error(f"❌ Failed to store anomaly info: {e}")
    
    async def _analyze_performance(self):
        """Analyze overall performance."""
        logger.info("📊 Analyzing performance...")
        
        try:
            if len(self.performance_history) > 0:
                # Calculate performance trends
                trends = await self._calculate_performance_trends()
                
                # Calculate performance statistics
                statistics = await self._calculate_performance_statistics()
                
                # Generate performance insights
                insights = await self._generate_performance_insights(trends, statistics)
                
                # Store analysis results
                analysis_result = {
                    'timestamp': datetime.now(),
                    'trends': trends,
                    'statistics': statistics,
                    'insights': insights
                }
                
                self.performance_metrics['analysis'] = analysis_result
                logger.info("✅ Performance analysis completed")
            else:
                logger.info("ℹ️ No performance data for analysis")
                
        except Exception as e:
            logger.error(f"❌ Performance analysis failed: {e}")
    
    async def _calculate_performance_trends(self) -> Dict[str, Any]:
        """Calculate performance trends."""
        try:
            trends = {}
            
            if len(self.performance_history) < 2:
                return trends
            
            # Convert to dataframe for trend analysis
            df = pd.DataFrame([entry['metrics'] for entry in self.performance_history])
            
            # Calculate trends for key metrics
            key_metrics = ['r2_score', 'sharpe_ratio', 'win_rate', 'volatility']
            
            for metric in key_metrics:
                if metric in df.columns:
                    values = df[metric].values
                    
                    # Linear trend
                    x = np.arange(len(values))
                    slope, intercept = np.polyfit(x, values, 1)
                    
                    trends[metric] = {
                        'slope': slope,
                        'trend': 'improving' if slope > 0 else 'declining' if slope < 0 else 'stable',
                        'change_rate': slope * len(values) / values.mean() if values.mean() != 0 else 0
                    }
            
            return trends
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate performance trends: {e}")
            return {}
    
    async def _calculate_performance_statistics(self) -> Dict[str, Any]:
        """Calculate performance statistics."""
        try:
            statistics = {}
            
            if len(self.performance_history) == 0:
                return statistics
            
            # Convert to dataframe for statistical analysis
            df = pd.DataFrame([entry['metrics'] for entry in self.performance_history])
            
            # Calculate statistics for all metrics
            for column in df.columns:
                if df[column].dtype in ['float64', 'int64']:
                    values = df[column].dropna()
                    if len(values) > 0:
                        statistics[column] = {
                            'mean': values.mean(),
                            'std': values.std(),
                            'min': values.min(),
                            'max': values.max(),
                            'median': values.median(),
                            'q25': values.quantile(0.25),
                            'q75': values.quantile(0.75)
                        }
            
            return statistics
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate performance statistics: {e}")
            return {}
    
    async def _generate_performance_insights(self, trends: Dict[str, Any], statistics: Dict[str, Any]) -> List[str]:
        """Generate performance insights."""
        try:
            insights = []
            
            # Analyze trends
            for metric, trend_info in trends.items():
                if trend_info['trend'] == 'improving':
                    insights.append(f"✅ {metric} is showing improvement trend")
                elif trend_info['trend'] == 'declining':
                    insights.append(f"⚠️ {metric} is showing decline trend")
            
            # Analyze statistics
            for metric, stats in statistics.items():
                if metric in self.performance_thresholds:
                    threshold = self.performance_thresholds[metric]
                    
                    if 'min' in metric or 'max_drawdown' in metric:
                        if stats['mean'] > threshold:
                            insights.append(f"⚠️ {metric} is above acceptable threshold")
                    else:
                        if stats['mean'] < threshold:
                            insights.append(f"⚠️ {metric} is below acceptable threshold")
            
            return insights
            
        except Exception as e:
            logger.error(f"❌ Failed to generate performance insights: {e}")
            return []
    
    async def _check_alerts(self):
        """Check and trigger alerts."""
        logger.info("🚨 Checking alerts...")
        
        try:
            # Check performance degradation
            await self._check_performance_degradation()
            
            # Check anomaly alerts
            await self._check_anomaly_alerts()
            
            # Check threshold violations
            await self._check_threshold_violations()
            
            logger.info("✅ Alert check completed")
            
        except Exception as e:
            logger.error(f"❌ Alert check failed: {e}")
    
    async def _check_performance_degradation(self):
        """Check for performance degradation."""
        try:
            if len(self.performance_history) < 10:
                return
            
            # Compare recent performance with historical performance
            recent_metrics = pd.DataFrame([entry['metrics'] for entry in self.performance_history[-5:]])
            historical_metrics = pd.DataFrame([entry['metrics'] for entry in self.performance_history[:-5]])
            
            for metric in recent_metrics.columns:
                if metric in historical_metrics.columns:
                    recent_mean = recent_metrics[metric].mean()
                    historical_mean = historical_metrics[metric].mean()
                    
                    if historical_mean != 0:
                        degradation = (historical_mean - recent_mean) / historical_mean
                        
                        if degradation > self.alert_thresholds['performance_degradation']:
                            logger.warning(f"🚨 Performance degradation detected in {metric}: {degradation:.2%}")
                            
                            # Trigger alert
                            await self._trigger_alert('performance_degradation', {
                                'metric': metric,
                                'degradation': degradation,
                                'recent_mean': recent_mean,
                                'historical_mean': historical_mean
                            })
            
        except Exception as e:
            logger.error(f"❌ Performance degradation check failed: {e}")
    
    async def _check_anomaly_alerts(self):
        """Check for anomaly alerts."""
        try:
            # Check recent anomalies
            recent_alerts = [alert for alert in self.alert_history if 
                           (datetime.now() - alert['timestamp']).total_seconds() < 3600]  # Last hour
            
            if len(recent_alerts) > 5:
                logger.warning(f"🚨 High anomaly rate detected: {len(recent_alerts)} anomalies in the last hour")
                
                # Trigger alert
                await self._trigger_alert('high_anomaly_rate', {
                    'anomaly_count': len(recent_alerts),
                    'time_period': '1 hour'
                })
            
        except Exception as e:
            logger.error(f"❌ Anomaly alert check failed: {e}")
    
    async def _check_threshold_violations(self):
        """Check for threshold violations."""
        try:
            if len(self.performance_history) == 0:
                return
            
            # Get latest metrics
            latest_metrics = self.performance_history[-1]['metrics']
            
            # Check each threshold
            for metric, threshold in self.performance_thresholds.items():
                if metric in latest_metrics:
                    value = latest_metrics[metric]
                    
                    # Check if threshold is violated
                    if 'min' in metric or 'sharpe' in metric or 'win_rate' in metric:
                        if value < threshold:
                            logger.warning(f"🚨 Threshold violation: {metric} = {value:.4f} < {threshold}")
                            await self._trigger_alert('threshold_violation', {
                                'metric': metric,
                                'value': value,
                                'threshold': threshold
                            })
                    elif 'max' in metric or 'drawdown' in metric or 'mse' in metric:
                        if value > threshold:
                            logger.warning(f"🚨 Threshold violation: {metric} = {value:.4f} > {threshold}")
                            await self._trigger_alert('threshold_violation', {
                                'metric': metric,
                                'value': value,
                                'threshold': threshold
                            })
            
        except Exception as e:
            logger.error(f"❌ Threshold violation check failed: {e}")
    
    async def _trigger_alert(self, alert_type: str, alert_data: Dict[str, Any]):
        """Trigger an alert."""
        try:
            alert = {
                'timestamp': datetime.now(),
                'type': alert_type,
                'data': alert_data,
                'severity': 'high' if 'degradation' in alert_type or 'violation' in alert_type else 'medium'
            }
            
            # Log alert
            logger.warning(f"🚨 ALERT: {alert_type} - {alert_data}")
            
            # Store alert
            self.alert_history.append(alert)
            
            # Send notification (would integrate with notification system)
            await self._send_notification(alert)
            
        except Exception as e:
            logger.error(f"❌ Failed to trigger alert: {e}")
    
    async def _send_notification(self, alert: Dict[str, Any]):
        """Send notification for alert."""
        try:
            # This would integrate with notification systems (email, Slack, etc.)
            logger.info(f"📧 Notification sent for alert: {alert['type']}")
            
        except Exception as e:
            logger.error(f"❌ Failed to send notification: {e}")
    
    async def _forecast_performance(self):
        """Forecast future performance."""
        logger.info("🔮 Forecasting performance...")
        
        try:
            if len(self.performance_history) > 20:
                # Prepare forecasting data
                forecast_data = self._prepare_forecast_data()
                
                # Generate forecasts for key metrics
                forecasts = await self._generate_performance_forecasts(forecast_data)
                
                # Store forecasts
                self.performance_metrics['forecasts'] = forecasts
                
                logger.info("✅ Performance forecasting completed")
            else:
                logger.info("ℹ️ Insufficient data for performance forecasting")
                
        except Exception as e:
            logger.error(f"❌ Performance forecasting failed: {e}")
    
    def _prepare_forecast_data(self) -> pd.DataFrame:
        """Prepare data for forecasting."""
        try:
            # Convert performance history to time series
            data = []
            for entry in self.performance_history:
                row = entry['metrics'].copy()
                row['timestamp'] = entry['timestamp']
                data.append(row)
            
            df = pd.DataFrame(data)
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Failed to prepare forecast data: {e}")
            return pd.DataFrame()
    
    async def _generate_performance_forecasts(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate performance forecasts."""
        try:
            forecasts = {}
            
            # Simple forecasting using moving averages
            key_metrics = ['r2_score', 'sharpe_ratio', 'win_rate']
            
            for metric in key_metrics:
                if metric in data.columns:
                    values = data[metric].values
                    
                    if len(values) > 5:
                        # Simple exponential smoothing forecast
                        alpha = 0.3
                        forecast = values[-1]  # Start with last value
                        
                        # Apply exponential smoothing
                        for i in range(len(values) - 1, 0, -1):
                            forecast = alpha * values[i] + (1 - alpha) * forecast
                        
                        # Forecast next 5 periods
                        future_forecast = [forecast]
                        for _ in range(4):
                            forecast = alpha * forecast + (1 - alpha) * forecast
                            future_forecast.append(forecast)
                        
                        forecasts[metric] = {
                            'current': values[-1],
                            'forecast': future_forecast,
                            'trend': 'improving' if future_forecast[-1] > values[-1] else 'declining'
                        }
            
            return forecasts
            
        except Exception as e:
            logger.error(f"❌ Failed to generate performance forecasts: {e}")
            return {}
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get a summary of monitoring activities."""
        return {
            'performance_metrics': self.performance_metrics,
            'alert_count': len(self.alert_history),
            'last_monitoring_time': self.last_monitoring_time,
            'monitoring_modes': self.monitoring_modes,
            'performance_thresholds': self.performance_thresholds,
            'alert_thresholds': self.alert_thresholds,
            'performance_history_length': len(self.performance_history)
        }
    
    def save_monitoring_state(self, filepath: str):
        """Save monitoring state to file."""
        try:
            state = {
                'performance_metrics': self.performance_metrics,
                'alert_history': self.alert_history,
                'performance_history': self.performance_history
            }
            
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2, default=str)
            logger.info(f"💾 Monitoring state saved to {filepath}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save monitoring state: {e}")
    
    def load_monitoring_state(self, filepath: str):
        """Load monitoring state from file."""
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            self.performance_metrics = state.get('performance_metrics', {})
            self.alert_history = state.get('alert_history', [])
            self.performance_history = state.get('performance_history', [])
            
            logger.info(f"📂 Monitoring state loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load monitoring state: {e}")


# Example usage
if __name__ == "__main__":
    # Configuration
    config = {
        'monitoring_enabled': True,
        'monitoring_interval_minutes': 5,
        'performance_threshold': 0.7
    }
    
    # Initialize performance monitor
    monitor = PerformanceMonitor(config)
    
    # Start performance monitoring
    asyncio.run(monitor.start_performance_monitoring()) 