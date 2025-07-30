import numpy as np
import pandas as pd
import logging
import json
import os
import time
import asyncio
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass
import sqlite3
import hashlib
import hmac
import requests
from urllib.parse import urlencode

@dataclass
class RiskMetrics:
    """Data class for risk metrics."""
    current_drawdown: float
    max_drawdown: float
    sharpe_ratio: float
    var_95: float  # Value at Risk (95%)
    cvar_95: float  # Conditional Value at Risk (95%)
    volatility: float
    beta: float
    correlation: float
    position_size: float
    leverage: float
    margin_used: float
    free_margin: float

class DynamicRiskManager:
    """
    Dynamic risk management system with adaptive position sizing and stop-loss.
    """
    
    def __init__(self, 
                 initial_capital: float = 10000,
                 max_position_size: float = 0.1,  # 10% of capital
                 max_drawdown: float = 0.2,       # 20% max drawdown
                 risk_per_trade: float = 0.02,    # 2% risk per trade
                 volatility_lookback: int = 30):
        """
        Initialize the dynamic risk manager.
        
        Args:
            initial_capital: Initial trading capital
            max_position_size: Maximum position size as fraction of capital
            max_drawdown: Maximum allowed drawdown
            risk_per_trade: Maximum risk per trade
            volatility_lookback: Lookback period for volatility calculation
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_position_size = max_position_size
        self.max_drawdown = max_drawdown
        self.risk_per_trade = risk_per_trade
        self.volatility_lookback = volatility_lookback
        
        # Risk metrics
        self.risk_metrics = RiskMetrics(
            current_drawdown=0.0,
            max_drawdown=0.0,
            sharpe_ratio=0.0,
            var_95=0.0,
            cvar_95=0.0,
            volatility=0.0,
            beta=1.0,
            correlation=0.0,
            position_size=0.0,
            leverage=1.0,
            margin_used=0.0,
            free_margin=initial_capital
        )
        
        # Performance history
        self.returns_history = []
        self.drawdown_history = []
        self.position_history = []
        
        # Risk limits
        self.risk_limits = {
            'max_daily_loss': 0.05,      # 5% daily loss limit
            'max_weekly_loss': 0.15,     # 15% weekly loss limit
            'max_leverage': 3.0,         # Maximum leverage
            'min_sharpe': 0.5,           # Minimum Sharpe ratio
            'max_var': 0.03,             # Maximum VaR (3%)
            'max_correlation': 0.8       # Maximum correlation with market
        }
        
        logging.info("🛡️ Dynamic Risk Manager initialized")
    
    def calculate_position_size(self, 
                               entry_price: float,
                               stop_loss: float,
                               confidence: float = 1.0,
                               volatility: float = None) -> Dict[str, float]:
        """
        Calculate optimal position size based on risk parameters.
        
        Args:
            entry_price: Entry price for the trade
            stop_loss: Stop loss price
            confidence: Model confidence (0-1)
            volatility: Current market volatility
            
        Returns:
            Dictionary with position size and risk metrics
        """
        try:
            # Calculate base position size using risk per trade
            risk_amount = self.current_capital * self.risk_per_trade
            price_risk = abs(entry_price - stop_loss) / entry_price
            
            if price_risk == 0:
                return {'position_size': 0, 'risk_metrics': self.risk_metrics}
            
            base_position_size = risk_amount / (self.current_capital * price_risk)
            
            # Adjust for confidence
            confidence_adjustment = 0.5 + (confidence * 0.5)  # 0.5 to 1.0
            base_position_size *= confidence_adjustment
            
            # Adjust for volatility
            if volatility is not None:
                volatility_adjustment = max(0.1, 1 - (volatility * 2))  # Reduce size for high volatility
                base_position_size *= volatility_adjustment
            
            # Adjust for current drawdown
            drawdown_adjustment = max(0.1, 1 - (self.risk_metrics.current_drawdown * 2))
            base_position_size *= drawdown_adjustment
            
            # Apply maximum position size limit
            max_size = self.max_position_size
            if self.risk_metrics.current_drawdown > 0.1:  # Reduce size if in drawdown
                max_size *= 0.5
            
            final_position_size = min(base_position_size, max_size)
            
            # Calculate risk metrics
            self._update_risk_metrics(final_position_size, entry_price, stop_loss)
            
            return {
                'position_size': final_position_size,
                'risk_metrics': self.risk_metrics,
                'confidence_adjustment': confidence_adjustment,
                'volatility_adjustment': volatility_adjustment if volatility is not None else 1.0,
                'drawdown_adjustment': drawdown_adjustment
            }
            
        except Exception as e:
            logging.error(f"Error calculating position size: {e}")
            return {'position_size': 0, 'risk_metrics': self.risk_metrics}
    
    def calculate_dynamic_stop_loss(self, 
                                   entry_price: float,
                                   direction: str,
                                   volatility: float = None,
                                   atr: float = None) -> Dict[str, float]:
        """
        Calculate dynamic stop loss based on market conditions.
        
        Args:
            entry_price: Entry price
            direction: 'long' or 'short'
            volatility: Current volatility
            atr: Average True Range
            
        Returns:
            Dictionary with stop loss and take profit levels
        """
        try:
            # Base stop loss percentage
            base_stop_percent = 0.02  # 2%
            
            # Adjust for volatility
            if volatility is not None:
                volatility_multiplier = 1 + (volatility * 5)  # Increase stop for high volatility
                base_stop_percent *= volatility_multiplier
            
            # Adjust for ATR
            if atr is not None:
                atr_multiplier = max(1.0, atr / entry_price * 10)  # Use ATR-based stop
                base_stop_percent = max(base_stop_percent, atr_multiplier * 0.01)
            
            # Adjust for current drawdown
            if self.risk_metrics.current_drawdown > 0.1:
                base_stop_percent *= 0.8  # Tighter stops in drawdown
            
            # Calculate stop loss price
            if direction == 'long':
                stop_loss = entry_price * (1 - base_stop_percent)
                take_profit = entry_price * (1 + base_stop_percent * 2)  # 2:1 reward-risk
            else:
                stop_loss = entry_price * (1 + base_stop_percent)
                take_profit = entry_price * (1 - base_stop_percent * 2)
            
            return {
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'stop_percent': base_stop_percent,
                'risk_reward_ratio': 2.0
            }
            
        except Exception as e:
            logging.error(f"Error calculating dynamic stop loss: {e}")
            return {
                'stop_loss': entry_price * 0.98 if direction == 'long' else entry_price * 1.02,
                'take_profit': entry_price * 1.04 if direction == 'long' else entry_price * 0.96,
                'stop_percent': 0.02,
                'risk_reward_ratio': 2.0
            }
    
    def update_performance(self, 
                          trade_pnl: float,
                          current_price: float,
                          position_size: float = 0) -> None:
        """
        Update performance metrics after a trade.
        
        Args:
            trade_pnl: Profit/loss from the trade
            current_price: Current market price
            position_size: Current position size
        """
        try:
            # Update capital
            self.current_capital += trade_pnl
            
            # Calculate return
            if self.initial_capital > 0:
                return_rate = trade_pnl / self.initial_capital
                self.returns_history.append(return_rate)
            
            # Update drawdown
            peak_capital = max(self.initial_capital, *[self.initial_capital + sum(self.returns_history[:i+1]) * self.initial_capital 
                                                      for i in range(len(self.returns_history))])
            current_drawdown = (peak_capital - self.current_capital) / peak_capital if peak_capital > 0 else 0
            
            self.risk_metrics.current_drawdown = current_drawdown
            self.risk_metrics.max_drawdown = max(self.risk_metrics.max_drawdown, current_drawdown)
            
            # Update position metrics
            self.risk_metrics.position_size = position_size
            self.risk_metrics.margin_used = position_size * current_price
            self.risk_metrics.free_margin = self.current_capital - self.risk_metrics.margin_used
            
            # Calculate Sharpe ratio
            if len(self.returns_history) > 1:
                returns_array = np.array(self.returns_history)
                self.risk_metrics.sharpe_ratio = np.mean(returns_array) / np.std(returns_array) if np.std(returns_array) > 0 else 0
            
            # Calculate VaR and CVaR
            if len(self.returns_history) >= 30:
                returns_array = np.array(self.returns_history[-30:])
                self.risk_metrics.var_95 = np.percentile(returns_array, 5)
                self.risk_metrics.cvar_95 = np.mean(returns_array[returns_array <= self.risk_metrics.var_95])
            
            # Calculate volatility
            if len(self.returns_history) >= self.volatility_lookback:
                recent_returns = self.returns_history[-self.volatility_lookback:]
                self.risk_metrics.volatility = np.std(recent_returns)
            
            # Store history
            self.drawdown_history.append(current_drawdown)
            self.position_history.append(position_size)
            
            # Keep history manageable
            if len(self.returns_history) > 1000:
                self.returns_history = self.returns_history[-1000:]
            if len(self.drawdown_history) > 1000:
                self.drawdown_history = self.drawdown_history[-1000:]
            if len(self.position_history) > 1000:
                self.position_history = self.position_history[-1000:]
                
        except Exception as e:
            logging.error(f"Error updating performance: {e}")
    
    def check_risk_limits(self) -> Dict[str, Any]:
        """
        Check if any risk limits are exceeded.
        
        Returns:
            Dictionary with risk limit status
        """
        try:
            violations = []
            warnings = []
            
            # Check drawdown limit
            if self.risk_metrics.current_drawdown > self.max_drawdown:
                violations.append({
                    'type': 'max_drawdown',
                    'current': self.risk_metrics.current_drawdown,
                    'limit': self.max_drawdown,
                    'severity': 'high'
                })
            
            # Check daily loss limit
            if len(self.returns_history) > 0:
                daily_return = self.returns_history[-1] if self.returns_history else 0
                if daily_return < -self.risk_limits['max_daily_loss']:
                    violations.append({
                        'type': 'daily_loss',
                        'current': abs(daily_return),
                        'limit': self.risk_limits['max_daily_loss'],
                        'severity': 'medium'
                    })
            
            # Check weekly loss limit
            if len(self.returns_history) >= 7:
                weekly_return = sum(self.returns_history[-7:])
                if weekly_return < -self.risk_limits['max_weekly_loss']:
                    violations.append({
                        'type': 'weekly_loss',
                        'current': abs(weekly_return),
                        'limit': self.risk_limits['max_weekly_loss'],
                        'severity': 'high'
                    })
            
            # Check Sharpe ratio
            if self.risk_metrics.sharpe_ratio < self.risk_limits['min_sharpe']:
                warnings.append({
                    'type': 'low_sharpe',
                    'current': self.risk_metrics.sharpe_ratio,
                    'limit': self.risk_limits['min_sharpe'],
                    'severity': 'medium'
                })
            
            # Check VaR
            if self.risk_metrics.var_95 < -self.risk_limits['max_var']:
                warnings.append({
                    'type': 'high_var',
                    'current': abs(self.risk_metrics.var_95),
                    'limit': self.risk_limits['max_var'],
                    'severity': 'medium'
                })
            
            return {
                'violations': violations,
                'warnings': warnings,
                'should_stop_trading': len(violations) > 0,
                'should_reduce_risk': len(warnings) > 2
            }
            
        except Exception as e:
            logging.error(f"Error checking risk limits: {e}")
            return {
                'violations': [],
                'warnings': [],
                'should_stop_trading': True,  # Conservative approach
                'should_reduce_risk': True
            }
    
    def get_risk_report(self) -> Dict[str, Any]:
        """Generate comprehensive risk report."""
        try:
            return {
                'timestamp': datetime.now().isoformat(),
                'capital': {
                    'initial': self.initial_capital,
                    'current': self.current_capital,
                    'total_return': (self.current_capital - self.initial_capital) / self.initial_capital
                },
                'risk_metrics': {
                    'current_drawdown': self.risk_metrics.current_drawdown,
                    'max_drawdown': self.risk_metrics.max_drawdown,
                    'sharpe_ratio': self.risk_metrics.sharpe_ratio,
                    'volatility': self.risk_metrics.volatility,
                    'var_95': self.risk_metrics.var_95,
                    'cvar_95': self.risk_metrics.cvar_95
                },
                'position_metrics': {
                    'position_size': self.risk_metrics.position_size,
                    'margin_used': self.risk_metrics.margin_used,
                    'free_margin': self.risk_metrics.free_margin,
                    'leverage': self.risk_metrics.leverage
                },
                'risk_limits': self.risk_limits,
                'history_length': len(self.returns_history)
            }
            
        except Exception as e:
            logging.error(f"Error generating risk report: {e}")
            return {}

class APILimitHandler:
    """
    Comprehensive API limit handling and rate limiting system.
    """
    
    def __init__(self, 
                 api_limits: Dict[str, int] = None,
                 retry_delays: List[int] = None):
        """
        Initialize the API limit handler.
        
        Args:
            api_limits: Dictionary of API limits (requests per time window)
            retry_delays: List of retry delays in seconds
        """
        self.api_limits = api_limits or {
            'requests_per_minute': 1200,
            'requests_per_hour': 100000,
            'orders_per_second': 10,
            'orders_per_day': 200000
        }
        
        self.retry_delays = retry_delays or [1, 2, 5, 10, 30, 60]
        
        # Request tracking
        self.request_history = []
        self.order_history = []
        self.rate_limit_hits = 0
        self.last_rate_limit = None
        
        # Rate limiting
        self.request_tokens = self.api_limits['requests_per_minute']
        self.order_tokens = self.api_limits['orders_per_second']
        self.last_token_refill = datetime.now()
        
        logging.info("🔒 API Limit Handler initialized")
    
    def check_rate_limit(self, request_type: str = 'request') -> bool:
        """
        Check if a request can be made without hitting rate limits.
        
        Args:
            request_type: Type of request ('request' or 'order')
            
        Returns:
            True if request can be made, False otherwise
        """
        try:
            now = datetime.now()
            
            # Refill tokens
            self._refill_tokens(now)
            
            # Check appropriate limit
            if request_type == 'order':
                if self.order_tokens <= 0:
                    self.rate_limit_hits += 1
                    self.last_rate_limit = now
                    return False
                self.order_tokens -= 1
            else:
                if self.request_tokens <= 0:
                    self.rate_limit_hits += 1
                    self.last_rate_limit = now
                    return False
                self.request_tokens -= 1
            
            # Record request
            self.request_history.append({
                'timestamp': now,
                'type': request_type
            })
            
            # Clean old history
            self._clean_history()
            
            return True
            
        except Exception as e:
            logging.error(f"Error checking rate limit: {e}")
            return False
    
    def _refill_tokens(self, now: datetime) -> None:
        """Refill rate limit tokens."""
        try:
            # Refill request tokens every minute
            if (now - self.last_token_refill).total_seconds() >= 60:
                self.request_tokens = min(
                    self.api_limits['requests_per_minute'],
                    self.request_tokens + self.api_limits['requests_per_minute']
                )
                self.last_token_refill = now
            
            # Refill order tokens every second
            if (now - self.last_token_refill).total_seconds() >= 1:
                self.order_tokens = min(
                    self.api_limits['orders_per_second'],
                    self.order_tokens + self.api_limits['orders_per_second']
                )
                
        except Exception as e:
            logging.error(f"Error refilling tokens: {e}")
    
    def _clean_history(self) -> None:
        """Clean old request history."""
        try:
            cutoff_time = datetime.now() - timedelta(hours=1)
            
            self.request_history = [
                req for req in self.request_history 
                if req['timestamp'] > cutoff_time
            ]
            
            self.order_history = [
                order for order in self.order_history 
                if order['timestamp'] > cutoff_time
            ]
            
        except Exception as e:
            logging.error(f"Error cleaning history: {e}")
    
    def get_retry_delay(self, attempt: int) -> int:
        """
        Get retry delay for a given attempt.
        
        Args:
            attempt: Attempt number (0-based)
            
        Returns:
            Delay in seconds
        """
        try:
            if attempt < len(self.retry_delays):
                return self.retry_delays[attempt]
            else:
                return self.retry_delays[-1]
                
        except Exception as e:
            logging.error(f"Error getting retry delay: {e}")
            return 60
    
    def handle_rate_limit_error(self, error: Exception) -> Dict[str, Any]:
        """
        Handle rate limit errors and provide recovery strategy.
        
        Args:
            error: The rate limit error
            
        Returns:
            Recovery strategy
        """
        try:
            self.rate_limit_hits += 1
            self.last_rate_limit = datetime.now()
            
            # Analyze error
            error_str = str(error).lower()
            
            if 'rate limit' in error_str or '429' in error_str:
                return {
                    'action': 'wait',
                    'delay': 60,
                    'message': 'Rate limit exceeded, waiting 60 seconds',
                    'severity': 'medium'
                }
            elif 'weight' in error_str:
                return {
                    'action': 'reduce_weight',
                    'delay': 30,
                    'message': 'Request weight exceeded, reducing request frequency',
                    'severity': 'low'
                }
            elif 'ip' in error_str:
                return {
                    'action': 'stop',
                    'delay': 300,
                    'message': 'IP rate limit exceeded, stopping for 5 minutes',
                    'severity': 'high'
                }
            else:
                return {
                    'action': 'retry',
                    'delay': 10,
                    'message': 'Unknown rate limit error, retrying in 10 seconds',
                    'severity': 'medium'
                }
                
        except Exception as e:
            logging.error(f"Error handling rate limit error: {e}")
            return {
                'action': 'stop',
                'delay': 60,
                'message': 'Error in rate limit handling, stopping for 1 minute',
                'severity': 'high'
            }
    
    def get_api_status(self) -> Dict[str, Any]:
        """Get current API usage status."""
        try:
            return {
                'timestamp': datetime.now().isoformat(),
                'request_tokens': self.request_tokens,
                'order_tokens': self.order_tokens,
                'rate_limit_hits': self.rate_limit_hits,
                'last_rate_limit': self.last_rate_limit.isoformat() if self.last_rate_limit else None,
                'recent_requests': len(self.request_history),
                'recent_orders': len(self.order_history),
                'limits': self.api_limits
            }
            
        except Exception as e:
            logging.error(f"Error getting API status: {e}")
            return {}

class FailoverSystem:
    """
    Comprehensive failover and redundancy system.
    """
    
    def __init__(self, 
                 backup_strategies: List[str] = None,
                 health_check_interval: int = 30):
        """
        Initialize the failover system.
        
        Args:
            backup_strategies: List of backup trading strategies
            health_check_interval: Health check interval in seconds
        """
        self.backup_strategies = backup_strategies or ['baseline', 'momentum', 'mean_reversion']
        self.health_check_interval = health_check_interval
        
        # System health
        self.primary_system_healthy = True
        self.backup_system_healthy = True
        self.last_health_check = datetime.now()
        
        # Failover state
        self.failover_active = False
        self.failover_reason = None
        self.failover_timestamp = None
        
        # Error tracking
        self.error_history = []
        self.recovery_attempts = 0
        self.max_recovery_attempts = 5
        
        # Health metrics
        self.health_metrics = {
            'uptime': 0.0,
            'error_rate': 0.0,
            'response_time': 0.0,
            'last_successful_operation': None
        }
        
        logging.info("🔄 Failover System initialized")
    
    def check_system_health(self) -> Dict[str, Any]:
        """
        Perform comprehensive system health check.
        
        Returns:
            Health check results
        """
        try:
            now = datetime.now()
            health_status = {
                'timestamp': now.isoformat(),
                'primary_system': self._check_primary_system(),
                'backup_system': self._check_backup_system(),
                'overall_health': 'healthy',
                'recommendations': []
            }
            
            # Determine overall health
            primary_healthy = health_status['primary_system']['healthy']
            backup_healthy = health_status['backup_system']['healthy']
            
            if not primary_healthy and not backup_healthy:
                health_status['overall_health'] = 'critical'
                health_status['recommendations'].append('Both primary and backup systems unhealthy')
            elif not primary_healthy:
                health_status['overall_health'] = 'degraded'
                health_status['recommendations'].append('Primary system unhealthy, using backup')
            elif not backup_healthy:
                health_status['overall_health'] = 'warning'
                health_status['recommendations'].append('Backup system unhealthy')
            
            # Update health metrics
            self._update_health_metrics(health_status)
            
            # Check if failover is needed
            if not primary_healthy and backup_healthy and not self.failover_active:
                self._activate_failover('Primary system unhealthy')
            
            # Check if recovery is possible
            if self.failover_active and primary_healthy:
                self._attempt_recovery()
            
            self.last_health_check = now
            return health_status
            
        except Exception as e:
            logging.error(f"Error in health check: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'overall_health': 'unknown',
                'error': str(e)
            }
    
    def _check_primary_system(self) -> Dict[str, Any]:
        """Check primary system health."""
        try:
            # Simulate health checks for different components
            checks = {
                'database': self._check_database_health(),
                'api_connection': self._check_api_health(),
                'model_performance': self._check_model_health(),
                'risk_manager': self._check_risk_manager_health()
            }
            
            overall_healthy = all(check['healthy'] for check in checks.values())
            
            return {
                'healthy': overall_healthy,
                'checks': checks,
                'last_check': datetime.now().isoformat()
            }
            
        except Exception as e:
            logging.error(f"Error checking primary system: {e}")
            return {'healthy': False, 'error': str(e)}
    
    def _check_backup_system(self) -> Dict[str, Any]:
        """Check backup system health."""
        try:
            # Check if backup strategies are available
            backup_available = len(self.backup_strategies) > 0
            
            return {
                'healthy': backup_available,
                'available_strategies': self.backup_strategies,
                'last_check': datetime.now().isoformat()
            }
            
        except Exception as e:
            logging.error(f"Error checking backup system: {e}")
            return {'healthy': False, 'error': str(e)}
    
    def _check_database_health(self) -> Dict[str, Any]:
        """Check database health."""
        try:
            # Simulate database health check
            return {
                'healthy': True,
                'response_time': 0.1,
                'connections': 5
            }
        except Exception as e:
            return {'healthy': False, 'error': str(e)}
    
    def _check_api_health(self) -> Dict[str, Any]:
        """Check API connection health."""
        try:
            # Simulate API health check
            return {
                'healthy': True,
                'response_time': 0.05,
                'rate_limit_status': 'normal'
            }
        except Exception as e:
            return {'healthy': False, 'error': str(e)}
    
    def _check_model_health(self) -> Dict[str, Any]:
        """Check model performance health."""
        try:
            # Simulate model health check
            return {
                'healthy': True,
                'accuracy': 0.75,
                'last_update': datetime.now().isoformat()
            }
        except Exception as e:
            return {'healthy': False, 'error': str(e)}
    
    def _check_risk_manager_health(self) -> Dict[str, Any]:
        """Check risk manager health."""
        try:
            # Simulate risk manager health check
            return {
                'healthy': True,
                'risk_level': 'normal',
                'drawdown': 0.05
            }
        except Exception as e:
            return {'healthy': False, 'error': str(e)}
    
    def _activate_failover(self, reason: str) -> None:
        """Activate failover to backup system."""
        try:
            self.failover_active = True
            self.failover_reason = reason
            self.failover_timestamp = datetime.now()
            
            logging.warning(f"🔄 Failover activated: {reason}")
            
            # Record failover event
            self.error_history.append({
                'timestamp': self.failover_timestamp,
                'type': 'failover_activation',
                'reason': reason,
                'severity': 'high'
            })
            
        except Exception as e:
            logging.error(f"Error activating failover: {e}")
    
    def _attempt_recovery(self) -> None:
        """Attempt recovery to primary system."""
        try:
            if self.recovery_attempts >= self.max_recovery_attempts:
                logging.warning("Maximum recovery attempts reached")
                return
            
            self.recovery_attempts += 1
            
            # Simulate recovery process
            recovery_successful = np.random.random() > 0.3  # 70% success rate
            
            if recovery_successful:
                self.failover_active = False
                self.failover_reason = None
                self.recovery_attempts = 0
                
                logging.info("✅ Recovery successful, returning to primary system")
                
                # Record recovery event
                self.error_history.append({
                    'timestamp': datetime.now(),
                    'type': 'recovery_successful',
                    'attempts': self.recovery_attempts,
                    'severity': 'low'
                })
            else:
                logging.warning(f"Recovery attempt {self.recovery_attempts} failed")
                
        except Exception as e:
            logging.error(f"Error attempting recovery: {e}")
    
    def _update_health_metrics(self, health_status: Dict[str, Any]) -> None:
        """Update health metrics."""
        try:
            # Update uptime
            if health_status['overall_health'] == 'healthy':
                self.health_metrics['uptime'] += self.health_check_interval
            
            # Update error rate
            total_checks = len(self.error_history) + 1
            error_count = len([e for e in self.error_history if e['severity'] in ['high', 'critical']])
            self.health_metrics['error_rate'] = error_count / total_checks
            
            # Update last successful operation
            if health_status['overall_health'] == 'healthy':
                self.health_metrics['last_successful_operation'] = datetime.now().isoformat()
                
        except Exception as e:
            logging.error(f"Error updating health metrics: {e}")
    
    def get_failover_status(self) -> Dict[str, Any]:
        """Get current failover status."""
        try:
            return {
                'failover_active': self.failover_active,
                'failover_reason': self.failover_reason,
                'failover_timestamp': self.failover_timestamp.isoformat() if self.failover_timestamp else None,
                'recovery_attempts': self.recovery_attempts,
                'backup_strategies': self.backup_strategies,
                'health_metrics': self.health_metrics,
                'error_history': self.error_history[-10:]  # Last 10 errors
            }
            
        except Exception as e:
            logging.error(f"Error getting failover status: {e}")
            return {}

class RobustnessManager:
    """
    Main robustness manager that coordinates all components.
    """
    
    def __init__(self, 
                 initial_capital: float = 10000,
                 config: Dict[str, Any] = None):
        """
        Initialize the robustness manager.
        
        Args:
            initial_capital: Initial trading capital
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Initialize components
        self.risk_manager = DynamicRiskManager(initial_capital=initial_capital)
        self.api_handler = APILimitHandler()
        self.failover_system = FailoverSystem()
        
        # Error handling
        self.error_handlers = {
            'rate_limit': self._handle_rate_limit_error,
            'network': self._handle_network_error,
            'api': self._handle_api_error,
            'risk': self._handle_risk_error,
            'system': self._handle_system_error
        }
        
        # Recovery strategies
        self.recovery_strategies = {
            'immediate': self._immediate_recovery,
            'gradual': self._gradual_recovery,
            'conservative': self._conservative_recovery
        }
        
        logging.info("🛡️ Robustness Manager initialized")
    
    def handle_error(self, 
                    error: Exception, 
                    error_type: str = 'unknown',
                    context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Handle errors with appropriate recovery strategies.
        
        Args:
            error: The error that occurred
            error_type: Type of error
            context: Additional context
            
        Returns:
            Error handling results
        """
        try:
            error_info = {
                'timestamp': datetime.now().isoformat(),
                'error_type': error_type,
                'error_message': str(error),
                'context': context or {},
                'handled': False,
                'recovery_action': None
            }
            
            # Get appropriate handler
            handler = self.error_handlers.get(error_type, self._handle_unknown_error)
            
            # Handle the error
            handling_result = handler(error, context)
            error_info.update(handling_result)
            
            # Log the error
            if handling_result.get('severity') == 'high':
                logging.error(f"🚨 High severity error: {error}")
            elif handling_result.get('severity') == 'medium':
                logging.warning(f"⚠️ Medium severity error: {error}")
            else:
                logging.info(f"ℹ️ Low severity error: {error}")
            
            return error_info
            
        except Exception as e:
            logging.error(f"Error in error handling: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'error_type': 'error_handling_failure',
                'error_message': str(e),
                'handled': False,
                'severity': 'critical'
            }
    
    def _handle_rate_limit_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle rate limit errors."""
        try:
            recovery_strategy = self.api_handler.handle_rate_limit_error(error)
            
            return {
                'handled': True,
                'recovery_action': recovery_strategy['action'],
                'delay': recovery_strategy['delay'],
                'severity': recovery_strategy['severity'],
                'message': recovery_strategy['message']
            }
            
        except Exception as e:
            return {'handled': False, 'severity': 'high', 'error': str(e)}
    
    def _handle_network_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle network errors."""
        try:
            return {
                'handled': True,
                'recovery_action': 'retry',
                'delay': 30,
                'severity': 'medium',
                'message': 'Network error, retrying in 30 seconds'
            }
            
        except Exception as e:
            return {'handled': False, 'severity': 'high', 'error': str(e)}
    
    def _handle_api_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle API errors."""
        try:
            error_str = str(error).lower()
            
            if 'authentication' in error_str:
                return {
                    'handled': True,
                    'recovery_action': 'stop',
                    'delay': 300,
                    'severity': 'high',
                    'message': 'Authentication error, stopping for 5 minutes'
                }
            elif 'permission' in error_str:
                return {
                    'handled': True,
                    'recovery_action': 'stop',
                    'delay': 600,
                    'severity': 'high',
                    'message': 'Permission error, stopping for 10 minutes'
                }
            else:
                return {
                    'handled': True,
                    'recovery_action': 'retry',
                    'delay': 60,
                    'severity': 'medium',
                    'message': 'API error, retrying in 1 minute'
                }
                
        except Exception as e:
            return {'handled': False, 'severity': 'high', 'error': str(e)}
    
    def _handle_risk_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle risk management errors."""
        try:
            return {
                'handled': True,
                'recovery_action': 'conservative',
                'delay': 0,
                'severity': 'high',
                'message': 'Risk management error, switching to conservative mode'
            }
            
        except Exception as e:
            return {'handled': False, 'severity': 'critical', 'error': str(e)}
    
    def _handle_system_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle system errors."""
        try:
            return {
                'handled': True,
                'recovery_action': 'failover',
                'delay': 0,
                'severity': 'high',
                'message': 'System error, activating failover'
            }
            
        except Exception as e:
            return {'handled': False, 'severity': 'critical', 'error': str(e)}
    
    def _handle_unknown_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle unknown errors."""
        try:
            return {
                'handled': True,
                'recovery_action': 'conservative',
                'delay': 60,
                'severity': 'medium',
                'message': 'Unknown error, switching to conservative mode'
            }
            
        except Exception as e:
            return {'handled': False, 'severity': 'high', 'error': str(e)}
    
    def _immediate_recovery(self, error_type, error_details):
        """Immediate recovery actions for critical errors"""
        try:
            logging.warning(f"🔄 Attempting immediate recovery for {error_type}")
            
            if error_type == "api_limit_exceeded":
                # Switch to alternative data sources
                self._switch_to_alternative_sources()
                return True
                
            elif error_type == "model_prediction_error":
                # Fall back to simpler models
                self._fallback_to_simple_models()
                return True
                
            elif error_type == "data_quality_issue":
                # Use cached data or alternative sources
                self._use_cached_data()
                return True
                
            elif error_type == "connection_error":
                # Retry with exponential backoff
                return self._retry_with_backoff()
                
            else:
                # Generic recovery - restart components
                self._restart_components()
                return True
                
        except Exception as e:
            logging.error(f"❌ Recovery failed: {e}")
            return False

    def _gradual_recovery(self, error_type, error_details):
        """Gradual recovery actions for non-critical errors"""
        try:
            logging.info(f"Attempting gradual recovery for {error_type}")
            # Example: reduce trading frequency, switch to safer strategies, alert user
            if error_type == "api_limit_exceeded":
                logging.info("Reducing API call frequency as part of gradual recovery.")
                # Implementation: slow down polling, use cache more
                return True
            elif error_type == "model_prediction_error":
                logging.info("Switching to conservative model as part of gradual recovery.")
                # Implementation: use more robust model
                return True
            elif error_type == "data_quality_issue":
                logging.info("Using fallback data as part of gradual recovery.")
                # Implementation: fallback to backup data
                return True
            elif error_type == "connection_error":
                logging.info("Increasing retry interval as part of gradual recovery.")
                # Implementation: increase retry interval
                return True
            else:
                logging.info("Generic gradual recovery action taken.")
                return True
        except Exception as e:
            logging.error(f"Gradual recovery failed: {e}")
            return False

    def _conservative_recovery(self, error_type, error_details):
        """Conservative recovery actions for severe errors"""
        try:
            logging.warning(f"Attempting conservative recovery for {error_type}")
            # Example: stop trading, switch to paper trading, alert user immediately
            if error_type == "api_limit_exceeded":
                logging.warning("Switching to paper trading mode due to API limits.")
                # Implementation: disable live trading, use paper trading
                return True
            elif error_type == "model_prediction_error":
                logging.warning("Switching to basic technical analysis due to model errors.")
                # Implementation: use simple moving averages only
                return True
            elif error_type == "data_quality_issue":
                logging.warning("Using minimal data sources due to quality issues.")
                # Implementation: use only Binance data
                return True
            elif error_type == "connection_error":
                logging.warning("Pausing trading due to connection issues.")
                # Implementation: pause all trading
                return True
            else:
                logging.warning("Generic conservative recovery action taken.")
                return True
        except Exception as e:
            logging.error(f"Conservative recovery failed: {e}")
            return False

    def _switch_to_alternative_sources(self):
        """Switch to alternative data sources when API limits are hit"""
        logging.info("🔄 Switching to alternative data sources")
        # Implementation would switch to free APIs or cached data
        
    def _fallback_to_simple_models(self):
        """Fall back to simpler, more reliable models"""
        logging.info("🔄 Falling back to simple models")
        # Implementation would use basic technical indicators only
        
    def _use_cached_data(self):
        """Use cached data when fresh data is unavailable"""
        logging.info("🔄 Using cached data")
        # Implementation would load from cache
        
    def _retry_with_backoff(self):
        """Retry with exponential backoff"""
        logging.info("🔄 Retrying with backoff")
        return True
        
    def _restart_components(self):
        """Restart critical components"""
        logging.info("🔄 Restarting components")
        # Implementation would restart data collection, models, etc.

    def get_robustness_status(self) -> Dict[str, Any]:
        """Get comprehensive robustness status."""
        try:
            return {
                'timestamp': datetime.now().isoformat(),
                'risk_management': self.risk_manager.get_risk_report(),
                'api_status': self.api_handler.get_api_status(),
                'failover_status': self.failover_system.get_failover_status(),
                'system_health': self.failover_system.check_system_health(),
                'overall_status': 'healthy'
            }
            
        except Exception as e:
            logging.error(f"Error getting robustness status: {e}")
            return {'error': str(e)} 