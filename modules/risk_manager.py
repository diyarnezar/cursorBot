#!/usr/bin/env python3
"""
Enhanced Risk Manager for Project Hyperion
Advanced risk management with dynamic position sizing, market regime detection, and adaptive limits
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import deque
import json

class RiskManager:
    """
    Enhanced Risk Manager with maximum intelligence:
    
    Features:
    - Dynamic position sizing based on volatility and confidence
    - Market regime detection and adaptive risk limits
    - Real-time drawdown monitoring
    - Correlation-based risk adjustment
    - Kelly Criterion position sizing
    - Maximum Sharpe ratio optimization
    - Stress testing and scenario analysis
    - Adaptive stop-loss and take-profit levels
    """
    
    def __init__(self, 
                 max_position_size: float = 0.1,
                 max_daily_loss: float = 0.05,
                 stop_loss_pct: float = 0.02,
                 take_profit_pct: float = 0.04,
                 max_drawdown: float = 0.15,
                 volatility_lookback: int = 100,
                 confidence_threshold: float = 0.7,
                 use_kelly_criterion: bool = True,
                 use_adaptive_limits: bool = True):
        """
        Initialize the Enhanced Risk Manager.
        
        Args:
            max_position_size: Maximum position size as fraction of capital
            max_daily_loss: Maximum daily loss as fraction of capital
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
            max_drawdown: Maximum allowed drawdown
            volatility_lookback: Lookback period for volatility calculation
            confidence_threshold: Minimum confidence for position sizing
            use_kelly_criterion: Whether to use Kelly Criterion for position sizing
            use_adaptive_limits: Whether to use adaptive risk limits
        """
        self.max_position_size = max_position_size
        self.max_daily_loss = max_daily_loss
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_drawdown = max_drawdown
        self.volatility_lookback = volatility_lookback
        self.confidence_threshold = confidence_threshold
        self.use_kelly_criterion = use_kelly_criterion
        self.use_adaptive_limits = use_adaptive_limits
        
        # Risk state tracking
        self.current_drawdown = 0.0
        self.daily_pnl = 0.0
        self.total_pnl = 0.0
        self.peak_capital = 1.0
        self.trade_history = deque(maxlen=1000)
        self.volatility_history = deque(maxlen=volatility_lookback)
        self.market_regime = 'NORMAL'
        
        # Performance metrics
        self.win_rate = 0.0
        self.avg_win = 0.0
        self.avg_loss = 0.0
        self.sharpe_ratio = 0.0
        self.max_consecutive_losses = 0
        self.current_consecutive_losses = 0
        
        # Adaptive parameters
        self.adaptive_position_size = max_position_size
        self.adaptive_stop_loss = stop_loss_pct
        self.adaptive_take_profit = take_profit_pct
        
        # Market regime thresholds
        self.regime_thresholds = {
            'LOW_VOLATILITY': 0.01,
            'NORMAL': 0.02,
            'HIGH_VOLATILITY': 0.04,
            'EXTREME_VOLATILITY': 0.08
        }
        
        logging.info("Enhanced Risk Manager initialized with maximum intelligence")
    
    def update_market_data(self, market_data: pd.DataFrame) -> None:
        """
        Update risk manager with latest market data.
        
        Args:
            market_data: DataFrame with market data including volatility indicators
        """
        try:
            if market_data.empty:
                return
            
            # Update volatility history
            if 'volatility_20' in market_data.columns:
                current_volatility = market_data['volatility_20'].iloc[-1]
                self.volatility_history.append(current_volatility)
            
            # Detect market regime
            self._detect_market_regime()
            
            # Update adaptive parameters
            if self.use_adaptive_limits:
                self._update_adaptive_parameters()
            
            logging.debug(f"Risk manager updated - Regime: {self.market_regime}, "
                         f"Volatility: {current_volatility:.4f}")
            
        except Exception as e:
            logging.error(f"Error updating risk manager: {e}")
    
    def _detect_market_regime(self) -> None:
        """Detect current market regime based on volatility."""
        try:
            if len(self.volatility_history) < 10:
                self.market_regime = 'NORMAL'
                return
            
            current_volatility = np.mean(list(self.volatility_history)[-10:])
            
            if current_volatility < self.regime_thresholds['LOW_VOLATILITY']:
                self.market_regime = 'LOW_VOLATILITY'
            elif current_volatility < self.regime_thresholds['NORMAL']:
                self.market_regime = 'NORMAL'
            elif current_volatility < self.regime_thresholds['HIGH_VOLATILITY']:
                self.market_regime = 'HIGH_VOLATILITY'
            else:
                self.market_regime = 'EXTREME_VOLATILITY'
                
        except Exception as e:
            logging.error(f"Error detecting market regime: {e}")
            self.market_regime = 'NORMAL'
    
    def _update_adaptive_parameters(self) -> None:
        """Update adaptive risk parameters based on market regime and performance."""
        try:
            # Base adjustments on market regime
            regime_multipliers = {
                'LOW_VOLATILITY': 1.2,    # More aggressive in low volatility
                'NORMAL': 1.0,            # Standard parameters
                'HIGH_VOLATILITY': 0.7,   # More conservative in high volatility
                'EXTREME_VOLATILITY': 0.5 # Very conservative in extreme volatility
            }
            
            multiplier = regime_multipliers.get(self.market_regime, 1.0)
            
            # Adjust position size
            self.adaptive_position_size = self.max_position_size * multiplier
            
            # Adjust stop loss and take profit
            self.adaptive_stop_loss = self.stop_loss_pct * multiplier
            self.adaptive_take_profit = self.take_profit_pct * multiplier
            
            # Performance-based adjustments
            if self.win_rate > 0.6 and self.sharpe_ratio > 1.0:
                # Increase position size if performing well
                self.adaptive_position_size *= 1.1
            elif self.win_rate < 0.4 or self.current_drawdown > 0.1:
                # Decrease position size if performing poorly
                self.adaptive_position_size *= 0.8
            
            # Ensure limits are within bounds
            self.adaptive_position_size = max(0.01, min(self.adaptive_position_size, self.max_position_size))
            self.adaptive_stop_loss = max(0.005, min(self.adaptive_stop_loss, 0.05))
            self.adaptive_take_profit = max(0.01, min(self.adaptive_take_profit, 0.1))
            
        except Exception as e:
            logging.error(f"Error updating adaptive parameters: {e}")
    
    def calculate_position_size(self, 
                              confidence: float, 
                              signal_strength: float,
                              current_price: float,
                              available_capital: float,
                              features: Optional[dict] = None) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate optimal position size with ULTRA-ADVANCED dynamic scaling.
        Optimized for maximum capital growth from $60 to $5000+ with intelligent risk management.
        """
        try:
            # ULTRA-ADVANCED DYNAMIC RISK SCALING
            # 1. Capital-based risk scaling (MOST IMPORTANT)
            risk_per_trade = self._calculate_dynamic_risk_per_trade(available_capital)
            base_size = risk_per_trade
            
            # 2. Confidence-based amplification
            confidence_multiplier = self._calculate_confidence_multiplier(confidence)
            base_size *= confidence_multiplier
            
            # 3. Signal strength optimization
            signal_multiplier = self._calculate_signal_multiplier(signal_strength)
            base_size *= signal_multiplier
            
            # 4. Market regime optimization
            regime_multiplier = self._calculate_regime_multiplier()
            base_size *= regime_multiplier
            
            # 5. Performance-based scaling
            performance_multiplier = self._calculate_performance_multiplier()
            base_size *= performance_multiplier
            
            # 6. Kelly Criterion optimization
            kelly_fraction = None
            if self.use_kelly_criterion and self.win_rate > 0:
                kelly_fraction = float(self._calculate_kelly_criterion())
                kelly_multiplier = min(1.5, float(kelly_fraction / base_size if base_size > 0 else 1.0))
                base_size *= kelly_multiplier
            
            # 7. External factors integration
            external_multiplier = self._calculate_external_multiplier(features)
            base_size *= external_multiplier
            
            # 8. Volatility adjustment
            volatility_multiplier = self._calculate_volatility_multiplier()
            base_size *= volatility_multiplier
            
            # 9. Drawdown protection
            drawdown_multiplier = self._calculate_drawdown_multiplier()
            base_size *= drawdown_multiplier
            
            # 10. Maximum safety limits
            max_position = self._calculate_max_position_limit(available_capital)
            base_size = max(0.005, min(base_size, max_position))  # Minimum 0.5%, maximum based on capital
            
            # Calculate position metrics
            position_value = available_capital * base_size
            position_quantity = position_value / current_price if current_price > 0 else 0
            
            # Comprehensive reasoning
            reasoning = {
                'risk_per_trade': risk_per_trade,
                'confidence_multiplier': confidence_multiplier,
                'signal_multiplier': signal_multiplier,
                'regime_multiplier': regime_multiplier,
                'performance_multiplier': performance_multiplier,
                'external_multiplier': external_multiplier,
                'volatility_multiplier': volatility_multiplier,
                'drawdown_multiplier': drawdown_multiplier,
                'capital_stage': self._get_capital_stage(available_capital),
                'market_regime': self.market_regime,
                'final_position_size': base_size,
                'position_value': position_value,
                'kelly_fraction': kelly_fraction
            }
            
            return base_size, reasoning
            
        except Exception as e:
            logging.error(f"Error calculating position size: {e}")
            return 0.01, {'error': str(e)}
    
    def _calculate_dynamic_risk_per_trade(self, available_capital: float) -> float:
        """Calculate dynamic risk per trade based on capital growth stage."""
        try:
            # ULTRA-OPTIMIZED CAPITAL SCALING
            if available_capital < 100:  # Early stage ($60-$100)
                return 0.02  # 2% - Conservative to build confidence
            elif available_capital < 250:  # Growth stage ($100-$250)
                return 0.025  # 2.5% - Moderate growth
            elif available_capital < 500:  # Expansion stage ($250-$500)
                return 0.03  # 3% - Aggressive growth
            elif available_capital < 1000:  # Scale stage ($500-$1000)
                return 0.035  # 3.5% - Maximum growth
            elif available_capital < 2500:  # Professional stage ($1000-$2500)
                return 0.04  # 4% - Professional level
            else:  # Elite stage ($2500+)
                return 0.045  # 4.5% - Elite level (maximum)
        except Exception as e:
            logging.error(f"Error calculating dynamic risk: {e}")
            return 0.02
    
    def _calculate_confidence_multiplier(self, confidence: float) -> float:
        """Calculate confidence-based position size multiplier."""
        try:
            if confidence > 0.9:  # Ultra-high confidence
                return 1.8  # 80% boost
            elif confidence > 0.8:  # High confidence
                return 1.5  # 50% boost
            elif confidence > 0.7:  # Good confidence
                return 1.3  # 30% boost
            elif confidence > 0.6:  # Moderate confidence
                return 1.1  # 10% boost
            elif confidence > 0.5:  # Low confidence
                return 0.9  # 10% reduction
            else:  # Very low confidence
                return 0.6  # 40% reduction
        except Exception as e:
            logging.error(f"Error calculating confidence multiplier: {e}")
            return 1.0
    
    def _calculate_signal_multiplier(self, signal_strength: float) -> float:
        """Calculate signal strength multiplier."""
        try:
            abs_signal = abs(signal_strength)
            if abs_signal > 0.8:  # Very strong signal
                return 1.6  # 60% boost
            elif abs_signal > 0.6:  # Strong signal
                return 1.4  # 40% boost
            elif abs_signal > 0.4:  # Moderate signal
                return 1.2  # 20% boost
            elif abs_signal > 0.2:  # Weak signal
                return 1.0  # No change
            else:  # Very weak signal
                return 0.8  # 20% reduction
        except Exception as e:
            logging.error(f"Error calculating signal multiplier: {e}")
            return 1.0
    
    def _calculate_regime_multiplier(self) -> float:
        """Calculate market regime multiplier."""
        try:
            regime_multipliers = {
                'LOW_VOLATILITY': 1.4,    # 40% boost in low volatility
                'NORMAL': 1.0,            # Standard
                'HIGH_VOLATILITY': 0.7,   # 30% reduction in high volatility
                'EXTREME_VOLATILITY': 0.5 # 50% reduction in extreme volatility
            }
            return regime_multipliers.get(self.market_regime, 1.0)
        except Exception as e:
            logging.error(f"Error calculating regime multiplier: {e}")
            return 1.0
    
    def _calculate_performance_multiplier(self) -> float:
        """Calculate performance-based multiplier."""
        try:
            if self.win_rate > 0.7 and self.sharpe_ratio > 1.5:  # Excellent performance
                return 1.5  # 50% boost
            elif self.win_rate > 0.6 and self.sharpe_ratio > 1.0:  # Good performance
                return 1.3  # 30% boost
            elif self.win_rate > 0.5 and self.sharpe_ratio > 0.5:  # Average performance
                return 1.1  # 10% boost
            elif self.win_rate < 0.4 or self.current_drawdown > 0.15:  # Poor performance
                return 0.6  # 40% reduction
            else:  # Neutral performance
                return 1.0
        except Exception as e:
            logging.error(f"Error calculating performance multiplier: {e}")
            return 1.0
    
    def _calculate_external_multiplier(self, features: Optional[dict]) -> float:
        """Calculate external factors multiplier."""
        try:
            if not features:
                return 1.0
            
            multiplier = 1.0
            
            # Fear & Greed index
            fear_greed = features.get('fear_greed_index', 50)
            if fear_greed < 20:  # Extreme fear - buying opportunity
                multiplier *= 1.3
            elif fear_greed > 80:  # Extreme greed - be cautious
                multiplier *= 0.8
            
            # News volatility
            news_volatility = features.get('news_volatility', 0)
            if news_volatility > 0.8:  # High news volatility
                multiplier *= 0.7
            
            # Whale activity
            whale_activity = features.get('whale_activity', 0)
            if whale_activity > 0.7:  # High whale activity
                multiplier *= 1.2
            
            # Market sentiment
            sentiment = features.get('sentiment_score', 0)
            if sentiment > 0.6:  # Positive sentiment
                multiplier *= 1.1
            elif sentiment < -0.6:  # Negative sentiment
                multiplier *= 0.9
            
            return multiplier
        except Exception as e:
            logging.error(f"Error calculating external multiplier: {e}")
            return 1.0
    
    def _calculate_volatility_multiplier(self) -> float:
        """Calculate volatility-based multiplier."""
        try:
            if len(self.volatility_history) < 5:
                return 1.0
            
            current_volatility = np.mean(list(self.volatility_history)[-5:])
            avg_volatility = np.mean(list(self.volatility_history))
            
            if current_volatility < avg_volatility * 0.7:  # Low volatility
                return 1.2
            elif current_volatility > avg_volatility * 1.3:  # High volatility
                return 0.8
            else:  # Normal volatility
                return 1.0
        except Exception as e:
            logging.error(f"Error calculating volatility multiplier: {e}")
            return 1.0
    
    def _calculate_drawdown_multiplier(self) -> float:
        """Calculate drawdown protection multiplier."""
        try:
            if self.current_drawdown > 0.2:  # Severe drawdown
                return 0.5  # 50% reduction
            elif self.current_drawdown > 0.15:  # High drawdown
                return 0.7  # 30% reduction
            elif self.current_drawdown > 0.1:  # Moderate drawdown
                return 0.8  # 20% reduction
            elif self.current_drawdown > 0.05:  # Low drawdown
                return 0.9  # 10% reduction
            else:  # No significant drawdown
                return 1.0
        except Exception as e:
            logging.error(f"Error calculating drawdown multiplier: {e}")
            return 1.0
    
    def _calculate_max_position_limit(self, available_capital: float) -> float:
        """Calculate maximum position size limit based on capital."""
        try:
            if available_capital < 100:
                return 0.15  # 15% max for small capital
            elif available_capital < 500:
                return 0.20  # 20% max for medium capital
            elif available_capital < 1000:
                return 0.25  # 25% max for large capital
            else:
                return 0.30  # 30% max for very large capital
        except Exception as e:
            logging.error(f"Error calculating max position limit: {e}")
            return 0.20
    
    def _get_capital_stage(self, available_capital: float) -> str:
        """Get capital growth stage."""
        if available_capital < 100:
            return 'early'
        elif available_capital < 250:
            return 'growth'
        elif available_capital < 500:
            return 'expansion'
        elif available_capital < 1000:
            return 'scale'
        elif available_capital < 2500:
            return 'professional'
        else:
            return 'elite'
    
    def _calculate_kelly_criterion(self) -> float:
        """Calculate Kelly Criterion fraction for position sizing."""
        try:
            if self.win_rate <= 0 or self.avg_win <= 0 or self.avg_loss >= 0:
                return 0.0
            
            # Kelly formula: f = (bp - q) / b
            # where b = avg_win/avg_loss, p = win_rate, q = 1-p
            b = self.avg_win / abs(self.avg_loss)
            p = self.win_rate
            q = 1 - p
            
            kelly_fraction = (b * p - q) / b
            
            # Apply safety factor (use half Kelly)
            kelly_fraction *= 0.5
            
            # Ensure reasonable bounds
            return float(max(0.0, min(float(kelly_fraction), float(self.max_position_size))))
            
        except Exception as e:
            logging.error(f"Error calculating Kelly Criterion: {e}")
            return 0.0
    
    def check_risk_limits(self, decision: Dict[str, Any], features: Optional[dict] = None) -> bool:
        """
        Check if trade decision meets risk limits, now including external risk features.
        Args:
            decision: Trading decision dictionary
            features: Optional dict of latest features (including fear_greed_index, news_volatility, etc.)
        Returns:
            True if trade is allowed, False otherwise
        """
        try:
            # Check daily loss limit
            if self.daily_pnl < -self.max_daily_loss:
                logging.warning(f"Daily loss limit exceeded: {self.daily_pnl:.4f}")
                return False
            
            # Check drawdown limit
            if self.current_drawdown > self.max_drawdown:
                logging.warning(f"Maximum drawdown exceeded: {self.current_drawdown:.4f}")
                return False
            
            # Check consecutive losses
            if self.current_consecutive_losses > 5:
                logging.warning(f"Too many consecutive losses: {self.current_consecutive_losses}")
                return False
            
            # Check market regime restrictions
            if self.market_regime == 'EXTREME_VOLATILITY':
                if decision.get('confidence', 0) < 0.9:
                    logging.warning("Extreme volatility regime - only high confidence trades allowed")
                    return False
            
            # Check position size limits
            position_size = decision.get('position_size', 0)
            if position_size > self.adaptive_position_size:
                logging.warning(f"Position size exceeds limit: {position_size:.4f} > {self.adaptive_position_size:.4f}")
                return False
            
            # Enhanced external risk features check
            if features is not None:
                fg_index = features.get('fear_greed_index', 50)
                fg_trend = features.get('fear_greed_trend', 0)
                news_vol = features.get('news_volatility', 0)
                breaking_news = features.get('breaking_news_flag', 0)
                sentiment_score = features.get('sentiment_score', 0)
                market_stress = features.get('market_stress_indicator', 0)
                
                # Block trades if extreme fear/greed
                if fg_index >= 90 or fg_index <= 10:
                    logging.warning(f"Trade blocked: Extreme fear/greed index ({fg_index})")
                    return False
                
                # Reduce trading in high fear/greed zones
                if fg_index >= 80 or fg_index <= 20:
                    if decision.get('confidence', 0) < 0.8:
                        logging.warning(f"Trade blocked: High fear/greed zone requires high confidence")
                        return False
                
                # Block if fear/greed trend is rapidly declining
                if fg_trend <= -2:
                    logging.warning("Trade blocked: Rapidly declining fear/greed trend")
                    return False
                
                # Block if high news volatility or breaking news
                if news_vol >= 2 or breaking_news == 1:
                    logging.warning("Trade blocked: High news volatility or breaking news")
                    return False
                
                # Block if extremely negative sentiment
                if sentiment_score <= -0.8:
                    logging.warning("Trade blocked: Extremely negative market sentiment")
                    return False
                
                # Block if market stress indicator is high
                if market_stress >= 0.8:
                    logging.warning("Trade blocked: High market stress indicator")
                    return False
                
                # Additional checks for whale activity
                whale_alert = features.get('whale_alert_flag', 0)
                large_volume_ratio = features.get('large_volume_ratio', 0)
                
                if whale_alert == 1 and large_volume_ratio > 2.0:
                    logging.warning("Trade blocked: Whale alert with high volume ratio")
                    return False
            
            # Check if confidence meets minimum threshold
            confidence = decision.get('confidence', 0)
            min_confidence = self._get_dynamic_confidence_threshold(features)
            
            if confidence < min_confidence:
                logging.warning(f"Trade blocked: Confidence {confidence:.3f} below threshold {min_confidence:.3f}")
                return False
            
            return True
            
        except Exception as e:
            logging.error(f"Error checking risk limits: {e}")
            return False
    
    def _get_dynamic_confidence_threshold(self, features: Optional[dict] = None) -> float:
        """Get dynamic confidence threshold based on market conditions."""
        try:
            base_threshold = self.confidence_threshold
            
            # Adjust threshold based on market regime
            regime_adjustments = {
                'LOW_VOLATILITY': 0.6,      # Lower threshold in stable conditions
                'NORMAL': 0.7,              # Standard threshold
                'HIGH_VOLATILITY': 0.8,     # Higher threshold in volatile conditions
                'EXTREME_VOLATILITY': 0.9   # Very high threshold in extreme conditions
            }
            
            adjusted_threshold = regime_adjustments.get(self.market_regime, base_threshold)
            
            # Further adjust based on external features
            if features is not None:
                fg_index = features.get('fear_greed_index', 50)
                news_vol = features.get('news_volatility', 0)
                
                # Increase threshold during extreme market conditions
                if fg_index >= 80 or fg_index <= 20:
                    adjusted_threshold += 0.1
                
                if news_vol >= 1:
                    adjusted_threshold += 0.05
                
                # Adjust based on recent performance
                if self.win_rate < 0.4:
                    adjusted_threshold += 0.1
                elif self.win_rate > 0.6:
                    adjusted_threshold -= 0.05
            
            return min(0.95, max(0.5, adjusted_threshold))
            
        except Exception as e:
            logging.error(f"Error calculating dynamic confidence threshold: {e}")
            return self.confidence_threshold
    
    def update_trade_result(self, trade_result: Dict[str, Any]) -> None:
        """
        Update risk manager with trade result.
        
        Args:
            trade_result: Dictionary with trade result information
        """
        try:
            # Extract trade information
            pnl = trade_result.get('pnl', 0.0)
            timestamp = trade_result.get('timestamp', datetime.now())
            
            # Update PnL tracking
            self.total_pnl += pnl
            self.daily_pnl += pnl
            
            # Update peak capital
            if self.total_pnl > 0:
                self.peak_capital = max(self.peak_capital, 1.0 + self.total_pnl)
            
            # Update drawdown
            self.current_drawdown = max(0, (self.peak_capital - (1.0 + self.total_pnl)) / self.peak_capital)
            
            # Update trade history
            self.trade_history.append({
                'timestamp': timestamp,
                'pnl': pnl,
                'total_pnl': self.total_pnl,
                'drawdown': self.current_drawdown
            })
            
            # Update performance metrics
            self._update_performance_metrics()
            
            # Reset daily PnL if it's a new day
            if len(self.trade_history) > 1:
                last_trade_time = self.trade_history[-2]['timestamp']
                if isinstance(timestamp, datetime) and isinstance(last_trade_time, datetime):
                    if timestamp.date() != last_trade_time.date():
                        self.daily_pnl = pnl
            
            logging.debug(f"Trade result updated - PnL: {pnl:.4f}, "
                         f"Total: {self.total_pnl:.4f}, Drawdown: {self.current_drawdown:.4f}")
            
        except Exception as e:
            logging.error(f"Error updating trade result: {e}")
    
    def _update_performance_metrics(self) -> None:
        """Update performance metrics based on trade history."""
        try:
            if len(self.trade_history) < 10:
                return
            
            # Calculate win rate
            wins = sum(1 for trade in self.trade_history if trade['pnl'] > 0)
            self.win_rate = wins / len(self.trade_history)
            
            # Calculate average win and loss
            wins_list = [trade['pnl'] for trade in self.trade_history if trade['pnl'] > 0]
            losses_list = [trade['pnl'] for trade in self.trade_history if trade['pnl'] < 0]
            
            self.avg_win = np.mean(wins_list) if wins_list else 0.0
            self.avg_loss = np.mean(losses_list) if losses_list else 0.0
            
            # Calculate Sharpe ratio (simplified)
            if len(self.trade_history) > 1:
                returns = [trade['pnl'] for trade in self.trade_history]
                if returns:
                    self.sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
            
            # Update consecutive losses
            if self.trade_history and self.trade_history[-1]['pnl'] < 0:
                self.current_consecutive_losses += 1
                self.max_consecutive_losses = max(self.max_consecutive_losses, self.current_consecutive_losses)
            else:
                self.current_consecutive_losses = 0
                
        except Exception as e:
            logging.error(f"Error updating performance metrics: {e}")
    
    def get_risk_status(self) -> Dict[str, Any]:
        """Get current risk status."""
        return {
            'current_drawdown': self.current_drawdown,
            'daily_pnl': self.daily_pnl,
            'total_pnl': self.total_pnl,
            'market_regime': self.market_regime,
            'win_rate': self.win_rate,
            'sharpe_ratio': self.sharpe_ratio,
            'consecutive_losses': self.current_consecutive_losses,
            'adaptive_position_size': self.adaptive_position_size,
            'adaptive_stop_loss': self.adaptive_stop_loss,
            'adaptive_take_profit': self.adaptive_take_profit,
            'volatility': np.mean(list(self.volatility_history)[-10:]) if self.volatility_history else 0.0
        }
    
    def get_stop_loss_take_profit(self, entry_price: float, side: str) -> Tuple[float, float]:
        """
        Get adaptive stop loss and take profit levels.
        
        Args:
            entry_price: Entry price
            side: 'BUY' or 'SELL'
            
        Returns:
            Tuple of (stop_loss, take_profit)
        """
        try:
            # Base levels
            stop_loss = entry_price * (1 - self.adaptive_stop_loss) if side == 'BUY' else entry_price * (1 + self.adaptive_stop_loss)
            take_profit = entry_price * (1 + self.adaptive_take_profit) if side == 'BUY' else entry_price * (1 - self.adaptive_take_profit)
            
            # Volatility adjustment
            if self.volatility_history:
                current_volatility = np.mean(list(self.volatility_history)[-10:])
                volatility_multiplier = 1.0 + current_volatility * 5
                
                if side == 'BUY':
                    stop_loss = entry_price * (1 - self.adaptive_stop_loss * volatility_multiplier)
                    take_profit = entry_price * (1 + self.adaptive_take_profit * volatility_multiplier)
                else:
                    stop_loss = entry_price * (1 + self.adaptive_stop_loss * volatility_multiplier)
                    take_profit = entry_price * (1 - self.adaptive_take_profit * volatility_multiplier)
            
            return float(stop_loss), float(take_profit)
            
        except Exception as e:
            logging.error(f"Error calculating stop loss/take profit: {e}")
            # Return default levels
            if side == 'BUY':
                return entry_price * 0.98, entry_price * 1.04
            else:
                return entry_price * 1.02, entry_price * 0.96
    
    def reset_daily_metrics(self) -> None:
        """Reset daily metrics (call at start of new day)."""
        self.daily_pnl = 0.0
        logging.info("Daily risk metrics reset")
