"""
Maximum Intelligence Risk Management
===================================

Part 3: Advanced Risk Management & Psychology
Focus: Maximum Profits, Minimum Losses

This module implements the smartest possible risk management that:
- Uses advanced position sizing for maximum profit potential
- Implements sophisticated stop-loss and take-profit strategies
- Incorporates trading psychology to avoid emotional decisions
- Uses dynamic risk adjustment based on market conditions
- Implements portfolio-level risk management
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import logging
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class MaximumIntelligenceRiskManager:
    """
    Maximum Intelligence Risk Manager
    Focus: Maximum profits with minimum risk through advanced risk management
    """
    
    def __init__(self, initial_capital: float = 10000.0):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = {}
        self.risk_metrics = {}
        self.psychology_state = {}
        self.market_regime = 'normal'
        self.volatility_regime = 'medium'
        
    def calculate_optimal_position_size(self, 
                                     signal_strength: float,
                                     volatility: float,
                                     confidence: float,
                                     current_drawdown: float = 0.0) -> float:
        """
        Calculate optimal position size using Kelly Criterion and advanced risk management
        Focus: Maximum profit potential while controlling risk
        """
        
        # Base Kelly Criterion
        win_rate = self._estimate_win_rate(signal_strength, confidence)
        avg_win = self._estimate_avg_win(signal_strength)
        avg_loss = self._estimate_avg_loss(volatility)
        
        # Kelly fraction
        kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        
        # Apply risk adjustments
        kelly_fraction = self._apply_risk_adjustments(kelly_fraction, 
                                                    volatility, 
                                                    current_drawdown)
        
        # Apply psychology adjustments
        kelly_fraction = self._apply_psychology_adjustments(kelly_fraction)
        
        # Apply market regime adjustments
        kelly_fraction = self._apply_market_regime_adjustments(kelly_fraction)
        
        # Ensure position size is within limits
        max_position = min(kelly_fraction, 0.25)  # Max 25% of capital
        min_position = max(max_position, 0.01)    # Min 1% of capital
        
        logger.info(f"🎯 Optimal position size: {min_position:.3f} ({min_position*100:.1f}% of capital)")
        logger.info(f"   • Signal strength: {signal_strength:.3f}")
        logger.info(f"   • Volatility: {volatility:.3f}")
        logger.info(f"   • Confidence: {confidence:.3f}")
        
        return min_position
    
    def calculate_dynamic_stop_loss(self, 
                                  entry_price: float,
                                  signal_direction: int,
                                  volatility: float,
                                  atr: float) -> float:
        """
        Calculate dynamic stop-loss using advanced techniques
        Focus: Protect capital while allowing for normal market fluctuations
        """
        
        # Base ATR-based stop loss
        base_stop_distance = atr * 2.0  # 2x ATR for normal volatility
        
        # Adjust for volatility regime
        if self.volatility_regime == 'high':
            base_stop_distance *= 1.5
        elif self.volatility_regime == 'low':
            base_stop_distance *= 0.7
        
        # Adjust for signal strength (stronger signals get tighter stops)
        signal_adjustment = 1.0 - (abs(signal_direction) * 0.2)
        base_stop_distance *= signal_adjustment
        
        # Calculate stop loss price
        if signal_direction > 0:  # Long position
            stop_loss = entry_price - base_stop_distance
        else:  # Short position
            stop_loss = entry_price + base_stop_distance
        
        # Ensure stop loss is reasonable (not too close, not too far)
        min_stop_distance = entry_price * 0.005  # 0.5% minimum
        max_stop_distance = entry_price * 0.15   # 15% maximum
        
        if abs(entry_price - stop_loss) < min_stop_distance:
            if signal_direction > 0:
                stop_loss = entry_price - min_stop_distance
            else:
                stop_loss = entry_price + min_stop_distance
        elif abs(entry_price - stop_loss) > max_stop_distance:
            if signal_direction > 0:
                stop_loss = entry_price - max_stop_distance
            else:
                stop_loss = entry_price + max_stop_distance
        
        logger.info(f"🛑 Dynamic stop loss: {stop_loss:.4f}")
        logger.info(f"   • Entry price: {entry_price:.4f}")
        logger.info(f"   • Stop distance: {abs(entry_price - stop_loss):.4f}")
        
        return stop_loss
    
    def calculate_dynamic_take_profit(self, 
                                    entry_price: float,
                                    signal_strength: float,
                                    volatility: float,
                                    atr: float) -> float:
        """
        Calculate dynamic take-profit using advanced techniques
        Focus: Maximize profit potential while being realistic
        """
        
        # Base take-profit based on risk-reward ratio
        risk_reward_ratio = self._calculate_optimal_risk_reward(signal_strength, volatility)
        
        # Calculate base take-profit distance
        base_tp_distance = atr * risk_reward_ratio
        
        # Adjust for signal strength (stronger signals get higher targets)
        signal_adjustment = 1.0 + (signal_strength * 0.5)
        base_tp_distance *= signal_adjustment
        
        # Adjust for volatility (higher volatility = higher targets)
        volatility_adjustment = 1.0 + (volatility * 2.0)
        base_tp_distance *= volatility_adjustment
        
        # Calculate take-profit price (assuming long position for now)
        take_profit = entry_price + base_tp_distance
        
        # Ensure take-profit is reasonable
        min_tp_distance = entry_price * 0.01   # 1% minimum
        max_tp_distance = entry_price * 0.50   # 50% maximum
        
        if take_profit - entry_price < min_tp_distance:
            take_profit = entry_price + min_tp_distance
        elif take_profit - entry_price > max_tp_distance:
            take_profit = entry_price + max_tp_distance
        
        logger.info(f"🎯 Dynamic take profit: {take_profit:.4f}")
        logger.info(f"   • Risk-reward ratio: {risk_reward_ratio:.2f}")
        logger.info(f"   • TP distance: {take_profit - entry_price:.4f}")
        
        return take_profit
    
    def update_psychology_state(self, 
                              recent_trades: List[Dict],
                              current_market_conditions: Dict) -> Dict:
        """
        Update trading psychology state based on recent performance
        Focus: Avoid emotional trading decisions
        """
        
        if not recent_trades:
            return self._get_neutral_psychology_state()
        
        # Calculate recent performance metrics
        recent_returns = [trade['return'] for trade in recent_trades[-10:]]
        recent_wins = [r for r in recent_returns if r > 0]
        recent_losses = [r for r in recent_returns if r < 0]
        
        # Calculate psychology metrics
        win_rate = len(recent_wins) / len(recent_returns) if recent_returns else 0.5
        avg_win = np.mean(recent_wins) if recent_wins else 0.0
        avg_loss = np.mean(recent_losses) if recent_losses else 0.0
        max_drawdown = self._calculate_max_drawdown(recent_returns)
        
        # Determine psychology state
        if win_rate > 0.7 and avg_win > abs(avg_loss):
            psychology_state = 'confident'
        elif win_rate < 0.3 or max_drawdown > 0.1:
            psychology_state = 'cautious'
        elif win_rate < 0.4:
            psychology_state = 'fearful'
        else:
            psychology_state = 'neutral'
        
        # Calculate confidence level
        confidence = self._calculate_confidence_level(win_rate, avg_win, avg_loss, max_drawdown)
        
        # Update psychology state
        self.psychology_state = {
            'state': psychology_state,
            'confidence': confidence,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'max_drawdown': max_drawdown,
            'recent_trades_count': len(recent_returns)
        }
        
        logger.info(f"🧠 Psychology state: {psychology_state} (confidence: {confidence:.3f})")
        logger.info(f"   • Win rate: {win_rate:.3f}")
        logger.info(f"   • Avg win/loss: {avg_win:.3f}/{avg_loss:.3f}")
        
        return self.psychology_state
    
    def should_trade(self, 
                    signal_strength: float,
                    market_conditions: Dict,
                    current_drawdown: float) -> bool:
        """
        Determine if we should trade based on risk management rules
        Focus: Only trade when conditions are optimal
        """
        
        # Check maximum drawdown limit
        if current_drawdown > 0.15:  # 15% maximum drawdown
            logger.warning("🚫 Trading stopped: Maximum drawdown exceeded")
            return False
        
        # Check psychology state
        if self.psychology_state.get('state') == 'fearful':
            if signal_strength < 0.8:  # Only trade very strong signals when fearful
                logger.warning("🚫 Trading stopped: Psychology state too fearful")
                return False
        
        # Check market conditions
        if market_conditions.get('volatility') > 0.05:  # 5% volatility threshold
            if signal_strength < 0.7:  # Need stronger signals in high volatility
                logger.warning("🚫 Trading stopped: High volatility, weak signal")
                return False
        
        # Check consecutive losses
        recent_trades = self._get_recent_trades()
        if len(recent_trades) >= 3:
            last_3_returns = [trade['return'] for trade in recent_trades[-3:]]
            if all(r < 0 for r in last_3_returns):
                logger.warning("🚫 Trading stopped: 3 consecutive losses")
                return False
        
        logger.info("✅ Trading conditions met")
        return True
    
    def update_market_regime(self, market_data: pd.DataFrame) -> str:
        """
        Update market regime classification
        Focus: Adapt risk management to market conditions
        """
        
        # Calculate regime indicators
        returns = market_data['close'].pct_change().dropna()
        volatility = returns.rolling(20).std()
        trend_strength = abs(market_data['close'] - market_data['close'].rolling(20).mean()) / market_data['close'].rolling(20).mean()
        volume_trend = market_data['volume'].rolling(20).mean() / market_data['volume'].rolling(50).mean()
        
        # Determine regime
        current_volatility = volatility.iloc[-1]
        current_trend = trend_strength.iloc[-1]
        current_volume = volume_trend.iloc[-1]
        
        if current_volatility > 0.03:  # High volatility
            if current_trend > 0.02:  # Strong trend
                regime = 'trending_volatile'
            else:
                regime = 'sideways_volatile'
        else:  # Low volatility
            if current_trend > 0.02:  # Strong trend
                regime = 'trending_calm'
            else:
                regime = 'sideways_calm'
        
        # Update volatility regime
        if current_volatility > 0.04:
            self.volatility_regime = 'high'
        elif current_volatility < 0.01:
            self.volatility_regime = 'low'
        else:
            self.volatility_regime = 'medium'
        
        self.market_regime = regime
        
        logger.info(f"📊 Market regime: {regime} (volatility: {self.volatility_regime})")
        
        return regime
    
    def _estimate_win_rate(self, signal_strength: float, confidence: float) -> float:
        """Estimate win rate based on signal strength and confidence"""
        base_win_rate = 0.5 + (signal_strength * 0.3)  # 50% to 80%
        confidence_adjustment = confidence * 0.1  # 0% to 10%
        return min(base_win_rate + confidence_adjustment, 0.9)  # Cap at 90%
    
    def _estimate_avg_win(self, signal_strength: float) -> float:
        """Estimate average win size"""
        return 0.02 + (signal_strength * 0.03)  # 2% to 5%
    
    def _estimate_avg_loss(self, volatility: float) -> float:
        """Estimate average loss size"""
        return 0.015 + (volatility * 0.5)  # 1.5% to 6.5%
    
    def _apply_risk_adjustments(self, kelly_fraction: float, volatility: float, drawdown: float) -> float:
        """Apply risk adjustments to Kelly fraction"""
        
        # Volatility adjustment
        volatility_penalty = volatility * 0.5
        kelly_fraction *= (1 - volatility_penalty)
        
        # Drawdown adjustment
        drawdown_penalty = drawdown * 0.3
        kelly_fraction *= (1 - drawdown_penalty)
        
        return max(kelly_fraction, 0.01)  # Minimum 1%
    
    def _apply_psychology_adjustments(self, kelly_fraction: float) -> float:
        """Apply psychology-based adjustments"""
        
        psychology_state = self.psychology_state.get('state', 'neutral')
        confidence = self.psychology_state.get('confidence', 0.5)
        
        if psychology_state == 'confident':
            adjustment = 1.0 + (confidence - 0.5) * 0.2
        elif psychology_state == 'cautious':
            adjustment = 0.8
        elif psychology_state == 'fearful':
            adjustment = 0.5
        else:  # neutral
            adjustment = 1.0
        
        return kelly_fraction * adjustment
    
    def _apply_market_regime_adjustments(self, kelly_fraction: float) -> float:
        """Apply market regime adjustments"""
        
        regime_adjustments = {
            'trending_volatile': 0.7,    # Reduce size in volatile trending markets
            'sideways_volatile': 0.5,    # Reduce size in volatile sideways markets
            'trending_calm': 1.2,        # Increase size in calm trending markets
            'sideways_calm': 0.8         # Reduce size in calm sideways markets
        }
        
        adjustment = regime_adjustments.get(self.market_regime, 1.0)
        return kelly_fraction * adjustment
    
    def _calculate_optimal_risk_reward(self, signal_strength: float, volatility: float) -> float:
        """Calculate optimal risk-reward ratio"""
        base_ratio = 2.0  # 2:1 base ratio
        signal_adjustment = signal_strength * 1.0  # 0 to 1 additional
        volatility_adjustment = volatility * 2.0   # 0 to 0.2 additional
        return base_ratio + signal_adjustment + volatility_adjustment
    
    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """Calculate maximum drawdown from returns"""
        if not returns:
            return 0.0
        
        cumulative = np.cumprod([1 + r for r in returns])
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return abs(min(drawdown))
    
    def _calculate_confidence_level(self, win_rate: float, avg_win: float, avg_loss: float, max_drawdown: float) -> float:
        """Calculate confidence level based on performance metrics"""
        
        # Base confidence from win rate
        confidence = win_rate
        
        # Adjust for profit factor
        if avg_loss != 0:
            profit_factor = avg_win / abs(avg_loss)
            confidence += (profit_factor - 1) * 0.1
        
        # Penalize for high drawdown
        confidence -= max_drawdown * 0.5
        
        return np.clip(confidence, 0.1, 0.9)
    
    def _get_neutral_psychology_state(self) -> Dict:
        """Get neutral psychology state"""
        return {
            'state': 'neutral',
            'confidence': 0.5,
            'win_rate': 0.5,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'max_drawdown': 0.0,
            'recent_trades_count': 0
        }
    
    def _get_recent_trades(self) -> List[Dict]:
        """Get recent trades (placeholder - implement based on your trade tracking)"""
        # This should be implemented based on your trade tracking system
        return []
    
    def get_risk_summary(self) -> Dict:
        """Get comprehensive risk management summary"""
        return {
            'current_capital': self.current_capital,
            'market_regime': self.market_regime,
            'volatility_regime': self.volatility_regime,
            'psychology_state': self.psychology_state,
            'risk_metrics': self.risk_metrics
        } 