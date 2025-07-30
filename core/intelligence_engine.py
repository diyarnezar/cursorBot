"""
Advanced Intelligence Engine for Project Hyperion
Implements 10X intelligence features including Kelly Criterion, Sharpe optimization, and risk management
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

from utils.logging.logger import get_logger

@dataclass
class RiskManagementConfig:
    """Risk management configuration"""
    max_position_size: float = 0.1  # 10% max position
    max_drawdown: float = 0.05      # 5% max drawdown
    stop_loss: float = 0.02         # 2% stop loss
    take_profit: float = 0.04       # 4% take profit
    correlation_threshold: float = 0.7
    volatility_threshold: float = 0.5
    risk_free_rate: float = 0.02    # 2% risk-free rate

@dataclass
class ProfitOptimizationConfig:
    """Profit optimization configuration"""
    kelly_criterion: bool = True
    sharpe_optimization: bool = True
    max_drawdown_control: bool = True
    risk_parity: bool = True
    volatility_targeting: bool = True
    position_sizing: str = 'adaptive'  # 'fixed', 'adaptive', 'kelly'

class AdvancedIntelligenceEngine:
    """
    Advanced Intelligence Engine with 10X features:
    
    1. Kelly Criterion Optimization
    2. Sharpe Ratio Optimization
    3. Risk Parity Strategies
    4. Volatility Targeting
    5. Adaptive Position Sizing
    6. Advanced Risk Management
    7. Market Regime Detection
    8. Performance Analytics
    """
    
    def __init__(self, risk_config: Optional[RiskManagementConfig] = None, 
                 profit_config: Optional[ProfitOptimizationConfig] = None):
        """Initialize the advanced intelligence engine"""
        self.logger = get_logger("hyperion.intelligence")
        
        # Configuration
        self.risk_config = risk_config or RiskManagementConfig()
        self.profit_config = profit_config or ProfitOptimizationConfig()
        
        # Performance tracking
        self.performance_history = []
        self.risk_metrics = {}
        self.optimization_results = {}
        
        # Market regime detection
        self.market_regime = 'normal'
        self.regime_history = []
        
        # Position sizing
        self.position_sizes = {}
        self.portfolio_weights = {}
        
        self.logger.info("Advanced Intelligence Engine initialized")
    
    def calculate_kelly_criterion(self, returns: pd.Series, win_rate: float = None) -> float:
        """
        Calculate Kelly Criterion optimal position size
        
        Args:
            returns: Series of returns
            win_rate: Win rate (if None, calculated from returns)
            
        Returns:
            Optimal Kelly fraction
        """
        try:
            if win_rate is None:
                win_rate = (returns > 0).mean()
            
            avg_win = returns[returns > 0].mean() if len(returns[returns > 0]) > 0 else 0.01
            avg_loss = abs(returns[returns < 0].mean()) if len(returns[returns < 0]) > 0 else 0.01
            
            # Kelly formula: f = (bp - q) / b
            # where b = odds received, p = probability of win, q = probability of loss
            if avg_loss > 0:
                kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
                # Constrain to reasonable bounds
                kelly_fraction = np.clip(kelly_fraction, 0.0, self.risk_config.max_position_size)
                return kelly_fraction
            else:
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Error calculating Kelly Criterion: {e}")
            return 0.0
    
    def calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """
        Calculate Sharpe ratio
        
        Args:
            returns: Series of returns
            
        Returns:
            Sharpe ratio
        """
        try:
            if len(returns) < 2:
                return 0.0
            
            excess_returns = returns - self.risk_config.risk_free_rate / 252  # Daily risk-free rate
            if excess_returns.std() == 0:
                return 0.0
            
            sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(252)  # Annualized
            return sharpe
            
        except Exception as e:
            self.logger.error(f"Error calculating Sharpe ratio: {e}")
            return 0.0
    
    def calculate_max_drawdown(self, cumulative_returns: pd.Series) -> float:
        """
        Calculate maximum drawdown
        
        Args:
            cumulative_returns: Series of cumulative returns
            
        Returns:
            Maximum drawdown as a fraction
        """
        try:
            if len(cumulative_returns) < 2:
                return 0.0
            
            # Calculate running maximum
            running_max = cumulative_returns.expanding().max()
            
            # Calculate drawdown
            drawdown = (cumulative_returns - running_max) / running_max
            
            # Get maximum drawdown
            max_drawdown = drawdown.min()
            
            return abs(max_drawdown)
            
        except Exception as e:
            self.logger.error(f"Error calculating max drawdown: {e}")
            return 0.0
    
    def calculate_volatility(self, returns: pd.Series, window: int = 20) -> float:
        """
        Calculate rolling volatility
        
        Args:
            returns: Series of returns
            window: Rolling window size
            
        Returns:
            Current volatility
        """
        try:
            if len(returns) < window:
                return returns.std() if len(returns) > 1 else 0.0
            
            return returns.rolling(window).std().iloc[-1]
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility: {e}")
            return 0.0
    
    def detect_market_regime(self, returns: pd.Series, volume: pd.Series = None) -> str:
        """
        Detect market regime based on volatility and trend
        
        Args:
            returns: Series of returns
            volume: Series of volume (optional)
            
        Returns:
            Market regime: 'low_volatility', 'normal', 'high_volatility', 'trending'
        """
        try:
            if len(returns) < 20:
                return 'normal'
            
            # Calculate volatility regime
            short_vol = returns.rolling(10).std()
            long_vol = returns.rolling(50).std()
            vol_ratio = (short_vol / (long_vol + 1e-8)).iloc[-1]
            
            # Calculate trend regime
            price_momentum = returns.rolling(20).mean().iloc[-1]
            trend_strength = abs(price_momentum)
            
            # Determine regime
            if vol_ratio > 2.0:
                return 'high_volatility'
            elif vol_ratio < 0.5:
                return 'low_volatility'
            elif trend_strength > 0.01:
                return 'trending'
            else:
                return 'normal'
                
        except Exception as e:
            self.logger.error(f"Error detecting market regime: {e}")
            return 'normal'
    
    def calculate_optimal_position_size(self, symbol: str, returns: pd.Series, 
                                      confidence: float = 0.5) -> float:
        """
        Calculate optimal position size using multiple strategies
        
        Args:
            symbol: Trading symbol
            returns: Historical returns
            confidence: Confidence level (0-1)
            
        Returns:
            Optimal position size as fraction of portfolio
        """
        try:
            if self.profit_config.position_sizing == 'kelly':
                # Kelly Criterion
                position_size = self.calculate_kelly_criterion(returns)
            elif self.profit_config.position_sizing == 'adaptive':
                # Adaptive sizing based on volatility and Sharpe ratio
                volatility = self.calculate_volatility(returns)
                sharpe = self.calculate_sharpe_ratio(returns)
                
                # Base size on Sharpe ratio
                base_size = min(abs(sharpe) * 0.1, self.risk_config.max_position_size)
                
                # Adjust for volatility
                vol_adjustment = max(0.1, 1 - volatility * 10)  # Reduce size for high volatility
                
                position_size = base_size * vol_adjustment * confidence
            else:
                # Fixed position size
                position_size = self.risk_config.max_position_size * confidence
            
            # Apply risk constraints
            position_size = min(position_size, self.risk_config.max_position_size)
            
            # Store for tracking
            self.position_sizes[symbol] = position_size
            
            return position_size
            
        except Exception as e:
            self.logger.error(f"Error calculating position size for {symbol}: {e}")
            return 0.0
    
    def optimize_portfolio_weights(self, returns_dict: Dict[str, pd.Series], 
                                 method: str = 'sharpe') -> Dict[str, float]:
        """
        Optimize portfolio weights using different methods
        
        Args:
            returns_dict: Dictionary of returns by symbol
            method: Optimization method ('sharpe', 'risk_parity', 'equal_weight')
            
        Returns:
            Dictionary of optimal weights
        """
        try:
            if method == 'equal_weight':
                # Equal weight allocation
                n_assets = len(returns_dict)
                weights = {symbol: 1.0 / n_assets for symbol in returns_dict.keys()}
                
            elif method == 'sharpe':
                # Sharpe ratio optimization
                sharpe_ratios = {}
                for symbol, returns in returns_dict.items():
                    sharpe_ratios[symbol] = self.calculate_sharpe_ratio(returns)
                
                # Weight by Sharpe ratio (normalized)
                total_sharpe = sum(abs(sharpe) for sharpe in sharpe_ratios.values())
                if total_sharpe > 0:
                    weights = {symbol: abs(sharpe) / total_sharpe for symbol, sharpe in sharpe_ratios.items()}
                else:
                    weights = {symbol: 1.0 / len(returns_dict) for symbol in returns_dict.keys()}
                    
            elif method == 'risk_parity':
                # Risk parity (equal risk contribution)
                volatilities = {}
                for symbol, returns in returns_dict.items():
                    volatilities[symbol] = self.calculate_volatility(returns)
                
                # Inverse volatility weighting
                total_inv_vol = sum(1 / vol for vol in volatilities.values() if vol > 0)
                if total_inv_vol > 0:
                    weights = {symbol: (1 / vol) / total_inv_vol for symbol, vol in volatilities.items() if vol > 0}
                else:
                    weights = {symbol: 1.0 / len(returns_dict) for symbol in returns_dict.keys()}
            
            else:
                # Default to equal weight
                weights = {symbol: 1.0 / len(returns_dict) for symbol in returns_dict.keys()}
            
            # Store for tracking
            self.portfolio_weights = weights
            
            return weights
            
        except Exception as e:
            self.logger.error(f"Error optimizing portfolio weights: {e}")
            return {symbol: 1.0 / len(returns_dict) for symbol in returns_dict.keys()}
    
    def calculate_risk_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """
        Calculate comprehensive risk metrics
        
        Args:
            returns: Series of returns
            
        Returns:
            Dictionary of risk metrics
        """
        try:
            metrics = {
                'sharpe_ratio': self.calculate_sharpe_ratio(returns),
                'max_drawdown': self.calculate_max_drawdown((1 + returns).cumprod()),
                'volatility': self.calculate_volatility(returns),
                'var_95': returns.quantile(0.05),  # 95% VaR
                'cvar_95': returns[returns <= returns.quantile(0.05)].mean(),  # 95% CVaR
                'win_rate': (returns > 0).mean(),
                'avg_win': returns[returns > 0].mean() if len(returns[returns > 0]) > 0 else 0.0,
                'avg_loss': returns[returns < 0].mean() if len(returns[returns < 0]) > 0 else 0.0,
                'profit_factor': abs(returns[returns > 0].sum() / returns[returns < 0].sum()) if returns[returns < 0].sum() != 0 else float('inf')
            }
            
            # Store for tracking
            self.risk_metrics = metrics
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating risk metrics: {e}")
            return {}
    
    def should_trade(self, symbol: str, current_volatility: float, 
                    current_drawdown: float) -> bool:
        """
        Determine if trading should continue based on risk constraints
        
        Args:
            symbol: Trading symbol
            current_volatility: Current volatility
            current_drawdown: Current drawdown
            
        Returns:
            True if trading should continue
        """
        try:
            # Check volatility threshold
            if current_volatility > self.risk_config.volatility_threshold:
                self.logger.warning(f"Volatility threshold exceeded for {symbol}: {current_volatility:.4f}")
                return False
            
            # Check drawdown threshold
            if current_drawdown > self.risk_config.max_drawdown:
                self.logger.warning(f"Max drawdown exceeded for {symbol}: {current_drawdown:.4f}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in should_trade check: {e}")
            return False
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of optimization results"""
        return {
            'position_sizes': self.position_sizes,
            'portfolio_weights': self.portfolio_weights,
            'risk_metrics': self.risk_metrics,
            'market_regime': self.market_regime,
            'performance_history_length': len(self.performance_history)
        } 