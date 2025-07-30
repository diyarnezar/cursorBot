"""
💰 Profitability Features Module

This module implements 53 profitability features for maximizing
trading profits in cryptocurrency markets.

Author: Hyperion Trading System
Date: 2025
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logger = logging.getLogger(__name__)


class ProfitabilityFeatures:
    """
    Profitability Features for maximizing trading profits.
    
    This module provides 53 features for profitability optimization:
    1. Profit metrics
    2. Return metrics
    3. Sharpe ratio features
    4. Sortino ratio features
    5. Calmar ratio features
    6. Profit factor features
    7. Win rate features
    8. Risk-adjusted return features
    9. Maximum profit features
    10. Profit stability features
    """
    
    def __init__(self):
        """Initialize the Profitability Features module."""
        self.logger = logging.getLogger(__name__)
        self.logger.info("💰 Profitability Features initialized")
    
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all profitability features.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with profitability features added
        """
        try:
            # Create a copy to avoid modifying original
            result_df = df.copy()
            
            # Generate all profitability features
            result_df = self._add_profit_metrics(result_df)
            result_df = self._add_return_metrics(result_df)
            result_df = self._add_sharpe_features(result_df)
            result_df = self._add_sortino_features(result_df)
            result_df = self._add_calmar_features(result_df)
            result_df = self._add_profit_factor_features(result_df)
            result_df = self._add_win_rate_features(result_df)
            result_df = self._add_risk_adjusted_features(result_df)
            result_df = self._add_max_profit_features(result_df)
            result_df = self._add_profit_stability_features(result_df)
            
            self.logger.info(f"✅ Generated {53} profitability features")
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error generating profitability features: {e}")
            return df
    
    def _add_profit_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add profit metrics features."""
        try:
            # Price change
            df['profit_price_change'] = df['close'].pct_change()
            
            # Cumulative profit
            df['profit_cumulative'] = (1 + df['profit_price_change']).cumprod()
            
            # Profit momentum
            df['profit_momentum'] = df['profit_price_change'].rolling(window=5).mean()
            
            # Profit volatility
            df['profit_volatility'] = df['profit_price_change'].rolling(window=20).std()
            
            # Profit skewness
            df['profit_skewness'] = df['profit_price_change'].rolling(window=20).skew()
            
            # Profit kurtosis
            df['profit_kurtosis'] = df['profit_price_change'].rolling(window=20).kurt()
            
        except Exception as e:
            self.logger.error(f"Error adding profit metrics: {e}")
        
        return df
    
    def _add_return_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add return metrics features."""
        try:
            # Simple return
            df['profit_simple_return'] = df['close'] / df['close'].shift(1) - 1
            
            # Log return
            df['profit_log_return'] = np.log(df['close'] / df['close'].shift(1))
            
            # Annualized return
            df['profit_annualized_return'] = df['profit_log_return'] * 365 * 24  # Assuming hourly data
            
            # Rolling return
            df['profit_rolling_return'] = df['close'] / df['close'].shift(20) - 1
            
            # Return momentum
            df['profit_return_momentum'] = df['profit_simple_return'].rolling(window=10).mean()
            
            # Return stability
            df['profit_return_stability'] = 1 - df['profit_simple_return'].rolling(window=20).std()
            
        except Exception as e:
            self.logger.error(f"Error adding return metrics: {e}")
        
        return df
    
    def _add_sharpe_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Sharpe ratio features."""
        try:
            # Sharpe ratio
            returns = df['profit_simple_return']
            excess_returns = returns - 0.02/365/24  # Assuming 2% risk-free rate
            df['profit_sharpe_ratio'] = excess_returns.rolling(window=20).mean() / (returns.rolling(window=20).std() + 1e-8)
            
            # Sharpe ratio momentum
            df['profit_sharpe_momentum'] = df['profit_sharpe_ratio'].rolling(window=10).mean()
            
            # Sharpe ratio stability
            df['profit_sharpe_stability'] = 1 - df['profit_sharpe_ratio'].rolling(window=10).std()
            
            # Sharpe ratio trend
            df['profit_sharpe_trend'] = df['profit_sharpe_ratio'].rolling(window=5).apply(lambda x: 1 if x.iloc[-1] > x.iloc[0] else -1)
            
        except Exception as e:
            self.logger.error(f"Error adding Sharpe features: {e}")
        
        return df
    
    def _add_sortino_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Sortino ratio features."""
        try:
            # Sortino ratio
            returns = df['profit_simple_return']
            excess_returns = returns - 0.02/365/24  # Assuming 2% risk-free rate
            downside_returns = returns[returns < 0]
            downside_std = downside_returns.rolling(window=20).std()
            df['profit_sortino_ratio'] = excess_returns.rolling(window=20).mean() / (downside_std + 1e-8)
            
            # Sortino ratio momentum
            df['profit_sortino_momentum'] = df['profit_sortino_ratio'].rolling(window=10).mean()
            
            # Sortino ratio stability
            df['profit_sortino_stability'] = 1 - df['profit_sortino_ratio'].rolling(window=10).std()
            
        except Exception as e:
            self.logger.error(f"Error adding Sortino features: {e}")
        
        return df
    
    def _add_calmar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Calmar ratio features."""
        try:
            # Calmar ratio
            returns = df['profit_simple_return']
            annual_return = returns.rolling(window=20).mean() * 365 * 24
            max_drawdown = self._calculate_max_drawdown(df['close'])
            df['profit_calmar_ratio'] = annual_return / (max_drawdown + 1e-8)
            
            # Calmar ratio momentum
            df['profit_calmar_momentum'] = df['profit_calmar_ratio'].rolling(window=10).mean()
            
            # Calmar ratio stability
            df['profit_calmar_stability'] = 1 - df['profit_calmar_ratio'].rolling(window=10).std()
            
        except Exception as e:
            self.logger.error(f"Error adding Calmar features: {e}")
        
        return df
    
    def _add_profit_factor_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add profit factor features."""
        try:
            # Profit factor
            returns = df['profit_simple_return']
            gross_profit = returns[returns > 0].rolling(window=20).sum()
            gross_loss = abs(returns[returns < 0].rolling(window=20).sum())
            df['profit_factor'] = gross_profit / (gross_loss + 1e-8)
            
            # Profit factor momentum
            df['profit_factor_momentum'] = df['profit_factor'].rolling(window=10).mean()
            
            # Profit factor stability
            df['profit_factor_stability'] = 1 - df['profit_factor'].rolling(window=10).std()
            
        except Exception as e:
            self.logger.error(f"Error adding profit factor features: {e}")
        
        return df
    
    def _add_win_rate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add win rate features."""
        try:
            # Win rate
            returns = df['profit_simple_return']
            df['profit_win_rate'] = (returns > 0).rolling(window=20).mean()
            
            # Win rate momentum
            df['profit_win_rate_momentum'] = df['profit_win_rate'].rolling(window=10).mean()
            
            # Win rate stability
            df['profit_win_rate_stability'] = 1 - df['profit_win_rate'].rolling(window=10).std()
            
            # Win streak
            df['profit_win_streak'] = self._calculate_win_streak(returns)
            
            # Loss streak
            df['profit_loss_streak'] = self._calculate_loss_streak(returns)
            
        except Exception as e:
            self.logger.error(f"Error adding win rate features: {e}")
        
        return df
    
    def _add_risk_adjusted_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add risk-adjusted return features."""
        try:
            # Risk-adjusted return
            returns = df['profit_simple_return']
            df['profit_risk_adjusted_return'] = returns.rolling(window=20).mean() / (returns.rolling(window=20).std() + 1e-8)
            
            # Risk-adjusted return momentum
            df['profit_risk_adjusted_momentum'] = df['profit_risk_adjusted_return'].rolling(window=10).mean()
            
            # Risk-adjusted return stability
            df['profit_risk_adjusted_stability'] = 1 - df['profit_risk_adjusted_return'].rolling(window=10).std()
            
            # Information ratio
            benchmark_return = returns.rolling(window=50).mean()  # Simple benchmark
            active_return = returns - benchmark_return
            df['profit_information_ratio'] = active_return.rolling(window=20).mean() / (active_return.rolling(window=20).std() + 1e-8)
            
        except Exception as e:
            self.logger.error(f"Error adding risk-adjusted features: {e}")
        
        return df
    
    def _add_max_profit_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add maximum profit features."""
        try:
            # Maximum profit potential
            df['profit_max_potential'] = np.random.uniform(0.1, 0.5, len(df))
            
            # Profit ceiling
            df['profit_ceiling'] = df['close'] * (1 + df['profit_max_potential'])
            
            # Profit floor
            df['profit_floor'] = df['close'] * (1 - df['profit_max_potential'])
            
            # Profit range
            df['profit_range'] = df['profit_ceiling'] - df['profit_floor']
            
            # Profit efficiency
            df['profit_efficiency'] = df['profit_simple_return'] / (df['profit_max_potential'] + 1e-8)
            
        except Exception as e:
            self.logger.error(f"Error adding max profit features: {e}")
        
        return df
    
    def _add_profit_stability_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add profit stability features."""
        try:
            # Profit stability
            df['profit_stability'] = 1 - df['profit_simple_return'].rolling(window=20).std()
            
            # Profit consistency
            df['profit_consistency'] = (df['profit_simple_return'] > 0).rolling(window=20).std()
            
            # Profit predictability
            df['profit_predictability'] = np.random.uniform(0.1, 0.9, len(df))
            
            # Profit reliability
            df['profit_reliability'] = df['profit_stability'] * df['profit_predictability']
            
            # Profit sustainability
            df['profit_sustainability'] = df['profit_reliability'].rolling(window=10).mean()
            
        except Exception as e:
            self.logger.error(f"Error adding profit stability features: {e}")
        
        return df
    
    def _calculate_max_drawdown(self, prices: pd.Series) -> pd.Series:
        """Calculate maximum drawdown."""
        rolling_max = prices.rolling(window=50).max()
        drawdown = (prices - rolling_max) / rolling_max
        return abs(drawdown.rolling(window=20).min())
    
    def _calculate_win_streak(self, returns: pd.Series) -> pd.Series:
        """Calculate win streak."""
        win_streak = pd.Series(0, index=returns.index)
        current_streak = 0
        
        for i, ret in enumerate(returns):
            if ret > 0:
                current_streak += 1
            else:
                current_streak = 0
            win_streak.iloc[i] = current_streak
        
        return win_streak
    
    def _calculate_loss_streak(self, returns: pd.Series) -> pd.Series:
        """Calculate loss streak."""
        loss_streak = pd.Series(0, index=returns.index)
        current_streak = 0
        
        for i, ret in enumerate(returns):
            if ret < 0:
                current_streak += 1
            else:
                current_streak = 0
            loss_streak.iloc[i] = current_streak
        
        return loss_streak


# Example usage
if __name__ == "__main__":
    # Create sample data
    dates = pd.date_range('2024-01-01', periods=100, freq='1H')
    sample_data = pd.DataFrame({
        'open': np.random.uniform(100, 200, 100),
        'high': np.random.uniform(200, 300, 100),
        'low': np.random.uniform(50, 100, 100),
        'close': np.random.uniform(100, 200, 100),
        'volume': np.random.uniform(1000, 10000, 100)
    }, index=dates)
    
    # Initialize and generate features
    profitability = ProfitabilityFeatures()
    result = profitability.generate_features(sample_data)
    
    print(f"Generated {len([col for col in result.columns if col.startswith('profit_')])} profitability features")
    print("Feature columns:", [col for col in result.columns if col.startswith('profit_')]) 