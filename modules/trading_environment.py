#!/usr/bin/env python3
"""
TRADING ENVIRONMENT - PHASE 4 IMPLEMENTATION
===========================================

This module implements the trading environment for reinforcement learning
as specified in Gemini's Phase 4 recommendations.
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import gym
from gym import spaces
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TradingEnvironment(gym.Env):
    """
    Advanced trading environment for reinforcement learning.
    
    This environment simulates the execution of trading strategies with
    realistic market conditions, order book dynamics, and execution costs.
    """
    
    def __init__(self, 
                 initial_balance: float = 10000.0,
                 maker_fee: float = 0.001,
                 taker_fee: float = 0.001,
                 max_positions: int = 8,
                 risk_per_trade: float = 0.02):
        """
        Initialize the trading environment.
        
        Args:
            initial_balance: Starting capital
            maker_fee: Maker fee rate
            taker_fee: Taker fee rate
            max_positions: Maximum concurrent positions
            risk_per_trade: Risk per trade as fraction of balance
        """
        super(TradingEnvironment, self).__init__()
        
        # Environment parameters
        self.initial_balance = initial_balance
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.max_positions = max_positions
        self.risk_per_trade = risk_per_trade
        
        # State variables
        self.balance = initial_balance
        self.positions = {}  # {symbol: {'quantity': float, 'entry_price': float}}
        self.current_step = 0
        self.total_trades = 0
        self.winning_trades = 0
        
        # Market data
        self.market_data = None
        self.current_prices = {}
        self.order_book_data = {}
        
        # Define action space
        # Actions: [0: Hold, 1: Buy, 2: Sell, 3: Close Position]
        self.action_space = spaces.Discrete(4)
        
        # Define observation space
        # Features: [balance, position_value, price_change, volume, rsi, macd, order_book_imbalance]
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(7,), 
            dtype=np.float32
        )
        
        # Performance tracking
        self.portfolio_values = []
        self.trade_history = []
        self.rewards_history = []
        
        logger.info(f"🎯 Trading Environment initialized with ${initial_balance:,.2f} capital")
    
    def reset(self) -> np.ndarray:
        """
        Reset the environment to initial state.
        
        Returns:
            Initial observation
        """
        self.balance = self.initial_balance
        self.positions = {}
        self.current_step = 0
        self.total_trades = 0
        self.winning_trades = 0
        self.portfolio_values = []
        self.trade_history = []
        self.rewards_history = []
        
        # Generate initial market data
        self._generate_market_data()
        
        # Get initial observation
        observation = self._get_observation()
        
        logger.info("🔄 Environment reset")
        
        return observation
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: Action to take (0: Hold, 1: Buy, 2: Sell, 3: Close)
            
        Returns:
            Tuple of (observation, reward, done, info)
        """
        # Store previous portfolio value
        prev_portfolio_value = self._get_portfolio_value()
        
        # Execute action
        self._execute_action(action)
        
        # Move to next step
        self.current_step += 1
        
        # Get new observation
        observation = self._get_observation()
        
        # Calculate reward
        current_portfolio_value = self._get_portfolio_value()
        reward = (current_portfolio_value - prev_portfolio_value) / prev_portfolio_value
        
        # Check if episode is done
        done = self._is_done()
        
        # Additional info
        info = {
            'portfolio_value': current_portfolio_value,
            'balance': self.balance,
            'positions': len(self.positions),
            'total_trades': self.total_trades,
            'win_rate': self.winning_trades / max(1, self.total_trades)
        }
        
        # Store history
        self.portfolio_values.append(current_portfolio_value)
        self.rewards_history.append(reward)
        
        return observation, reward, done, info
    
    def _generate_market_data(self):
        """Generate realistic market data for the current step."""
        # Simulate price movement
        price_change = np.random.normal(0, 0.02)  # 2% volatility
        volume = np.random.exponential(1000)
        rsi = 50 + np.random.normal(0, 10)
        macd = np.random.normal(0, 1)
        
        # Simulate order book imbalance
        bid_volume = np.random.exponential(500)
        ask_volume = np.random.exponential(500)
        imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)
        
        self.current_prices = {
            'BTC': 50000 * (1 + price_change),
            'ETH': 3000 * (1 + price_change * 0.8),
            'SOL': 100 * (1 + price_change * 1.2)
        }
        
        self.order_book_data = {
            'imbalance': imbalance,
            'spread': 0.001 + np.random.exponential(0.001),
            'depth': volume
        }
    
    def _get_observation(self) -> np.ndarray:
        """
        Get current state observation.
        
        Returns:
            Observation array
        """
        # Calculate position value
        position_value = sum(
            abs(pos['quantity']) * self.current_prices.get(symbol, 0)
            for symbol, pos in self.positions.items()
        )
        
        # Get market features
        price_change = np.random.normal(0, 0.01)  # Simulated
        volume = self.order_book_data.get('depth', 1000)
        rsi = 50 + np.random.normal(0, 10)
        macd = np.random.normal(0, 1)
        imbalance = self.order_book_data.get('imbalance', 0)
        
        observation = np.array([
            self.balance,
            position_value,
            price_change,
            volume,
            rsi,
            macd,
            imbalance
        ], dtype=np.float32)
        
        return observation
    
    def _execute_action(self, action: int):
        """
        Execute the given action.
        
        Args:
            action: Action to execute
        """
        if action == 0:  # Hold
            return
        
        elif action == 1:  # Buy
            self._execute_buy()
        
        elif action == 2:  # Sell
            self._execute_sell()
        
        elif action == 3:  # Close position
            self._close_all_positions()
    
    def _execute_buy(self):
        """Execute a buy order."""
        if len(self.positions) >= self.max_positions:
            return
        
        # Calculate position size
        risk_amount = self.balance * self.risk_per_trade
        symbol = 'BTC'  # Simplified - trade BTC only
        price = self.current_prices[symbol]
        
        # Calculate quantity
        quantity = risk_amount / price
        
        # Apply maker fee
        cost = quantity * price * (1 + self.maker_fee)
        
        if cost <= self.balance:
            # Execute trade
            self.balance -= cost
            
            if symbol in self.positions:
                # Add to existing position
                self.positions[symbol]['quantity'] += quantity
                # Update average entry price
                total_cost = self.positions[symbol]['quantity'] * self.positions[symbol]['entry_price']
                new_cost = quantity * price
                self.positions[symbol]['entry_price'] = (total_cost + new_cost) / (self.positions[symbol]['quantity'] + quantity)
            else:
                # New position
                self.positions[symbol] = {
                    'quantity': quantity,
                    'entry_price': price
                }
            
            self.total_trades += 1
            self.trade_history.append({
                'step': self.current_step,
                'action': 'buy',
                'symbol': symbol,
                'quantity': quantity,
                'price': price,
                'cost': cost
            })
    
    def _execute_sell(self):
        """Execute a sell order."""
        symbol = 'BTC'  # Simplified
        
        if symbol in self.positions and self.positions[symbol]['quantity'] > 0:
            # Close long position
            quantity = self.positions[symbol]['quantity']
            price = self.current_prices[symbol]
            
            # Calculate proceeds (apply maker fee)
            proceeds = quantity * price * (1 - self.maker_fee)
            
            # Update balance
            self.balance += proceeds
            
            # Calculate P&L
            entry_price = self.positions[symbol]['entry_price']
            pnl = (price - entry_price) * quantity
            
            if pnl > 0:
                self.winning_trades += 1
            
            # Remove position
            del self.positions[symbol]
            
            self.total_trades += 1
            self.trade_history.append({
                'step': self.current_step,
                'action': 'sell',
                'symbol': symbol,
                'quantity': quantity,
                'price': price,
                'proceeds': proceeds,
                'pnl': pnl
            })
    
    def _close_all_positions(self):
        """Close all open positions."""
        for symbol in list(self.positions.keys()):
            if self.positions[symbol]['quantity'] > 0:
                self._execute_sell()
    
    def _get_portfolio_value(self) -> float:
        """
        Calculate total portfolio value.
        
        Returns:
            Total portfolio value
        """
        position_value = sum(
            pos['quantity'] * self.current_prices.get(symbol, 0)
            for symbol, pos in self.positions.items()
        )
        
        return self.balance + position_value
    
    def _is_done(self) -> bool:
        """
        Check if episode is done.
        
        Returns:
            True if episode should end
        """
        # End if balance is too low
        if self.balance < self.initial_balance * 0.1:
            return True
        
        # End if maximum steps reached
        if self.current_step >= 1000:
            return True
        
        return False
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Calculate performance metrics.
        
        Returns:
            Dictionary of performance metrics
        """
        if not self.portfolio_values:
            return {}
        
        returns = np.diff(self.portfolio_values) / self.portfolio_values[:-1]
        
        metrics = {
            'total_return': (self.portfolio_values[-1] - self.initial_balance) / self.initial_balance,
            'sharpe_ratio': np.mean(returns) / (np.std(returns) + 1e-8),
            'max_drawdown': self._calculate_max_drawdown(),
            'win_rate': self.winning_trades / max(1, self.total_trades),
            'total_trades': self.total_trades,
            'final_balance': self.portfolio_values[-1]
        }
        
        return metrics
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown."""
        if not self.portfolio_values:
            return 0.0
        
        peak = self.portfolio_values[0]
        max_dd = 0.0
        
        for value in self.portfolio_values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            max_dd = max(max_dd, dd)
        
        return max_dd
    
    def render(self, mode='human'):
        """Render the environment (not implemented for trading)."""
        pass

def main():
    """Test the trading environment."""
    logger.info("🧪 Testing Trading Environment...")
    
    # Create environment
    env = TradingEnvironment(initial_balance=10000.0)
    
    # Test reset
    obs = env.reset()
    logger.info(f"Initial observation shape: {obs.shape}")
    
    # Test a few steps
    for step in range(10):
        action = np.random.randint(0, 4)  # Random action
        obs, reward, done, info = env.step(action)
        
        logger.info(f"Step {step}: Action={action}, Reward={reward:.4f}, Portfolio=${info['portfolio_value']:.2f}")
        
        if done:
            break
    
    # Get performance metrics
    metrics = env.get_performance_metrics()
    logger.info(f"Performance metrics: {metrics}")
    
    logger.info("✅ Trading Environment test completed")
    
    return env

if __name__ == "__main__":
    main() 