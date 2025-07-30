#!/usr/bin/env python3
"""
Shadow Deployment Module
Implements shadow runs and canary deployment for safe model validation
"""

import numpy as np
import pandas as pd
import logging
import threading
import time
from typing import Dict, List, Tuple, Any, Optional, Callable
from datetime import datetime, timedelta
from collections import deque
import json
import os
from dataclasses import dataclass

@dataclass
class ShadowTrade:
    """Data class for shadow trade information."""
    timestamp: datetime
    symbol: str
    side: str  # 'BUY' or 'SELL'
    quantity: float
    price: float
    confidence: float
    model_prediction: float
    actual_price: float
    pnl: float
    slippage: float
    fees: float
    execution_time: float

class ShadowDeployment:
    """
    Shadow Deployment System for Safe Model Validation
    
    Features:
    - Shadow runs with live data feeds
    - Canary deployment with gradual rollout
    - Performance comparison between shadow and paper trading
    - Real-time discrepancy detection
    - Safe model validation before live deployment
    """
    
    def __init__(self, 
                 initial_capital: float = 10000.0,
                 max_shadow_trades: int = 1000,
                 performance_threshold: float = 0.8,
                 discrepancy_threshold: float = 0.1):
        """
        Initialize Shadow Deployment System.
        
        Args:
            initial_capital: Initial capital for shadow trading
            max_shadow_trades: Maximum number of shadow trades to track
            performance_threshold: Minimum performance threshold for deployment
            discrepancy_threshold: Maximum allowed discrepancy between shadow and paper
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_shadow_trades = max_shadow_trades
        self.performance_threshold = performance_threshold
        self.discrepancy_threshold = discrepancy_threshold
        
        # Shadow trading state
        self.shadow_trades = deque(maxlen=max_shadow_trades)
        self.shadow_positions = {}
        self.is_shadow_running = False
        self.shadow_thread = None
        
        # Performance tracking
        self.shadow_performance = {
            'total_trades': 0,
            'winning_trades': 0,
            'total_pnl': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'win_rate': 0.0,
            'avg_trade_pnl': 0.0
        }
        
        # Comparison tracking
        self.paper_performance = {}
        self.performance_discrepancies = deque(maxlen=100)
        
        # Deployment state
        self.canary_deployment_active = False
        self.canary_traffic_percentage = 0.0
        self.deployment_ready = False
        
        logging.info("Shadow Deployment System initialized")
    
    def start_shadow_run(self, 
                        data_feed_callback: Callable,
                        model_prediction_callback: Callable,
                        paper_trading_callback: Callable = None):
        """
        Start shadow run with live data feed.
        
        Args:
            data_feed_callback: Function to get live market data
            model_prediction_callback: Function to get model predictions
            paper_trading_callback: Function to get paper trading results
        """
        try:
            if self.is_shadow_running:
                logging.warning("Shadow run already active")
                return
            
            self.is_shadow_running = True
            self.shadow_thread = threading.Thread(
                target=self._shadow_run_loop,
                args=(data_feed_callback, model_prediction_callback, paper_trading_callback),
                daemon=True
            )
            self.shadow_thread.start()
            
            logging.info("🚀 Shadow run started")
            
        except Exception as e:
            logging.error(f"Error starting shadow run: {e}")
    
    def stop_shadow_run(self):
        """Stop shadow run."""
        try:
            self.is_shadow_running = False
            if self.shadow_thread:
                self.shadow_thread.join(timeout=5)
            
            logging.info("🛑 Shadow run stopped")
            
        except Exception as e:
            logging.error(f"Error stopping shadow run: {e}")
    
    def _shadow_run_loop(self, 
                        data_feed_callback: Callable,
                        model_prediction_callback: Callable,
                        paper_trading_callback: Callable):
        """Main shadow run loop."""
        try:
            while self.is_shadow_running:
                # Get live market data
                market_data = data_feed_callback()
                if market_data is None:
                    time.sleep(1)
                    continue
                
                # Get model predictions
                predictions = model_prediction_callback(market_data)
                if predictions is None:
                    time.sleep(1)
                    continue
                
                # Execute shadow trade
                shadow_trade = self._execute_shadow_trade(market_data, predictions)
                if shadow_trade:
                    self.shadow_trades.append(shadow_trade)
                    self._update_shadow_performance(shadow_trade)
                
                # Compare with paper trading if available
                if paper_trading_callback:
                    paper_results = paper_trading_callback(market_data)
                    if paper_results:
                        self._compare_with_paper_trading(shadow_trade, paper_results)
                
                # Check deployment readiness
                self._check_deployment_readiness()
                
                time.sleep(1)  # 1-second intervals
                
        except Exception as e:
            logging.error(f"Error in shadow run loop: {e}")
            self.is_shadow_running = False
    
    def _execute_shadow_trade(self, 
                            market_data: Dict[str, Any], 
                            predictions: Dict[str, Any]) -> Optional[ShadowTrade]:
        """Execute a shadow trade based on predictions."""
        try:
            symbol = market_data.get('symbol', 'ETHFDUSD')
            current_price = market_data.get('price', 0.0)
            timestamp = market_data.get('timestamp', datetime.now())
            
            if current_price <= 0:
                return None
            
            # Get prediction and confidence
            prediction = predictions.get('prediction', 0.0)
            confidence = predictions.get('confidence', 0.0)
            
            # Determine trade direction
            if confidence < 0.6:  # Low confidence - no trade
                return None
            
            side = 'BUY' if prediction > 0 else 'SELL' if prediction < 0 else None
            if side is None:
                return None
            
            # Calculate position size based on confidence and capital
            position_size = self._calculate_position_size(confidence, current_price)
            
            # Simulate execution with realistic slippage and fees
            execution_price = self._simulate_execution(current_price, side, position_size)
            slippage = abs(execution_price - current_price) / current_price
            fees = self._calculate_fees(execution_price, position_size)
            
            # Calculate PnL
            pnl = self._calculate_trade_pnl(side, position_size, execution_price, current_price)
            
            # Create shadow trade
            shadow_trade = ShadowTrade(
                timestamp=timestamp,
                symbol=symbol,
                side=side,
                quantity=position_size,
                price=execution_price,
                confidence=confidence,
                model_prediction=prediction,
                actual_price=current_price,
                pnl=pnl,
                slippage=slippage,
                fees=fees,
                execution_time=time.time()
            )
            
            # Update capital and positions
            self.current_capital += pnl - fees
            self._update_positions(symbol, side, position_size, execution_price)
            
            return shadow_trade
            
        except Exception as e:
            logging.error(f"Error executing shadow trade: {e}")
            return None
    
    def _calculate_position_size(self, confidence: float, price: float) -> float:
        """Calculate position size based on confidence and risk management."""
        try:
            # Base position size as percentage of capital
            base_size_pct = 0.02  # 2% of capital per trade
            
            # Adjust for confidence
            confidence_multiplier = min(confidence, 0.9)  # Cap at 90%
            
            # Calculate position size
            position_value = self.current_capital * base_size_pct * confidence_multiplier
            position_size = position_value / price
            
            # Ensure minimum and maximum sizes
            min_size = 0.001  # Minimum ETH
            max_size = self.current_capital * 0.1 / price  # Maximum 10% of capital
            
            return np.clip(position_size, min_size, max_size)
            
        except Exception as e:
            logging.error(f"Error calculating position size: {e}")
            return 0.001
    
    def _simulate_execution(self, 
                          market_price: float, 
                          side: str, 
                          quantity: float) -> float:
        """Simulate realistic execution with slippage."""
        try:
            # Base slippage based on order size
            base_slippage = 0.0001  # 0.01%
            
            # Size-based slippage (larger orders = more slippage)
            size_multiplier = min(quantity * market_price / 1000, 5.0)  # Cap at 5x
            
            # Market volatility impact (simplified)
            volatility_impact = 0.0002  # Additional 0.02%
            
            # Calculate total slippage
            total_slippage = base_slippage * size_multiplier + volatility_impact
            
            # Apply slippage
            if side == 'BUY':
                execution_price = market_price * (1 + total_slippage)
            else:  # SELL
                execution_price = market_price * (1 - total_slippage)
            
            return execution_price
            
        except Exception as e:
            logging.error(f"Error simulating execution: {e}")
            return market_price
    
    def _calculate_fees(self, price: float, quantity: float) -> float:
        """Calculate trading fees."""
        try:
            # Maker fee (0% for limit orders)
            maker_fee = 0.0
            
            # Taker fee (0.1% for market orders)
            taker_fee = 0.001
            
            # Assume 80% maker, 20% taker orders
            avg_fee = maker_fee * 0.8 + taker_fee * 0.2
            
            return price * quantity * avg_fee
            
        except Exception as e:
            logging.error(f"Error calculating fees: {e}")
            return 0.0
    
    def _calculate_trade_pnl(self, 
                           side: str, 
                           quantity: float, 
                           entry_price: float, 
                           current_price: float) -> float:
        """Calculate trade PnL."""
        try:
            if side == 'BUY':
                return quantity * (current_price - entry_price)
            else:  # SELL
                return quantity * (entry_price - current_price)
        except Exception as e:
            logging.error(f"Error calculating trade PnL: {e}")
            return 0.0
    
    def _update_positions(self, symbol: str, side: str, quantity: float, price: float):
        """Update shadow positions."""
        try:
            if symbol not in self.shadow_positions:
                self.shadow_positions[symbol] = 0.0
            
            if side == 'BUY':
                self.shadow_positions[symbol] += quantity
            else:  # SELL
                self.shadow_positions[symbol] -= quantity
                
        except Exception as e:
            logging.error(f"Error updating positions: {e}")
    
    def _update_shadow_performance(self, trade: ShadowTrade):
        """Update shadow performance metrics."""
        try:
            self.shadow_performance['total_trades'] += 1
            self.shadow_performance['total_pnl'] += trade.pnl
            
            if trade.pnl > 0:
                self.shadow_performance['winning_trades'] += 1
            
            # Update win rate
            self.shadow_performance['win_rate'] = (
                self.shadow_performance['winning_trades'] / 
                self.shadow_performance['total_trades']
            )
            
            # Update average trade PnL
            self.shadow_performance['avg_trade_pnl'] = (
                self.shadow_performance['total_pnl'] / 
                self.shadow_performance['total_trades']
            )
            
            # Calculate Sharpe ratio (simplified)
            if len(self.shadow_trades) > 1:
                returns = [t.pnl for t in self.shadow_trades]
                if np.std(returns) > 0:
                    self.shadow_performance['sharpe_ratio'] = np.mean(returns) / np.std(returns)
            
            # Calculate max drawdown
            cumulative_pnl = np.cumsum([t.pnl for t in self.shadow_trades])
            running_max = np.maximum.accumulate(cumulative_pnl)
            drawdown = (cumulative_pnl - running_max) / (running_max + self.initial_capital)
            self.shadow_performance['max_drawdown'] = abs(drawdown.min())
            
        except Exception as e:
            logging.error(f"Error updating shadow performance: {e}")
    
    def _compare_with_paper_trading(self, 
                                  shadow_trade: ShadowTrade, 
                                  paper_results: Dict[str, Any]):
        """Compare shadow trading with paper trading results."""
        try:
            paper_pnl = paper_results.get('pnl', 0.0)
            paper_trades = paper_results.get('total_trades', 0)
            
            # Calculate discrepancy
            pnl_discrepancy = abs(shadow_trade.pnl - paper_pnl) / max(abs(paper_pnl), 0.01)
            
            self.performance_discrepancies.append({
                'timestamp': shadow_trade.timestamp,
                'shadow_pnl': shadow_trade.pnl,
                'paper_pnl': paper_pnl,
                'discrepancy': pnl_discrepancy,
                'shadow_trades': self.shadow_performance['total_trades'],
                'paper_trades': paper_trades
            })
            
            # Alert if discrepancy is too high
            if pnl_discrepancy > self.discrepancy_threshold:
                logging.warning(f"⚠️ High discrepancy detected: {pnl_discrepancy:.2%}")
                
        except Exception as e:
            logging.error(f"Error comparing with paper trading: {e}")
    
    def _check_deployment_readiness(self):
        """Check if model is ready for deployment."""
        try:
            if len(self.shadow_trades) < 50:  # Need minimum trades
                return
            
            # Check performance threshold
            performance_score = self._calculate_deployment_score()
            
            if performance_score >= self.performance_threshold:
                self.deployment_ready = True
                logging.info(f"✅ Model ready for deployment (score: {performance_score:.3f})")
            else:
                self.deployment_ready = False
                
        except Exception as e:
            logging.error(f"Error checking deployment readiness: {e}")
    
    def _calculate_deployment_score(self) -> float:
        """Calculate deployment readiness score."""
        try:
            # Weighted combination of metrics
            win_rate_score = self.shadow_performance['win_rate']
            sharpe_score = min(self.shadow_performance['sharpe_ratio'] / 2.0, 1.0)  # Cap at 1.0
            drawdown_score = max(0, 1 - self.shadow_performance['max_drawdown'] / 0.2)  # Cap at 20%
            
            # Overall score
            deployment_score = (
                win_rate_score * 0.4 +
                sharpe_score * 0.4 +
                drawdown_score * 0.2
            )
            
            return deployment_score
            
        except Exception as e:
            logging.error(f"Error calculating deployment score: {e}")
            return 0.0
    
    def start_canary_deployment(self, traffic_percentage: float = 0.1):
        """
        Start canary deployment with gradual rollout.
        
        Args:
            traffic_percentage: Percentage of traffic to route to new model
        """
        try:
            if not self.deployment_ready:
                logging.warning("Model not ready for deployment")
                return
            
            self.canary_deployment_active = True
            self.canary_traffic_percentage = traffic_percentage
            
            logging.info(f"🚀 Canary deployment started with {traffic_percentage:.1%} traffic")
            
        except Exception as e:
            logging.error(f"Error starting canary deployment: {e}")
    
    def update_canary_traffic(self, new_percentage: float):
        """Update canary deployment traffic percentage."""
        try:
            if not self.canary_deployment_active:
                logging.warning("Canary deployment not active")
                return
            
            self.canary_traffic_percentage = new_percentage
            logging.info(f"📊 Canary traffic updated to {new_percentage:.1%}")
            
        except Exception as e:
            logging.error(f"Error updating canary traffic: {e}")
    
    def stop_canary_deployment(self):
        """Stop canary deployment."""
        try:
            self.canary_deployment_active = False
            self.canary_traffic_percentage = 0.0
            
            logging.info("🛑 Canary deployment stopped")
            
        except Exception as e:
            logging.error(f"Error stopping canary deployment: {e}")
    
    def get_shadow_summary(self) -> Dict[str, Any]:
        """Get comprehensive shadow deployment summary."""
        try:
            return {
                'shadow_performance': self.shadow_performance,
                'current_capital': self.current_capital,
                'total_return': (self.current_capital - self.initial_capital) / self.initial_capital,
                'total_trades': len(self.shadow_trades),
                'deployment_ready': self.deployment_ready,
                'canary_active': self.canary_deployment_active,
                'canary_traffic': self.canary_traffic_percentage,
                'recent_discrepancies': list(self.performance_discrepancies)[-10:],
                'positions': self.shadow_positions.copy()
            }
            
        except Exception as e:
            logging.error(f"Error getting shadow summary: {e}")
            return {}
    
    def save_shadow_results(self, filepath: str):
        """Save shadow deployment results."""
        try:
            results = {
                'shadow_trades': [vars(trade) for trade in self.shadow_trades],
                'shadow_performance': self.shadow_performance,
                'performance_discrepancies': list(self.performance_discrepancies),
                'deployment_ready': self.deployment_ready,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logging.info(f"Shadow results saved to {filepath}")
            
        except Exception as e:
            logging.error(f"Error saving shadow results: {e}")
    
    def load_shadow_results(self, filepath: str):
        """Load shadow deployment results."""
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    results = json.load(f)
                
                # Restore shadow trades
                self.shadow_trades.clear()
                for trade_data in results.get('shadow_trades', []):
                    trade = ShadowTrade(**trade_data)
                    self.shadow_trades.append(trade)
                
                # Restore performance
                self.shadow_performance = results.get('shadow_performance', {})
                
                # Restore discrepancies
                self.performance_discrepancies.clear()
                for disc in results.get('performance_discrepancies', []):
                    self.performance_discrepancies.append(disc)
                
                self.deployment_ready = results.get('deployment_ready', False)
                
                logging.info(f"Shadow results loaded from {filepath}")
            else:
                logging.warning(f"Shadow results file not found: {filepath}")
                
        except Exception as e:
            logging.error(f"Error loading shadow results: {e}") 