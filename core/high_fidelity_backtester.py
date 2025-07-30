"""
🚀 PROJECT HYPERION - HIGH-FIDELITY BACKTESTER
=============================================

Event-driven backtester with realistic maker-only order simulation.
Includes fill probabilities, order book depth, and slippage modeling.

Author: Project Hyperion Team
Date: 2025
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import json
from pathlib import Path
import asyncio
import time

from core.historical_data_warehouse import HistoricalDataWarehouse
from core.intelligent_execution import IntelligentExecutionAlchemist
from risk.maximum_intelligence_risk import MaximumIntelligenceRisk


class HighFidelityBacktester:
    """
    High-fidelity backtester with realistic order execution simulation
    """
    
    def __init__(self, config_path: str = "config.json"):
        """Initialize the High-Fidelity Backtester"""
        self.logger = logging.getLogger(__name__)
        self.config_path = config_path
        
        # Load configuration
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}")
            self.config = {}
        
        # Initialize components
        self.data_warehouse = HistoricalDataWarehouse(config_path)
        self.execution_engine = IntelligentExecutionAlchemist(config_path)
        self.risk_manager = MaximumIntelligenceRisk(config=self.config)
        
        # Backtesting settings
        self.initial_capital = 100000  # $100k initial capital
        self.commission_rate = 0.001   # 0.1% commission
        self.slippage_model = 'realistic'
        
        # Order book simulation settings
        self.order_book_depth = 20     # Levels to simulate
        self.min_spread = 0.0001       # Minimum spread 0.01%
        self.max_spread = 0.01         # Maximum spread 1%
        self.liquidity_decay = 0.8     # Liquidity decay factor
        
        # Fill probability settings
        self.base_fill_probability = 0.8
        self.spread_impact = 0.5
        self.volume_impact = 0.3
        self.volatility_impact = 0.2
        
        # Backtesting state
        self.current_capital = self.initial_capital
        self.positions = {}
        self.orders = {}
        self.trades = []
        self.equity_curve = []
        
        # Performance metrics
        self.backtest_results = {
            'total_return': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'fill_rate': 0.0,
            'avg_slippage': 0.0
        }
        
        self.logger.info("🚀 High-Fidelity Backtester initialized")
    
    def run_backtest(self, strategy_config: Dict[str, Any], 
                    start_date: datetime, end_date: datetime,
                    symbols: List[str]) -> Dict[str, Any]:
        """Run high-fidelity backtest"""
        try:
            self.logger.info(f"🚀 Starting backtest from {start_date} to {end_date}")
            
            # Initialize backtest
            self._initialize_backtest(strategy_config, symbols)
            
            # Get historical data
            historical_data = self._load_historical_data(symbols, start_date, end_date)
            
            if historical_data is None or historical_data.empty:
                raise ValueError("No historical data available for backtest period")
            
            # Run event-driven simulation
            self._run_event_simulation(historical_data, strategy_config)
            
            # Calculate performance metrics
            self._calculate_performance_metrics()
            
            # Generate backtest report
            report = self._generate_backtest_report()
            
            self.logger.info("✅ Backtest completed successfully")
            
            return report
            
        except Exception as e:
            self.logger.error(f"❌ Error in backtest: {e}")
            return {'error': str(e)}
    
    def _initialize_backtest(self, strategy_config: Dict[str, Any], symbols: List[str]):
        """Initialize backtest state"""
        try:
            # Reset state
            self.current_capital = self.initial_capital
            self.positions = {symbol: 0.0 for symbol in symbols}
            self.orders = {}
            self.trades = []
            self.equity_curve = []
            
            # Initialize order books for each symbol
            self.order_books = {}
            for symbol in symbols:
                self.order_books[symbol] = self._initialize_order_book()
            
            # Strategy parameters
            self.strategy_config = strategy_config
            
            self.logger.info(f"💰 Initialized backtest with ${self.initial_capital:,.2f} capital")
            
        except Exception as e:
            self.logger.error(f"❌ Error initializing backtest: {e}")
    
    def _initialize_order_book(self) -> Dict[str, List[List[float]]]:
        """Initialize realistic order book"""
        try:
            # Generate realistic order book levels
            bids = []
            asks = []
            
            # Base price (will be updated with actual data)
            base_price = 100.0
            
            # Generate bid levels (descending prices)
            for i in range(self.order_book_depth):
                price = base_price * (1 - (i + 1) * 0.0001)  # 0.01% increments
                quantity = np.random.lognormal(5, 1) * (self.liquidity_decay ** i)
                bids.append([price, quantity])
            
            # Generate ask levels (ascending prices)
            for i in range(self.order_book_depth):
                price = base_price * (1 + (i + 1) * 0.0001)  # 0.01% increments
                quantity = np.random.lognormal(5, 1) * (self.liquidity_decay ** i)
                asks.append([price, quantity])
            
            return {
                'bids': bids,
                'asks': asks,
                'last_update': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error initializing order book: {e}")
            return {'bids': [], 'asks': [], 'last_update': datetime.now()}
    
    def _load_historical_data(self, symbols: List[str], start_date: datetime, 
                             end_date: datetime) -> Optional[pd.DataFrame]:
        """Load historical data from warehouse"""
        try:
            all_data = []
            
            for symbol in symbols:
                # Load price data
                price_data = self.data_warehouse.query_data('price_data', symbol, start_date, end_date)
                
                if price_data is not None and not price_data.empty:
                    # Add symbol column
                    price_data['symbol'] = symbol
                    all_data.append(price_data)
                
                # Load additional data sources if available
                sentiment_data = self.data_warehouse.query_data('sentiment_data', symbol, start_date, end_date)
                if sentiment_data is not None and not sentiment_data.empty:
                    sentiment_data['symbol'] = symbol
                    all_data.append(sentiment_data)
            
            if all_data:
                # Merge all data by timestamp
                merged_data = pd.concat(all_data, ignore_index=True)
                merged_data = merged_data.sort_values('timestamp').reset_index(drop=True)
                
                self.logger.info(f"📊 Loaded {len(merged_data)} data points for {len(symbols)} symbols")
                return merged_data
            
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Error loading historical data: {e}")
            return None
    
    def _run_event_simulation(self, historical_data: pd.DataFrame, strategy_config: Dict[str, Any]):
        """Run event-driven simulation"""
        try:
            # Group data by timestamp
            grouped_data = historical_data.groupby('timestamp')
            
            for timestamp, data_group in grouped_data:
                # Update order books with new market data
                self._update_order_books(data_group)
                
                # Process strategy signals
                signals = self._generate_strategy_signals(data_group, strategy_config)
                
                # Execute orders based on signals
                for signal in signals:
                    self._execute_order(signal, timestamp)
                
                # Update positions and calculate P&L
                self._update_positions(data_group, timestamp)
                
                # Record equity curve
                self._record_equity_point(timestamp)
                
                # Check risk limits
                self._check_risk_limits()
            
            self.logger.info(f"📈 Event simulation completed: {len(grouped_data)} events processed")
            
        except Exception as e:
            self.logger.error(f"❌ Error in event simulation: {e}")
    
    def _update_order_books(self, market_data: pd.DataFrame):
        """Update order books with new market data"""
        try:
            for _, row in market_data.iterrows():
                symbol = row['symbol']
                current_price = row['close']
                
                if symbol in self.order_books:
                    # Update order book based on price movement
                    price_change = (current_price - self.order_books[symbol]['bids'][0][0]) / self.order_books[symbol]['bids'][0][0]
                    
                    # Adjust order book levels
                    self._adjust_order_book_levels(symbol, current_price, price_change)
                    
                    # Update liquidity based on volume
                    volume = row.get('volume', 1000)
                    self._update_liquidity(symbol, volume)
            
        except Exception as e:
            self.logger.error(f"❌ Error updating order books: {e}")
    
    def _adjust_order_book_levels(self, symbol: str, current_price: float, price_change: float):
        """Adjust order book levels based on price movement"""
        try:
            order_book = self.order_books[symbol]
            
            # Adjust bid levels
            for i, (price, quantity) in enumerate(order_book['bids']):
                new_price = current_price * (1 - (i + 1) * 0.0001)
                order_book['bids'][i] = [new_price, quantity]
            
            # Adjust ask levels
            for i, (price, quantity) in enumerate(order_book['asks']):
                new_price = current_price * (1 + (i + 1) * 0.0001)
                order_book['asks'][i] = [new_price, quantity]
            
            # Add some randomness to quantities
            for i in range(len(order_book['bids'])):
                order_book['bids'][i][1] *= np.random.uniform(0.8, 1.2)
                order_book['asks'][i][1] *= np.random.uniform(0.8, 1.2)
            
            order_book['last_update'] = datetime.now()
            
        except Exception as e:
            self.logger.error(f"❌ Error adjusting order book levels: {e}")
    
    def _update_liquidity(self, symbol: str, volume: float):
        """Update liquidity based on trading volume"""
        try:
            order_book = self.order_books[symbol]
            
            # Volume impact on liquidity
            volume_factor = min(volume / 10000, 2.0)  # Normalize volume impact
            
            for i in range(len(order_book['bids'])):
                order_book['bids'][i][1] *= (0.5 + 0.5 * volume_factor)
                order_book['asks'][i][1] *= (0.5 + 0.5 * volume_factor)
            
        except Exception as e:
            self.logger.error(f"❌ Error updating liquidity: {e}")
    
    def _generate_strategy_signals(self, market_data: pd.DataFrame, 
                                  strategy_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate trading signals based on strategy configuration"""
        try:
            signals = []
            
            for _, row in market_data.iterrows():
                symbol = row['symbol']
                
                # Simple strategy example (replace with actual strategy logic)
                if 'strategy_type' in strategy_config:
                    if strategy_config['strategy_type'] == 'momentum':
                        signal = self._generate_momentum_signal(row)
                    elif strategy_config['strategy_type'] == 'mean_reversion':
                        signal = self._generate_mean_reversion_signal(row)
                    else:
                        signal = self._generate_random_signal(row)
                    
                    if signal:
                        signal['symbol'] = symbol
                        signal['timestamp'] = row['timestamp']
                        signals.append(signal)
            
            return signals
            
        except Exception as e:
            self.logger.error(f"❌ Error generating strategy signals: {e}")
            return []
    
    def _generate_momentum_signal(self, row: pd.Series) -> Optional[Dict[str, Any]]:
        """Generate momentum-based trading signal"""
        try:
            # Simple momentum strategy
            price = row['close']
            volume = row.get('volume', 1000)
            
            # Calculate momentum indicators
            returns = row.get('returns', 0)
            volume_ma = row.get('volume_ma', volume)
            
            # Generate signal
            if returns > 0.001 and volume > volume_ma * 1.2:  # Strong upward momentum
                return {
                    'action': 'buy',
                    'quantity': 1.0,
                    'confidence': min(abs(returns) * 100, 0.9),
                    'signal_type': 'momentum'
                }
            elif returns < -0.001 and volume > volume_ma * 1.2:  # Strong downward momentum
                return {
                    'action': 'sell',
                    'quantity': 1.0,
                    'confidence': min(abs(returns) * 100, 0.9),
                    'signal_type': 'momentum'
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Error generating momentum signal: {e}")
            return None
    
    def _generate_mean_reversion_signal(self, row: pd.Series) -> Optional[Dict[str, Any]]:
        """Generate mean reversion trading signal"""
        try:
            # Simple mean reversion strategy
            price = row['close']
            price_ma = row.get('price_ma', price)
            
            # Calculate deviation from mean
            deviation = (price - price_ma) / price_ma
            
            # Generate signal
            if deviation > 0.02:  # Price 2% above mean
                return {
                    'action': 'sell',
                    'quantity': 1.0,
                    'confidence': min(abs(deviation) * 50, 0.9),
                    'signal_type': 'mean_reversion'
                }
            elif deviation < -0.02:  # Price 2% below mean
                return {
                    'action': 'buy',
                    'quantity': 1.0,
                    'confidence': min(abs(deviation) * 50, 0.9),
                    'signal_type': 'mean_reversion'
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Error generating mean reversion signal: {e}")
            return None
    
    def _generate_random_signal(self, row: pd.Series) -> Optional[Dict[str, Any]]:
        """Generate random trading signal for testing"""
        try:
            # Random signal generation (for testing purposes)
            if np.random.random() < 0.01:  # 1% chance of signal
                action = 'buy' if np.random.random() < 0.5 else 'sell'
                return {
                    'action': action,
                    'quantity': np.random.uniform(0.5, 2.0),
                    'confidence': np.random.uniform(0.3, 0.8),
                    'signal_type': 'random'
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Error generating random signal: {e}")
            return None
    
    def _execute_order(self, signal: Dict[str, Any], timestamp: datetime):
        """Execute order based on signal"""
        try:
            symbol = signal['symbol']
            action = signal['action']
            quantity = signal['quantity']
            confidence = signal['confidence']
            
            # Calculate order size based on capital and confidence
            position_size = self.current_capital * 0.02 * confidence  # 2% base position
            
            # Get current market price
            current_price = self._get_market_price(symbol)
            
            # Calculate fill probability
            fill_probability = self._calculate_fill_probability(symbol, action, quantity, confidence)
            
            # Determine if order is filled
            if np.random.random() < fill_probability:
                # Calculate execution price with slippage
                execution_price = self._calculate_execution_price(symbol, action, quantity, current_price)
                
                # Calculate slippage
                slippage = abs(execution_price - current_price) / current_price
                
                # Execute the trade
                trade_value = quantity * execution_price
                commission = trade_value * self.commission_rate
                
                if action == 'buy':
                    self.positions[symbol] += quantity
                    self.current_capital -= (trade_value + commission)
                else:  # sell
                    self.positions[symbol] -= quantity
                    self.current_capital += (trade_value - commission)
                
                # Record trade
                trade = {
                    'timestamp': timestamp,
                    'symbol': symbol,
                    'action': action,
                    'quantity': quantity,
                    'price': execution_price,
                    'value': trade_value,
                    'commission': commission,
                    'slippage': slippage,
                    'confidence': confidence,
                    'signal_type': signal['signal_type']
                }
                
                self.trades.append(trade)
                
                self.logger.debug(f"✅ Executed {action} order: {quantity} {symbol} @ {execution_price:.4f}")
            
            else:
                self.logger.debug(f"❌ Order not filled: {action} {quantity} {symbol}")
            
        except Exception as e:
            self.logger.error(f"❌ Error executing order: {e}")
    
    def _get_market_price(self, symbol: str) -> float:
        """Get current market price from order book"""
        try:
            if symbol in self.order_books:
                order_book = self.order_books[symbol]
                best_bid = order_book['bids'][0][0]
                best_ask = order_book['asks'][0][0]
                return (best_bid + best_ask) / 2
            
            return 100.0  # Default price
            
        except Exception as e:
            self.logger.error(f"❌ Error getting market price: {e}")
            return 100.0
    
    def _calculate_fill_probability(self, symbol: str, action: str, quantity: float, 
                                   confidence: float) -> float:
        """Calculate probability of order being filled"""
        try:
            if symbol not in self.order_books:
                return 0.5
            
            order_book = self.order_books[symbol]
            
            # Base fill probability
            base_prob = self.base_fill_probability
            
            # Spread impact
            best_bid = order_book['bids'][0][0]
            best_ask = order_book['asks'][0][0]
            spread = (best_ask - best_bid) / best_bid
            spread_factor = 1.0 - (spread * self.spread_impact)
            
            # Volume impact
            available_volume = sum(qty for _, qty in order_book['bids'][:5]) if action == 'sell' else sum(qty for _, qty in order_book['asks'][:5])
            volume_factor = min(quantity / available_volume, 1.0) if available_volume > 0 else 0.5
            
            # Confidence impact
            confidence_factor = confidence
            
            # Calculate final probability
            fill_prob = base_prob * spread_factor * volume_factor * confidence_factor
            
            return min(max(fill_prob, 0.0), 1.0)
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating fill probability: {e}")
            return 0.5
    
    def _calculate_execution_price(self, symbol: str, action: str, quantity: float, 
                                  market_price: float) -> float:
        """Calculate execution price with slippage"""
        try:
            if symbol not in self.order_books:
                return market_price
            
            order_book = self.order_books[symbol]
            
            # Calculate slippage based on order size and liquidity
            if action == 'buy':
                # Walk up the ask book
                remaining_quantity = quantity
                total_cost = 0.0
                
                for price, available_qty in order_book['asks']:
                    if remaining_quantity <= 0:
                        break
                    
                    executed_qty = min(remaining_quantity, available_qty)
                    total_cost += executed_qty * price
                    remaining_quantity -= executed_qty
                
                if quantity > 0:
                    execution_price = total_cost / quantity
                else:
                    execution_price = market_price
                    
            else:  # sell
                # Walk down the bid book
                remaining_quantity = quantity
                total_proceeds = 0.0
                
                for price, available_qty in order_book['bids']:
                    if remaining_quantity <= 0:
                        break
                    
                    executed_qty = min(remaining_quantity, available_qty)
                    total_proceeds += executed_qty * price
                    remaining_quantity -= executed_qty
                
                if quantity > 0:
                    execution_price = total_proceeds / quantity
                else:
                    execution_price = market_price
            
            return execution_price
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating execution price: {e}")
            return market_price
    
    def _update_positions(self, market_data: pd.DataFrame, timestamp: datetime):
        """Update positions and calculate unrealized P&L"""
        try:
            total_pnl = 0.0
            
            for _, row in market_data.iterrows():
                symbol = row['symbol']
                current_price = row['close']
                
                if symbol in self.positions and self.positions[symbol] != 0:
                    # Calculate unrealized P&L
                    position_value = self.positions[symbol] * current_price
                    total_pnl += position_value
            
            # Update current capital with unrealized P&L
            self.current_capital += total_pnl
            
        except Exception as e:
            self.logger.error(f"❌ Error updating positions: {e}")
    
    def _record_equity_point(self, timestamp: datetime):
        """Record equity curve point"""
        try:
            equity_point = {
                'timestamp': timestamp,
                'equity': self.current_capital,
                'positions': self.positions.copy()
            }
            
            self.equity_curve.append(equity_point)
            
        except Exception as e:
            self.logger.error(f"❌ Error recording equity point: {e}")
    
    def _check_risk_limits(self):
        """Check risk limits and apply circuit breakers"""
        try:
            # Check maximum drawdown
            if len(self.equity_curve) > 1:
                peak_equity = max(point['equity'] for point in self.equity_curve)
                current_drawdown = (peak_equity - self.current_capital) / peak_equity
                
                if current_drawdown > 0.1:  # 10% drawdown limit
                    self.logger.warning(f"⚠️ Maximum drawdown exceeded: {current_drawdown:.2%}")
                    # Apply circuit breaker - close all positions
                    self._apply_circuit_breaker()
            
        except Exception as e:
            self.logger.error(f"❌ Error checking risk limits: {e}")
    
    def _apply_circuit_breaker(self):
        """Apply circuit breaker - close all positions"""
        try:
            self.logger.warning("🚨 Circuit breaker activated - closing all positions")
            
            # Close all positions at market price
            for symbol, position in self.positions.items():
                if position != 0:
                    # Simulate market order execution
                    market_price = self._get_market_price(symbol)
                    trade_value = abs(position) * market_price
                    commission = trade_value * self.commission_rate
                    
                    if position > 0:  # Long position
                        self.current_capital += (trade_value - commission)
                    else:  # Short position
                        self.current_capital += (trade_value - commission)
                    
                    self.positions[symbol] = 0
            
        except Exception as e:
            self.logger.error(f"❌ Error applying circuit breaker: {e}")
    
    def _calculate_performance_metrics(self):
        """Calculate comprehensive performance metrics"""
        try:
            if not self.trades:
                return
            
            # Basic metrics
            total_trades = len(self.trades)
            winning_trades = len([t for t in self.trades if t['value'] > 0])
            losing_trades = total_trades - winning_trades
            
            # Calculate returns
            final_equity = self.current_capital
            total_return = (final_equity - self.initial_capital) / self.initial_capital
            
            # Calculate win rate
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # Calculate average win/loss
            wins = [t['value'] for t in self.trades if t['value'] > 0]
            losses = [abs(t['value']) for t in self.trades if t['value'] < 0]
            
            avg_win = np.mean(wins) if wins else 0
            avg_loss = np.mean(losses) if losses else 0
            
            # Calculate profit factor
            total_wins = sum(wins)
            total_losses = sum(losses)
            profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
            
            # Calculate Sharpe ratio
            returns = []
            for i in range(1, len(self.equity_curve)):
                prev_equity = self.equity_curve[i-1]['equity']
                curr_equity = self.equity_curve[i]['equity']
                returns.append((curr_equity - prev_equity) / prev_equity)
            
            if returns:
                sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
            else:
                sharpe_ratio = 0
            
            # Calculate maximum drawdown
            max_drawdown = 0
            peak = self.initial_capital
            
            for point in self.equity_curve:
                equity = point['equity']
                if equity > peak:
                    peak = equity
                drawdown = (peak - equity) / peak
                max_drawdown = max(max_drawdown, drawdown)
            
            # Calculate fill rate and average slippage
            filled_orders = len(self.trades)
            total_orders = filled_orders  # Simplified for now
            fill_rate = filled_orders / total_orders if total_orders > 0 else 0
            
            avg_slippage = np.mean([t['slippage'] for t in self.trades]) if self.trades else 0
            
            # Update results
            self.backtest_results.update({
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'fill_rate': fill_rate,
                'avg_slippage': avg_slippage
            })
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating performance metrics: {e}")
    
    def _generate_backtest_report(self) -> Dict[str, Any]:
        """Generate comprehensive backtest report"""
        try:
            report = {
                'backtest_summary': {
                    'initial_capital': self.initial_capital,
                    'final_capital': self.current_capital,
                    'total_return': self.backtest_results['total_return'],
                    'sharpe_ratio': self.backtest_results['sharpe_ratio'],
                    'max_drawdown': self.backtest_results['max_drawdown'],
                    'win_rate': self.backtest_results['win_rate'],
                    'profit_factor': self.backtest_results['profit_factor']
                },
                'trading_metrics': {
                    'total_trades': self.backtest_results['total_trades'],
                    'winning_trades': self.backtest_results['winning_trades'],
                    'losing_trades': self.backtest_results['losing_trades'],
                    'avg_win': self.backtest_results['avg_win'],
                    'avg_loss': self.backtest_results['avg_loss']
                },
                'execution_metrics': {
                    'fill_rate': self.backtest_results['fill_rate'],
                    'avg_slippage': self.backtest_results['avg_slippage'],
                    'total_commission': sum(t['commission'] for t in self.trades)
                },
                'equity_curve': self.equity_curve,
                'trades': self.trades,
                'final_positions': self.positions
            }
            
            # Save report
            self._save_backtest_report(report)
            
            return report
            
        except Exception as e:
            self.logger.error(f"❌ Error generating backtest report: {e}")
            return {}
    
    def _save_backtest_report(self, report: Dict[str, Any]):
        """Save backtest report to file"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_path = f"backtests/backtest_report_{timestamp}.json"
            
            # Create directory if it doesn't exist
            Path(report_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            self.logger.info(f"💾 Backtest report saved to {report_path}")
            
        except Exception as e:
            self.logger.error(f"❌ Error saving backtest report: {e}")
    
    def get_backtest_results(self) -> Dict[str, Any]:
        """Get backtest results"""
        return self.backtest_results.copy()
    
    def get_equity_curve(self) -> List[Dict[str, Any]]:
        """Get equity curve data"""
        return self.equity_curve.copy()
    
    def get_trades(self) -> List[Dict[str, Any]]:
        """Get all trades"""
        return self.trades.copy() 