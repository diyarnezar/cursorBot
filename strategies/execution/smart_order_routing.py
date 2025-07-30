"""
🚀 Smart Order Routing Module

This module implements smart order routing for optimal trade execution
in cryptocurrency trading with multi-exchange support.

Author: Hyperion Trading System
Date: 2025
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
import asyncio
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logger = logging.getLogger(__name__)

class SmartOrderRouter:
    """
    🚀 Smart Order Routing System
    
    Implements intelligent order routing for optimal trade execution
    across multiple exchanges and order types.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the smart order router.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.exchanges = {}
        self.order_history = []
        self.execution_metrics = {}
        self.routing_strategies = {}
        
        # Routing parameters
        self.routing_params = {
            'max_slippage': 0.001,  # 0.1% maximum slippage
            'min_liquidity': 1000,  # Minimum liquidity required
            'max_spread': 0.002,  # 0.2% maximum spread
            'execution_timeout': 30,  # 30 seconds timeout
            'retry_attempts': 3,  # Number of retry attempts
            'smart_order_types': ['market', 'limit', 'stop', 'twap', 'vwap'],
            'priority_exchanges': ['binance', 'coinbase', 'kraken'],
            'fallback_exchanges': ['huobi', 'okx', 'bybit']
        }
        
        # Execution strategies
        self.execution_strategies = {
            'aggressive': {
                'slippage_tolerance': 0.002,
                'time_priority': True,
                'price_priority': False
            },
            'conservative': {
                'slippage_tolerance': 0.0005,
                'time_priority': False,
                'price_priority': True
            },
            'balanced': {
                'slippage_tolerance': 0.001,
                'time_priority': True,
                'price_priority': True
            }
        }
        
        logger.info("🚀 Smart Order Router initialized")
    
    def add_exchange(self, exchange_name: str, exchange_config: Dict[str, Any]):
        """Add an exchange to the router."""
        try:
            self.exchanges[exchange_name] = {
                'config': exchange_config,
                'status': 'active',
                'metrics': {
                    'latency': 0,
                    'success_rate': 1.0,
                    'avg_slippage': 0,
                    'liquidity_score': 1.0
                },
                'order_book': {},
                'recent_trades': []
            }
            
            logger.info(f"✅ Added exchange: {exchange_name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to add exchange {exchange_name}: {e}")
    
    def update_exchange_metrics(self, exchange_name: str, metrics: Dict[str, Any]):
        """Update exchange performance metrics."""
        try:
            if exchange_name in self.exchanges:
                self.exchanges[exchange_name]['metrics'].update(metrics)
                logger.info(f"✅ Updated metrics for {exchange_name}")
            else:
                logger.warning(f"⚠️ Exchange {exchange_name} not found")
                
        except Exception as e:
            logger.error(f"❌ Failed to update metrics for {exchange_name}: {e}")
    
    def analyze_market_conditions(self, symbol: str) -> Dict[str, Any]:
        """Analyze current market conditions for optimal routing."""
        try:
            market_analysis = {
                'volatility': 0,
                'liquidity': 0,
                'spread': 0,
                'volume': 0,
                'best_exchanges': [],
                'execution_strategy': 'balanced'
            }
            
            # Analyze each exchange
            exchange_scores = {}
            
            for exchange_name, exchange_data in self.exchanges.items():
                if exchange_data['status'] != 'active':
                    continue
                
                # Get order book data
                order_book = exchange_data.get('order_book', {}).get(symbol, {})
                if not order_book:
                    continue
                
                # Calculate metrics
                bid = order_book.get('bids', [0])[0] if order_book.get('bids') else 0
                ask = order_book.get('asks', [0])[0] if order_book.get('asks') else 0
                spread = (ask - bid) / bid if bid > 0 else float('inf')
                
                # Calculate liquidity
                bid_volume = sum(bid_vol for _, bid_vol in order_book.get('bids', [])[:5])
                ask_volume = sum(ask_vol for _, ask_vol in order_book.get('asks', [])[:5])
                liquidity = min(bid_volume, ask_volume)
                
                # Calculate score
                metrics = exchange_data['metrics']
                score = (
                    metrics['success_rate'] * 0.3 +
                    (1 - metrics['avg_slippage']) * 0.3 +
                    (1 - spread) * 0.2 +
                    (liquidity / self.routing_params['min_liquidity']) * 0.2
                )
                
                exchange_scores[exchange_name] = {
                    'score': score,
                    'spread': spread,
                    'liquidity': liquidity,
                    'latency': metrics['latency']
                }
            
            # Sort exchanges by score
            sorted_exchanges = sorted(exchange_scores.items(), key=lambda x: x[1]['score'], reverse=True)
            
            # Update market analysis
            if sorted_exchanges:
                market_analysis['best_exchanges'] = [ex[0] for ex in sorted_exchanges[:3]]
                market_analysis['spread'] = sorted_exchanges[0][1]['spread']
                market_analysis['liquidity'] = sorted_exchanges[0][1]['liquidity']
                
                # Determine execution strategy
                if market_analysis['spread'] > self.routing_params['max_spread']:
                    market_analysis['execution_strategy'] = 'conservative'
                elif market_analysis['liquidity'] < self.routing_params['min_liquidity']:
                    market_analysis['execution_strategy'] = 'aggressive'
            
            return market_analysis
            
        except Exception as e:
            logger.error(f"❌ Failed to analyze market conditions: {e}")
            return {}
    
    def select_optimal_exchange(self, symbol: str, order_type: str, 
                               order_size: float) -> Tuple[str, Dict[str, Any]]:
        """Select the optimal exchange for order execution."""
        try:
            market_analysis = self.analyze_market_conditions(symbol)
            
            if not market_analysis.get('best_exchanges'):
                logger.warning("⚠️ No suitable exchanges found")
                return None, {}
            
            # Get execution strategy
            strategy = market_analysis['execution_strategy']
            strategy_params = self.execution_strategies[strategy]
            
            # Select exchange based on strategy
            if strategy_params['price_priority']:
                # Price priority: select exchange with best spread
                best_exchange = None
                best_spread = float('inf')
                
                for exchange_name in market_analysis['best_exchanges']:
                    exchange_data = self.exchanges[exchange_name]
                    order_book = exchange_data.get('order_book', {}).get(symbol, {})
                    
                    if order_book:
                        bid = order_book.get('bids', [0])[0] if order_book.get('bids') else 0
                        ask = order_book.get('asks', [0])[0] if order_book.get('asks') else 0
                        spread = (ask - bid) / bid if bid > 0 else float('inf')
                        
                        if spread < best_spread:
                            best_spread = spread
                            best_exchange = exchange_name
                
                selected_exchange = best_exchange
            else:
                # Time priority: select exchange with lowest latency
                selected_exchange = market_analysis['best_exchanges'][0]
            
            if selected_exchange:
                execution_config = {
                    'exchange': selected_exchange,
                    'strategy': strategy,
                    'strategy_params': strategy_params,
                    'market_analysis': market_analysis
                }
                
                logger.info(f"✅ Selected exchange: {selected_exchange} with strategy: {strategy}")
                return selected_exchange, execution_config
            
            return None, {}
            
        except Exception as e:
            logger.error(f"❌ Failed to select optimal exchange: {e}")
            return None, {}
    
    def calculate_optimal_order_size(self, symbol: str, exchange: str, 
                                   total_size: float) -> List[Dict[str, Any]]:
        """Calculate optimal order size and splitting strategy."""
        try:
            exchange_data = self.exchanges[exchange]
            order_book = exchange_data.get('order_book', {}).get(symbol, {})
            
            if not order_book:
                return [{'size': total_size, 'type': 'market'}]
            
            # Get available liquidity at different price levels
            bids = order_book.get('bids', [])
            asks = order_book.get('asks', [])
            
            orders = []
            remaining_size = total_size
            
            # Calculate optimal order sizes based on liquidity
            for price, volume in asks[:5]:  # Top 5 ask levels
                if remaining_size <= 0:
                    break
                
                order_size = min(remaining_size, volume * 0.1)  # Use 10% of available volume
                if order_size > 0:
                    orders.append({
                        'size': order_size,
                        'price': price,
                        'type': 'limit'
                    })
                    remaining_size -= order_size
            
            # If still have remaining size, add market order
            if remaining_size > 0:
                orders.append({
                    'size': remaining_size,
                    'type': 'market'
                })
            
            return orders
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate optimal order size: {e}")
            return [{'size': total_size, 'type': 'market'}]
    
    async def execute_order(self, symbol: str, side: str, size: float, 
                           order_type: str = 'market') -> Dict[str, Any]:
        """Execute an order using smart routing."""
        try:
            logger.info(f"🚀 Executing {side} order for {size} {symbol}")
            
            # Select optimal exchange
            selected_exchange, execution_config = self.select_optimal_exchange(symbol, order_type, size)
            
            if not selected_exchange:
                return {'success': False, 'error': 'No suitable exchange found'}
            
            # Calculate optimal order sizes
            order_splits = self.calculate_optimal_order_size(symbol, selected_exchange, size)
            
            # Execute orders
            execution_results = []
            total_executed = 0
            total_cost = 0
            
            for order_split in order_splits:
                # Execute individual order
                result = await self._execute_single_order(
                    exchange=selected_exchange,
                    symbol=symbol,
                    side=side,
                    size=order_split['size'],
                    order_type=order_split['type'],
                    price=order_split.get('price'),
                    execution_config=execution_config
                )
                
                if result['success']:
                    total_executed += result['executed_size']
                    total_cost += result['total_cost']
                
                execution_results.append(result)
            
            # Calculate execution metrics
            avg_price = total_cost / total_executed if total_executed > 0 else 0
            execution_efficiency = total_executed / size if size > 0 else 0
            
            # Store order history
            order_record = {
                'timestamp': datetime.now(),
                'symbol': symbol,
                'side': side,
                'size': size,
                'executed_size': total_executed,
                'avg_price': avg_price,
                'efficiency': execution_efficiency,
                'exchange': selected_exchange,
                'strategy': execution_config['strategy'],
                'results': execution_results
            }
            
            self.order_history.append(order_record)
            
            # Update execution metrics
            self._update_execution_metrics(order_record)
            
            logger.info(f"✅ Order executed - Efficiency: {execution_efficiency:.2%}")
            
            return {
                'success': True,
                'executed_size': total_executed,
                'avg_price': avg_price,
                'efficiency': execution_efficiency,
                'exchange': selected_exchange,
                'results': execution_results
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to execute order: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _execute_single_order(self, exchange: str, symbol: str, side: str, 
                                   size: float, order_type: str, price: float = None,
                                   execution_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute a single order on the specified exchange."""
        try:
            # Simulate order execution (replace with actual exchange API calls)
            await asyncio.sleep(0.1)  # Simulate network latency
            
            # Simulate execution results
            executed_size = size * 0.95  # 95% fill rate
            execution_price = price if price else 1000  # Simulated price
            
            # Calculate slippage
            expected_price = execution_price
            actual_price = execution_price * (1 + np.random.normal(0, 0.001))  # Small slippage
            slippage = abs(actual_price - expected_price) / expected_price
            
            # Check if slippage is within tolerance
            strategy_params = execution_config.get('strategy_params', {})
            slippage_tolerance = strategy_params.get('slippage_tolerance', 0.001)
            
            if slippage > slippage_tolerance:
                logger.warning(f"⚠️ High slippage detected: {slippage:.4f}")
            
            result = {
                'success': True,
                'executed_size': executed_size,
                'execution_price': actual_price,
                'slippage': slippage,
                'total_cost': executed_size * actual_price,
                'exchange': exchange,
                'order_type': order_type
            }
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Failed to execute single order: {e}")
            return {
                'success': False,
                'error': str(e),
                'executed_size': 0,
                'total_cost': 0
            }
    
    def _update_execution_metrics(self, order_record: Dict[str, Any]):
        """Update execution metrics with order results."""
        try:
            # Update exchange metrics
            exchange = order_record['exchange']
            if exchange in self.exchanges:
                metrics = self.exchanges[exchange]['metrics']
                
                # Update success rate
                success_rate = metrics['success_rate']
                metrics['success_rate'] = success_rate * 0.9 + 0.1  # Exponential moving average
                
                # Update average slippage
                avg_slippage = metrics['avg_slippage']
                current_slippage = order_record.get('slippage', 0)
                metrics['avg_slippage'] = avg_slippage * 0.9 + current_slippage * 0.1
            
            # Update overall execution metrics
            self.execution_metrics['total_orders'] = self.execution_metrics.get('total_orders', 0) + 1
            self.execution_metrics['total_volume'] = self.execution_metrics.get('total_volume', 0) + order_record['executed_size']
            self.execution_metrics['avg_efficiency'] = (
                self.execution_metrics.get('avg_efficiency', 0) * 0.9 + 
                order_record['efficiency'] * 0.1
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to update execution metrics: {e}")
    
    def get_routing_summary(self) -> Dict[str, Any]:
        """Get a summary of routing activities."""
        return {
            'total_exchanges': len(self.exchanges),
            'active_exchanges': len([ex for ex in self.exchanges.values() if ex['status'] == 'active']),
            'total_orders': len(self.order_history),
            'execution_metrics': self.execution_metrics,
            'routing_params': self.routing_params,
            'execution_strategies': list(self.execution_strategies.keys())
        }
    
    def save_routing_state(self, filepath: str):
        """Save routing state to file."""
        try:
            import pickle
            
            routing_state = {
                'exchanges': self.exchanges,
                'order_history': self.order_history,
                'execution_metrics': self.execution_metrics,
                'routing_params': self.routing_params
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(routing_state, f)
            
            logger.info(f"💾 Routing state saved to {filepath}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save routing state: {e}")
    
    def load_routing_state(self, filepath: str):
        """Load routing state from file."""
        try:
            import pickle
            
            with open(filepath, 'rb') as f:
                routing_state = pickle.load(f)
            
            self.exchanges = routing_state['exchanges']
            self.order_history = routing_state['order_history']
            self.execution_metrics = routing_state['execution_metrics']
            self.routing_params = routing_state['routing_params']
            
            logger.info(f"📂 Routing state loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load routing state: {e}")


# Example usage
if __name__ == "__main__":
    # Configuration
    config = {
        'smart_routing_enabled': True,
        'max_slippage': 0.001,
        'execution_timeout': 30
    }
    
    # Initialize smart order router
    router = SmartOrderRouter(config)
    
    # Add exchanges
    router.add_exchange('binance', {'api_key': 'test', 'api_secret': 'test'})
    router.add_exchange('coinbase', {'api_key': 'test', 'api_secret': 'test'})
    
    # Update exchange metrics
    router.update_exchange_metrics('binance', {
        'latency': 50,
        'success_rate': 0.98,
        'avg_slippage': 0.0005,
        'liquidity_score': 0.9
    })
    
    # Execute order
    async def test_execution():
        result = await router.execute_order('BTC/USDT', 'buy', 0.1, 'market')
        print(f"Execution result: {result}")
    
    # Run test
    asyncio.run(test_execution())
    
    # Get routing summary
    summary = router.get_routing_summary()
    print(f"Smart order router initialized with {summary['total_exchanges']} exchanges") 