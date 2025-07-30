import logging
import time
import asyncio
from decimal import Decimal, ROUND_DOWN
from binance.client import Client
from binance.exceptions import BinanceAPIException
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

class ExecutionEngine:
    """
    ULTRA-ADVANCED Execution Engine with hybrid execution, smart order routing, and slippage optimization.
    Optimized for maximum profitability with intelligent order management.
    """
    def __init__(self, api_key: str, api_secret: str, maker_fee: float = 0.001, taker_fee: float = 0.001, test_mode: bool = False):
        """
        Initializes the ULTRA-ADVANCED Binance client and execution parameters.
        
        Args:
            api_key: Binance API key
            api_secret: Binance API secret
            maker_fee: Maker fee rate (default 0.1%)
            taker_fee: Taker fee rate (default 0.1%)
            test_mode: If True, runs in test mode without API calls
        """
        self.test_mode = test_mode
        if not test_mode:
        self.client = Client(api_key, api_secret)
        else:
            self.client = None
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.active_orders = {}  # Track active orders
        self.symbol_info_cache = {}  # Cache symbol information
        
        # Performance tracking
        self.total_executed_volume = 0.0
        self.avg_execution_time = 0.0
        self.successful_maker_orders = 0
        self.failed_maker_orders = 0
        self.emergency_taker_orders = 0
        
        # GEMINI PHASE 3: ENHANCED EMERGENCY CIRCUIT BREAKER
        self.emergency_circuit_breaker = {
            'enabled': True,
            'stop_loss_timeout': 5,  # 5 seconds for stop-loss breach
            'liquidity_collapse_threshold': 0.01,  # 1% spread threshold
            'system_health_threshold': 0.8,  # 80% system health minimum
            'emergency_orders': [],
            'trigger_history': [],
            'last_health_check': time.time()
        }
        
        # System health monitoring
        self.system_health = {
            'api_latency': 0.0,
            'order_fill_rate': 1.0,
            'error_rate': 0.0,
            'memory_usage': 0.0,
            'cpu_usage': 0.0,
            'overall_health': 1.0
        }
        
        # Order tracking for emergency triggers
        self.pending_orders = {}
        
        logging.info("ULTRA-ADVANCED Execution Engine initialized with maximum intelligence.")

    def get_symbol_info(self, pair: str) -> Optional[Dict]:
        """
        Gets and caches symbol information including precision requirements.
        
        Args:
            pair: Trading pair symbol (e.g., 'ETHFDUSD')
            
        Returns:
            Symbol information dictionary or None on error
        """
        if self.test_mode:
            # Return mock symbol info for testing
            return {
                'symbol': pair,
                'status': 'TRADING',
                'baseAsset': pair.replace('FDUSD', ''),
                'quoteAsset': 'FDUSD',
                'filters': [
                    {'filterType': 'PRICE_FILTER', 'minPrice': '0.01', 'maxPrice': '1000000', 'tickSize': '0.01'},
                    {'filterType': 'LOT_SIZE', 'minQty': '0.001', 'maxQty': '1000000', 'stepSize': '0.001'},
                    {'filterType': 'MIN_NOTIONAL', 'minNotional': '1.00'}
                ]
            }
        
        if pair not in self.symbol_info_cache:
            try:
                self.symbol_info_cache[pair] = self.client.get_symbol_info(pair)
            except BinanceAPIException as e:
                logging.error(f"Could not fetch symbol info for {pair}: {e}")
                return None
        return self.symbol_info_cache[pair]

    def get_current_price(self, pair: str) -> float:
        """
        Gets the last traded price for a given pair.
        
        Args:
            pair: Trading pair symbol (e.g., 'ETHFDUSD')
            
        Returns:
            Current price as a float, or 0.0 on error
        """
        try:
            return float(self.client.get_symbol_ticker(symbol=pair)['price'])
        except BinanceAPIException as e:
            logging.error(f"Could not fetch current price for {pair}: {e}")
            return 0.0
    # Add this new function inside the ExecutionEngine class
    def get_order_book_ticker(self, pair: str) -> Optional[Dict[str, float]]:
        """
        Gets the best bid and ask prices from the order book.
        :param pair: The trading pair (e.g., 'ETHFDUSD').
        :return: A dictionary with 'bid_price' and 'ask_price' or None on error.
        """
        try:
            self.ticker = self.client.get_orderbook_ticker(symbol=pair)
            return {
                'bid_price': float(ticker['bidPrice']),
                'ask_price': float(ticker['askPrice'])
            }
        except BinanceAPIException as e:
            logging.error(f"Could not fetch order book ticker for {pair}: {e}")
            return None

    def get_account_balance(self, asset: str) -> Dict[str, float]:
        """
        Gets both free and locked balances for a specific asset.
        
        Args:
            asset: Asset symbol (e.g., 'FDUSD', 'ETH')
            
        Returns:
            Dictionary with 'free' and 'locked' balances
        """
        try:
            self.balance_info = self.client.get_account()['balances']
            for balance in self.balance_info:
                if balance['asset'] == asset:
                return {
                        'free': float(balance['free']),
                        'locked': float(balance['locked']),
                        'total': float(balance['free']) + float(balance['locked'])
                }
            return {'free': 0.0, 'locked': 0.0, 'total': 0.0}
        except BinanceAPIException as e:
            logging.error(f"Could not fetch balance for asset {asset}: {e}")
            return {'free': 0.0, 'locked': 0.0, 'total': 0.0}
            
    def get_order_book(self, pair: str, limit: int = 20) -> Dict:
        """
        Gets the full order book to assess market depth.
        
        Args:
            pair: Trading pair symbol
            limit: Number of price levels to fetch (default 20)
            
        Returns:
            Order book with bids and asks
        """
        try:
            return self.client.get_order_book(symbol=pair, limit=limit)
        except BinanceAPIException as e:
            logging.error(f"Could not fetch order book for {pair}: {e}")
            return {"bids": [], "asks": []}
            
    def calculate_optimal_quantity(self, pair: str, side: str, 
                                   available_balance: float, 
                                   max_percentage: float = 0.25, 
                                   min_depth_ratio: float = 3.0) -> Tuple[float, float]:
        """
        Calculates the optimal order quantity based on available balance and order book liquidity.
        
        Args:
            pair: Trading pair symbol
            side: 'BUY' or 'SELL'
            available_balance: Available balance to use for the trade
            max_percentage: Maximum percentage of balance to use (default 25%)
            min_depth_ratio: Minimum ratio of order book depth to order size
            
        Returns:
            Tuple of (quantity, price)
        """
        # Get symbol information for precision
        self.symbol_info = self.get_symbol_info(pair)
        if not self.symbol_info:
            return 0.0, 0.0
            
        # Get order book to analyze liquidity
        self.order_book = self.get_order_book(pair)
        if not self.order_book:
            return 0.0, 0.0
            
        # Calculate base quantity from balance and max percentage
        self.max_quantity = available_balance * max_percentage
        self.current_price = self.get_current_price(pair)
        if self.current_price <= 0:
            return 0.0, 0.0
        
        # Determine which side to analyze for quantity
        if side == 'BUY':
            # For buy orders, we analyze ask side liquidity
            self.order_book_side = self.order_book['asks']
            self.price_threshold = self.current_price * 1.0005 # Slightly above current price
        else:  # SELL
            # For sell orders, we analyze bid side liquidity
            self.order_book_side = self.order_book['bids']
            self.price_threshold = self.current_price * 0.9995 # Slightly below current price

        # Filter order book to only include prices below/above the threshold
        self.filtered_orders = [item for item in self.order_book_side if float(item[0]) <= self.price_threshold]
        
        if self.filtered_orders:
            self.best_price = float(self.filtered_orders[0][0])
            self.available_liquidity = sum(float(volume) for price, volume in self.filtered_orders)
            
            # Ensure our order is not too large relative to available liquidity
            self.liquidity_based_quantity = self.available_liquidity / min_depth_ratio
            self.final_quantity = min(self.max_quantity, self.liquidity_based_quantity)
        else:
            self.final_quantity = 0.0
            
        return self.final_quantity, self.best_price

    def execute_smart_order(self, pair: str, side: str, quantity: float, 
                        urgency: str = 'normal', max_slippage: float = 0.005) -> Dict[str, Any]:
        """
        ULTRA-ADVANCED smart order execution with hybrid routing and slippage optimization.
        
        Args:
            pair: Trading pair symbol
            side: 'BUY' or 'SELL'
            quantity: Order quantity
            urgency: 'low', 'normal', 'high', 'urgent'
            max_slippage: Maximum allowed slippage
            
        Returns:
            Execution result with performance metrics
        """
        try:
            self.start_time = time.time()
            
            # Estimate slippage and market impact
            self.estimated_slippage = self._estimate_slippage(pair, quantity, side)
            self.market_impact = self._estimate_market_impact(pair, quantity)
            
            # Determine execution strategy based on urgency and slippage
            if urgency == 'urgent' or self.estimated_slippage > max_slippage:
                # Use taker order for urgent execution
                self.result = self.place_taker_order(pair, side, quantity)
                self.execution_type = 'taker'
            elif urgency == 'low' and self.estimated_slippage < max_slippage * 0.5:
                # Use maker order for low urgency
                self.result = self.place_maker_order(pair, side, quantity, self._calculate_optimal_maker_price(pair, side, self.current_price))
                self.execution_type = 'maker'
            else:
                # Use hybrid approach
                self.result = self._execute_hybrid_order(pair, side, quantity, self.current_price)
                self.execution_type = 'hybrid'
            
            self.execution_time = time.time() - self.start_time
            self._update_execution_metrics(self.execution_type, self.execution_time, self.estimated_slippage, self.result)
            
            return {
                'success': self.result.get('success', False),
                'execution_type': self.execution_type,
                'execution_time': self.execution_time,
                'estimated_slippage': self.estimated_slippage,
                'market_impact': self.market_impact,
                'result': self.result
            }
            
        except Exception as e:
            logging.error(f"Error in smart order execution: {e}")
            return {'success': False, 'error': str(e)}

    def _estimate_slippage(self, pair: str, quantity: float, side: str) -> float:
        """Estimate slippage based on order book depth and historical data."""
        try:
            self.order_book = self.get_order_book(pair)
            self.current_price = self.get_current_price(pair)
            if not self.order_book or self.current_price <= 0:
                return 0.001  # Default 0.1% slippage
            
            # Calculate slippage from order book
            if side == 'BUY':
                self.orders = self.order_book['asks']
                cumulative_volume = 0.0
                for price, volume in self.orders:
                    cumulative_volume += float(volume)
                    if cumulative_volume >= quantity:
                        self.slippage = (float(price) - self.current_price) / self.current_price
                        return min(self.slippage, 0.05)  # Cap at 5%
            else:  # SELL
                self.orders = self.order_book['bids']
                cumulative_volume = 0.0
                for price, volume in self.orders:
                    cumulative_volume += float(volume)
                    if cumulative_volume >= quantity:
                        self.slippage = (self.current_price - float(price)) / self.current_price
                        return min(self.slippage, 0.05)  # Cap at 5%
            
            # Fallback to historical estimates if available
            if pair in self.slippage_estimates:
                return self.slippage_estimates[pair].get(str(quantity), 0.001)
            
            return 0.001  # Default slippage
            
        except Exception as e:
            logging.error(f"Error estimating slippage: {e}")
            return 0.001

    def _estimate_market_impact(self, pair: str, quantity: float) -> float:
        """Estimate market impact of the order."""
        try:
            # Simple market impact model based on order size relative to average volume
            # In a real implementation, this would use more sophisticated models
            self.order_book = self.get_order_book(pair)
            self.current_price = self.get_current_price(pair)
            if not self.order_book or self.current_price <= 0:
                return 0.001
            
            self.avg_volume = sum(float(volume) for price, volume in self.order_book['bids']) + sum(float(volume) for price, volume in self.order_book['asks'])
            if self.avg_volume == 0:
                return 0.001
                
            self.impact_factor = min(quantity / self.avg_volume, 0.01) # 0.01% per unit, max 1%
            return self.impact_factor
            
        except Exception as e:
            logging.error(f"Error estimating market impact: {e}")
            return 0.001

    def _calculate_optimal_maker_price(self, pair: str, side: str, current_price: float) -> float:
        """Calculate optimal price for maker orders."""
        try:
            self.ticker = self.get_order_book_ticker(pair)
            if not self.ticker:
                return current_price
            
            if side == 'BUY':
                # Place slightly below ask for buy orders
                return float(self.ticker['ask_price']) * 0.9995
            else:
                # Place slightly above bid for sell orders
                return float(self.ticker['bid_price']) * 1.0005
                
        except Exception as e:
            logging.error(f"Error calculating optimal maker price: {e}")
            return current_price

    def _get_large_order_threshold(self, pair: str) -> float:
        """Get threshold for considering an order 'large'."""
        try:
            # This would be based on average daily volume
            # For now, use a simple threshold
            return 100.0  # 100 units
        except Exception as e:
            logging.error(f"Error getting large order threshold: {e}")
            return 50.0

    def _execute_hybrid_order(self, pair: str, side: str, quantity: float, current_price: float) -> Dict[str, Any]:
        """Execute order using hybrid maker-taker approach."""
        try:
            # Try maker order first
            self.maker_result = self.place_maker_order(pair, side, quantity, self._calculate_optimal_maker_price(pair, side, current_price))
            if self.maker_result.get('success', False):
                return self.maker_result
            
            # If maker order fails, use taker order
            logging.info(f"Maker order failed, falling back to taker order for {pair}")
            return self.place_taker_order(pair, side, quantity)
            
        except Exception as e:
            logging.error(f"Error in hybrid order execution: {e}")
            return {'success': False, 'error': str(e)}

    def _execute_twap_order(self, pair: str, side: str, quantity: float) -> Dict[str, Any]:
        """Execute order using Time-Weighted Average Price (TWAP) strategy."""
        try:
            self.twap_chunks = 3 # Number of chunks
            self.chunk_size = quantity / self.twap_chunks
            self.chunk_interval = 1 # Seconds between chunks
            
            self.total_executed = 0.0
            self.results = []
            
            for i in range(self.twap_chunks):
                # Execute chunk
                self.chunk_result = self.place_maker_order(pair, side, self.chunk_size, self._calculate_optimal_maker_price(pair, side, self.current_price))
                self.results.append(self.chunk_result)
                self.total_executed += self.chunk_size
                
                # Wait before next chunk (except for last chunk)
                if i < self.twap_chunks - 1:
                    time.sleep(self.chunk_interval)
            
            # Calculate average execution price
            self.executed_orders = [r for r in self.results if r.get('success', False)]
            if self.executed_orders:
                self.avg_price = sum(r['avg_price'] for r in self.executed_orders) / len(self.executed_orders)
            else:
                self.avg_price = 0
            
            return {
                'success': self.total_executed > 0,
                'execution_type': 'twap',
                'total_executed': self.total_executed,
                'avg_price': self.avg_price,
                'chunk_results': self.results
            }
            
        except Exception as e:
            logging.error(f"Error in TWAP execution: {e}")
            return {'success': False, 'error': str(e)}

    def _update_execution_metrics(self, execution_type: str, execution_time: float, 
                                slippage: float, result: Dict[str, Any]) -> None:
        """Update execution performance metrics."""
        try:
            self.execution_history.append({
                'timestamp': time.time(),
                'execution_type': execution_type,
                'execution_time': execution_time,
                'slippage': slippage,
                'success': result.get('success', False)
            })
            
            # Update counters
            if result.get('success', False):
                if execution_type == 'maker':
                    self.successful_maker_orders += 1
                elif execution_type == 'taker':
                    self.emergency_taker_orders += 1
            else:
                if execution_type == 'maker':
                    self.failed_maker_orders += 1
            
            # Update average execution time
            if self.execution_history:
                self.avg_execution_time = sum(h['execution_time'] for h in self.execution_history) / len(self.execution_history)
            
        except Exception as e:
            logging.error(f"Error updating execution metrics: {e}")

    def place_maker_order(self, pair: str, side: str, quantity: float, price: float) -> Dict[str, Any]:
        """
        Places a post-only limit order on Binance.
        
        Args:
            pair: The trading pair (e.g., 'ETHFDUSD')
            side: 'BUY' or 'SELL'
            quantity: The amount of the base asset to trade
            price: The price at which to place the order
            
        Returns:
            The order response dictionary from Binance or an error dictionary
        """
        try:
            logging.info(f"Placing {side} maker order for {quantity:.6f} {pair} at {price:.2f}")
            
            # Get symbol info for precision rules
            self.info = self.get_symbol_info(pair)
            if not self.info:
                return {"error": f"Could not get symbol info for {pair}"}
                
            self.price_precision = self._get_price_precision(pair)
            self.quantity_precision = self._get_quantity_precision(pair)

            if not self._validate_order_params(pair, quantity, price):
                return {"error": "Order does not meet minimum requirements"}

            # Format price and quantity
            self.formatted_price = f"{price:.{self.price_precision}f}"
            self.formatted_quantity = f"{quantity:.{self.quantity_precision}f}"
            
            # Place post-only limit order
            self.order = self.client.create_order(
                symbol=pair,
                side=side,
                type='LIMIT_MAKER',
                timeInForce='GTC',
                quantity=self.formatted_quantity,
                price=self.formatted_price
            )
            
            # Track the order
            self.active_orders[self.order['orderId']] = {
                'pair': pair,
                'side': side,
                'quantity': quantity,
                'price': price,
                'status': self.order['status'],
                'type': 'maker',
                'time': time.time()
            }
            
            logging.info(f"Order placed successfully: {self.order['orderId']}")
            return self.order
        except BinanceAPIException as e:
            logging.error(f"Failed to place maker order due to API error: {e}")
            return {"error": str(e)}
        except Exception as e:
            logging.error(f"An unexpected error occurred during maker order placement: {e}")
            return {"error": str(e)}
            
    def place_taker_order(self, pair: str, side: str, quantity: float) -> Dict[str, Any]:
        """
        Places a market (taker) order for emergency situations like stop-loss execution.
        
        Args:
            pair: The trading pair (e.g., 'ETHFDUSD')
            side: 'BUY' or 'SELL'
            quantity: The amount of the base asset to trade
            
        Returns:
            The order response dictionary from Binance or an error dictionary
        """
        try:
            logging.warning(f"Placing EMERGENCY {side} taker order for {quantity:.6f} {pair}")
            
            # Get symbol info for precision
            self.info = self.get_symbol_info(pair)
            if not self.info:
                return {"error": f"Could not get symbol info for {pair}"}
                
            self.quantity_precision = self._get_quantity_precision(pair)
            self.formatted_quantity = f"{quantity:.{self.quantity_precision}f}"
            
            # Place market order
            self.order = self.client.create_order(
                symbol=pair,
                side=side,
                type='MARKET',
                quantity=self.formatted_quantity
            )
            
            logging.warning(f"Emergency taker order executed: {self.order['orderId']}")
            return self.order
        except BinanceAPIException as e:
            logging.error(f"Failed to place taker order due to API error: {e}")
            return {"error": str(e)}
        except Exception as e:
            logging.error(f"An unexpected error occurred during taker order placement: {e}")
            return {"error": str(e)}
            
    def cancel_order(self, pair: str, order_id: str) -> Dict[str, Any]:
        """
        Cancels an existing order.
        
        Args:
            pair: Trading pair symbol
            order_id: Order ID to cancel
            
        Returns:
            Cancellation response from Binance
        """
        try:
            self.result = self.client.cancel_order(
                symbol=pair,
                orderId=order_id
            )
            if self.result['orderId'] in self.active_orders:
                self.active_orders.pop(self.result['orderId'])
                
            logging.info(f"Successfully cancelled order {self.result['orderId']}")
            return self.result
        except BinanceAPIException as e:
            logging.error(f"Failed to cancel order {order_id}: {e}")
            return {"error": str(e)}
            
    def cancel_all_orders(self, pair: str) -> Dict[str, Any]:
        """
        Cancels all open orders for a specific trading pair.
        
        Args:
            pair: Trading pair symbol
            
        Returns:
            Dictionary with success/failure counts
        """
        try:
            self.result = self.client.get_open_orders(symbol=pair)
            self.active_orders = {k: v for k, v in self.active_orders.items() if v['pair'] != pair}
            
            logging.info(f"Cancelled all open orders for {pair}")
            return {"success": True, "message": f"All orders for {pair} cancelled"}
        except BinanceAPIException as e:
            logging.error(f"Failed to cancel all orders for {pair}: {e}")
            return {"error": str(e)}
            
    def get_open_orders(self, pair: str = None) -> List[Dict[str, Any]]:
        """
        Gets all open orders, optionally filtered by trading pair.
        
        Args:
            pair: Optional trading pair to filter
            
        Returns:
            List of open orders
        """
        try:
            if pair:
                self.orders = self.client.get_open_orders(symbol=pair)
            else:
                self.orders = self.client.get_open_orders()
                
            return [
                {
                    'orderId': order['orderId'],
                    'symbol': order['symbol'],
                    'side': order['side'],
                    'type': order['type'],
                    'origQty': float(order['origQty']),
                    'executedQty': float(order['executedQty']),
                    'price': float(order['price']),
                    'status': order['status'],
                    'time': order['time'] / 1000  # Convert from milliseconds
                } for order in self.orders
            ]
        except BinanceAPIException as e:
            logging.error(f"Failed to get open orders: {e}")
            return []
            
    def place_trailing_stop_order(self, pair: str, quantity: float, 
                                 activation_price: float, callback_rate: float) -> Dict[str, Any]:
        """
        Places a trailing stop order to protect positions.
        
        Args:
            pair: Trading pair symbol
            quantity: Quantity to sell
            activation_price: Price at which trailing stop activates
            callback_rate: Callback rate in percentage
            
        Returns:
            Order response from Binance
        """
        try:
            # Get symbol info for precision
            self.info = self.get_symbol_info(pair)
            if not self.info:
                return {"error": f"Could not get symbol info for {pair}"}
                
            self.quantity_precision = self._get_quantity_precision(pair)
            self.formatted_quantity = f"{quantity:.{self.quantity_precision}f}"
            
            # Place trailing stop order - note that this requires Binance Futures API
            # For spot trading, we would need to implement our own trailing stop logic
            self.order = self.client.create_order(
                symbol=pair,
                side='SELL', # Always sell for trailing stop
                type='TRAILING_STOP_MARKET',
                quantity=self.formatted_quantity,
                callbackRate=callback_rate,
                activationPrice=activation_price
            )
            
            logging.info(f"Trailing stop order placed: {self.order['orderId']}")
            return self.order
        except BinanceAPIException as e:
            logging.error(f"Failed to place trailing stop: {e}")
            return {"error": str(e)}
            
    def _validate_order_params(self, pair: str, quantity: float, price: float) -> bool:
        """
        Validates order parameters before placement.
        
        Args:
            pair: Trading pair symbol
            quantity: Order quantity
            price: Order price
            
        Returns:
            True if parameters are valid, False otherwise
        """
        try:
            self.symbol_info = self.get_symbol_info(pair)
            if not self.symbol_info:
                return False
                
            # Check minimum quantity
            self.min_qty = float(self.symbol_info['filters'][1]['minQty']) # LOT_SIZE filter
            if quantity < self.min_qty:
                logging.warning(f"Quantity {quantity} below minimum {self.min_qty} for {pair}")
                        return False
                        
            # Check price precision
            self.price_precision = self._get_price_precision(pair)
            if len(str(price).split('.')[-1]) > self.price_precision:
                logging.warning(f"Price precision exceeds limit for {pair}")
                return False
                
            return True
            
        except Exception as e:
            logging.error(f"Error validating order parameters: {e}")
            return False

    def _get_price_precision(self, pair: str) -> int:
        """Get the price precision for a given trading pair."""
        try:
            self.symbol_info = self.get_symbol_info(pair)
            if not self.symbol_info:
                return 8 # Default precision
            for f in self.symbol_info['filters']:
                if f['filterType'] == 'PRICE_FILTER':
                    return len(str(f['tickSize']).split('.')[-1])
            return 8 # Default precision
        except Exception as e:
            logging.error(f"Error getting price precision for {pair}: {e}")
            return 8

    def _get_quantity_precision(self, pair: str) -> int:
        """Get the quantity precision for a given trading pair."""
        try:
            self.symbol_info = self.get_symbol_info(pair)
            if not self.symbol_info:
                return 8 # Default precision
            for f in self.symbol_info['filters']:
                if f['filterType'] == 'LOT_SIZE':
                    return len(str(f['stepSize']).split('.')[-1])
            return 8 # Default precision
        except Exception as e:
            logging.error(f"Error getting quantity precision for {pair}: {e}")
            return 8

    # GEMINI PHASE 3: ENHANCED EMERGENCY CIRCUIT BREAKER METHODS
    
    def check_emergency_triggers(self, pair: str, position_data: Dict = None) -> Dict[str, Any]:
        """
        Check all emergency circuit breaker triggers as specified in Gemini's Phase 3.
        
        Args:
            pair: Trading pair symbol
            position_data: Current position data for stop-loss checking
            
        Returns:
            Dictionary with trigger status and recommended actions
        """
        try:
            self.triggers = {
                'stop_loss_breach': False,
                'liquidity_collapse': False,
                'system_health_failure': False,
                'emergency_action_required': False,
                'recommended_action': None
            }
            
            # Trigger 1: Stop-Loss Failure
            if position_data and self._check_stop_loss_breach(pair, position_data):
                self.triggers['stop_loss_breach'] = True
                self.triggers['emergency_action_required'] = True
                self.triggers['recommended_action'] = 'emergency_market_exit'
                logging.warning(f"🚨 STOP-LOSS BREACH DETECTED for {pair}")
            
            # Trigger 2: Liquidity Collapse
            if self._check_liquidity_collapse(pair):
                self.triggers['liquidity_collapse'] = True
                self.triggers['emergency_action_required'] = True
                self.triggers['recommended_action'] = 'emergency_market_exit'
                logging.warning(f"🚨 LIQUIDITY COLLAPSE DETECTED for {pair}")
            
            # Trigger 3: System Health Failure
            if self._check_system_health_failure():
                self.triggers['system_health_failure'] = True
                self.triggers['emergency_action_required'] = True
                self.triggers['recommended_action'] = 'pause_trading'
                logging.warning(f"🚨 SYSTEM HEALTH FAILURE DETECTED")
            
            # Log trigger history
            if any([self.triggers['stop_loss_breach'], self.triggers['liquidity_collapse'], self.triggers['system_health_failure']]):
                self.emergency_circuit_breaker['trigger_history'].append({
                    'timestamp': time.time(),
                    'pair': pair,
                    'triggers': self.triggers,
                    'position_data': position_data
                })
            
            return self.triggers
            
        except Exception as e:
            logging.error(f"Error checking emergency triggers: {e}")
            return {'emergency_action_required': False, 'recommended_action': None}
    
    def _check_stop_loss_breach(self, pair: str, position_data: Dict) -> bool:
        """
        Check if stop-loss is breached and maker exit order is not filled within 5 seconds.
        
        Args:
            pair: Trading pair symbol
            position_data: Position data with stop-loss information
            
        Returns:
            True if stop-loss breach detected
        """
        try:
            if not position_data or 'stop_loss_price' not in position_data:
                return False
            
            self.current_price = self.get_current_price(pair)
            self.position_side = position_data.get('position_side', 'LONG') # Default to LONG
            self.stop_loss_price = float(position_data['stop_loss_price'])

            self.breach_detected = False
            if self.position_side == 'LONG' and self.current_price <= self.stop_loss_price:
                self.breach_detected = True
            elif self.position_side == 'SHORT' and self.current_price > self.stop_loss_price:
                self.breach_detected = True

            if self.breach_detected:
                # Check if maker exit order is pending and unfilled
                self.pending_order_id = self.emergency_circuit_breaker['emergency_orders'][-1].get('orderId') if self.emergency_circuit_breaker['emergency_orders'] else None
                if self.pending_order_id:
                    self.order_time = time.time() - self.active_orders[self.pending_order_id]['time']
                    if self.order_time > self.emergency_circuit_breaker['stop_loss_timeout']:
                        logging.warning(f"Stop-loss breached for {pair}, maker order unfilled for {self.order_time:.1f}s")
                        return True
            
            return False
            
        except Exception as e:
            logging.error(f"Error checking stop-loss breach: {e}")
            return False
    
    def _check_liquidity_collapse(self, pair: str) -> bool:
        """
        Check if bid-ask spread widens beyond 1% indicating liquidity collapse.
        
        Args:
            pair: Trading pair symbol
            
        Returns:
            True if liquidity collapse detected
        """
        try:
            self.order_book_ticker = self.get_order_book_ticker(pair)
            if not self.order_book_ticker:
                return False
            
            self.bid_price = float(self.order_book_ticker['bid_price'])
            self.ask_price = float(self.order_book_ticker['ask_price'])
            
            if self.bid_price <= 0 or self.ask_price <= 0:
                return False
            
            self.spread = (self.ask_price - self.bid_price) / self.ask_price
            if self.spread > self.emergency_circuit_breaker['liquidity_collapse_threshold']:
                logging.warning(f"Liquidity collapse detected for {pair}: {self.spread:.3%} spread")
                return True
            
            return False
            
        except Exception as e:
            logging.error(f"Error checking liquidity collapse: {e}")
            return False
    
    def _check_system_health_failure(self) -> bool:
        """
        Check if system health drops below 80% threshold.
        
        Returns:
            True if system health failure detected
        """
        try:
            # Update system health metrics
            self._update_system_health()
            
            if self.system_health['overall_health'] < self.emergency_circuit_breaker['system_health_threshold']:
                logging.warning(f"System health failure: {self.system_health['overall_health']:.1%} below {self.emergency_circuit_breaker['system_health_threshold']:.1%}")
                return True
            
            return False
            
        except Exception as e:
            logging.error(f"Error checking system health: {e}")
            return False
    
    def _update_system_health(self):
        """Update system health metrics."""
        try:
            # Calculate order fill rate
            self.total_orders = self.successful_maker_orders + self.emergency_taker_orders
            if self.total_orders > 0:
                self.system_health['order_fill_rate'] = self.successful_maker_orders / self.total_orders
            
            # Calculate error rate (simplified)
            self.system_health['error_rate'] = min(self.failed_maker_orders / max(self.total_orders, 1), 1.0)
            
            # Calculate overall health score
            self.health_factors = [
                self.system_health['order_fill_rate'],
                1.0 - self.system_health['error_rate'],
                min(1.0, 1.0 - self.system_health['api_latency'] / 1000),  # Normalize latency
            ]
            
            self.system_health['overall_health'] = sum(self.health_factors) / len(self.health_factors)
            
        except Exception as e:
            logging.error(f"Error updating system health: {e}")
    
    def execute_emergency_action(self, pair: str, action: str, position_data: Dict = None) -> Dict[str, Any]:
        """
        Execute emergency action based on circuit breaker triggers.
        
        Args:
            pair: Trading pair symbol
            action: Emergency action to take ('emergency_market_exit', 'pause_trading')
            position_data: Position data for market exit
            
        Returns:
            Execution result
        """
        try:
            if action == 'emergency_market_exit':
                return self._execute_emergency_market_exit(pair, position_data)
            elif action == 'pause_trading':
                return self._execute_pause_trading()
            else:
                logging.error(f"Unknown emergency action: {action}")
                return {'success': False, 'error': f'Unknown action: {action}'}
                
        except Exception as e:
            logging.error(f"Error executing emergency action: {e}")
            return {'success': False, 'error': str(e)}
    
    def _execute_emergency_market_exit(self, pair: str, position_data: Dict) -> Dict[str, Any]:
        """
        Execute emergency market exit for a position.
        
        Args:
            pair: Trading pair symbol
            position_data: Position data with quantity and side
            
        Returns:
            Execution result
        """
        try:
            if not position_data:
                return {'success': False, 'error': 'No position data provided'}
            
            self.quantity = float(position_data.get('quantity', 0))
            if self.quantity <= 0:
                return {'success': False, 'error': 'Invalid quantity'}
            
            # Determine exit side (opposite of position side)
            self.exit_side = 'SELL' if position_data.get('position_side', 'LONG') == 'LONG' else 'BUY'
            
            # Cancel any pending emergency orders
            self.pending_order_id = self.emergency_circuit_breaker['emergency_orders'][-1].get('orderId') if self.emergency_circuit_breaker['emergency_orders'] else None
            if self.pending_order_id:
                self.cancel_order(pair, self.pending_order_id)
            
            # Execute market order for immediate exit
            self.result = self.place_taker_order(pair, self.exit_side, self.quantity)
            if self.result.get('success', False):
                self.emergency_taker_orders += 1
                logging.warning(f"🚨 EMERGENCY MARKET EXIT EXECUTED for {pair}: {self.exit_side} {self.quantity}")
            
            return self.result
            
        except Exception as e:
            logging.error(f"Error executing emergency market exit: {e}")
            return {'success': False, 'error': str(e)}
    
    def _execute_pause_trading(self) -> Dict[str, Any]:
        """
        Execute pause trading action.
        
        Returns:
            Execution result
        """
        try:
            # Cancel all open orders
            for pair in self.active_orders.keys():
                self.cancel_all_orders(pair)
            
            # Disable trading
            self.emergency_circuit_breaker['enabled'] = False
            
            logging.warning("🚨 TRADING PAUSED due to system health failure")
            
            return {
                'success': True,
                'action': 'trading_paused',
                'message': 'Trading paused due to system health failure'
            }
            
        except Exception as e:
            logging.error(f"Error executing pause trading: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_emergency_status(self) -> Dict[str, Any]:
        """
        Get current emergency circuit breaker status.
        
        Returns:
            Status dictionary
        """
        return {
            'enabled': self.emergency_circuit_breaker['enabled'],
            'system_health': self.system_health,
            'trigger_history': self.emergency_circuit_breaker['trigger_history'][-10:],  # Last 10 triggers
            'emergency_orders_count': self.emergency_taker_orders,
            'last_health_check': self.emergency_circuit_breaker['last_health_check']
        }
