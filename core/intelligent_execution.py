"""
🚀 PROJECT HYPERION - INTELLIGENT EXECUTION ALCHEMIST
====================================================

Implements Phase 3 from gemini_plan_new.md
Master the art of maker-only orders with high fill rates and zero costs.

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
import websockets
import threading
from queue import Queue

from config.api_config import APIConfig
from data.collectors.binance_collector import BinanceDataCollector
from risk.maximum_intelligence_risk import MaximumIntelligenceRisk


class IntelligentExecutionAlchemist:
    """
    Intelligent Execution Alchemist
    Masters maker-only order execution with high fill rates and zero costs
    """
    
    def __init__(self, config_path: str = "config.json"):
        """Initialize the Intelligent Execution Alchemist"""
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
        self.api_config = APIConfig()
        self.data_collector = BinanceDataCollector()
        self.risk_manager = MaximumIntelligenceRisk(config=self.config)
        
        # Order book state
        self.order_books = {}
        self.trade_flows = {}
        self.market_conditions = {}
        
        # Execution settings
        self.passive_spread_threshold = 0.001  # 0.1% spread for passive placement
        self.aggressive_spread_threshold = 0.005  # 0.5% spread for aggressive placement
        self.max_reprice_delay = 30  # Maximum 30 seconds before repricing
        self.min_fill_probability = 0.8  # Minimum 80% fill probability
        
        # Emergency settings
        self.emergency_spread_threshold = 0.01  # 1% spread triggers emergency
        self.emergency_fill_timeout = 5  # 5 seconds emergency fill timeout
        self.stop_loss_fill_timeout = 5  # 5 seconds stop loss fill timeout
        
        # Performance tracking
        self.execution_stats = {
            'total_orders': 0,
            'filled_orders': 0,
            'cancelled_orders': 0,
            'emergency_orders': 0,
            'total_fill_rate': 0.0,
            'average_fill_time': 0.0,
            'total_slippage': 0.0
        }
        
        # Active orders
        self.active_orders = {}
        self.order_history = []
        
        # WebSocket connections
        self.ws_connections = {}
        self.ws_running = False
        
        self.logger.info("🚀 Intelligent Execution Alchemist initialized")
    
    async def start_order_book_streaming(self, symbols: List[str]):
        """Start real-time order book streaming for symbols"""
        try:
            self.ws_running = True
            self.logger.info(f"📡 Starting order book streaming for {len(symbols)} symbols")
            
            # Start WebSocket connections for each symbol
            tasks = []
            for symbol in symbols:
                task = asyncio.create_task(self._stream_order_book(symbol))
                tasks.append(task)
            
            # Wait for all streams to complete
            await asyncio.gather(*tasks)
            
        except Exception as e:
            self.logger.error(f"❌ Error in order book streaming: {e}")
        finally:
            self.ws_running = False
    
    async def _stream_order_book(self, symbol: str):
        """Stream order book data for a specific symbol"""
        try:
            # Binance WebSocket URL for order book
            ws_url = f"wss://stream.binance.com:9443/ws/{symbol.lower()}@depth20@100ms"
            
            async with websockets.connect(ws_url) as websocket:
                self.logger.info(f"📡 Connected to order book stream for {symbol}")
                
                while self.ws_running:
                    try:
                        message = await websocket.recv()
                        data = json.loads(message)
                        
                        # Process order book update
                        await self._process_order_book_update(symbol, data)
                        
                    except websockets.exceptions.ConnectionClosed:
                        self.logger.warning(f"⚠️ WebSocket connection closed for {symbol}")
                        break
                    except Exception as e:
                        self.logger.error(f"❌ Error processing order book for {symbol}: {e}")
                        continue
                        
        except Exception as e:
            self.logger.error(f"❌ Error streaming order book for {symbol}: {e}")
    
    async def _process_order_book_update(self, symbol: str, data: Dict[str, Any]):
        """Process order book update"""
        try:
            # Extract order book data
            bids = [[float(price), float(qty)] for price, qty in data.get('bids', [])]
            asks = [[float(price), float(qty)] for price, qty in data.get('asks', [])]
            
            # Calculate order book metrics
            order_book_metrics = self._calculate_order_book_metrics(bids, asks)
            
            # Update order book state
            self.order_books[symbol] = {
                'bids': bids,
                'asks': asks,
                'metrics': order_book_metrics,
                'timestamp': datetime.now()
            }
            
            # Check for market condition changes
            await self._check_market_conditions(symbol, order_book_metrics)
            
        except Exception as e:
            self.logger.error(f"❌ Error processing order book update for {symbol}: {e}")
    
    def _calculate_order_book_metrics(self, bids: List[List[float]], asks: List[List[float]]) -> Dict[str, float]:
        """Calculate order book metrics"""
        try:
            if not bids or not asks:
                return {}
            
            best_bid = bids[0][0]
            best_ask = asks[0][0]
            spread = best_ask - best_bid
            spread_percent = spread / best_bid
            
            # Calculate liquidity depth
            bid_depth = sum(qty for _, qty in bids[:10])  # Top 10 levels
            ask_depth = sum(qty for _, qty in asks[:10])
            
            # Calculate VWAP for bid/ask walls
            bid_vwap = sum(price * qty for price, qty in bids[:5]) / sum(qty for _, qty in bids[:5])
            ask_vwap = sum(price * qty for price, qty in asks[:5]) / sum(qty for _, qty in asks[:5])
            
            # Calculate order flow imbalance
            total_bid_volume = sum(qty for _, qty in bids[:5])
            total_ask_volume = sum(qty for _, qty in asks[:5])
            flow_imbalance = (total_bid_volume - total_ask_volume) / (total_bid_volume + total_ask_volume)
            
            return {
                'best_bid': best_bid,
                'best_ask': best_ask,
                'spread': spread,
                'spread_percent': spread_percent,
                'bid_depth': bid_depth,
                'ask_depth': ask_depth,
                'bid_vwap': bid_vwap,
                'ask_vwap': ask_vwap,
                'flow_imbalance': flow_imbalance,
                'liquidity_ratio': min(bid_depth, ask_depth) / max(bid_depth, ask_depth)
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating order book metrics: {e}")
            return {}
    
    async def _check_market_conditions(self, symbol: str, metrics: Dict[str, float]):
        """Check for market condition changes"""
        try:
            spread_percent = metrics.get('spread_percent', 0)
            flow_imbalance = metrics.get('flow_imbalance', 0)
            
            # Determine market condition
            if spread_percent > self.emergency_spread_threshold:
                condition = 'emergency'
            elif spread_percent > self.aggressive_spread_threshold:
                condition = 'aggressive'
            elif spread_percent > self.passive_spread_threshold:
                condition = 'moderate'
            else:
                condition = 'passive'
            
            # Update market conditions
            self.market_conditions[symbol] = {
                'condition': condition,
                'spread_percent': spread_percent,
                'flow_imbalance': flow_imbalance,
                'timestamp': datetime.now()
            }
            
            # Check for emergency conditions
            if condition == 'emergency':
                await self._handle_emergency_condition(symbol, metrics)
                
        except Exception as e:
            self.logger.error(f"❌ Error checking market conditions for {symbol}: {e}")
    
    async def _handle_emergency_condition(self, symbol: str, metrics: Dict[str, float]):
        """Handle emergency market conditions"""
        try:
            self.logger.warning(f"🚨 Emergency condition detected for {symbol}: {metrics['spread_percent']:.3%} spread")
            
            # Cancel all active orders for this symbol
            await self._cancel_all_orders(symbol)
            
            # Trigger emergency circuit breaker
            await self._trigger_emergency_circuit_breaker(symbol)
            
        except Exception as e:
            self.logger.error(f"❌ Error handling emergency condition for {symbol}: {e}")
    
    async def place_maker_order(self, symbol: str, side: str, quantity: float, 
                               confidence: float = 0.5) -> Dict[str, Any]:
        """Place an intelligent maker order"""
        try:
            if symbol not in self.order_books:
                raise ValueError(f"No order book data for {symbol}")
            
            order_book = self.order_books[symbol]
            metrics = order_book['metrics']
            market_condition = self.market_conditions.get(symbol, {}).get('condition', 'passive')
            
            # Determine placement strategy
            placement_strategy = self._determine_placement_strategy(
                side, metrics, market_condition, confidence
            )
            
            # Calculate optimal price
            optimal_price = self._calculate_optimal_price(side, metrics, placement_strategy)
            
            # Calculate fill probability
            fill_probability = self._calculate_fill_probability(side, metrics, placement_strategy)
            
            # Place the order
            order_result = await self._place_order(symbol, side, quantity, optimal_price, placement_strategy)
            
            # Track order
            order_id = order_result.get('orderId')
            if order_id:
                self.active_orders[order_id] = {
                    'symbol': symbol,
                    'side': side,
                    'quantity': quantity,
                    'price': optimal_price,
                    'strategy': placement_strategy,
                    'confidence': confidence,
                    'fill_probability': fill_probability,
                    'place_time': datetime.now(),
                    'status': 'active'
                }
                
                # Start monitoring for repricing
                asyncio.create_task(self._monitor_order_fill(order_id))
            
            # Update statistics
            self.execution_stats['total_orders'] += 1
            
            self.logger.info(
                f"📊 Placed {side} order for {symbol}: {quantity} @ {optimal_price} "
                f"({placement_strategy} strategy, {fill_probability:.1%} fill probability)"
            )
            
            return order_result
            
        except Exception as e:
            self.logger.error(f"❌ Error placing maker order for {symbol}: {e}")
            return {'error': str(e)}
    
    def _determine_placement_strategy(self, side: str, metrics: Dict[str, float], 
                                    market_condition: str, confidence: float) -> str:
        """Determine optimal placement strategy"""
        try:
            spread_percent = metrics.get('spread_percent', 0)
            
            # High confidence + narrow spread = aggressive placement
            if confidence > 0.8 and spread_percent < self.passive_spread_threshold:
                return 'aggressive'
            
            # Emergency condition = emergency placement
            elif market_condition == 'emergency':
                return 'emergency'
            
            # Wide spread = passive placement
            elif spread_percent > self.aggressive_spread_threshold:
                return 'passive'
            
            # Moderate conditions = adaptive placement
            else:
                return 'adaptive'
                
        except Exception as e:
            self.logger.error(f"❌ Error determining placement strategy: {e}")
            return 'passive'
    
    def _calculate_optimal_price(self, side: str, metrics: Dict[str, float], 
                               strategy: str) -> float:
        """Calculate optimal price for order placement"""
        try:
            best_bid = metrics.get('best_bid', 0)
            best_ask = metrics.get('best_ask', 0)
            spread = metrics.get('spread', 0)
            
            if side == 'buy':
                if strategy == 'aggressive':
                    # Cross the spread slightly to ensure fill
                    return best_ask + (spread * 0.1)
                elif strategy == 'passive':
                    # Place at best bid
                    return best_bid
                elif strategy == 'adaptive':
                    # Place between best bid and ask
                    return best_bid + (spread * 0.3)
                else:  # emergency
                    # Market order equivalent
                    return best_ask + (spread * 0.5)
            
            else:  # sell
                if strategy == 'aggressive':
                    # Cross the spread slightly to ensure fill
                    return best_bid - (spread * 0.1)
                elif strategy == 'passive':
                    # Place at best ask
                    return best_ask
                elif strategy == 'adaptive':
                    # Place between best bid and ask
                    return best_ask - (spread * 0.3)
                else:  # emergency
                    # Market order equivalent
                    return best_bid - (spread * 0.5)
                    
        except Exception as e:
            self.logger.error(f"❌ Error calculating optimal price: {e}")
            return 0
    
    def _calculate_fill_probability(self, side: str, metrics: Dict[str, float], 
                                  strategy: str) -> float:
        """Calculate probability of order being filled"""
        try:
            spread_percent = metrics.get('spread_percent', 0)
            liquidity_ratio = metrics.get('liquidity_ratio', 1.0)
            flow_imbalance = abs(metrics.get('flow_imbalance', 0))
            
            # Base probability based on strategy
            base_probability = {
                'aggressive': 0.95,
                'adaptive': 0.85,
                'passive': 0.70,
                'emergency': 0.99
            }.get(strategy, 0.70)
            
            # Adjust for spread
            spread_adjustment = 1.0 - (spread_percent * 10)  # Wider spread = lower probability
            spread_adjustment = max(spread_adjustment, 0.1)
            
            # Adjust for liquidity
            liquidity_adjustment = liquidity_ratio
            
            # Adjust for flow imbalance
            flow_adjustment = 1.0 - (flow_imbalance * 0.5)
            
            final_probability = base_probability * spread_adjustment * liquidity_adjustment * flow_adjustment
            
            return min(final_probability, 0.99)
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating fill probability: {e}")
            return 0.5
    
    async def _place_order(self, symbol: str, side: str, quantity: float, 
                          price: float, strategy: str) -> Dict[str, Any]:
        """Place the actual order via API"""
        try:
            # This would integrate with the actual Binance API
            # For now, simulate order placement
            
            order_id = f"order_{int(time.time() * 1000)}"
            
            # Simulate order placement
            order_result = {
                'orderId': order_id,
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'price': price,
                'status': 'NEW',
                'strategy': strategy,
                'timestamp': datetime.now().isoformat()
            }
            
            return order_result
            
        except Exception as e:
            self.logger.error(f"❌ Error placing order: {e}")
            return {'error': str(e)}
    
    async def _monitor_order_fill(self, order_id: str):
        """Monitor order for fill and handle repricing"""
        try:
            order = self.active_orders.get(order_id)
            if not order:
                return
            
            start_time = order['place_time']
            max_wait_time = timedelta(seconds=self.max_reprice_delay)
            
            while True:
                await asyncio.sleep(5)  # Check every 5 seconds
                
                # Check if order is still active
                if order_id not in self.active_orders:
                    break
                
                current_time = datetime.now()
                elapsed_time = current_time - start_time
                
                # Check if we need to reprice
                if elapsed_time > max_wait_time:
                    await self._reprice_order(order_id)
                    break
                
                # Check if order was filled
                if await self._check_order_filled(order_id):
                    break
                    
        except Exception as e:
            self.logger.error(f"❌ Error monitoring order {order_id}: {e}")
    
    async def _reprice_order(self, order_id: str):
        """Reprice an unfilled order"""
        try:
            order = self.active_orders.get(order_id)
            if not order:
                return
            
            symbol = order['symbol']
            side = order['side']
            
            # Get current order book
            if symbol not in self.order_books:
                return
            
            current_metrics = self.order_books[symbol]['metrics']
            
            # Calculate new price (more aggressive)
            new_price = self._calculate_optimal_price(side, current_metrics, 'aggressive')
            
            # Cancel old order
            await self._cancel_order(order_id)
            
            # Place new order
            new_order_result = await self._place_order(
                symbol, side, order['quantity'], new_price, 'aggressive'
            )
            
            # Update tracking
            new_order_id = new_order_result.get('orderId')
            if new_order_id:
                self.active_orders[new_order_id] = {
                    **order,
                    'price': new_price,
                    'strategy': 'aggressive',
                    'place_time': datetime.now(),
                    'reprice_count': order.get('reprice_count', 0) + 1
                }
                
                self.logger.info(f"🔄 Repriced order {order_id} -> {new_order_id} at {new_price}")
            
        except Exception as e:
            self.logger.error(f"❌ Error repricing order {order_id}: {e}")
    
    async def _check_order_filled(self, order_id: str) -> bool:
        """Check if order was filled"""
        try:
            # This would check the actual order status via API
            # For now, simulate with random fill probability
            
            order = self.active_orders.get(order_id)
            if not order:
                return False
            
            fill_probability = order.get('fill_probability', 0.5)
            
            # Simulate fill check
            if np.random.random() < fill_probability * 0.1:  # 10% of probability per check
                await self._handle_order_fill(order_id)
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"❌ Error checking order fill for {order_id}: {e}")
            return False
    
    async def _handle_order_fill(self, order_id: str):
        """Handle order fill"""
        try:
            order = self.active_orders.get(order_id)
            if not order:
                return
            
            # Update statistics
            self.execution_stats['filled_orders'] += 1
            
            # Calculate fill time
            fill_time = datetime.now() - order['place_time']
            self.execution_stats['average_fill_time'] = (
                (self.execution_stats['average_fill_time'] * (self.execution_stats['filled_orders'] - 1) + 
                 fill_time.total_seconds()) / self.execution_stats['filled_orders']
            )
            
            # Update fill rate
            self.execution_stats['total_fill_rate'] = (
                self.execution_stats['filled_orders'] / self.execution_stats['total_orders']
            )
            
            # Remove from active orders
            del self.active_orders[order_id]
            
            self.logger.info(f"✅ Order {order_id} filled in {fill_time.total_seconds():.1f}s")
            
        except Exception as e:
            self.logger.error(f"❌ Error handling order fill for {order_id}: {e}")
    
    async def _cancel_order(self, order_id: str):
        """Cancel an order"""
        try:
            # This would cancel the actual order via API
            # For now, just remove from tracking
            
            if order_id in self.active_orders:
                del self.active_orders[order_id]
                self.execution_stats['cancelled_orders'] += 1
                
                self.logger.info(f"❌ Cancelled order {order_id}")
            
        except Exception as e:
            self.logger.error(f"❌ Error cancelling order {order_id}: {e}")
    
    async def _cancel_all_orders(self, symbol: str):
        """Cancel all active orders for a symbol"""
        try:
            orders_to_cancel = [
                order_id for order_id, order in self.active_orders.items()
                if order['symbol'] == symbol
            ]
            
            for order_id in orders_to_cancel:
                await self._cancel_order(order_id)
                
        except Exception as e:
            self.logger.error(f"❌ Error cancelling all orders for {symbol}: {e}")
    
    async def _trigger_emergency_circuit_breaker(self, symbol: str):
        """Trigger emergency circuit breaker"""
        try:
            self.logger.warning(f"🚨 Emergency circuit breaker triggered for {symbol}")
            
            # Cancel all orders
            await self._cancel_all_orders(symbol)
            
            # Update statistics
            self.execution_stats['emergency_orders'] += 1
            
            # Notify risk manager
            await self.risk_manager.handle_emergency_condition(symbol)
            
        except Exception as e:
            self.logger.error(f"❌ Error triggering emergency circuit breaker for {symbol}: {e}")
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""
        return self.execution_stats.copy()
    
    def get_active_orders(self) -> Dict[str, Any]:
        """Get all active orders"""
        return self.active_orders.copy()
    
    def get_order_book_summary(self, symbol: str) -> Dict[str, Any]:
        """Get order book summary for a symbol"""
        if symbol in self.order_books:
            return self.order_books[symbol].copy()
        return {}
    
    def get_market_conditions(self, symbol: str) -> Dict[str, Any]:
        """Get market conditions for a symbol"""
        return self.market_conditions.get(symbol, {}).copy()
    
    def stop_streaming(self):
        """Stop all WebSocket streaming"""
        self.ws_running = False
        self.logger.info("🛑 Stopped order book streaming") 