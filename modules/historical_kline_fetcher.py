#!/usr/bin/env python3
"""
Historical Kline Fetcher
Implements proper strategy for fetching historical klines without triggering rate limits
Based on Binance API specifications - FIXED for 1-minute intervals
"""

import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import requests
from modules.binance_rate_limiter import binance_limiter

class HistoricalKlineFetcher:
    """
    Fetches historical klines with proper rate limiting
    Strategy: Extended sequential per-symbol with proper delays
    FIXED: Now properly handles 1-minute intervals for 15 days
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.base_url = "https://api.binance.com"
        
        # Strategy parameters
        self.max_limit_per_call = 1000  # Maximum klines per API call
        self.inter_call_delay = 0.1     # 100ms between calls (conservative)
        self.symbol_delay = 1.0         # 1 second between symbols
        
        # Data span: 15 days = 15 × 24 × 60 = 21,600 minutes
        self.days_to_fetch = 15
        self.minutes_per_day = 24 * 60
        self.total_minutes = self.days_to_fetch * self.minutes_per_day
        
        # Calculate calls per symbol (CORRECTED for 1-minute intervals)
        self.calls_per_symbol = (self.total_minutes + self.max_limit_per_call - 1) // self.max_limit_per_call
        
        # Rate limit analysis
        self.weight_per_call = 2  # /api/v3/klines with limit=1000 has weight=2
        self.total_weight_per_symbol = self.calls_per_symbol * self.weight_per_call
        
        self.logger.info("📊 Historical Kline Fetcher initialized (FIXED for 1-minute intervals)")
        self.logger.info(f"   Data span: {self.days_to_fetch} days = {self.total_minutes} minutes")
        self.logger.info(f"   Calls per symbol: {self.calls_per_symbol}")
        self.logger.info(f"   Weight per symbol: {self.total_weight_per_symbol}")
        self.logger.info(f"   Inter-call delay: {self.inter_call_delay}s")
        self.logger.info(f"   Symbol delay: {self.symbol_delay}s")
    
    def get_fetch_strategy_summary(self, symbols: List[str]) -> Dict[str, Any]:
        """Get summary of fetch strategy (CORRECTED for 1-minute intervals)"""
        total_calls = len(symbols) * self.calls_per_symbol
        total_weight = total_calls * self.weight_per_call
        
        # Time estimates (more realistic for 1-minute intervals)
        sequential_time = total_calls * self.inter_call_delay + len(symbols) * self.symbol_delay
        parallel_time = (total_calls * self.inter_call_delay * 2) + (len(symbols) // 2) * self.symbol_delay * 2
        
        # Rate limit analysis
        weight_usage_percent = (total_weight / 1200) * 100
        calls_per_minute = total_calls / (sequential_time / 60) if sequential_time > 0 else 0
        
        return {
            'symbols_count': len(symbols),
            'days_to_fetch': self.days_to_fetch,
            'total_minutes': self.total_minutes,
            'calls_per_symbol': self.calls_per_symbol,
            'weight_per_symbol': self.total_weight_per_symbol,
            'total_calls': total_calls,
            'total_weight': total_weight,
            'weight_usage_percent': weight_usage_percent,
            'calls_per_minute': calls_per_minute,
            'sequential_time_estimate': sequential_time,
            'parallel_time_estimate': parallel_time,
            'inter_call_delay': self.inter_call_delay,
            'symbol_delay': self.symbol_delay,
            'max_limit_per_call': self.max_limit_per_call,
            'rate_limit_safe': weight_usage_percent <= 100
        }
    
    def validate_strategy(self, symbols: List[str]) -> bool:
        """Validate that the strategy won't exceed rate limits (CORRECTED)"""
        summary = self.get_fetch_strategy_summary(symbols)
        
        self.logger.info("🔍 Strategy validation (1-minute intervals):")
        self.logger.info(f"   Total calls: {summary['total_calls']}")
        self.logger.info(f"   Total weight: {summary['total_weight']}")
        self.logger.info(f"   Weight usage: {summary['weight_usage_percent']:.1f}%")
        self.logger.info(f"   Sequential time: {summary['sequential_time_estimate']:.0f}s ({summary['sequential_time_estimate']/60:.1f}min)")
        
        # Check weight limit
        if summary['total_weight'] > 1200:
            self.logger.error(f"❌ Strategy exceeds weight limit: {summary['total_weight']} > 1200")
            self.logger.error(f"   This would require {summary['total_weight']/1200:.1f} minutes to complete safely")
            return False
        
        # Check if we can complete within reasonable time
        if summary['sequential_time_estimate'] > 3600:  # 1 hour
            self.logger.warning(f"⚠️ Strategy will take long: {summary['sequential_time_estimate']:.0f}s ({summary['sequential_time_estimate']/60:.1f}min)")
        
        self.logger.info(f"✅ Strategy validation passed")
        return True
    
    def get_optimized_strategy(self, symbols: List[str]) -> Dict[str, Any]:
        """Get optimized strategy for 1-minute intervals"""
        summary = self.get_fetch_strategy_summary(symbols)
        
        if not summary['rate_limit_safe']:
            # Calculate how to make it safe
            required_minutes = summary['total_weight'] / 1200
            safe_delay = (required_minutes * 60) / summary['total_calls']
            
            return {
                'current_strategy': summary,
                'optimization_needed': True,
                'required_minutes': required_minutes,
                'safe_inter_call_delay': safe_delay,
                'recommendation': f"Use {safe_delay:.3f}s delay between calls or split into {required_minutes:.1f} minute sessions"
            }
        else:
            return {
                'current_strategy': summary,
                'optimization_needed': False,
                'recommendation': "Current strategy is safe"
            }
    
    def fetch_klines_for_symbol(self, symbol: str, start_time: Optional[datetime] = None) -> List[List]:
        """
        Fetch all klines for a single symbol (1-minute intervals)
        Returns: List of kline data
        """
        if start_time is None:
            start_time = datetime.now() - timedelta(days=self.days_to_fetch)
        
        end_time = datetime.now()
        all_klines = []
        
        self.logger.info(f"📈 Fetching 1-minute klines for {symbol} from {start_time} to {end_time}")
        self.logger.info(f"   Expected calls: {self.calls_per_symbol}")
        
        # Convert to milliseconds
        start_ms = int(start_time.timestamp() * 1000)
        end_ms = int(end_time.timestamp() * 1000)
        current_start = start_ms
        
        call_count = 0
        
        while current_start < end_ms and call_count < self.calls_per_symbol:
            # Prepare parameters
            params = {
                'symbol': symbol,
                'interval': '1m',
                'startTime': current_start,
                'limit': self.max_limit_per_call
            }
            
            # Wait for rate limiter
            wait_time = binance_limiter.wait_if_needed('/api/v3/klines', params)
            if wait_time > 0:
                self.logger.debug(f"⏳ Waited {wait_time:.2f}s for rate limiter")
            
            # Make API call
            try:
                response = requests.get(f"{self.base_url}/api/v3/klines", params=params, timeout=30)
                
                # Handle response headers
                header_info = binance_limiter.handle_response_headers(response)
                
                if response.status_code == 200:
                    klines = response.json()
                    all_klines.extend(klines)
                    
                    self.logger.debug(f"✅ {symbol}: Got {len(klines)} klines (call {call_count + 1}/{self.calls_per_symbol})")
                    
                    # Update start time for next call
                    if klines:
                        last_open_time = klines[-1][0]  # openTime of last candle
                        current_start = last_open_time + 60000  # Add 1 minute in milliseconds
                    else:
                        # No more data available
                        break
                    
                    call_count += 1
                    
                    # Inter-call delay
                    time.sleep(self.inter_call_delay)
                    
                else:
                    self.logger.error(f"❌ {symbol}: API error {response.status_code}: {response.text}")
                    break
                    
            except Exception as e:
                self.logger.error(f"❌ {symbol}: Request failed: {e}")
                break
        
        self.logger.info(f"✅ {symbol}: Completed with {len(all_klines)} total 1-minute klines")
        return all_klines
    
    def fetch_klines_for_multiple_symbols(self, symbols: List[str]) -> Dict[str, List[List]]:
        """
        Fetch klines for multiple symbols sequentially (1-minute intervals)
        Returns: Dict mapping symbol to kline data
        """
        results = {}
        
        # Get strategy analysis
        strategy = self.get_optimized_strategy(symbols)
        self.logger.info(f"🚀 Starting 1-minute kline fetch for {len(symbols)} symbols")
        self.logger.info(f"   Strategy: {strategy['recommendation']}")
        
        if strategy['optimization_needed']:
            self.logger.warning(f"⚠️ Strategy requires optimization: {strategy['recommendation']}")
        
        for i, symbol in enumerate(symbols):
            self.logger.info(f"📊 Processing symbol {i+1}/{len(symbols)}: {symbol}")
            
            try:
                klines = self.fetch_klines_for_symbol(symbol)
                results[symbol] = klines
                
                # Symbol delay (except for last symbol)
                if i < len(symbols) - 1:
                    time.sleep(self.symbol_delay)
                    
            except Exception as e:
                self.logger.error(f"❌ Failed to fetch {symbol}: {e}")
                results[symbol] = []
        
        # Log summary
        total_klines = sum(len(klines) for klines in results.values())
        successful_symbols = sum(1 for klines in results.values() if klines)
        
        self.logger.info(f"🎯 1-minute kline fetch completed:")
        self.logger.info(f"   Successful symbols: {successful_symbols}/{len(symbols)}")
        self.logger.info(f"   Total 1-minute klines: {total_klines}")
        
        return results

# Global instance
kline_fetcher = HistoricalKlineFetcher() 