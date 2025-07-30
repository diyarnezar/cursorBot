"""
Parallel Data Collector for Project Hyperion
Handles multiple symbols with ultra-conservative rate limiting
"""

import logging
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Any
import pandas as pd
import multiprocessing
from datetime import datetime

from .binance_collector import BinanceDataCollector, BinanceConfig

logger = logging.getLogger(__name__)


class ParallelDataCollector:
    """
    Parallel data collector with ultra-conservative rate limiting
    
    Features:
    - Parallel symbol processing
    - Rate limit management across all workers
    - Automatic retry logic
    - Progress tracking
    - Memory-efficient processing
    """
    
    def __init__(self, config: BinanceConfig, max_workers: Optional[int] = None):
        """
        Initialize parallel data collector
        
        Args:
            config: Binance configuration
            max_workers: Maximum number of parallel workers (default: 90% of CPU cores)
        """
        self.config = config
        
        # Calculate optimal number of workers
        if max_workers is None:
            cpu_count = multiprocessing.cpu_count()
            self.max_workers = max(1, int(cpu_count * 0.9))  # Use 90% of CPU cores
        else:
            self.max_workers = max_workers
            
        logger.info(f"🔄 Parallel data collector initialized with {self.max_workers} workers")
        
        # Global rate limiting across all workers
        self.global_rate_limiter = GlobalRateLimiter(
            max_requests_per_minute=100,  # Ultra-conservative: 100 requests per minute
            max_requests_per_day=10000    # 10,000 requests per day
        )
        
        # Thread-safe data storage
        self.collected_data = {}
        self.data_lock = threading.Lock()
        
        # Progress tracking
        self.progress = {
            'total_symbols': 0,
            'completed_symbols': 0,
            'failed_symbols': 0,
            'start_time': None,
            'estimated_completion': None
        }
        self.progress_lock = threading.Lock()
    
    def collect_data_parallel(self, symbols: List[str], days: int, interval: str = '1m') -> Dict[str, pd.DataFrame]:
        """
        Collect data for multiple symbols in parallel
        
        Args:
            symbols: List of trading symbols
            days: Number of days to fetch
            interval: Time interval
            
        Returns:
            Dictionary mapping symbols to DataFrames
        """
        logger.info(f"🚀 Starting parallel data collection for {len(symbols)} symbols")
        logger.info(f"📊 Target: {days} days of {interval} data")
        logger.info(f"⚡ Using {self.max_workers} parallel workers")
        
        # Initialize progress tracking
        with self.progress_lock:
            self.progress['total_symbols'] = len(symbols)
            self.progress['completed_symbols'] = 0
            self.progress['failed_symbols'] = 0
            self.progress['start_time'] = datetime.now()
        
        # Clear previous data
        self.collected_data.clear()
        
        # Calculate optimal batch size based on rate limits
        # We want to ensure we don't exceed 100 requests per minute
        requests_per_symbol = self._calculate_requests_per_symbol(days, interval)
        safe_batch_size = max(1, min(self.max_workers, 100 // requests_per_symbol))
        
        logger.info(f"📦 Processing in batches of {safe_batch_size} symbols")
        
        # Process symbols in batches
        all_results = {}
        
        for i in range(0, len(symbols), safe_batch_size):
            batch = symbols[i:i + safe_batch_size]
            batch_num = i // safe_batch_size + 1
            total_batches = (len(symbols) + safe_batch_size - 1) // safe_batch_size
            
            logger.info(f"📦 Processing batch {batch_num}/{total_batches} ({len(batch)} symbols)")
            
            # Process this batch
            batch_results = self._process_batch(batch, days, interval)
            all_results.update(batch_results)
            
            # Rate limiting between batches
            if i + safe_batch_size < len(symbols):
                batch_delay = 60.0 / 100  # 60 seconds / 100 requests per minute
                logger.info(f"⏳ Rate limiting: waiting {batch_delay:.1f}s between batches")
                time.sleep(batch_delay)
        
        # Final progress report
        self._log_final_progress()
        
        return all_results
    
    def _process_batch(self, symbols: List[str], days: int, interval: str) -> Dict[str, pd.DataFrame]:
        """Process a batch of symbols in parallel"""
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_symbol = {
                executor.submit(self._collect_single_symbol, symbol, days, interval): symbol
                for symbol in symbols
            }
            
            # Collect results
            batch_results = {}
            
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    symbol_data = future.result()
                    if not symbol_data.empty:
                        batch_results[symbol] = symbol_data
                        self._update_progress('completed')
                        logger.info(f"✅ Completed {symbol}: {len(symbol_data)} records")
                    else:
                        self._update_progress('failed')
                        logger.warning(f"⚠️ No data for {symbol}")
                        
                except Exception as e:
                    self._update_progress('failed')
                    logger.error(f"❌ Failed to collect data for {symbol}: {e}")
        
        return batch_results
    
    def _collect_single_symbol(self, symbol: str, days: int, interval: str) -> pd.DataFrame:
        """Collect data for a single symbol with rate limiting"""
        
        # Wait for global rate limiter
        self.global_rate_limiter.wait()
        
        # Create individual collector for this symbol
        collector = BinanceDataCollector(config=self.config)
        
        try:
            # Fetch data with retry logic
            data = collector.fetch_historical_data(
                symbol=symbol,
                days=days,
                interval=interval
            )
            
            return data
            
        except Exception as e:
            logger.error(f"❌ Error collecting data for {symbol}: {e}")
            return pd.DataFrame()
    
    def _calculate_requests_per_symbol(self, days: int, interval: str) -> int:
        """Calculate number of API requests needed per symbol"""
        records_per_day = self._get_records_per_day(interval)
        total_records = days * records_per_day
        max_records_per_request = 1000
        return (total_records + max_records_per_request - 1) // max_records_per_request
    
    def _get_records_per_day(self, interval: str) -> int:
        """Calculate records per day for given interval"""
        interval_map = {
            '1m': 1440,   # 24 * 60
            '3m': 480,    # 24 * 20
            '5m': 288,    # 24 * 12
            '15m': 96,    # 24 * 4
            '30m': 48,    # 24 * 2
            '1h': 24,
            '2h': 12,
            '4h': 6,
            '6h': 4,
            '8h': 3,
            '12h': 2,
            '1d': 1,
            '3d': 1,
            '1w': 1,
            '1M': 1
        }
        return interval_map.get(interval, 1440)
    
    def _update_progress(self, status: str):
        """Update progress tracking"""
        with self.progress_lock:
            if status == 'completed':
                self.progress['completed_symbols'] += 1
            elif status == 'failed':
                self.progress['failed_symbols'] += 1
            
            # Calculate estimated completion
            if self.progress['completed_symbols'] > 0:
                elapsed = (datetime.now() - self.progress['start_time']).total_seconds()
                rate = self.progress['completed_symbols'] / elapsed
                remaining = self.progress['total_symbols'] - self.progress['completed_symbols']
                eta_seconds = remaining / rate if rate > 0 else 0
                self.progress['estimated_completion'] = datetime.now().timestamp() + eta_seconds
    
    def _log_final_progress(self):
        """Log final progress report"""
        with self.progress_lock:
            total_time = (datetime.now() - self.progress['start_time']).total_seconds()
            success_rate = (self.progress['completed_symbols'] / self.progress['total_symbols']) * 100
            
            logger.info(f"🎉 Parallel data collection completed!")
            logger.info(f"📊 Results:")
            logger.info(f"   ✅ Successful: {self.progress['completed_symbols']}")
            logger.info(f"   ❌ Failed: {self.progress['failed_symbols']}")
            logger.info(f"   📈 Success rate: {success_rate:.1f}%")
            logger.info(f"   ⏱️  Total time: {total_time:.1f}s")
            logger.info(f"   🚀 Average: {total_time/self.progress['total_symbols']:.1f}s per symbol")


class GlobalRateLimiter:
    """Global rate limiter for coordinating across multiple workers"""
    
    def __init__(self, max_requests_per_minute: int = 100, max_requests_per_day: int = 10000):
        self.max_requests_per_minute = max_requests_per_minute
        self.max_requests_per_day = max_requests_per_day
        self.lock = threading.Lock()
        
        # Tracking
        self.requests_this_minute = 0
        self.requests_today = 0
        self.last_minute_reset = datetime.now()
        self.last_day_reset = datetime.now()
    
    def wait(self):
        """Wait if rate limits would be exceeded"""
        with self.lock:
            now = datetime.now()
            
            # Reset counters if needed
            if (now - self.last_minute_reset).total_seconds() >= 60:
                self.requests_this_minute = 0
                self.last_minute_reset = now
                
            if (now - self.last_day_reset).total_seconds() >= 86400:
                self.requests_today = 0
                self.last_day_reset = now
            
            # Check if we need to wait
            if (self.requests_this_minute >= self.max_requests_per_minute or 
                self.requests_today >= self.max_requests_per_day):
                
                # Calculate wait time
                wait_time = 60.0  # Default to 1 minute
                
                if self.requests_this_minute >= self.max_requests_per_minute:
                    # Wait until next minute
                    elapsed = (now - self.last_minute_reset).total_seconds()
                    wait_time = max(wait_time, 60.0 - elapsed)
                
                if self.requests_today >= self.max_requests_per_day:
                    # Wait until next day
                    elapsed = (now - self.last_day_reset).total_seconds()
                    wait_time = max(wait_time, 86400.0 - elapsed)
                
                logger.debug(f"⏳ Rate limit reached, waiting {wait_time:.1f}s")
                time.sleep(wait_time)
                
                # Reset counters after waiting
                self.requests_this_minute = 0
                self.requests_today = 0
                self.last_minute_reset = datetime.now()
                self.last_day_reset = datetime.now()
            
            # Increment counters
            self.requests_this_minute += 1
            self.requests_today += 1
    
    def get_status(self) -> Dict[str, Any]:
        """Get current rate limiter status"""
        with self.lock:
            return {
                'requests_per_minute': self.requests_this_minute,
                'max_per_minute': self.max_requests_per_minute,
                'requests_per_day': self.requests_today,
                'max_per_day': self.max_requests_per_day
            } 