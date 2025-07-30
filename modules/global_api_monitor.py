#!/usr/bin/env python3
"""
Global API Monitor
Tracks ALL API calls across the entire system to ensure Binance limits are never exceeded
"""

import time
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, Callable
from collections import deque
import functools

class GlobalAPIMonitor:
    """
    Global monitor that tracks ALL API calls across the entire system
    Ensures Binance API limits are never exceeded from any source
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Global tracking
        self.global_request_history = deque(maxlen=1000)
        self.global_second_history = deque(maxlen=1000)
        
        # Source tracking
        self.source_requests = {}  # Track requests by source
        self.source_stats = {}     # Track statistics by source
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Binance API limits (CORRECTED)
        self.binance_requests_per_minute = 1200  # 1,200 requests per MINUTE
        self.binance_requests_per_second = 20    # 20 requests per SECOND
        
        # Conservative safety limits (stay well under Binance limits)
        self.global_safety_limit_per_second = 20  # Use 100% of Binance limit (20/sec)
        self.global_safety_limit_per_minute = 1200  # Use 100% of Binance limit (1200/min)
        
        # Statistics
        self.global_stats = {
            'total_requests': 0,
            'blocked_requests': 0,
            'delayed_requests': 0,
            'sources': set()
        }
        
        self.logger.info("🌍 Global API Monitor initialized with CORRECT Binance limits")
        self.logger.info(f"   Binance limits: {self.binance_requests_per_minute}/min, {self.binance_requests_per_second}/sec")
        self.logger.info(f"   Safety limits: {self.global_safety_limit_per_minute}/min, {self.global_safety_limit_per_second}/sec")
    
    def register_api_call(self, source: str = 'unknown', call_type: str = 'api') -> bool:
        """
        Register an API call from any source
        Returns True if allowed, False if blocked
        """
        with self.lock:
            now = datetime.now()
            
            # Clean old history
            cutoff_time = now - timedelta(seconds=1)
            while self.global_second_history and self.global_second_history[0] < cutoff_time:
                self.global_second_history.popleft()
            
            # Check global safety limit (15 requests per second)
            requests_last_second = len(self.global_second_history)
            if requests_last_second >= self.global_safety_limit_per_second:
                self.global_stats['blocked_requests'] += 1
                self.logger.warning(f"🚫 GLOBAL SECOND LIMIT EXCEEDED: {requests_last_second} requests in last second from {source}")
                return False
            
            # Check minute limit (1000 requests per minute)
            minute_cutoff = now - timedelta(minutes=1)
            while self.global_request_history and self.global_request_history[0] < minute_cutoff:
                self.global_request_history.popleft()
            
            requests_last_minute = len(self.global_request_history)
            if requests_last_minute >= self.global_safety_limit_per_minute:
                self.global_stats['blocked_requests'] += 1
                self.logger.warning(f"🚫 GLOBAL MINUTE LIMIT EXCEEDED: {requests_last_minute} requests in last minute from {source}")
                return False
            
            # Record the request
            self.global_request_history.append(now)
            self.global_second_history.append(now)
            
            # Track by source
            if source not in self.source_requests:
                self.source_requests[source] = deque(maxlen=1000)
                self.source_stats[source] = {
                    'total_requests': 0,
                    'last_request': None,
                    'requests_last_second': 0
                }
            
            self.source_requests[source].append(now)
            self.source_stats[source]['total_requests'] += 1
            self.source_stats[source]['last_request'] = now
            
            # Update global stats
            self.global_stats['total_requests'] += 1
            self.global_stats['sources'].add(source)
            
            return True
    
    def wait_if_global_limit_approaching(self, source: str = 'unknown') -> float:
        """
        Wait if global limit is approaching
        Returns wait time
        """
        with self.lock:
            now = datetime.now()
            wait_time = 0.0
            
            # Clean old history
            cutoff_time = now - timedelta(seconds=1)
            while self.global_second_history and self.global_second_history[0] < cutoff_time:
                self.global_second_history.popleft()
            
            # Check if we're approaching the second limit (80% threshold)
            requests_last_second = len(self.global_second_history)
            second_threshold = int(self.global_safety_limit_per_second * 0.8)  # 12 requests
            
            if requests_last_second >= second_threshold:
                # Calculate wait time
                oldest_request = self.global_second_history[0]
                next_available = oldest_request + timedelta(seconds=1)
                wait_time = max(0.0, (next_available - now).total_seconds())
                
                if wait_time > 0:
                    self.global_stats['delayed_requests'] += 1
                    self.logger.warning(f"⚠️ Global second limit approaching ({requests_last_second}/{self.global_safety_limit_per_second}). Waiting {wait_time:.2f}s")
            
            # Check minute limit (80% threshold)
            minute_cutoff = now - timedelta(minutes=1)
            while self.global_request_history and self.global_request_history[0] < minute_cutoff:
                self.global_request_history.popleft()
            
            requests_last_minute = len(self.global_request_history)
            minute_threshold = int(self.global_safety_limit_per_minute * 0.8)  # 800 requests
            
            if requests_last_minute >= minute_threshold:
                oldest_minute_request = self.global_request_history[0]
                next_minute_available = oldest_minute_request + timedelta(minutes=1)
                minute_wait = max(0.0, (next_minute_available - now).total_seconds())
                wait_time = max(wait_time, minute_wait)
                
                if minute_wait > 0:
                    self.global_stats['delayed_requests'] += 1
                    self.logger.warning(f"⚠️ Global minute limit approaching ({requests_last_minute}/{self.global_safety_limit_per_minute}). Waiting {minute_wait:.2f}s")
            
            # Apply wait if needed
            if wait_time > 0:
                time.sleep(wait_time)
            
            return wait_time
    
    def get_global_stats(self) -> Dict[str, Any]:
        """Get global statistics"""
        with self.lock:
            return {
                'total_requests': self.global_stats['total_requests'],
                'blocked_requests': self.global_stats['blocked_requests'],
                'delayed_requests': self.global_stats['delayed_requests'],
                'active_sources': len(self.global_stats['sources']),
                'requests_last_second': len(self.global_second_history),
                'requests_last_minute': len(self.global_request_history),
                'global_safety_limit_per_second': self.global_safety_limit_per_second,
                'global_safety_limit_per_minute': self.global_safety_limit_per_minute,
                'binance_requests_per_second': self.binance_requests_per_second,
                'binance_requests_per_minute': self.binance_requests_per_minute,
                'available_requests_per_second': self.global_safety_limit_per_second - len(self.global_second_history),
                'available_requests_per_minute': self.global_safety_limit_per_minute - len(self.global_request_history),
                'second_usage_percent': (len(self.global_second_history) / self.global_safety_limit_per_second) * 100,
                'minute_usage_percent': (len(self.global_request_history) / self.global_safety_limit_per_minute) * 100
            }
    
    def get_source_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics by source"""
        with self.lock:
            source_stats = {}
            for source, stats in self.source_stats.items():
                # Calculate requests in last second for this source
                now = datetime.now()
                cutoff_time = now - timedelta(seconds=1)
                recent_requests = sum(1 for req_time in self.source_requests[source] 
                                    if req_time > cutoff_time)
                
                source_stats[source] = {
                    'total_requests': stats['total_requests'],
                    'requests_last_second': recent_requests,
                    'last_request': stats['last_request'],
                    'last_request_ago': (now - stats['last_request']).total_seconds() if stats['last_request'] else None
                }
            
            return source_stats
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        global_stats = self.get_global_stats()
        source_stats = self.get_source_stats()
        
        # Determine status based on usage
        second_usage = global_stats['second_usage_percent']
        minute_usage = global_stats['minute_usage_percent']
        
        if second_usage >= 95 or minute_usage >= 95:
            status = 'CRITICAL'
        elif second_usage >= 80 or minute_usage >= 80:
            status = 'WARNING'
        else:
            status = 'SAFE'
        
        return {
            'global': global_stats,
            'sources': source_stats,
            'status': status,
            'timestamp': datetime.now().isoformat()
        }

# Global monitor instance
global_api_monitor = GlobalAPIMonitor()

def monitor_api_call(source: str = 'unknown'):
    """
    Decorator to monitor API calls
    Usage: @monitor_api_call('data_collection')
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Wait if global limit is approaching
            global_api_monitor.wait_if_global_limit_approaching(source)
            
            # Register the API call
            if global_api_monitor.register_api_call(source, func.__name__):
                return func(*args, **kwargs)
            else:
                raise Exception(f"API call blocked by global rate limiter: {source}")
        
        return wrapper
    return decorator 