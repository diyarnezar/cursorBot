#!/usr/bin/env python3
"""
Intelligent Rate Limiter
Advanced rate limiting system to prevent API violations
"""

import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Any
from collections import deque
import threading

class IntelligentRateLimiter:
    """Advanced rate limiter with intelligent backoff"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Binance rate limits (CORRECTED)
        self.requests_per_minute = 1200  # 1,200 requests per MINUTE
        self.requests_per_second = 20    # 20 requests per SECOND
        self.burst_limit = 50            # Conservative burst limit
        
        # SAFETY LIMIT: Conservative limit to stay well under Binance limits
        self.safety_limit_per_second = 15  # Stay under 20/sec limit
        self.safety_limit_per_minute = 1000  # Stay under 1200/min limit
        
        # Request tracking
        self.request_history = deque(maxlen=1000)
        self.second_history = deque(maxlen=1000)  # Track last second
        self.last_request_time = datetime.now()
        self.current_burst = 0
        self.burst_start_time = datetime.now()
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'delayed_requests': 0,
            'rate_limited_requests': 0,
            'safety_limited_requests': 0,
            'global_blocked_requests': 0
        }
        
        self.logger.info("🚦 Intelligent Rate Limiter initialized with CORRECT Binance limits")
        self.logger.info(f"   Binance limits: {self.requests_per_minute}/min, {self.requests_per_second}/sec")
        self.logger.info(f"   Safety limits: {self.safety_limit_per_minute}/min, {self.safety_limit_per_second}/sec")
    
    def wait_if_needed(self, source: str = 'rate_limiter') -> float:
        """Wait if rate limit would be exceeded"""
        with self.lock:
            now = datetime.now()
            wait_time = 0.0
            
            # 🌍 GLOBAL API MONITOR CHECK
            try:
                from modules.global_api_monitor import global_api_monitor
                # Wait if global limit is approaching
                global_wait = global_api_monitor.wait_if_global_limit_approaching(source)
                wait_time = max(wait_time, global_wait)
                
                # Register this request with global monitor
                if not global_api_monitor.register_api_call(source, 'rate_limited_request'):
                    self.stats['global_blocked_requests'] += 1
                    self.logger.error(f"🚫 Request blocked by global API monitor: {source}")
                    raise Exception(f"Global API limit exceeded for {source}")
                    
            except ImportError:
                self.logger.warning("Global API monitor not available, using local limits only")
            
            # Clean old requests (older than 1 minute)
            cutoff_time = now - timedelta(minutes=1)
            while self.request_history and self.request_history[0] < cutoff_time:
                self.request_history.popleft()
            
            # Clean old second history (older than 1 second)
            second_cutoff = now - timedelta(seconds=1)
            while self.second_history and self.second_history[0] < second_cutoff:
                self.second_history.popleft()
            
            # 🚨 SAFETY CHECK: Never exceed 15 requests per second (under 20/sec limit)
            requests_last_second = len(self.second_history)
            if requests_last_second >= self.safety_limit_per_second:
                # Wait for next second
                oldest_second_request = self.second_history[0]
                next_second = oldest_second_request + timedelta(seconds=1)
                safety_wait = max(0.0, (next_second - now).total_seconds())
                wait_time = max(wait_time, safety_wait)
                
                if safety_wait > 0:
                    self.stats['safety_limited_requests'] += 1
                    self.logger.warning(f"🛡️ Safety limit exceeded ({requests_last_second}/{self.safety_limit_per_second}). Waiting {safety_wait:.2f}s")
            
            # Check minute limit (safety limit: 1000/min under 1200/min)
            requests_last_minute = len(self.request_history)
            if requests_last_minute >= self.safety_limit_per_minute:
                oldest_request = self.request_history[0]
                next_available = oldest_request + timedelta(minutes=1)
                minute_wait = max(0.0, (next_available - now).total_seconds())
                wait_time = max(wait_time, minute_wait)
                
                if minute_wait > 0:
                    self.stats['rate_limited_requests'] += 1
                    self.logger.warning(f"🚫 Minute limit exceeded ({requests_last_minute}/{self.safety_limit_per_minute}). Waiting {minute_wait:.2f}s")
            
            # Check burst limit
            time_since_burst_start = (now - self.burst_start_time).total_seconds()
            if time_since_burst_start < 1.0:
                if self.current_burst >= self.burst_limit:
                    burst_wait = max(wait_time, 1.0 - time_since_burst_start)
                    wait_time = max(wait_time, burst_wait)
                    self.stats['delayed_requests'] += 1
            else:
                self.current_burst = 0
                self.burst_start_time = now
            
            # Check minimum interval between requests
            time_since_last = (now - self.last_request_time).total_seconds()
            min_interval = 1.0 / self.requests_per_second  # 50ms
            if time_since_last < min_interval:
                additional_wait = min_interval - time_since_last
                wait_time = max(wait_time, additional_wait)
            
            # Apply wait if needed
            if wait_time > 0:
                time.sleep(wait_time)
            
            # Record this request
            self.request_history.append(now)
            self.second_history.append(now)
            self.current_burst += 1
            self.last_request_time = datetime.now()
            self.stats['total_requests'] += 1
            
            return wait_time
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics"""
        with self.lock:
            stats = {
                'total_requests': self.stats['total_requests'],
                'delayed_requests': self.stats['delayed_requests'],
                'rate_limited_requests': self.stats['rate_limited_requests'],
                'safety_limited_requests': self.stats['safety_limited_requests'],
                'global_blocked_requests': self.stats['global_blocked_requests'],
                'requests_last_minute': len(self.request_history),
                'requests_last_second': len(self.second_history),
                'current_burst': self.current_burst,
                'available_requests_per_minute': self.safety_limit_per_minute - len(self.request_history),
                'available_requests_per_second': self.safety_limit_per_second - len(self.second_history),
                'binance_minute_limit': self.requests_per_minute,
                'binance_second_limit': self.requests_per_second
            }
            
            # Add global monitor stats if available
            try:
                from modules.global_api_monitor import global_api_monitor
                global_stats = global_api_monitor.get_global_stats()
                stats['global_monitor'] = global_stats
            except ImportError:
                stats['global_monitor'] = {'status': 'not_available'}
            
            return stats
    
    def get_current_limits(self) -> Dict[str, Any]:
        """Get current limit status"""
        with self.lock:
            return {
                'binance_minute_limit': self.requests_per_minute,
                'binance_second_limit': self.requests_per_second,
                'safety_minute_limit': self.safety_limit_per_minute,
                'safety_second_limit': self.safety_limit_per_second,
                'current_minute_usage': len(self.request_history),
                'current_second_usage': len(self.second_history),
                'minute_usage_percent': (len(self.request_history) / self.safety_limit_per_minute) * 100,
                'second_usage_percent': (len(self.second_history) / self.safety_limit_per_second) * 100
            }

# Global rate limiter instance
rate_limiter = IntelligentRateLimiter() 