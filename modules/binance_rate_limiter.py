#!/usr/bin/env python3
"""
Binance Rate Limiter
Comprehensive rate limiting based on official Binance API specifications
"""

import time
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from collections import deque
import requests

class BinanceRateLimiter:
    """
    Comprehensive Binance API rate limiter with proper weight tracking
    Based on official Binance API specifications
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Binance Rate Limits (Official)
        self.REQUEST_WEIGHT_1M = 1200      # 1,200 weight per minute (sliding)
        self.RAW_REQUESTS_5M = 6100        # 6,100 calls per 5 minutes (sliding)
        self.ORDERS_10S = 50               # 50 orders per 10 seconds (sliding)
        self.ORDERS_1D = 160000            # 160,000 orders per day (fixed)
        
        # SAPI Limits
        self.SAPI_IP_WEIGHT_1M = 12000     # 12,000 weight per minute (IP-based)
        self.SAPI_UID_WEIGHT_1M = 180000   # 180,000 weight per minute (UID-based)
        
        # Safety margins (stay under limits)
        self.SAFETY_MARGIN = 1.0  # Use 100% of limits (as recommended by ChatGPT)
        
        # Request tracking
        self.request_weight_history = deque(maxlen=1200)  # Track last 1200 requests
        self.raw_requests_history = deque(maxlen=6100)    # Track last 6100 requests
        self.orders_10s_history = deque(maxlen=50)        # Track last 50 orders
        self.orders_1d_history = deque(maxlen=160000)     # Track last 160k orders
        
        # SAPI tracking
        self.sapi_ip_weight_history = deque(maxlen=12000)
        self.sapi_uid_weight_history = deque(maxlen=180000)
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'weight_limited_requests': 0,
            'raw_limited_requests': 0,
            'order_limited_requests': 0,
            'sapi_limited_requests': 0,
            'retry_after_delays': 0,
            'ip_bans': 0,
            'waf_violations': 0
        }
        
        # Endpoint weights (common endpoints)
        self.endpoint_weights = {
            '/api/v3/klines': 2,           # Historical klines
            '/api/v3/ticker/24hr': 1,      # 24hr ticker
            '/api/v3/ticker/price': 1,     # Price ticker
            '/api/v3/ticker/bookTicker': 1, # Book ticker
            '/api/v3/depth': 1,            # Order book
            '/api/v3/trades': 1,           # Recent trades
            '/api/v3/aggTrades': 1,        # Aggregated trades
            '/api/v3/exchangeInfo': 10,    # Exchange info
            '/api/v3/time': 1,             # Server time
            '/api/v3/ping': 1,             # Ping
            '/sapi/v1/account': 10,        # Account info
            '/sapi/v1/order': 1,           # Place order
            '/sapi/v1/openOrders': 1,      # Open orders
            '/sapi/v1/allOrders': 5,       # All orders
            '/sapi/v1/myTrades': 5,        # My trades
        }
        
        self.logger.info("🔧 Binance Rate Limiter initialized with official specifications")
        self.logger.info(f"   REQUEST_WEIGHT: {self.REQUEST_WEIGHT_1M}/min")
        self.logger.info(f"   RAW_REQUESTS: {self.RAW_REQUESTS_5M}/5min")
        self.logger.info(f"   ORDERS: {self.ORDERS_10S}/10s, {self.ORDERS_1D}/day")
        self.logger.info(f"   SAPI: {self.SAPI_IP_WEIGHT_1M}/min (IP), {self.SAPI_UID_WEIGHT_1M}/min (UID)")
    
    def get_endpoint_weight(self, endpoint: str, params: Dict = None) -> int:
        """Get weight for an endpoint"""
        base_weight = self.endpoint_weights.get(endpoint, 1)
        
        # Special cases
        if endpoint == '/api/v3/klines':
            limit = params.get('limit', 500) if params else 500
            if limit > 1000:
                return 5
            elif limit > 500:
                return 2
            else:
                return 1
        
        return base_weight
    
    def wait_if_needed(self, endpoint: str, params: Dict = None, is_sapi: bool = False) -> float:
        """Wait if any rate limit would be exceeded"""
        with self.lock:
            now = datetime.now()
            wait_time = 0.0
            
            # Get endpoint weight
            weight = self.get_endpoint_weight(endpoint, params)
            
            # Clean old history
            self._clean_old_history(now)
            
            # Check REQUEST_WEIGHT (1 minute sliding window)
            current_weight = sum(self.request_weight_history)
            if current_weight + weight > self.REQUEST_WEIGHT_1M * self.SAFETY_MARGIN:
                oldest_weight = self.request_weight_history[0] if self.request_weight_history else 0
                weight_wait = self._calculate_wait_time(now, oldest_weight, 60)
                wait_time = max(wait_time, weight_wait)
                
                if weight_wait > 0:
                    self.stats['weight_limited_requests'] += 1
                    self.logger.warning(f"🔄 REQUEST_WEIGHT limit approaching ({current_weight}/{self.REQUEST_WEIGHT_1M}). Waiting {weight_wait:.2f}s")
            
            # Check RAW_REQUESTS (5 minute sliding window)
            current_raw = len(self.raw_requests_history)
            if current_raw + 1 > self.RAW_REQUESTS_5M * self.SAFETY_MARGIN:
                oldest_raw = self.raw_requests_history[0] if self.raw_requests_history else now
                raw_wait = self._calculate_wait_time(now, oldest_raw, 300)
                wait_time = max(wait_time, raw_wait)
                
                if raw_wait > 0:
                    self.stats['raw_limited_requests'] += 1
                    self.logger.warning(f"🔄 RAW_REQUESTS limit approaching ({current_raw}/{self.RAW_REQUESTS_5M}). Waiting {raw_wait:.2f}s")
            
            # Check SAPI limits if applicable
            if is_sapi:
                # SAPI IP weight
                current_sapi_ip = sum(self.sapi_ip_weight_history)
                if current_sapi_ip + weight > self.SAPI_IP_WEIGHT_1M * self.SAFETY_MARGIN:
                    oldest_sapi_ip = self.sapi_ip_weight_history[0] if self.sapi_ip_weight_history else 0
                    sapi_ip_wait = self._calculate_wait_time(now, oldest_sapi_ip, 60)
                    wait_time = max(wait_time, sapi_ip_wait)
                    
                    if sapi_ip_wait > 0:
                        self.stats['sapi_limited_requests'] += 1
                        self.logger.warning(f"🔄 SAPI IP weight limit approaching ({current_sapi_ip}/{self.SAPI_IP_WEIGHT_1M}). Waiting {sapi_ip_wait:.2f}s")
                
                # SAPI UID weight
                current_sapi_uid = sum(self.sapi_uid_weight_history)
                if current_sapi_uid + weight > self.SAPI_UID_WEIGHT_1M * self.SAFETY_MARGIN:
                    oldest_sapi_uid = self.sapi_uid_weight_history[0] if self.sapi_uid_weight_history else 0
                    sapi_uid_wait = self._calculate_wait_time(now, oldest_sapi_uid, 60)
                    wait_time = max(wait_time, sapi_uid_wait)
                    
                    if sapi_uid_wait > 0:
                        self.stats['sapi_limited_requests'] += 1
                        self.logger.warning(f"🔄 SAPI UID weight limit approaching ({current_sapi_uid}/{self.SAPI_UID_WEIGHT_1M}). Waiting {sapi_uid_wait:.2f}s")
            
            # Apply wait if needed
            if wait_time > 0:
                time.sleep(wait_time)
            
            # Record this request
            self.request_weight_history.append(weight)
            self.raw_requests_history.append(now)
            
            if is_sapi:
                self.sapi_ip_weight_history.append(weight)
                self.sapi_uid_weight_history.append(weight)
            
            self.stats['total_requests'] += 1
            
            return wait_time
    
    def _clean_old_history(self, now: datetime):
        """Clean old history entries"""
        # Clean REQUEST_WEIGHT (1 minute)
        cutoff_1m = now - timedelta(minutes=1)
        while self.request_weight_history and len(self.request_weight_history) > 0:
            # For weight history, we store weights, not timestamps
            # So we just limit the size
            if len(self.request_weight_history) > self.REQUEST_WEIGHT_1M:
                self.request_weight_history.popleft()
            else:
                break
        
        # Clean RAW_REQUESTS (5 minutes)
        cutoff_5m = now - timedelta(minutes=5)
        while self.raw_requests_history and self.raw_requests_history[0] < cutoff_5m:
            self.raw_requests_history.popleft()
        
        # Clean ORDERS 10s
        cutoff_10s = now - timedelta(seconds=10)
        while self.orders_10s_history and self.orders_10s_history[0] < cutoff_10s:
            self.orders_10s_history.popleft()
        
        # Clean ORDERS 1d
        cutoff_1d = now - timedelta(days=1)
        while self.orders_1d_history and self.orders_1d_history[0] < cutoff_1d:
            self.orders_1d_history.popleft()
        
        # Clean SAPI (1 minute)
        while self.sapi_ip_weight_history and len(self.sapi_ip_weight_history) > self.SAPI_IP_WEIGHT_1M:
            self.sapi_ip_weight_history.popleft()
        
        while self.sapi_uid_weight_history and len(self.sapi_uid_weight_history) > self.SAPI_UID_WEIGHT_1M:
            self.sapi_uid_weight_history.popleft()
    
    def _calculate_wait_time(self, now: datetime, oldest_time, window_seconds: int) -> float:
        """Calculate wait time for sliding window"""
        if isinstance(oldest_time, (int, float)):
            # For weight-based limits, we need to estimate
            return 60.0 / self.REQUEST_WEIGHT_1M  # Conservative estimate
        else:
            # For time-based limits
            next_available = oldest_time + timedelta(seconds=window_seconds)
            return max(0.0, (next_available - now).total_seconds())
    
    def handle_response_headers(self, response: requests.Response) -> Dict[str, Any]:
        """Process response headers for rate limit monitoring"""
        headers = response.headers
        info = {}
        
        # Extract rate limit headers
        info['X-MBX-USED-WEIGHT-1M'] = int(headers.get('X-MBX-USED-WEIGHT-1M', 0))
        info['X-MBX-USED-WEIGHT-5M'] = int(headers.get('X-MBX-USED-WEIGHT-5M', 0))
        info['X-MBX-ORDER-COUNT-10S'] = int(headers.get('X-MBX-ORDER-COUNT-10S', 0))
        info['X-MBX-ORDER-COUNT-1D'] = int(headers.get('X-MBX-ORDER-COUNT-1D', 0))
        info['X-SAPI-USED-IP-WEIGHT-1M'] = int(headers.get('X-SAPI-USED-IP-WEIGHT-1M', 0))
        info['X-SAPI-USED-UID-WEIGHT-1M'] = int(headers.get('X-SAPI-USED-UID-WEIGHT-1M', 0))
        
        # Check for rate limit violations
        if response.status_code == 429:
            retry_after = int(headers.get('Retry-After', 60))
            self.stats['retry_after_delays'] += 1
            self.logger.warning(f"🚫 Rate limit exceeded (429). Retry-After: {retry_after}s")
            time.sleep(retry_after)
            info['retry_after'] = retry_after
        
        elif response.status_code == 418:
            self.stats['ip_bans'] += 1
            self.logger.error("🚫 IP banned (418). Check your rate limiting!")
            info['ip_banned'] = True
        
        elif response.status_code == 403:
            self.stats['waf_violations'] += 1
            self.logger.error("🚫 WAF limit violated (403). Check request patterns!")
            info['waf_violation'] = True
        
        return info
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics"""
        with self.lock:
            now = datetime.now()
            self._clean_old_history(now)
            
            return {
                'total_requests': self.stats['total_requests'],
                'weight_limited_requests': self.stats['weight_limited_requests'],
                'raw_limited_requests': self.stats['raw_limited_requests'],
                'order_limited_requests': self.stats['order_limited_requests'],
                'sapi_limited_requests': self.stats['sapi_limited_requests'],
                'retry_after_delays': self.stats['retry_after_delays'],
                'ip_bans': self.stats['ip_bans'],
                'waf_violations': self.stats['waf_violations'],
                'current_weight_1m': sum(self.request_weight_history),
                'current_raw_5m': len(self.raw_requests_history),
                'current_orders_10s': len(self.orders_10s_history),
                'current_orders_1d': len(self.orders_1d_history),
                'current_sapi_ip_1m': sum(self.sapi_ip_weight_history),
                'current_sapi_uid_1m': sum(self.sapi_uid_weight_history),
                'available_weight_1m': self.REQUEST_WEIGHT_1M - sum(self.request_weight_history),
                'available_raw_5m': self.RAW_REQUESTS_5M - len(self.raw_requests_history),
                'weight_usage_percent': (sum(self.request_weight_history) / self.REQUEST_WEIGHT_1M) * 100,
                'raw_usage_percent': (len(self.raw_requests_history) / self.RAW_REQUESTS_5M) * 100
            }
    
    def get_current_limits(self) -> Dict[str, Any]:
        """Get current limit status"""
        with self.lock:
            return {
                'REQUEST_WEIGHT_1M': self.REQUEST_WEIGHT_1M,
                'RAW_REQUESTS_5M': self.RAW_REQUESTS_5M,
                'ORDERS_10S': self.ORDERS_10S,
                'ORDERS_1D': self.ORDERS_1D,
                'SAPI_IP_WEIGHT_1M': self.SAPI_IP_WEIGHT_1M,
                'SAPI_UID_WEIGHT_1M': self.SAPI_UID_WEIGHT_1M,
                'SAFETY_MARGIN': self.SAFETY_MARGIN
            }

# Global Binance rate limiter instance
binance_limiter = BinanceRateLimiter() 