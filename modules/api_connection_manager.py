import time
import logging
import requests
from typing import Optional, Dict, Any, Callable
from datetime import datetime, timedelta
from functools import wraps
import random

class APIConnectionManager:
    """
    Advanced API Connection Manager with intelligent retry logic and fallback strategies.
    Prevents immediate fallback to synthetic data by implementing proper retry mechanisms.
    """
    
    def __init__(self, 
                 max_retries: int = 5,
                 base_delay: float = 1.0,
                 max_delay: float = 60.0,
                 exponential_base: float = 2.0,
                 jitter: bool = True,
                 timeout: int = 30):
        """
        Initialize the API Connection Manager.
        
        Args:
            max_retries: Maximum number of retry attempts
            base_delay: Initial delay between retries in seconds
            max_delay: Maximum delay between retries in seconds
            exponential_base: Base for exponential backoff
            jitter: Whether to add random jitter to delays
            timeout: Request timeout in seconds
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.timeout = timeout
        
        # Connection status tracking
        self.connection_status = {
            'binance': {'status': 'unknown', 'last_check': datetime.now(), 'failures': 0},
            'alternative_apis': {'status': 'unknown', 'last_check': datetime.now(), 'failures': 0}
        }
        
        # Rate limiting tracking
        self.rate_limits = {
            'binance': {'last_request': datetime.now(), 'requests_per_minute': 0},
            'alternative_apis': {'last_request': datetime.now(), 'requests_per_minute': 0}
        }
        
        logging.info("🔗 API Connection Manager initialized with intelligent retry logic")
    
    def retry_with_backoff(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute a function with intelligent retry logic and exponential backoff.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result or None if all retries fail
        """
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                # Check if we need to wait due to rate limiting
                self._check_rate_limits(func.__name__)
                
                # Execute the function
                result = func(*args, **kwargs)
                
                # Update connection status on success
                self._update_connection_status(func.__name__, 'success')
                
                if attempt > 0:
                    logging.info(f"✅ API call succeeded on attempt {attempt + 1}")
                
                return result
                
            except requests.exceptions.ConnectionError as e:
                last_exception = e
                logging.warning(f"🔌 Connection error on attempt {attempt + 1}: {e}")
                self._update_connection_status(func.__name__, 'connection_error')
                
            except requests.exceptions.Timeout as e:
                last_exception = e
                logging.warning(f"⏰ Timeout error on attempt {attempt + 1}: {e}")
                self._update_connection_status(func.__name__, 'timeout')
                
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:  # Rate limit exceeded
                    last_exception = e
                    logging.warning(f"🚫 Rate limit exceeded on attempt {attempt + 1}")
                    self._update_connection_status(func.__name__, 'rate_limited')
                    # Wait longer for rate limit errors
                    wait_time = min(self.max_delay * 2, 120)
                else:
                    # For other HTTP errors, don't retry
                    logging.error(f"❌ HTTP error {e.response.status_code}: {e}")
                    raise e
                
            except Exception as e:
                last_exception = e
                logging.warning(f"⚠️ Unexpected error on attempt {attempt + 1}: {e}")
                self._update_connection_status(func.__name__, 'error')
            
            # Calculate delay for next attempt
            if attempt < self.max_retries:
                delay = self._calculate_delay(attempt)
                logging.info(f"⏳ Waiting {delay:.1f}s before retry {attempt + 2}/{self.max_retries + 1}")
                time.sleep(delay)
        
        # All retries failed
        logging.error(f"❌ All {self.max_retries + 1} attempts failed for {func.__name__}")
        self._update_connection_status(func.__name__, 'failed')
        return None
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for exponential backoff with optional jitter."""
        delay = min(self.base_delay * (self.exponential_base ** attempt), self.max_delay)
        
        if self.jitter:
            # Add random jitter (±25% of delay)
            jitter_factor = 0.75 + random.random() * 0.5
            delay *= jitter_factor
        
        return delay
    
    def _check_rate_limits(self, api_name: str):
        """Check and enforce rate limits."""
        now = datetime.now()
        rate_limit = self.rate_limits.get(api_name, {})
        
        if rate_limit.get('last_request'):
            time_since_last = (now - rate_limit['last_request']).total_seconds()
            
            # Binance rate limits: 1200 requests per minute
            if api_name == 'binance' and time_since_last < 0.05:  # 50ms between requests
                sleep_time = 0.05 - time_since_last
                logging.debug(f"Rate limiting: waiting {sleep_time:.3f}s")
                time.sleep(sleep_time)
            
            # Alternative APIs: 60 requests per minute
            elif api_name == 'alternative_apis' and time_since_last < 1.0:  # 1s between requests
                sleep_time = 1.0 - time_since_last
                logging.debug(f"Rate limiting: waiting {sleep_time:.3f}s")
                time.sleep(sleep_time)
        
        # Update rate limit tracking
        if api_name not in self.rate_limits:
            self.rate_limits[api_name] = {}
        self.rate_limits[api_name]['last_request'] = now
    
    def _update_connection_status(self, api_name: str, status: str):
        """Update connection status tracking."""
        if api_name not in self.connection_status:
            self.connection_status[api_name] = {'status': 'unknown', 'last_check': datetime.now(), 'failures': 0}
        
        self.connection_status[api_name]['status'] = status
        self.connection_status[api_name]['last_check'] = datetime.now()
        
        if status in ['connection_error', 'timeout', 'rate_limited', 'error', 'failed']:
            self.connection_status[api_name]['failures'] += 1
        else:
            self.connection_status[api_name]['failures'] = 0
    
    def wait_for_connection(self, api_type: str, max_wait_time: int = 60, endpoint: str = None) -> bool:
        """
        Wait for API connection to be available.
        
        Args:
            api_type: Type of API ('binance', 'alternative_apis')
            max_wait_time: Maximum time to wait in seconds
            endpoint: Specific endpoint to test (optional)
            
        Returns:
            True if connection is available, False otherwise
        """
        try:
            start_time = time.time()
            
            while time.time() - start_time < max_wait_time:
                if self._test_connection(api_type, endpoint):
                    self.connection_status[api_type]['status'] = 'connected'
                    self.connection_status[api_type]['last_check'] = datetime.now()
                    return True
                
                logging.info(f"⏳ {api_type} not available, waiting 5s...")
                time.sleep(5)
            
            self.connection_status[api_type]['status'] = 'failed'
            self.connection_status[api_type]['last_check'] = datetime.now()
            return False
            
        except Exception as e:
            logging.error(f"Error waiting for {api_type} connection: {e}")
            return False
    
    def _test_connection(self, api_type: str, endpoint: str = None) -> bool:
        """
        Test connection to specific API type.
        
        Args:
            api_type: Type of API to test
            endpoint: Specific endpoint to test (optional)
            
        Returns:
            True if connection successful, False otherwise
        """
        try:
            if api_type == 'binance':
                # Use specific endpoint if provided, otherwise use default
                base_url = endpoint if endpoint is not None else 'https://api.binance.com'
                response = requests.get(f"{base_url}/api/v3/ping", timeout=10)
                return response.status_code == 200
            elif api_type == 'alternative_apis':
                # Test alternative APIs
                test_urls = [
                    'https://api.coingecko.com/api/v3/ping',
                    'https://api.coinmarketcap.com/v1/ticker/',
                    'https://api.cryptocompare.com/data/price'
                ]
                
                for url in test_urls:
                    try:
                        response = requests.get(url, timeout=5)
                        if response.status_code == 200:
                            return True
                    except:
                        continue
                
                return False
            else:
                return False
                
        except Exception as e:
            logging.debug(f"Connection test failed for {api_type}: {e}")
            return False
    
    def get_connection_status(self) -> Dict[str, Any]:
        """Get current connection status for all APIs."""
        return {
            'connection_status': self.connection_status,
            'rate_limits': self.rate_limits,
            'recommendation': self._get_recommendation()
        }
    
    def _get_recommendation(self) -> str:
        """Get recommendation based on current connection status."""
        binance_status = self.connection_status.get('binance', {})
        alt_status = self.connection_status.get('alternative_apis', {})
        
        if binance_status.get('status') == 'success':
            return "Use Binance API - connection stable"
        elif binance_status.get('failures', 0) < 3:
            return "Retry Binance API - temporary issues"
        elif alt_status.get('status') == 'success':
            return "Use alternative APIs - Binance unavailable"
        else:
            return "All APIs unavailable - consider network issues"
    
    def reset_connection_status(self, api_name: Optional[str] = None):
        """Reset connection status for specified API or all APIs."""
        if api_name:
            if api_name in self.connection_status:
                self.connection_status[api_name] = {'status': 'unknown', 'last_check': datetime.now(), 'failures': 0}
        else:
            for api in self.connection_status:
                self.connection_status[api] = {'status': 'unknown', 'last_check': datetime.now(), 'failures': 0}
        
        logging.info(f"🔄 Connection status reset for {api_name or 'all APIs'}")


# Decorator for easy use
def with_retry(max_retries: int = 5, base_delay: float = 1.0):
    """
    Decorator to add retry logic to any function.
    
    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries in seconds
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            manager = APIConnectionManager(max_retries=max_retries, base_delay=base_delay)
            return manager.retry_with_backoff(func, *args, **kwargs)
        return wrapper
    return decorator 