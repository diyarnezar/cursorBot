"""
API configuration for Project Hyperion
"""

import json
import os
from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class APIConfig:
    """API configuration settings"""
    
    # Binance API credentials
    binance_api_key: Optional[str] = None
    binance_api_secret: Optional[str] = None
    use_testnet: bool = False
    
    # Binance API endpoints
    BINANCE_BASE_URL: str = "https://api.binance.com"
    BINANCE_TESTNET_URL: str = "https://testnet.binance.vision"
    
    # API endpoints
    ENDPOINTS: Dict[str, str] = None
    
    # Rate limiting configuration
    RATE_LIMITS: Dict[str, Any] = None
    
    # Request configuration
    REQUEST_CONFIG: Dict[str, Any] = None
    
    def __init__(self, config_path: str = "config.json"):
        """Initialize API configuration from config file"""
        self._load_config(config_path)
        self.__post_init__()
    
    def _load_config(self, config_path: str):
        """Load configuration from JSON file"""
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                # Load API keys from api_keys section
                api_keys = config.get('api_keys', {})
                self.binance_api_key = api_keys.get('binance_api_key')
                self.binance_api_secret = api_keys.get('binance_api_secret')
                
                # Load testnet setting from binance_credentials section
                binance_creds = config.get('binance_credentials', {})
                self.use_testnet = binance_creds.get('testnet', False)
                
                # Load rate limits from cache_settings if available
                cache_settings = config.get('cache_settings', {})
                rate_limiting = cache_settings.get('rate_limiting', {})
                if rate_limiting:
                    self.RATE_LIMITS = rate_limiting.get('api_specific_limits', {}).get('binance', {})
                
        except Exception as e:
            print(f"Warning: Could not load API config from {config_path}: {e}")
            # Use environment variables as fallback
            self.binance_api_key = os.getenv('BINANCE_API_KEY')
            self.binance_api_secret = os.getenv('BINANCE_API_SECRET')
            self.use_testnet = os.getenv('BINANCE_TESTNET', 'false').lower() == 'true'
    
    def __post_init__(self):
        """Initialize default configurations"""
        if self.ENDPOINTS is None:
            self.ENDPOINTS = {
                'klines': '/api/v3/klines',
                'ticker_24hr': '/api/v3/ticker/24hr',
                'ticker_price': '/api/v3/ticker/price',
                'order_book': '/api/v3/depth',
                'exchange_info': '/api/v3/exchangeInfo',
                'server_time': '/api/v3/time',
                'ping': '/api/v3/ping'
            }
        
        if self.RATE_LIMITS is None:
            self.RATE_LIMITS = {
                'weight_limit_per_minute': 1200,
                'raw_requests_per_5min': 6100,
                'orders_per_10s': 50,
                'orders_per_day': 160000,
                'sapi_ip_weight_per_minute': 12000,
                'sapi_uid_weight_per_minute': 180000,
                'safety_margin': 1.0  # 100% of limits
            }
        
        if self.REQUEST_CONFIG is None:
            self.REQUEST_CONFIG = {
                'timeout': 30,
                'max_retries': 3,
                'retry_delay': 1,
                'headers': {
                    'User-Agent': 'Project-Hyperion/1.0',
                    'Accept': 'application/json'
                }
            }
    
    def get_endpoint_url(self, endpoint: str, testnet: bool = None) -> str:
        """Get full URL for an endpoint"""
        if testnet is None:
            testnet = self.use_testnet
        base_url = self.BINANCE_TESTNET_URL if testnet else self.BINANCE_BASE_URL
        endpoint_path = self.ENDPOINTS.get(endpoint, endpoint)
        return f"{base_url}{endpoint_path}"
    
    def get_rate_limit(self, limit_type: str) -> int:
        """Get rate limit for a specific type"""
        return self.RATE_LIMITS.get(limit_type, 0)
    
    def get_safe_rate_limit(self, limit_type: str) -> int:
        """Get safe rate limit (with safety margin)"""
        base_limit = self.get_rate_limit(limit_type)
        safety_margin = self.RATE_LIMITS.get('safety_margin', 1.0)
        return int(base_limit * safety_margin)

# Global API config instance
api_config = APIConfig() 