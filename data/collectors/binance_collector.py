"""
Binance Data Collector for Real-time and Historical Data
Part of Project Hyperion - Ultimate Autonomous Trading Bot
"""

import logging
import time
import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
import pickle
import os
from urllib3.exceptions import NameResolutionError, ConnectTimeoutError
import socket
import dns.resolver
from dataclasses import dataclass

try:
    from binance.client import Client
except ImportError:
    Client = None

logger = logging.getLogger(__name__)


@dataclass
class BinanceConfig:
    """Configuration for Binance data collection"""
    api_key: str = ""
    api_secret: str = ""
    base_url: str = "https://api.binance.com"
    ws_url: str = "wss://stream.binance.com:9443/ws"
    # Ultra-conservative rate limits - Binance allows 1200 requests per minute, we use only 10% of that
    rate_limit_per_second: int = 2  # 2 requests per second = 120 per minute (10% of limit)
    max_retries: int = 3
    retry_delay: float = 2.0  # Increased delay
    timeout: int = 30
    # Additional safety parameters
    max_concurrent_requests: int = 1  # Only one request at a time
    safety_margin: float = 0.5  # Use only 50% of calculated safe limits
    cooldown_period: float = 5.0  # 5 seconds cooldown after any error
    # Network connectivity settings
    dns_timeout: int = 10
    connection_timeout: int = 15
    enable_dns_check: bool = True
    enable_connection_test: bool = True


class NetworkConnectivityChecker:
    """Enhanced network connectivity checker with DNS resolution and connection testing"""
    
    def __init__(self, config: BinanceConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def check_dns_resolution(self, hostname: str = "api.binance.com") -> bool:
        """Check if DNS resolution is working"""
        try:
            self.logger.info(f"🔍 Checking DNS resolution for {hostname}...")
            resolver = dns.resolver.Resolver()
            resolver.timeout = self.config.dns_timeout
            resolver.lifetime = self.config.dns_timeout
            
            answers = resolver.resolve(hostname, 'A')
            ip_addresses = [str(answer) for answer in answers]
            
            self.logger.info(f"✅ DNS resolution successful: {hostname} -> {ip_addresses}")
            return True
            
        except dns.resolver.NXDOMAIN:
            self.logger.error(f"❌ DNS resolution failed: {hostname} does not exist")
            return False
        except dns.resolver.Timeout:
            self.logger.error(f"❌ DNS resolution timeout for {hostname}")
            return False
        except dns.resolver.NoAnswer:
            self.logger.error(f"❌ DNS resolution failed: No answer for {hostname}")
            return False
        except Exception as e:
            self.logger.error(f"❌ DNS resolution error for {hostname}: {e}")
            return False
    
    def check_connection(self, hostname: str = "api.binance.com", port: int = 443) -> bool:
        """Check if connection to hostname:port is possible"""
        try:
            self.logger.info(f"🔍 Testing connection to {hostname}:{port}...")
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.config.connection_timeout)
            result = sock.connect_ex((hostname, port))
            sock.close()
            
            if result == 0:
                self.logger.info(f"✅ Connection test successful: {hostname}:{port}")
                return True
            else:
                self.logger.error(f"❌ Connection test failed: {hostname}:{port} (error code: {result})")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Connection test error for {hostname}:{port}: {e}")
            return False
    
    def check_http_connectivity(self, url: str = "https://api.binance.com/api/v3/ping") -> bool:
        """Check HTTP connectivity to Binance API"""
        try:
            self.logger.info(f"🔍 Testing HTTP connectivity to {url}...")
            response = requests.get(url, timeout=self.config.connection_timeout)
            
            if response.status_code == 200:
                self.logger.info(f"✅ HTTP connectivity test successful: {url}")
                return True
            else:
                self.logger.warning(f"⚠️ HTTP connectivity test returned status {response.status_code}: {url}")
                return False
                
        except requests.exceptions.ConnectionError as e:
            self.logger.error(f"❌ HTTP connection error: {e}")
            return False
        except requests.exceptions.Timeout as e:
            self.logger.error(f"❌ HTTP timeout error: {e}")
            return False
        except requests.exceptions.RequestException as e:
            self.logger.error(f"❌ HTTP request error: {e}")
            return False
        except Exception as e:
            self.logger.error(f"❌ HTTP connectivity test error: {e}")
            return False
    
    def comprehensive_network_check(self) -> Dict[str, bool]:
        """Perform comprehensive network connectivity check"""
        self.logger.info("🌐 Starting comprehensive network connectivity check...")
        
        results = {}
        
        # Check DNS resolution
        if self.config.enable_dns_check:
            results['dns_resolution'] = self.check_dns_resolution()
        else:
            results['dns_resolution'] = True
        
        # Check TCP connection
        if self.config.enable_connection_test:
            results['tcp_connection'] = self.check_connection()
        else:
            results['tcp_connection'] = True
        
        # Check HTTP connectivity
        results['http_connectivity'] = self.check_http_connectivity()
        
        # Overall connectivity status
        results['overall_connectivity'] = all(results.values())
        
        if results['overall_connectivity']:
            self.logger.info("✅ All network connectivity checks passed")
        else:
            self.logger.error("❌ Network connectivity check failed")
            for check, status in results.items():
                if not status:
                    self.logger.error(f"   - {check}: FAILED")
        
        return results


class BinanceDataCollector:
    """
    Advanced Binance Data Collector with enhanced network error handling
    
    Features:
    - Real-time WebSocket data streaming
    - Historical data collection
    - Multiple timeframes support
    - Rate limiting and error handling
    - Data validation and cleaning
    - Multi-pair support
    - Enhanced network connectivity checking
    """
    
    def __init__(self, config: Optional[BinanceConfig] = None):
        """Initialize Binance Data Collector with enhanced network checking"""
        self.config = config or BinanceConfig()
        self.session = requests.Session()
        self.rate_limiter = RateLimiter(
            requests_per_second=self.config.rate_limit_per_second,
            max_requests_per_minute=self.config.rate_limit_per_second * 60
        )
        self.network_checker = NetworkConnectivityChecker(self.config)
        self.last_network_check = 0
        self.network_check_interval = 300  # 5 minutes
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Initialize Binance client
        try:
            # Assuming Client is defined elsewhere or will be added
            # For now, we'll just set it to None as a placeholder
            self.client = None 
        except Exception as e:
            self.logger.warning(f"Failed to initialize Binance client: {e}")
            self.client = None
        
        self.logger.info("Binance Data Collector initialized with enhanced network checking")

    def _ensure_network_connectivity(self) -> bool:
        """Ensure network connectivity before making requests"""
        try:
            # Skip network checks if disabled
            if not self.config.enable_dns_check and not self.config.enable_connection_test:
                self.logger.info("🔧 Network connectivity checks disabled")
                return True
            
            # Perform comprehensive network check
            network_status = self.network_checker.comprehensive_network_check()
            
            # Check if any critical connectivity failed
            critical_failures = []
            if not network_status.get('dns_resolution', True):
                critical_failures.append("DNS resolution")
            if not network_status.get('tcp_connection', True): # Changed from 'connection_test' to 'tcp_connection'
                critical_failures.append("TCP connection")
            if not network_status.get('http_connectivity', True):
                critical_failures.append("HTTP connectivity")
            
            if critical_failures:
                self.logger.error(f"❌ Network connectivity issues detected: {', '.join(critical_failures)}")
                self.logger.warning("⚠️ Will attempt to use cached data or generate synthetic data")
                return False
            else:
                self.logger.info("✅ Network connectivity verified")
                return True
                
        except Exception as e:
            self.logger.error(f"❌ Error checking network connectivity: {e}")
            return False

    def get_exchange_info(self) -> Dict[str, Any]:
        """Get exchange information including trading pairs"""
        if not self._ensure_network_connectivity():
            self.logger.error("❌ Cannot get exchange info - network connectivity failed")
            return {}
            
        try:
            url = f"{self.config.base_url}/api/v3/exchangeInfo"
            response = self.session.get(url, timeout=self.config.timeout)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.logger.error(f"Failed to get exchange info: {e}")
            return {}

    def get_trading_pairs(self, quote_asset: str = "USDT") -> List[str]:
        """Get available trading pairs"""
        exchange_info = self.get_exchange_info()
        pairs = []
        
        if 'symbols' in exchange_info:
            for symbol_info in exchange_info['symbols']:
                if (symbol_info['status'] == 'TRADING' and 
                    symbol_info['quoteAsset'] == quote_asset):
                    pairs.append(symbol_info['symbol'])
        
        self.logger.info(f"Found {len(pairs)} trading pairs for {quote_asset}")
        return pairs

    def get_klines(self, symbol: str, interval: str = '1m', start_time: Optional[int] = None, 
                   end_time: Optional[int] = None, limit: int = 1000) -> pd.DataFrame:
        """Get kline/candlestick data with enhanced retry logic and NO synthetic data"""
        try:
            # Ensure network connectivity first
            if not self._ensure_network_connectivity():
                self.logger.error(f"❌ Network connectivity failed for {symbol}")
                return pd.DataFrame()
            
            # Wait for rate limiter
            self.rate_limiter.wait()
            
            # Prepare parameters
            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': min(limit, 1000)  # Binance max is 1000
            }
            
            if start_time:
                params['startTime'] = start_time
            if end_time:
                params['endTime'] = end_time
            
            # Enhanced retry logic with exponential backoff
            max_retries = 5
            base_delay = 2.0
            
            for attempt in range(max_retries):
                try:
                    self.logger.debug(f"🔍 Fetching {symbol} {interval} data (attempt {attempt + 1}/{max_retries})")
                    
                    # Make API request
                    url = f"{self.config.base_url}/api/v3/klines"
                    response = self.session.get(url, params=params, timeout=self.config.timeout)
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        if data and len(data) > 0:
                            # Process the data
                            df = self._process_klines(data, symbol)
                            self.logger.info(f"✅ Successfully fetched {len(df)} klines for {symbol}")
                            return df
                        else:
                            self.logger.warning(f"⚠️ No data returned for {symbol}")
                            return pd.DataFrame()
                    
                    elif response.status_code == 429:  # Rate limit exceeded
                        retry_after = int(response.headers.get('Retry-After', base_delay * (2 ** attempt)))
                        self.logger.warning(f"⚠️ Rate limit exceeded for {symbol}, waiting {retry_after}s")
                        self.rate_limiter.record_error()
                        time.sleep(retry_after)
                        continue
                    
                    elif response.status_code == 418:  # IP banned
                        self.logger.error(f"❌ IP banned for {symbol}, waiting 60s")
                        time.sleep(60)
                        continue
                    
                    else:
                        self.logger.error(f"❌ HTTP {response.status_code} for {symbol}: {response.text}")
                        if attempt < max_retries - 1:
                            delay = base_delay * (2 ** attempt)
                            self.logger.info(f"🔄 Retrying in {delay}s...")
                            time.sleep(delay)
                        continue
                
                except requests.exceptions.Timeout:
                    self.logger.warning(f"⏰ Timeout for {symbol} (attempt {attempt + 1})")
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        self.logger.info(f"🔄 Retrying in {delay}s...")
                        time.sleep(delay)
                    continue
                
                except requests.exceptions.ConnectionError:
                    self.logger.warning(f"🌐 Connection error for {symbol} (attempt {attempt + 1})")
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        self.logger.info(f"🔄 Retrying in {delay}s...")
                        time.sleep(delay)
                    continue
                
                except Exception as e:
                    self.logger.error(f"❌ Unexpected error for {symbol}: {e}")
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        self.logger.info(f"🔄 Retrying in {delay}s...")
                        time.sleep(delay)
                    continue
            
            # If all retries failed, log the failure but don't generate synthetic data
            self.logger.error(f"❌ All {max_retries} attempts failed for {symbol}")
            self.logger.error(f"❌ No real data available for {symbol} - returning empty DataFrame")
            return pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"❌ Error in get_klines for {symbol}: {e}")
            return pd.DataFrame()

    def get_historical_data(self, symbols: List[str], intervals: List[str],
                          start_date: str, end_date: str = None) -> Dict[str, pd.DataFrame]:
        """Get historical data for multiple symbols and intervals"""
        
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        historical_data = {}
        
        for symbol in symbols:
            for interval in intervals:
                key = f"{symbol}_{interval}"
                
                try:
                    # Convert dates to timestamps
                    start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
                    end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)
                    
                    df = self.get_klines(symbol, interval, str(start_ts), str(end_ts))
                    
                    if not df.empty:
                        historical_data[key] = df
                        
                except Exception as e:
                    self.logger.error(f"Failed to get historical data for {key}: {e}")
        
        self.historical_data.update(historical_data)
        self.logger.info(f"Collected historical data for {len(historical_data)} symbol-interval combinations")
        
        return historical_data

    def get_order_book(self, symbol: str, limit: int = 100) -> Dict[str, Any]:
        """Get order book data"""
        
        self.rate_limiter.wait()
        
        try:
            url = f"{self.config.base_url}/api/v3/depth"
            params = {
                'symbol': symbol,
                'limit': limit
            }
            
            response = self.session.get(url, params=params, timeout=self.config.timeout)
            response.raise_for_status()
            
            data = response.json()
            
            # Convert to DataFrame
            bids_df = pd.DataFrame(data['bids'], columns=['price', 'quantity'])
            asks_df = pd.DataFrame(data['asks'], columns=['price', 'quantity'])
            
            # Convert to numeric
            for df in [bids_df, asks_df]:
                df['price'] = pd.to_numeric(df['price'])
                df['quantity'] = pd.to_numeric(df['quantity'])
            
            order_book = {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'bids': bids_df,
                'asks': asks_df,
                'lastUpdateId': data['lastUpdateId']
            }
            
            self.order_book_data[symbol] = order_book
            return order_book
            
        except Exception as e:
            self.logger.error(f"Failed to get order book for {symbol}: {e}")
            return {}

    def get_ticker_24hr(self, symbol: str = None) -> Dict[str, Any]:
        """Get 24hr ticker statistics"""
        
        self.rate_limiter.wait()
        
        try:
            url = f"{self.config.base_url}/api/v3/ticker/24hr"
            params = {}
            
            if symbol:
                params['symbol'] = symbol
            
            response = self.session.get(url, params=params, timeout=self.config.timeout)
            response.raise_for_status()
            
            data = response.json()
            
            if symbol:
                return data
            else:
                return {item['symbol']: item for item in data}
                
        except Exception as e:
            self.logger.error(f"Failed to get 24hr ticker: {e}")
            return {}

    def get_recent_trades(self, symbol: str, limit: int = 1000) -> pd.DataFrame:
        """Get recent trades"""
        
        self.rate_limiter.wait()
        
        try:
            url = f"{self.config.base_url}/api/v3/trades"
            params = {
                'symbol': symbol,
                'limit': limit
            }
            
            response = self.session.get(url, params=params, timeout=self.config.timeout)
            response.raise_for_status()
            
            data = response.json()
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Convert types
            numeric_columns = ['price', 'qty', 'quoteQty']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df['time'] = pd.to_datetime(df['time'], unit='ms')
            df['symbol'] = symbol
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to get recent trades for {symbol}: {e}")
            return pd.DataFrame()

    def start_realtime_stream(self, symbols: List[str], streams: List[str],
                            callback: Optional[Callable] = None):
        """Start real-time data streaming via WebSocket"""
        
        for symbol in symbols:
            symbol_lower = symbol.lower()
            
            # Create WebSocket URL
            stream_names = [f"{symbol_lower}@{stream}" for stream in streams]
            ws_url = f"{self.config.ws_url}/{'/'.join(stream_names)}"
            
            # Initialize data buffer
            self.data_buffers[symbol] = deque(maxlen=10000)
            
            # Store callback
            if callback:
                self.callbacks[symbol] = callback
            
            # Start WebSocket connection
            self._start_websocket_connection(symbol, ws_url)
            
        self.is_running = True
        self.logger.info(f"Started real-time streaming for {len(symbols)} symbols")

    def _start_websocket_connection(self, symbol: str, ws_url: str):
        """Start WebSocket connection for a symbol"""
        
        def on_message(ws, message):
            try:
                data = json.loads(message)
                
                # Add timestamp
                data['timestamp'] = datetime.now()
                data['symbol'] = symbol
                
                # Store in buffer
                self.data_buffers[symbol].append(data)
                
                # Store in realtime data
                if symbol not in self.realtime_data:
                    self.realtime_data[symbol] = []
                self.realtime_data[symbol].append(data)
                
                # Call callback if provided
                if symbol in self.callbacks:
                    self.callbacks[symbol](data)
                    
            except Exception as e:
                self.logger.error(f"Error processing WebSocket message for {symbol}: {e}")
        
        def on_error(ws, error):
            self.logger.error(f"WebSocket error for {symbol}: {error}")
        
        def on_close(ws, close_status_code, close_msg):
            self.logger.info(f"WebSocket connection closed for {symbol}")
        
        def on_open(ws):
            self.logger.info(f"WebSocket connection opened for {symbol}")
        
        # Create WebSocket connection
        ws = websocket.WebSocketApp(
            ws_url,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
            on_open=on_open
        )
        
        # Start WebSocket in separate thread
        ws_thread = threading.Thread(target=ws.run_forever)
        ws_thread.daemon = True
        ws_thread.start()
        
        self.ws_connections[symbol] = ws
        self.ws_threads[symbol] = ws_thread

    def stop_realtime_stream(self, symbols: List[str] = None):
        """Stop real-time data streaming"""
        
        if symbols is None:
            symbols = list(self.ws_connections.keys())
        
        for symbol in symbols:
            if symbol in self.ws_connections:
                self.ws_connections[symbol].close()
                del self.ws_connections[symbol]
                del self.ws_threads[symbol]
        
        self.is_running = False
        self.logger.info(f"Stopped real-time streaming for {len(symbols)} symbols")

    def get_realtime_data(self, symbol: str, limit: int = 1000) -> List[Dict[str, Any]]:
        """Get recent real-time data for a symbol"""
        
        if symbol in self.realtime_data:
            return self.realtime_data[symbol][-limit:]
        return []

    def get_data_buffer(self, symbol: str, limit: int = 1000) -> List[Dict[str, Any]]:
        """Get data from buffer for a symbol"""
        
        if symbol in self.data_buffers:
            return list(self.data_buffers[symbol])[-limit:]
        return []

    def save_data(self, filepath: str, data_type: str = 'historical'):
        """Save collected data to file"""
        
        try:
            if data_type == 'historical':
                data_to_save = self.historical_data
            elif data_type == 'realtime':
                data_to_save = self.realtime_data
            elif data_type == 'orderbook':
                data_to_save = self.order_book_data
            else:
                raise ValueError(f"Unknown data type: {data_type}")
            
            # Save as pickle
            import pickle
            with open(filepath, 'wb') as f:
                pickle.dump(data_to_save, f)
            
            self.logger.info(f"Saved {data_type} data to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to save data: {e}")

    def load_data(self, filepath: str, data_type: str = 'historical'):
        """Load data from file"""
        
        try:
            import pickle
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            if data_type == 'historical':
                self.historical_data = data
            elif data_type == 'realtime':
                self.realtime_data = data
            elif data_type == 'orderbook':
                self.order_book_data = data
            
            self.logger.info(f"Loaded {data_type} data from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to load data: {e}")

    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary of collected data"""
        
        summary = {
            'historical_data': {
                'num_symbols': len(self.historical_data),
                'symbols': list(self.historical_data.keys())
            },
            'realtime_data': {
                'num_symbols': len(self.realtime_data),
                'symbols': list(self.realtime_data.keys())
            },
            'order_book_data': {
                'num_symbols': len(self.order_book_data),
                'symbols': list(self.order_book_data.keys())
            },
            'websocket_connections': {
                'active': len(self.ws_connections),
                'symbols': list(self.ws_connections.keys())
            }
        }
        
        return summary

    def fetch_historical_data(self, symbol: str, days: int, interval: str = '1m') -> pd.DataFrame:
        """
        Fetch large amounts of historical data with ultra-conservative rate limiting
        
        Args:
            symbol: Trading symbol
            days: Number of days to fetch
            interval: Time interval (1m, 5m, 15m, 1h, etc.)
            
        Returns:
            DataFrame with historical data
        """
        self.logger.info(f"🔄 Fetching {days} days of {interval} data for {symbol}")
        
        # Calculate total records needed
        records_per_day = self._get_records_per_day(interval)
        total_records_needed = int(days * records_per_day)
        max_records_per_request = 1000  # Binance limit
        
        # Calculate number of requests needed
        num_requests = int((total_records_needed + max_records_per_request - 1) // max_records_per_request)
        
        self.logger.info(f"📊 Need {total_records_needed} records, will make {num_requests} requests")
        
        # Ultra-conservative rate limiting: max 2 requests per second
        requests_per_second = 2
        delay_between_requests = 1.0 / requests_per_second
        
        all_data = []
        failed_requests = 0
        max_failed_requests = 3
        
        # Calculate the end time (current time)
        end_time = int(datetime.now().timestamp() * 1000)
        
        for i in range(num_requests):
            # Rate limiting: wait between requests
            if i > 0:
                self.logger.info(f"⏳ Rate limiting: waiting {delay_between_requests:.2f}s between requests")
                time.sleep(delay_between_requests)
            
            # Calculate start and end times for this chunk
            if i == 0:
                # First request: get most recent data
                chunk_end_time = end_time
                chunk_start_time = chunk_end_time - (max_records_per_request * self._get_interval_ms(interval))
            else:
                # Subsequent requests: get older data
                chunk_end_time = chunk_start_time
                chunk_start_time = chunk_end_time - (max_records_per_request * self._get_interval_ms(interval))
            
            self.logger.info(f"🔄 Request {i+1}/{num_requests} for {symbol} (time range: {datetime.fromtimestamp(chunk_start_time/1000)} to {datetime.fromtimestamp(chunk_end_time/1000)})")
            
            # Fetch this chunk with retry logic
            chunk_data = self._fetch_chunk_with_retry(
                symbol=symbol,
                interval=interval,
                start_time=chunk_start_time,
                end_time=chunk_end_time,
                limit=max_records_per_request,
                max_retries=3
            )
            
            if chunk_data is not None and not chunk_data.empty:
                all_data.append(chunk_data)
                self.logger.info(f"✅ Got {len(chunk_data)} records for chunk {i+1}")
                failed_requests = 0  # Reset failed counter on success
            else:
                failed_requests += 1
                self.logger.warning(f"⚠️ No data for chunk {i+1}, failed requests: {failed_requests}")
                
                # If too many consecutive failures, stop
                if failed_requests >= max_failed_requests:
                    self.logger.error(f"❌ Too many consecutive failures for {symbol}, stopping")
                    break
        
        if all_data:
            # Combine all chunks
            combined_data = pd.concat(all_data, ignore_index=False)
            combined_data = combined_data.sort_index()
            
            # Remove duplicates
            combined_data = combined_data[~combined_data.index.duplicated(keep='first')]
            
            self.logger.info(f"✅ Successfully fetched {len(combined_data)} total records for {symbol}")
            
            # If we got significantly less data than expected, log a warning
            if len(combined_data) < total_records_needed * 0.8:  # Less than 80% of expected
                self.logger.warning(f"⚠️ Got {len(combined_data)} records, expected {total_records_needed} for {symbol}")
            
            return combined_data
        else:
            self.logger.error(f"❌ No data fetched for {symbol}")
            # Fall back to synthetic data when network connectivity fails
            self.logger.info(f"🔄 Falling back to synthetic data for {symbol}")
            try:
                # Generate synthetic data for the requested period
                end_time = int(datetime.now().timestamp() * 1000)
                start_time = end_time - (days * 24 * 60 * 60 * 1000)  # days to milliseconds
                total_records_needed = int(days * records_per_day)
                
                synthetic_data = self._generate_synthetic_data(
                    symbol=symbol,
                    limit=total_records_needed,
                    start_time=start_time,
                    end_time=end_time
                )
                
                if not synthetic_data.empty:
                    self.logger.info(f"✅ Generated {len(synthetic_data)} synthetic data points for {symbol}")
                    return synthetic_data
                else:
                    self.logger.error(f"❌ Failed to generate synthetic data for {symbol}")
                    return pd.DataFrame()
            except Exception as e:
                self.logger.error(f"❌ Error generating synthetic data for {symbol}: {e}")
                return pd.DataFrame()
    
    def _fetch_chunk_with_retry(self, symbol: str, interval: str, start_time: int, end_time: int, limit: int, max_retries: int = 3) -> Optional[pd.DataFrame]:
        """Fetch a chunk of data with retry logic"""
        for attempt in range(max_retries):
            try:
                # Try to fetch real data
                klines = self.get_klines(
                    symbol=symbol,
                    interval=interval,
                    start_time=start_time,
                    end_time=end_time,
                    limit=limit
                )
                
                if klines is not None and len(klines) > 0:
                    return klines  # Return the processed data directly
                else:
                    self.logger.warning(f"⚠️ Empty response for {symbol} (attempt {attempt + 1})")
                    
            except Exception as e:
                self.logger.error(f"❌ Error fetching chunk for {symbol} (attempt {attempt + 1}): {e}")
                
                # If this is the last attempt, log the failure but don't generate synthetic data
                if attempt == max_retries - 1:
                    self.logger.error(f"❌ All retry attempts failed for {symbol}")
        
        return None
    
    def _generate_synthetic_data(self, symbol: str, limit: int, start_time: int, end_time: int) -> pd.DataFrame:
        """REMOVED: Synthetic data generation is not allowed - only real data"""
        self.logger.error(f"❌ Synthetic data generation attempted for {symbol} - NOT ALLOWED")
        self.logger.error(f"❌ Only real data from Binance API is permitted")
        return pd.DataFrame()
    
    def _process_klines(self, klines: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Process raw kline data into a clean DataFrame."""
        # Convert to DataFrame
        df = pd.DataFrame(klines, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        # Convert types
        numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'quote_volume']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Convert timestamps
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        
        # Set index
        df.set_index('open_time', inplace=True)
        
        # Drop unnecessary columns
        df = df[['open', 'high', 'low', 'close', 'volume']]
        
        # Add symbol column
        df['symbol'] = symbol
        
        return df
    
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
            '3d': 1,      # Approximate
            '1w': 1,      # Approximate
            '1M': 1       # Approximate
        }
        return interval_map.get(interval, 1440)  # Default to 1 minute

    def _get_interval_ms(self, interval: str) -> int:
        """Convert interval string to milliseconds"""
        interval_map = {
            '1m': 60 * 1000,
            '3m': 3 * 60 * 1000,
            '5m': 5 * 60 * 1000,
            '15m': 15 * 60 * 1000,
            '30m': 30 * 60 * 1000,
            '1h': 60 * 60 * 1000,
            '2h': 2 * 60 * 60 * 1000,
            '4h': 4 * 60 * 60 * 1000,
            '6h': 6 * 60 * 60 * 1000,
            '8h': 8 * 60 * 60 * 1000,
            '12h': 12 * 60 * 60 * 1000,
            '1d': 24 * 60 * 60 * 1000,
            '3d': 3 * 24 * 60 * 60 * 1000,
            '1w': 7 * 24 * 60 * 60 * 1000,
            '1M': 30 * 24 * 60 * 60 * 1000
        }
        return interval_map.get(interval, 60 * 1000)  # Default to 1 minute


class RateLimiter:
    """Ultra-conservative rate limiter to prevent any API limit violations"""
    
    def __init__(self, requests_per_second: int = 2, max_requests_per_minute: int = 100):
        self.calls_per_second = requests_per_second
        self.min_interval = 1.0 / requests_per_second
        self.last_call_time = 0
        self.request_count = 0
        self.minute_start = time.time()
        self.daily_start = time.time()
        self.daily_requests = 0
        
        # Ultra-conservative limits
        self.max_requests_per_minute = max_requests_per_minute  # Binance allows 1200, we use only 100
        self.max_requests_per_day = 10000   # Binance allows 100000, we use only 10000
        self.max_concurrent_requests = 1    # Only one request at a time
        
        # Safety tracking
        self.error_count = 0
        self.last_error_time = 0
        self.cooldown_until = 0
        
        logger.info(f"🛡️ Ultra-conservative rate limiter initialized: {requests_per_second} req/sec, max {self.max_requests_per_minute}/min")
    
    def wait(self):
        """Wait for rate limit with multiple safety checks"""
        current_time = time.time()
        
        # 1. Check cooldown period after errors
        if current_time < self.cooldown_until:
            sleep_time = self.cooldown_until - current_time
            logger.warning(f"🛡️ Rate limiter: Cooling down for {sleep_time:.2f}s after error")
            time.sleep(sleep_time)
            current_time = time.time()
        
        # 2. Check daily limit
        if current_time - self.daily_start > 86400:  # 24 hours
            self.daily_requests = 0
            self.daily_start = current_time
        
        if self.daily_requests >= self.max_requests_per_day:
            sleep_time = 86400 - (current_time - self.daily_start)
            logger.warning(f"🛡️ Rate limiter: Daily limit reached, sleeping for {sleep_time:.2f}s")
            time.sleep(sleep_time)
            self.daily_requests = 0
            self.daily_start = time.time()
            current_time = time.time()
        
        # 3. Check minute limit
        if current_time - self.minute_start > 60:
            self.request_count = 0
            self.minute_start = current_time
        
        if self.request_count >= self.max_requests_per_minute:
            sleep_time = 60 - (current_time - self.minute_start)
            logger.warning(f"🛡️ Rate limiter: Minute limit reached, sleeping for {sleep_time:.2f}s")
            time.sleep(sleep_time)
            self.request_count = 0
            self.minute_start = time.time()
            current_time = time.time()
        
        # 4. Check minimum interval between requests
        time_since_last = current_time - self.last_call_time
        if time_since_last < self.min_interval:
            sleep_time = self.min_interval - time_since_last
            time.sleep(sleep_time)
            current_time = time.time()
        
        # 5. Update counters
        self.last_call_time = current_time
        self.request_count += 1
        self.daily_requests += 1
        
        # 6. Log for monitoring
        if self.request_count % 10 == 0:
            logger.info(f"🛡️ Rate limiter: {self.request_count}/min, {self.daily_requests}/day requests")
    
    def record_error(self):
        """Record an error and increase cooldown period"""
        self.error_count += 1
        self.last_error_time = time.time()
        
        # Exponential backoff: 5s, 10s, 20s, 40s...
        cooldown_time = min(5 * (2 ** (self.error_count - 1)), 300)  # Max 5 minutes
        self.cooldown_until = time.time() + cooldown_time
        
        logger.warning(f"🛡️ Rate limiter: Error recorded, cooldown for {cooldown_time}s (error #{self.error_count})")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current rate limiter status"""
        current_time = time.time()
        return {
            'requests_per_minute': self.request_count,
            'requests_per_day': self.daily_requests,
            'max_per_minute': self.max_requests_per_minute,
            'max_per_day': self.max_requests_per_day,
            'error_count': self.error_count,
            'time_since_last_error': current_time - self.last_error_time if self.last_error_time > 0 else None,
            'cooldown_active': current_time < self.cooldown_until,
            'cooldown_remaining': max(0, self.cooldown_until - current_time) if self.cooldown_until > current_time else 0
        }


class AsyncBinanceCollector:
    """Asynchronous version of Binance data collector"""
    
    def __init__(self, config: BinanceConfig):
        self.config = config
        self.session = None
        self.data_buffers = {}
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get_klines_async(self, symbol: str, interval: str, 
                              start_time: Optional[str] = None,
                              end_time: Optional[str] = None,
                              limit: int = 1000) -> pd.DataFrame:
        """Get historical kline data asynchronously"""
        
        try:
            url = f"{self.config.base_url}/api/v3/klines"
            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': limit
            }
            
            if start_time:
                params['startTime'] = start_time
            if end_time:
                params['endTime'] = end_time
            
            async with self.session.get(url, params=params, timeout=self.config.timeout) as response:
                response.raise_for_status()
                data = await response.json()
            
            # Convert to DataFrame (same as synchronous version)
            df = pd.DataFrame(data, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Convert types
            numeric_columns = ['open', 'high', 'low', 'close', 'volume', 
                             'quote_asset_volume', 'taker_buy_base_asset_volume', 
                             'taker_buy_quote_asset_volume']
            
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
            df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
            df.set_index('open_time', inplace=True)
            df['symbol'] = symbol
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to get klines for {symbol}: {e}")
            return pd.DataFrame()
    
    async def get_multiple_klines_async(self, symbols: List[str], interval: str,
                                      start_time: Optional[str] = None,
                                      end_time: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """Get klines for multiple symbols concurrently"""
        
        tasks = []
        for symbol in symbols:
            task = self.get_klines_async(symbol, interval, start_time, end_time)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        data = {}
        for symbol, result in zip(symbols, results):
            if isinstance(result, pd.DataFrame) and not result.empty:
                data[symbol] = result
        
        return data 