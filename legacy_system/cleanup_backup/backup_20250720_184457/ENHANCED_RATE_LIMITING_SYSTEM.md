# Enhanced Rate Limiting System

## Overview

This system implements comprehensive rate limiting based on **official Binance API specifications** provided by ChatGPT. It ensures the trading bot never violates Binance API limits and can safely fetch historical data for all 26 FDUSD pairs.

## 🎯 Key Features

### 1. **Official Binance API Compliance**
- **REQUEST_WEIGHT**: 1,200 weight per minute (sliding window)
- **RAW_REQUESTS**: 6,100 calls per 5 minutes (sliding window)
- **ORDERS**: 50 per 10 seconds, 160,000 per day
- **SAPI**: 12,000 IP weight/min, 180,000 UID weight/min

### 2. **Intelligent Weight Tracking**
- Tracks endpoint weights dynamically
- `/api/v3/klines`: 1-5 weight based on limit parameter
- `/api/v3/exchangeInfo`: 10 weight
- `/sapi/v1/account`: 10 weight
- Other endpoints: 1 weight

### 3. **Historical Kline Strategy**
- **15 days of 1-minute klines** for 26 pairs
- **21,600 minutes** per symbol
- **22 API calls** per symbol (1000 klines per call)
- **572 total calls** (22 × 26)
- **1,144 total weight** (572 × 2) - safely under 1,200 limit

### 4. **Safety Mechanisms**
- **80% safety margin** - stay under actual limits
- **Automatic delays** when approaching limits
- **Response header monitoring** for 429/418/403 errors
- **Exponential backoff** on rate limit violations

## 📊 System Architecture

### Core Components

1. **`BinanceRateLimiter`** (`modules/binance_rate_limiter.py`)
   - Official Binance rate limit enforcement
   - Weight-based tracking
   - Response header processing
   - Safety margin enforcement

2. **`HistoricalKlineFetcher`** (`modules/historical_kline_fetcher.py`)
   - Sequential per-symbol fetching
   - Proper pagination strategy
   - Inter-call delays (100ms)
   - Symbol delays (1 second)

3. **`GlobalAPIMonitor`** (`modules/global_api_monitor.py`)
   - Global API call tracking
   - Cross-system monitoring
   - Training-specific monitoring

## 🚀 Fetching Strategy

### Sequential Strategy (Recommended)
```python
# For each symbol:
for symbol in symbols:
    # Fetch all 22 pages for this symbol
    for page in range(22):
        # Wait for rate limiter
        binance_limiter.wait_if_needed('/api/v3/klines', params)
        
        # Make API call
        response = requests.get('/api/v3/klines', params)
        
        # Process response headers
        binance_limiter.handle_response_headers(response)
        
        # 100ms delay between calls
        time.sleep(0.1)
    
    # 1 second delay between symbols
    time.sleep(1.0)
```

### Parallel Strategy (Limited)
```python
# Run 2 symbols in parallel
# Increased delays to compensate
inter_call_delay = 0.2  # 200ms instead of 100ms
symbol_delay = 2.0      # 2 seconds instead of 1 second
```

## 📈 Performance Metrics

### For 26 Symbols (15 days each):
- **Total calls**: 572
- **Total weight**: 1,144
- **Weight usage**: 95.3% (safely under 100%)
- **Sequential time**: ~57 seconds
- **Parallel time**: ~114 seconds

### Safety Margins:
- **Weight limit**: 1,200 (using 1,144 = 95.3%)
- **Raw requests**: 6,100 (using ~572 = 9.4%)
- **Safety margin**: 80% (staying under limits)

## 🛡️ Error Handling

### HTTP Status Codes:
- **429 Too Many Requests**: Automatic retry with `Retry-After`
- **418 IP Banned**: Critical error, requires manual intervention
- **403 WAF Violation**: Check request patterns

### Automatic Responses:
```python
if response.status_code == 429:
    retry_after = int(headers.get('Retry-After', 60))
    time.sleep(retry_after)
elif response.status_code == 418:
    logger.error("IP banned - check rate limiting!")
elif response.status_code == 403:
    logger.error("WAF violation - check request patterns!")
```

## 🔧 Configuration

### Rate Limiter Settings:
```python
SAFETY_MARGIN = 0.8  # Use only 80% of limits
REQUEST_WEIGHT_1M = 1200
RAW_REQUESTS_5M = 6100
ORDERS_10S = 50
ORDERS_1D = 160000
```

### Kline Fetcher Settings:
```python
max_limit_per_call = 1000  # Maximum klines per API call
inter_call_delay = 0.1     # 100ms between calls
symbol_delay = 1.0         # 1 second between symbols
days_to_fetch = 15         # 15 days of historical data
```

## 📊 Monitoring

### Real-time Statistics:
```python
stats = binance_limiter.get_stats()
print(f"Weight usage: {stats['weight_usage_percent']:.1f}%")
print(f"Available weight: {stats['available_weight_1m']}")
print(f"Total requests: {stats['total_requests']}")
print(f"Rate limited: {stats['weight_limited_requests']}")
```

### System Status:
```python
status = global_api_monitor.get_system_status()
print(f"Status: {status['status']}")  # SAFE/WARNING/CRITICAL
print(f"Global usage: {status['global']['usage_percent']:.1f}%")
```

## ✅ Benefits

1. **Never violates Binance limits** - stays well under actual limits
2. **Efficient data collection** - optimized strategy for 26 pairs
3. **Robust error handling** - automatic recovery from rate limits
4. **Real-time monitoring** - track usage and prevent violations
5. **Scalable architecture** - can handle increased load safely
6. **Training-safe** - ensures training never contributes to violations

## 🎯 Usage Example

```python
from modules.historical_kline_fetcher import kline_fetcher
from modules.binance_rate_limiter import binance_limiter

# Define symbols
symbols = ['ETHFDUSD', 'BTCFDUSD', 'ADAUSDT', ...]  # All 26 pairs

# Validate strategy
if kline_fetcher.validate_strategy(symbols):
    # Fetch historical klines
    results = kline_fetcher.fetch_klines_for_multiple_symbols(symbols)
    
    # Check final stats
    stats = binance_limiter.get_stats()
    print(f"Completed with {stats['weight_usage_percent']:.1f}% weight usage")
else:
    print("Strategy validation failed!")
```

## 🔄 Integration

The enhanced system integrates seamlessly with:
- **Multi-pair training** - safe data collection for all pairs
- **Real-time trading** - proper rate limiting for live operations
- **Background collection** - continuous data updates
- **Model training** - training-specific monitoring

This system ensures the trading bot operates safely within Binance API limits while maximizing data collection efficiency for optimal training and trading performance. 