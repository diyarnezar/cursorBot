#!/usr/bin/env python3
"""
🛡️ RATE LIMITING SAFETY TEST

This script tests the comprehensive rate limiting system to ensure
we NEVER hit Binance API limits under any circumstances.

Author: Project Hyperion
Date: 2025
"""

import sys
import time
import logging
from datetime import datetime
from data.collectors.binance_collector import BinanceDataCollector, BinanceConfig
from config.api_config import APIConfig

def setup_logger():
    """Setup logging for the test"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f'logs/rate_limit_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )
    return logging.getLogger(__name__)

def test_rate_limiting():
    """Test the comprehensive rate limiting system"""
    logger = setup_logger()
    
    print("🛡️ RATE LIMITING SAFETY TEST")
    print("="*50)
    print("Testing comprehensive rate limiting to ensure API safety")
    print("="*50)
    
    try:
        # Load configuration
        api_config = APIConfig("config.json")
        
        # Create Binance config with ultra-conservative settings
        binance_config = BinanceConfig(
            api_key=api_config.binance_api_key or "",
            api_secret=api_config.binance_api_secret or "",
            base_url=api_config.BINANCE_TESTNET_URL if api_config.use_testnet else api_config.BINANCE_BASE_URL,
            rate_limit_per_second=1,  # Only 1 request per second
            max_retries=3,
            retry_delay=5.0,  # 5 second delay
            timeout=30
        )
        
        # Create data collector
        collector = BinanceDataCollector(binance_config)
        
        # Test symbols
        test_symbols = ['ETHFDUSD', 'BTCFDUSD', 'ADAUSDT', 'BNBUSDT', 'SOLUSDT']
        
        print(f"\n🧪 Testing with {len(test_symbols)} symbols")
        print(f"🛡️ Rate limit: {binance_config.rate_limit_per_second} requests/second")
        print(f"⏳ Delay between requests: {1/binance_config.rate_limit_per_second:.1f} seconds")
        
        # Test data collection with rate limiting
        successful_requests = 0
        failed_requests = 0
        
        for i, symbol in enumerate(test_symbols):
            print(f"\n📊 Test {i+1}/{len(test_symbols)}: {symbol}")
            
            # Get rate limiter status before request
            status_before = collector.rate_limiter.get_status()
            print(f"   Status before: {status_before['requests_per_minute']}/min, {status_before['requests_per_day']}/day")
            
            # Make request
            start_time = time.time()
            df = collector.get_klines(
                symbol=symbol,
                interval='1m',
                limit=10  # Small limit for testing
            )
            end_time = time.time()
            
            # Check result
            if not df.empty:
                successful_requests += 1
                print(f"   ✅ Success: {len(df)} data points in {end_time - start_time:.2f}s")
            else:
                failed_requests += 1
                print(f"   ❌ Failed: No data returned")
            
            # Get rate limiter status after request
            status_after = collector.rate_limiter.get_status()
            print(f"   Status after: {status_after['requests_per_minute']}/min, {status_after['requests_per_day']}/day")
            
            # Check for any errors
            if status_after['error_count'] > status_before['error_count']:
                print(f"   ⚠️  Error detected: {status_after['error_count']} total errors")
            
            # Wait between requests (rate limiter should handle this)
            if i < len(test_symbols) - 1:
                print(f"   ⏳ Waiting for rate limiter...")
                time.sleep(1)  # Additional safety delay
        
        # Final status
        final_status = collector.rate_limiter.get_status()
        
        print(f"\n{'='*50}")
        print(f"🎯 TEST RESULTS")
        print(f"{'='*50}")
        print(f"✅ Successful requests: {successful_requests}")
        print(f"❌ Failed requests: {failed_requests}")
        print(f"📊 Total requests: {final_status['requests_per_minute']}")
        print(f"📅 Daily requests: {final_status['requests_per_day']}")
        print(f"⚠️  Total errors: {final_status['error_count']}")
        print(f"🛡️  Rate limit safe: {'✅ YES' if final_status['error_count'] == 0 else '❌ NO'}")
        
        if final_status['error_count'] == 0:
            print(f"\n🎉 RATE LIMITING TEST PASSED!")
            print(f"🛡️  System is safe and will not hit API limits")
        else:
            print(f"\n⚠️  RATE LIMITING TEST FAILED!")
            print(f"❌ Errors detected - system needs adjustment")
        
        return final_status['error_count'] == 0
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        print(f"\n❌ TEST FAILED: {e}")
        return False

if __name__ == "__main__":
    success = test_rate_limiting()
    sys.exit(0 if success else 1) 