#!/usr/bin/env python3
"""
Simple Test for Rate Limiting Integration
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.binance_rate_limiter import binance_limiter
from modules.historical_kline_fetcher import kline_fetcher
from modules.global_api_monitor import global_api_monitor
from modules.training_api_monitor import training_monitor
from datetime import datetime, timedelta

def test_fixed_integration():
    """Test the fixed integration"""
    print("🧪 Testing Fixed Rate Limiting Integration")
    print("=" * 50)
    
    # Test 1: Rate limiting modules
    print("Test 1: Rate Limiting Modules")
    try:
        stats = binance_limiter.get_stats()
        print(f"✅ Binance limiter: {stats.get('available_weight_1m', 0)} weight available")
    except Exception as e:
        print(f"❌ Binance limiter failed: {e}")
        return False
    
    # Test 2: Kline fetcher with correct parameters
    print("\nTest 2: Kline Fetcher (Fixed)")
    try:
        start_time = datetime.now() - timedelta(days=0.1)  # 2.4 hours
        klines = kline_fetcher.fetch_klines_for_symbol('ETHFDUSD', start_time=start_time)
        if klines and len(klines) > 0:
            print(f"✅ Kline fetcher: {len(klines)} klines fetched")
        else:
            print("❌ Kline fetcher: No data")
            return False
    except Exception as e:
        print(f"❌ Kline fetcher failed: {e}")
        return False
    
    # Test 3: Multi-pair strategy
    print("\nTest 3: Multi-Pair Strategy")
    try:
        test_symbols = ['ETHFDUSD', 'BTCFDUSD']
        is_valid = kline_fetcher.validate_strategy(test_symbols)
        if is_valid:
            print("✅ Multi-pair strategy validation passed")
        else:
            print("❌ Multi-pair strategy validation failed")
            return False
    except Exception as e:
        print(f"❌ Multi-pair strategy failed: {e}")
        return False
    
    print("\n🎉 All tests passed!")
    return True

if __name__ == "__main__":
    success = test_fixed_integration()
    if success:
        print("\n✅ Ready to use fixed training script!")
        print("   python ultra_train_enhanced_rate_limited_fixed.py")
    else:
        print("\n❌ Tests failed!")
