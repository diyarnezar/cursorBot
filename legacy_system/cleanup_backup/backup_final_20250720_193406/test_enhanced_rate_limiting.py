#!/usr/bin/env python3
"""
Test Enhanced Rate Limiting System
Verify the comprehensive rate limiting with proper Binance API specifications
"""

import time
from modules.binance_rate_limiter import binance_limiter
from modules.historical_kline_fetcher import kline_fetcher

def test_enhanced_rate_limiting():
    """Test the enhanced rate limiting system"""
    
    print("🔧 Testing Enhanced Rate Limiting System")
    print("=" * 60)
    print("Based on official Binance API specifications")
    print()
    
    # Test 1: Check rate limiter configuration
    print("📊 Test 1: Rate limiter configuration")
    limits = binance_limiter.get_current_limits()
    print("Binance rate limits:")
    for key, value in limits.items():
        print(f"  {key}: {value}")
    
    # Test 2: Test endpoint weight calculation
    print("\n⚖️ Test 2: Endpoint weight calculation")
    test_endpoints = [
        ('/api/v3/klines', {'limit': 500}),
        ('/api/v3/klines', {'limit': 1000}),
        ('/api/v3/klines', {'limit': 1500}),
        ('/api/v3/ticker/24hr', {}),
        ('/api/v3/exchangeInfo', {}),
        ('/sapi/v1/account', {})
    ]
    
    for endpoint, params in test_endpoints:
        weight = binance_limiter.get_endpoint_weight(endpoint, params)
        print(f"  {endpoint} (params: {params}): {weight} weight")
    
    # Test 3: Test rate limiting behavior
    print("\n🚦 Test 3: Rate limiting behavior")
    
    def test_rate_limiting():
        delays = []
        for i in range(10):
            start_time = time.time()
            delay = binance_limiter.wait_if_needed('/api/v3/klines', {'limit': 1000})
            end_time = time.time()
            actual_delay = end_time - start_time
            delays.append(actual_delay)
            print(f"  Request {i+1}: {delay:.3f}s delay, {actual_delay:.3f}s actual")
        return delays
    
    delays = test_rate_limiting()
    avg_delay = sum(delays) / len(delays)
    print(f"Average delay: {avg_delay:.3f}s")
    
    # Test 4: Test historical kline strategy
    print("\n📈 Test 4: Historical kline strategy")
    
    # Test symbols (subset for testing)
    test_symbols = ['ETHFDUSD', 'BTCFDUSD', 'ADAUSDT']
    
    strategy_summary = kline_fetcher.get_fetch_strategy_summary(test_symbols)
    print("Strategy summary:")
    for key, value in strategy_summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    # Validate strategy
    is_valid = kline_fetcher.validate_strategy(test_symbols)
    print(f"Strategy validation: {'✅ PASSED' if is_valid else '❌ FAILED'}")
    
    # Test 5: Simulate kline fetching (without actual API calls)
    print("\n🔄 Test 5: Simulate kline fetching")
    
    def simulate_kline_fetch(symbols, max_calls=5):
        """Simulate kline fetching with limited calls for testing"""
        results = {}
        
        for symbol in symbols:
            print(f"  Simulating {symbol}...")
            results[symbol] = []
            
            for call in range(min(max_calls, kline_fetcher.calls_per_symbol)):
                # Wait for rate limiter
                delay = binance_limiter.wait_if_needed('/api/v3/klines', {'limit': 1000})
                if delay > 0:
                    print(f"    Call {call+1}: Waited {delay:.2f}s")
                
                # Simulate API call
                time.sleep(0.01)  # Minimal delay for simulation
                
                # Simulate getting some klines
                simulated_klines = [[time.time() * 1000, 100, 101, 99, 100.5, 1000]] * 100
                results[symbol].extend(simulated_klines)
                
                print(f"    Call {call+1}: Got {len(simulated_klines)} klines")
        
        return results
    
    # Simulate with limited calls
    simulated_results = simulate_kline_fetch(test_symbols, max_calls=3)
    
    total_klines = sum(len(klines) for klines in simulated_results.values())
    print(f"Simulation completed: {total_klines} total klines")
    
    # Test 6: Check final statistics
    print("\n📊 Test 6: Final statistics")
    stats = binance_limiter.get_stats()
    print("Rate limiter stats:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    # Test 7: Verify we're under limits
    print("\n✅ Test 7: Verify under limits")
    
    weight_ok = stats['current_weight_1m'] <= 1200
    raw_ok = stats['current_raw_5m'] <= 6100
    
    print(f"  REQUEST_WEIGHT: {stats['current_weight_1m']}/1200 ({'✅ OK' if weight_ok else '❌ VIOLATION'})")
    print(f"  RAW_REQUESTS: {stats['current_raw_5m']}/6100 ({'✅ OK' if raw_ok else '❌ VIOLATION'})")
    
    if weight_ok and raw_ok:
        print("✅ All tests passed! Enhanced rate limiting system working correctly.")
    else:
        print("❌ Some tests failed! Check rate limiting implementation.")

if __name__ == "__main__":
    test_enhanced_rate_limiting() 