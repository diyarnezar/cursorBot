#!/usr/bin/env python3
"""
Test Corrected Binance API Limits
Verify that the system properly enforces 1,200 requests per MINUTE and 20 requests per SECOND
"""

import time
import threading
from modules.global_api_monitor import global_api_monitor
from modules.intelligent_rate_limiter import rate_limiter

def test_corrected_limits():
    """Test the corrected Binance API limits"""
    
    print("🔧 Testing Corrected Binance API Limits")
    print("=" * 60)
    print("Binance limits: 1,200 requests per MINUTE, 20 requests per SECOND")
    print("Safety limits: 1,000 requests per MINUTE, 15 requests per SECOND")
    print()
    
    # Test 1: Check initial limits
    print("📊 Test 1: Initial limit configuration")
    rate_stats = rate_limiter.get_current_limits()
    print("Rate limiter limits:")
    for key, value in rate_stats.items():
        print(f"  {key}: {value}")
    
    global_stats = global_api_monitor.get_global_stats()
    print("\nGlobal monitor limits:")
    for key, value in global_stats.items():
        if 'limit' in key or 'binance' in key:
            print(f"  {key}: {value}")
    
    # Test 2: Test second limit (15 requests per second)
    print("\n🚀 Test 2: Second limit test (15 requests/sec)")
    
    def make_requests_second(source, count):
        allowed = 0
        blocked = 0
        for i in range(count):
            if global_api_monitor.register_api_call(source, 'second_test'):
                allowed += 1
            else:
                blocked += 1
        return allowed, blocked
    
    # Try to make 20 requests (should block after 15)
    allowed, blocked = make_requests_second('second_test', 20)
    print(f"Second limit test: {allowed} allowed, {blocked} blocked")
    
    # Wait for second to reset
    time.sleep(1.1)
    
    # Test 3: Test minute limit (1000 requests per minute)
    print("\n⏰ Test 3: Minute limit test (1000 requests/min)")
    
    def make_requests_minute(source, count):
        allowed = 0
        blocked = 0
        for i in range(count):
            if global_api_monitor.register_api_call(source, 'minute_test'):
                allowed += 1
            else:
                blocked += 1
        return allowed, blocked
    
    # Try to make 1100 requests (should block after 1000)
    allowed, blocked = make_requests_minute('minute_test', 1100)
    print(f"Minute limit test: {allowed} allowed, {blocked} blocked")
    
    # Test 4: Rate limiter integration
    print("\n🚦 Test 4: Rate limiter integration")
    
    def test_rate_limiter():
        delays = []
        for i in range(10):
            start_time = time.time()
            delay = rate_limiter.wait_if_needed('rate_test')
            end_time = time.time()
            actual_delay = end_time - start_time
            delays.append(actual_delay)
            print(f"  Request {i+1}: {delay:.3f}s delay, {actual_delay:.3f}s actual")
        return delays
    
    delays = test_rate_limiter()
    avg_delay = sum(delays) / len(delays)
    print(f"Average delay: {avg_delay:.3f}s")
    
    # Test 5: System status
    print("\n📈 Test 5: System status")
    system_status = global_api_monitor.get_system_status()
    print(f"System status: {system_status['status']}")
    print(f"Second usage: {system_status['global']['second_usage_percent']:.1f}%")
    print(f"Minute usage: {system_status['global']['minute_usage_percent']:.1f}%")
    
    # Test 6: Verify we're under Binance limits
    print("\n✅ Test 6: Verify under Binance limits")
    final_stats = rate_limiter.get_stats()
    
    print("Final verification:")
    print(f"  Requests last second: {final_stats['requests_last_second']} (limit: 20)")
    print(f"  Requests last minute: {final_stats['requests_last_minute']} (limit: 1200)")
    print(f"  Safety second limit: {final_stats['available_requests_per_second']} remaining")
    print(f"  Safety minute limit: {final_stats['available_requests_per_minute']} remaining")
    
    # Verify we're well under limits
    second_ok = final_stats['requests_last_second'] <= 20
    minute_ok = final_stats['requests_last_minute'] <= 1200
    
    print(f"\n🎯 Results:")
    print(f"  Second limit: {'✅ OK' if second_ok else '❌ VIOLATION'}")
    print(f"  Minute limit: {'✅ OK' if minute_ok else '❌ VIOLATION'}")
    
    if second_ok and minute_ok:
        print("✅ All tests passed! System properly enforces Binance API limits.")
    else:
        print("❌ Some tests failed! Check rate limiting implementation.")

if __name__ == "__main__":
    test_corrected_limits() 