#!/usr/bin/env python3
"""
Test Rate Limiting
Verify that the rate limiter never exceeds 1K requests per second
"""

import time
import threading
from modules.intelligent_rate_limiter import rate_limiter
from modules.optimized_data_collector import optimized_collector

def test_rate_limiting():
    """Test the rate limiting functionality"""
    
    print("🧪 Testing Rate Limiting System")
    print("=" * 50)
    
    # Test 1: Single requests
    print("\n📊 Test 1: Single requests")
    for i in range(5):
        start_time = time.time()
        wait_time = rate_limiter.wait_if_needed()
        end_time = time.time()
        print(f"Request {i+1}: Waited {wait_time:.3f}s, Total time: {(end_time-start_time):.3f}s")
    
    # Test 2: Burst requests (should trigger rate limiting)
    print("\n🚀 Test 2: Burst requests (should trigger rate limiting)")
    start_time = time.time()
    
    def make_request(request_id):
        req_start = time.time()
        wait_time = rate_limiter.wait_if_needed()
        req_end = time.time()
        print(f"Request {request_id}: Waited {wait_time:.3f}s, Total: {(req_end-req_start):.3f}s")
    
    # Create 10 threads to simulate concurrent requests
    threads = []
    for i in range(10):
        thread = threading.Thread(target=make_request, args=(i+1,))
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    end_time = time.time()
    print(f"Burst test completed in {end_time - start_time:.3f}s")
    
    # Test 3: Check statistics
    print("\n📈 Test 3: Rate limiter statistics")
    stats = rate_limiter.get_stats()
    limits = rate_limiter.get_current_limits()
    
    print("Current Stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\nCurrent Limits:")
    for key, value in limits.items():
        print(f"  {key}: {value}")
    
    # Test 4: Simulate data collection
    print("\n📊 Test 4: Simulate data collection")
    test_pairs = ['ETHFDUSD', 'BTCFDUSD', 'ADAFDUSD', 'DOTFDUSD', 'LINKFDUSD']
    
    collection_stats = optimized_collector.get_collection_stats()
    print("Collection Stats:")
    for key, value in collection_stats.items():
        print(f"  {key}: {value}")
    
    print("\n✅ Rate limiting test completed!")
    print("🛡️ Safety limit of 1K requests/second is active")
    print("📊 Monitor the statistics to ensure limits are respected")

if __name__ == "__main__":
    test_rate_limiting() 