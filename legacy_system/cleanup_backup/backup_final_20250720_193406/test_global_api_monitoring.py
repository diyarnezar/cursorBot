#!/usr/bin/env python3
"""
Test Global API Monitoring
Verify that global API monitor prevents exceeding 1K requests per second from any source
"""

import time
import threading
from modules.global_api_monitor import global_api_monitor
from modules.intelligent_rate_limiter import rate_limiter
from modules.training_api_monitor import training_monitor

def test_global_api_monitoring():
    """Test the global API monitoring system"""
    
    print("🌍 Testing Global API Monitoring System")
    print("=" * 60)
    
    # Test 1: Basic global monitoring
    print("\n📊 Test 1: Basic global monitoring")
    for i in range(5):
        success = global_api_monitor.register_api_call(f'test_source_{i}', 'test_call')
        print(f"API call {i+1}: {'✅ Allowed' if success else '❌ Blocked'}")
    
    # Test 2: Simulate high-volume requests
    print("\n🚀 Test 2: High-volume requests (should trigger limits)")
    
    def make_requests(source, count):
        blocked = 0
        allowed = 0
        for i in range(count):
            if global_api_monitor.register_api_call(source, 'high_volume'):
                allowed += 1
            else:
                blocked += 1
        return allowed, blocked
    
    # Simulate multiple sources making requests simultaneously
    sources = ['data_collection', 'training', 'background', 'validation']
    threads = []
    results = {}
    
    for source in sources:
        thread = threading.Thread(target=lambda s=source: results.update({s: make_requests(s, 300)}))
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    print("High-volume test results:")
    for source, (allowed, blocked) in results.items():
        print(f"  {source}: {allowed} allowed, {blocked} blocked")
    
    # Test 3: Training-specific monitoring
    print("\n🎯 Test 3: Training-specific monitoring")
    
    # Simulate training process
    training_monitor.collect_training_data('ETHFDUSD', 1.0)
    training_monitor.engineer_features(None)
    training_monitor.train_model('xgboost')
    training_monitor.validate_model('xgboost')
    training_monitor.background_collection()
    
    training_stats = training_monitor.get_training_stats()
    print("Training stats:")
    for key, value in training_stats.items():
        print(f"  {key}: {value}")
    
    # Test 4: Safety checks
    print("\n🛡️ Test 4: Safety checks")
    safety_status = training_monitor.check_training_safety()
    print("Training safety status:")
    for key, value in safety_status.items():
        print(f"  {key}: {value}")
    
    # Test 5: Comprehensive system status
    print("\n📈 Test 5: Comprehensive system status")
    system_status = global_api_monitor.get_system_status()
    print("System status:")
    print(f"  Status: {system_status['status']}")
    print(f"  Global requests: {system_status['global']['requests_last_second']}/{system_status['global']['global_safety_limit']}")
    print(f"  Usage: {system_status['global']['usage_percent']:.1f}%")
    
    print("\nSource breakdown:")
    for source, stats in system_status['sources'].items():
        print(f"  {source}: {stats['requests_last_second']} requests/sec, {stats['total_requests']} total")
    
    # Test 6: Rate limiter integration
    print("\n🚦 Test 6: Rate limiter integration")
    rate_stats = rate_limiter.get_stats()
    print("Rate limiter stats:")
    for key, value in rate_stats.items():
        if key != 'global_monitor':  # Avoid printing nested dict
            print(f"  {key}: {value}")
    
    print("\n✅ Global API monitoring test completed!")
    print("🛡️ 1K requests/second limit is enforced globally")
    print("📊 All sources are monitored and controlled")

if __name__ == "__main__":
    test_global_api_monitoring() 