#!/usr/bin/env python3
"""
Test Rate Limited Training Integration
Verifies that all training modes work correctly with enhanced rate limiting
"""

import os
import sys
import time
import logging
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import rate limiting modules
from modules.binance_rate_limiter import binance_limiter
from modules.historical_kline_fetcher import kline_fetcher
from modules.global_api_monitor import global_api_monitor
from modules.training_api_monitor import training_monitor

def setup_test_logging():
    """Setup logging for tests"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f'test_rate_limited_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )
    return logging.getLogger(__name__)

def test_rate_limiting_modules():
    """Test that all rate limiting modules are working"""
    logger = logging.getLogger(__name__)
    logger.info("🧪 Testing Rate Limiting Modules")
    
    tests_passed = 0
    total_tests = 4
    
    try:
        # Test 1: Binance Rate Limiter
        logger.info("Test 1: Binance Rate Limiter")
        stats = binance_limiter.get_stats()
        if stats:
            logger.info(f"✅ Binance limiter working: {stats.get('available_weight_1m', 0)} weight available")
            tests_passed += 1
        else:
            logger.error("❌ Binance limiter not working")
    except Exception as e:
        logger.error(f"❌ Binance limiter test failed: {e}")
    
    try:
        # Test 2: Global API Monitor
        logger.info("Test 2: Global API Monitor")
        stats = global_api_monitor.get_global_stats()
        if stats:
            logger.info(f"✅ Global monitor working: {stats.get('total_requests', 0)} total requests")
            tests_passed += 1
        else:
            logger.error("❌ Global monitor not working")
    except Exception as e:
        logger.error(f"❌ Global monitor test failed: {e}")
    
    try:
        # Test 3: Training API Monitor
        logger.info("Test 3: Training API Monitor")
        stats = training_monitor.get_training_stats()
        if stats is not None:
            logger.info(f"✅ Training monitor working: {stats.get('total_training_requests', 0)} training requests")
            tests_passed += 1
        else:
            logger.error("❌ Training monitor not working")
    except Exception as e:
        logger.error(f"❌ Training monitor test failed: {e}")
    
    try:
        # Test 4: Kline Fetcher
        logger.info("Test 4: Kline Fetcher")
        # Test with minimal data to avoid rate limits
        klines = kline_fetcher.fetch_klines_for_symbol('ETHFDUSD', days=0.1)  # 2.4 hours
        if klines and len(klines) > 0:
            logger.info(f"✅ Kline fetcher working: {len(klines)} klines fetched")
            tests_passed += 1
        else:
            logger.error("❌ Kline fetcher not working")
    except Exception as e:
        logger.error(f"❌ Kline fetcher test failed: {e}")
    
    logger.info(f"📊 Rate Limiting Modules Test: {tests_passed}/{total_tests} passed")
    return tests_passed == total_tests

def test_training_modes():
    """Test different training modes with rate limiting"""
    logger = logging.getLogger(__name__)
    logger.info("🧪 Testing Training Modes")
    
    # Define test modes
    test_modes = [
        {'name': 'ultra_short', 'days': 0.021, 'minutes': 30, 'description': 'Ultra-Short Test (30 minutes)'},
        {'name': 'ultra_fast', 'days': 0.083, 'minutes': 120, 'description': 'Ultra-Fast Testing (2 hours)'},
        {'name': 'quick', 'days': 1.0, 'minutes': 1440, 'description': 'Quick Training (1 day)'},
        {'name': 'test', 'days': 0.01, 'minutes': 15, 'description': 'Fast Test (15 minutes)'}
    ]
    
    tests_passed = 0
    total_tests = len(test_modes)
    
    for mode in test_modes:
        try:
            logger.info(f"Testing mode: {mode['description']}")
            
            # Get initial stats
            initial_stats = binance_limiter.get_stats()
            initial_weight = initial_stats.get('available_weight_1m', 0)
            
            # Test data collection
            start_time = time.time()
            klines = kline_fetcher.fetch_klines_for_symbol('ETHFDUSD', days=mode['days'])
            end_time = time.time()
            
            # Get final stats
            final_stats = binance_limiter.get_stats()
            final_weight = final_stats.get('available_weight_1m', 0)
            weight_used = initial_weight - final_weight
            
            if klines and len(klines) > 0:
                logger.info(f"✅ {mode['name']}: {len(klines)} klines in {end_time - start_time:.2f}s")
                logger.info(f"   Weight used: {weight_used}, Remaining: {final_weight}")
                tests_passed += 1
            else:
                logger.error(f"❌ {mode['name']}: No data collected")
                
        except Exception as e:
            logger.error(f"❌ {mode['name']} test failed: {e}")
    
    logger.info(f"📊 Training Modes Test: {tests_passed}/{total_tests} passed")
    return tests_passed == total_tests

def test_multi_pair_strategy():
    """Test multi-pair training strategy"""
    logger = logging.getLogger(__name__)
    logger.info("🧪 Testing Multi-Pair Strategy")
    
    try:
        # Test strategy validation
        test_symbols = ['ETHFDUSD', 'BTCFDUSD', 'ADAUSDT']
        
        is_valid = kline_fetcher.validate_strategy(test_symbols)
        if is_valid:
            logger.info("✅ Multi-pair strategy validation passed")
            
            # Test with small subset
            logger.info("Testing multi-pair data collection...")
            results = kline_fetcher.fetch_klines_for_multiple_symbols(test_symbols, days=0.1)
            
            if results and len(results) > 0:
                total_klines = sum(len(klines) for klines in results.values() if klines)
                logger.info(f"✅ Multi-pair collection: {len(results)} pairs, {total_klines} total klines")
                return True
            else:
                logger.error("❌ Multi-pair collection failed")
                return False
        else:
            logger.error("❌ Multi-pair strategy validation failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Multi-pair test failed: {e}")
        return False

def test_rate_limit_safety():
    """Test that rate limiting prevents API limit violations"""
    logger = logging.getLogger(__name__)
    logger.info("🧪 Testing Rate Limit Safety")
    
    try:
        # Get initial stats
        initial_stats = binance_limiter.get_stats()
        initial_weight = initial_stats.get('available_weight_1m', 0)
        
        logger.info(f"Initial weight available: {initial_weight}")
        
        # Test rapid requests
        logger.info("Testing rapid request handling...")
        
        for i in range(10):
            # This should be rate limited
            binance_limiter.wait_if_needed('/api/v3/klines', {'limit': 1000})
            
            # Simulate API call
            time.sleep(0.1)
        
        # Get final stats
        final_stats = binance_limiter.get_stats()
        final_weight = final_stats.get('available_weight_1m', 0)
        weight_used = initial_weight - final_weight
        
        logger.info(f"Final weight available: {final_weight}")
        logger.info(f"Weight used: {weight_used}")
        
        # Check if we're still within limits
        if final_weight > 0:
            logger.info("✅ Rate limiting working correctly")
            return True
        else:
            logger.error("❌ Rate limiting failed - exceeded limits")
            return False
            
    except Exception as e:
        logger.error(f"❌ Rate limit safety test failed: {e}")
        return False

def test_integration_with_training():
    """Test integration with the training script"""
    logger = logging.getLogger(__name__)
    logger.info("🧪 Testing Training Integration")
    
    try:
        # Check if the enhanced training script exists
        script_path = "ultra_train_enhanced_rate_limited.py"
        
        if not os.path.exists(script_path):
            logger.warning(f"⚠️ Enhanced training script not found: {script_path}")
            logger.info("   Run integrate_rate_limiting.py first to create it")
            return False
        
        logger.info(f"✅ Enhanced training script found: {script_path}")
        
        # Check if it has the required imports
        with open(script_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        required_imports = [
            'from modules.binance_rate_limiter import binance_limiter',
            'from modules.historical_kline_fetcher import kline_fetcher',
            'from modules.global_api_monitor import global_api_monitor',
            'from modules.training_api_monitor import training_monitor'
        ]
        
        missing_imports = []
        for imp in required_imports:
            if imp not in content:
                missing_imports.append(imp)
        
        if missing_imports:
            logger.error(f"❌ Missing imports in training script: {missing_imports}")
            return False
        
        logger.info("✅ All required imports found in training script")
        
        # Check for rate limiting methods
        required_methods = [
            'get_rate_limit_status',
            'log_rate_limit_status'
        ]
        
        missing_methods = []
        for method in required_methods:
            if f'def {method}' not in content:
                missing_methods.append(method)
        
        if missing_methods:
            logger.error(f"❌ Missing methods in training script: {missing_methods}")
            return False
        
        logger.info("✅ All required methods found in training script")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Training integration test failed: {e}")
        return False

def main():
    """Run all tests"""
    logger = setup_test_logging()
    
    logger.info("🚀 Starting Rate Limited Training Tests")
    logger.info("=" * 60)
    
    test_results = []
    
    # Test 1: Rate limiting modules
    logger.info("\n" + "=" * 60)
    result1 = test_rate_limiting_modules()
    test_results.append(("Rate Limiting Modules", result1))
    
    # Test 2: Training modes
    logger.info("\n" + "=" * 60)
    result2 = test_training_modes()
    test_results.append(("Training Modes", result2))
    
    # Test 3: Multi-pair strategy
    logger.info("\n" + "=" * 60)
    result3 = test_multi_pair_strategy()
    test_results.append(("Multi-Pair Strategy", result3))
    
    # Test 4: Rate limit safety
    logger.info("\n" + "=" * 60)
    result4 = test_rate_limit_safety()
    test_results.append(("Rate Limit Safety", result4))
    
    # Test 5: Training integration
    logger.info("\n" + "=" * 60)
    result5 = test_integration_with_training()
    test_results.append(("Training Integration", result5))
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 TEST SUMMARY")
    logger.info("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("\n🎉 ALL TESTS PASSED!")
        logger.info("✅ Rate limiting integration is working correctly")
        logger.info("✅ Ready to use enhanced training script")
        logger.info("\n🚀 Next steps:")
        logger.info("   python ultra_train_enhanced_rate_limited.py")
    else:
        logger.info(f"\n⚠️ {total - passed} tests failed")
        logger.info("❌ Rate limiting integration needs attention")
        logger.info("\n🔧 Troubleshooting:")
        logger.info("   1. Check if all rate limiting modules are installed")
        logger.info("   2. Run integrate_rate_limiting.py to update training script")
        logger.info("   3. Verify API keys are configured correctly")

if __name__ == "__main__":
    main() 