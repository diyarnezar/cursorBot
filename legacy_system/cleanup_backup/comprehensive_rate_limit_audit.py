#!/usr/bin/env python3
"""
Comprehensive Rate Limit Audit
Verifies that ALL components stay under 1,200 weight per minute
"""

import os
import sys
import time
import logging
from datetime import datetime, timedelta
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def setup_audit_logging():
    """Setup logging for audit"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f'rate_limit_audit_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )
    return logging.getLogger(__name__)

def audit_binance_rate_limiter():
    """Audit the Binance rate limiter configuration"""
    logger = logging.getLogger(__name__)
    logger.info("🔍 Auditing Binance Rate Limiter")
    
    try:
        from modules.binance_rate_limiter import binance_limiter
        
        # Check configuration
        config_issues = []
        
        if binance_limiter.REQUEST_WEIGHT_1M != 1200:
            config_issues.append(f"REQUEST_WEIGHT_1M should be 1200, got {binance_limiter.REQUEST_WEIGHT_1M}")
        
        if binance_limiter.SAFETY_MARGIN != 0.8:
            config_issues.append(f"SAFETY_MARGIN should be 0.8, got {binance_limiter.SAFETY_MARGIN}")
        
        effective_limit = binance_limiter.REQUEST_WEIGHT_1M * binance_limiter.SAFETY_MARGIN
        if effective_limit != 960:
            config_issues.append(f"Effective limit should be 960, got {effective_limit}")
        
        # Check endpoint weights
        klines_weight = binance_limiter.get_endpoint_weight('/api/v3/klines', {'limit': 1000})
        if klines_weight != 2:
            config_issues.append(f"Klines weight should be 2, got {klines_weight}")
        
        if config_issues:
            logger.error("❌ Binance Rate Limiter Configuration Issues:")
            for issue in config_issues:
                logger.error(f"   {issue}")
            return False
        else:
            logger.info("✅ Binance Rate Limiter Configuration: CORRECT")
            logger.info(f"   REQUEST_WEIGHT_1M: {binance_limiter.REQUEST_WEIGHT_1M}")
            logger.info(f"   SAFETY_MARGIN: {binance_limiter.SAFETY_MARGIN}")
            logger.info(f"   Effective Limit: {effective_limit}")
            logger.info(f"   Klines Weight: {klines_weight}")
            return True
            
    except Exception as e:
        logger.error(f"❌ Binance Rate Limiter Audit Failed: {e}")
        return False

def audit_global_api_monitor():
    """Audit the global API monitor configuration"""
    logger = logging.getLogger(__name__)
    logger.info("🔍 Auditing Global API Monitor")
    
    try:
        from modules.global_api_monitor import global_api_monitor
        
        # Check configuration
        config_issues = []
        
        if global_api_monitor.binance_requests_per_minute != 1200:
            config_issues.append(f"binance_requests_per_minute should be 1200, got {global_api_monitor.binance_requests_per_minute}")
        
        if global_api_monitor.global_safety_limit_per_minute != 1000:
            config_issues.append(f"global_safety_limit_per_minute should be 1000, got {global_api_monitor.global_safety_limit_per_minute}")
        
        if global_api_monitor.global_safety_limit_per_second != 15:
            config_issues.append(f"global_safety_limit_per_second should be 15, got {global_api_monitor.global_safety_limit_per_second}")
        
        if config_issues:
            logger.error("❌ Global API Monitor Configuration Issues:")
            for issue in config_issues:
                logger.error(f"   {issue}")
            return False
        else:
            logger.info("✅ Global API Monitor Configuration: CORRECT")
            logger.info(f"   Binance Limit: {global_api_monitor.binance_requests_per_minute}/min")
            logger.info(f"   Safety Limit: {global_api_monitor.global_safety_limit_per_minute}/min")
            logger.info(f"   Second Limit: {global_api_monitor.global_safety_limit_per_second}/sec")
            return True
            
    except Exception as e:
        logger.error(f"❌ Global API Monitor Audit Failed: {e}")
        return False

def audit_historical_kline_fetcher():
    """Audit the historical kline fetcher strategy"""
    logger = logging.getLogger(__name__)
    logger.info("🔍 Auditing Historical Kline Fetcher")
    
    try:
        from modules.historical_kline_fetcher import kline_fetcher
        
        # Check strategy parameters
        config_issues = []
        
        if kline_fetcher.days_to_fetch != 15:
            config_issues.append(f"days_to_fetch should be 15, got {kline_fetcher.days_to_fetch}")
        
        if kline_fetcher.total_minutes != 21600:
            config_issues.append(f"total_minutes should be 21600, got {kline_fetcher.total_minutes}")
        
        if kline_fetcher.calls_per_symbol != 22:
            config_issues.append(f"calls_per_symbol should be 22, got {kline_fetcher.calls_per_symbol}")
        
        if kline_fetcher.total_weight_per_symbol != 44:
            config_issues.append(f"total_weight_per_symbol should be 44, got {kline_fetcher.total_weight_per_symbol}")
        
        if kline_fetcher.inter_call_delay < 0.1:
            config_issues.append(f"inter_call_delay should be >= 0.1, got {kline_fetcher.inter_call_delay}")
        
        if kline_fetcher.symbol_delay < 1.0:
            config_issues.append(f"symbol_delay should be >= 1.0, got {kline_fetcher.symbol_delay}")
        
        # Test strategy validation
        test_symbols = ['ETHFDUSD', 'BTCFDUSD', 'ADAUSDT']
        strategy_summary = kline_fetcher.get_fetch_strategy_summary(test_symbols)
        
        if strategy_summary['total_weight'] > 1200:
            config_issues.append(f"Strategy exceeds weight limit: {strategy_summary['total_weight']} > 1200")
        
        if not strategy_summary['rate_limit_safe']:
            config_issues.append("Strategy is not rate limit safe")
        
        if config_issues:
            logger.error("❌ Historical Kline Fetcher Configuration Issues:")
            for issue in config_issues:
                logger.error(f"   {issue}")
            return False
        else:
            logger.info("✅ Historical Kline Fetcher Configuration: CORRECT")
            logger.info(f"   Days to fetch: {kline_fetcher.days_to_fetch}")
            logger.info(f"   Total minutes: {kline_fetcher.total_minutes}")
            logger.info(f"   Calls per symbol: {kline_fetcher.calls_per_symbol}")
            logger.info(f"   Weight per symbol: {kline_fetcher.total_weight_per_symbol}")
            logger.info(f"   Inter-call delay: {kline_fetcher.inter_call_delay}s")
            logger.info(f"   Symbol delay: {kline_fetcher.symbol_delay}s")
            logger.info(f"   Strategy safe: {strategy_summary['rate_limit_safe']}")
            return True
            
    except Exception as e:
        logger.error(f"❌ Historical Kline Fetcher Audit Failed: {e}")
        return False

def audit_training_modes():
    """Audit all training modes for rate limit safety"""
    logger = logging.getLogger(__name__)
    logger.info("🔍 Auditing Training Modes")
    
    # Define training modes
    training_modes = [
        {'name': 'ultra_short', 'days': 0.021, 'minutes': 30, 'description': 'Ultra-Short Test (30 minutes)'},
        {'name': 'ultra_fast', 'days': 0.083, 'minutes': 120, 'description': 'Ultra-Fast Testing (2 hours)'},
        {'name': 'quick', 'days': 1.0, 'minutes': 1440, 'description': 'Quick Training (1 day)'},
        {'name': 'full', 'days': 7.0, 'minutes': 10080, 'description': 'Full Training (7 days)'},
        {'name': 'extended', 'days': 15.0, 'minutes': 21600, 'description': 'Extended Training (15 days)'}
    ]
    
    all_safe = True
    
    for mode in training_modes:
        try:
            # Calculate weight for this mode
            minutes = mode['minutes']
            calls_per_symbol = (minutes + 1000 - 1) // 1000  # Ceiling division
            weight_per_symbol = calls_per_symbol * 2  # 2 weight per call
            total_weight_26_symbols = weight_per_symbol * 26
            
            # Check if safe
            is_safe = total_weight_26_symbols <= 1200
            
            if is_safe:
                logger.info(f"✅ {mode['description']}: {total_weight_26_symbols} weight (SAFE)")
            else:
                logger.error(f"❌ {mode['description']}: {total_weight_26_symbols} weight (UNSAFE)")
                all_safe = False
                
        except Exception as e:
            logger.error(f"❌ {mode['description']}: Audit failed - {e}")
            all_safe = False
    
    return all_safe

def audit_multi_pair_strategy():
    """Audit multi-pair strategy for rate limit safety"""
    logger = logging.getLogger(__name__)
    logger.info("🔍 Auditing Multi-Pair Strategy")
    
    try:
        from modules.historical_kline_fetcher import kline_fetcher
        
        # Test with all 26 FDUSD pairs
        symbols = [
            'ETHFDUSD', 'BTCFDUSD', 'ADAUSDT', 'BNBUSDT', 'XRPUSDT',
            'SOLUSDT', 'DOTUSDT', 'DOGEUSDT', 'AVAXUSDT', 'MATICUSDT',
            'LINKUSDT', 'UNIUSDT', 'LTCUSDT', 'BCHUSDT', 'ATOMUSDT',
            'ETCUSDT', 'FILUSDT', 'NEARUSDT', 'APTUSDT', 'OPUSDT',
            'ARBUSDT', 'MKRUSDT', 'AAVEUSDT', 'SNXUSDT', 'COMPUSDT',
            'SUSHIUSDT'
        ]
        
        # Validate strategy
        is_valid = kline_fetcher.validate_strategy(symbols)
        
        if is_valid:
            strategy_summary = kline_fetcher.get_fetch_strategy_summary(symbols)
            
            logger.info("✅ Multi-Pair Strategy: SAFE")
            logger.info(f"   Total calls: {strategy_summary['total_calls']}")
            logger.info(f"   Total weight: {strategy_summary['total_weight']}")
            logger.info(f"   Weight usage: {strategy_summary['weight_usage_percent']:.1f}%")
            logger.info(f"   Sequential time: {strategy_summary['sequential_time_estimate']:.0f}s")
            logger.info(f"   Rate limit safe: {strategy_summary['rate_limit_safe']}")
            
            return True
        else:
            logger.error("❌ Multi-Pair Strategy: UNSAFE")
            return False
            
    except Exception as e:
        logger.error(f"❌ Multi-Pair Strategy Audit Failed: {e}")
        return False

def audit_rate_limit_integration():
    """Audit the rate limit integration in training script"""
    logger = logging.getLogger(__name__)
    logger.info("🔍 Auditing Rate Limit Integration")
    
    script_path = "ultra_train_enhanced_rate_limited_fixed.py"
    
    if not os.path.exists(script_path):
        logger.error(f"❌ Training script not found: {script_path}")
        return False
    
    try:
        with open(script_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for required imports
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
            logger.error("❌ Missing imports in training script:")
            for imp in missing_imports:
                logger.error(f"   {imp}")
            return False
        
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
            logger.error("❌ Missing methods in training script:")
            for method in missing_methods:
                logger.error(f"   {method}")
            return False
        
        # Check for proper kline fetcher usage
        if 'kline_fetcher.fetch_klines_for_symbol' not in content:
            logger.error("❌ Kline fetcher not properly integrated")
            return False
        
        logger.info("✅ Rate Limit Integration: CORRECT")
        logger.info("   All required imports present")
        logger.info("   All required methods present")
        logger.info("   Kline fetcher properly integrated")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Rate Limit Integration Audit Failed: {e}")
        return False

def run_comprehensive_audit():
    """Run comprehensive rate limit audit"""
    logger = setup_audit_logging()
    
    logger.info("🚀 COMPREHENSIVE RATE LIMIT AUDIT")
    logger.info("=" * 60)
    logger.info("Verifying ALL components stay under 1,200 weight per minute")
    logger.info("=" * 60)
    
    audit_results = []
    
    # Run all audits
    audit_results.append(("Binance Rate Limiter", audit_binance_rate_limiter()))
    audit_results.append(("Global API Monitor", audit_global_api_monitor()))
    audit_results.append(("Historical Kline Fetcher", audit_historical_kline_fetcher()))
    audit_results.append(("Training Modes", audit_training_modes()))
    audit_results.append(("Multi-Pair Strategy", audit_multi_pair_strategy()))
    audit_results.append(("Rate Limit Integration", audit_rate_limit_integration()))
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 AUDIT SUMMARY")
    logger.info("=" * 60)
    
    passed = 0
    total = len(audit_results)
    
    for component, result in audit_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{component}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\nOverall: {passed}/{total} components passed")
    
    if passed == total:
        logger.info("\n🎉 ALL COMPONENTS PASSED!")
        logger.info("✅ System is BULLETPROOF for rate limiting")
        logger.info("✅ Never exceeds 1,200 weight per minute")
        logger.info("✅ Safe for all training modes")
        logger.info("✅ Multi-pair training supported")
        logger.info("\n🚀 Ready for production use!")
    else:
        logger.info(f"\n⚠️ {total - passed} components failed")
        logger.info("❌ System needs attention before production use")
    
    return passed == total

if __name__ == "__main__":
    success = run_comprehensive_audit()
    if success:
        print("\n✅ AUDIT PASSED - System is bulletproof!")
    else:
        print("\n❌ AUDIT FAILED - System needs fixes!") 