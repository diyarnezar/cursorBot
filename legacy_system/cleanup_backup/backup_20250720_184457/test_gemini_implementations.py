#!/usr/bin/env python3
"""
TEST GEMINI IMPLEMENTATIONS
===========================

This script tests all the implementations based on Gemini's blueprint:
1. Data Leakage Detector
2. Historical Data Pipeline
3. Baseline Performance Validation
"""

import os
import sys
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_data_leakage_detector():
    """Test the data leakage detector"""
    logger.info("🧪 Testing Data Leakage Detector...")
    
    try:
        from modules.data_leakage_detector import audit_features, validate_baseline
        
        # Create test data with some suspicious features
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=1000, freq='1min')
        
        test_data = {
            'timestamp': dates,
            'open': np.random.normal(100, 1, 1000),
            'high': np.random.normal(101, 1, 1000),
            'low': np.random.normal(99, 1, 1000),
            'close': np.random.normal(100, 1, 1000),
            'volume': np.random.normal(1000, 100, 1000),
            'target': np.random.normal(0, 1, 1000),
            'future_price': np.random.normal(100, 1, 1000),  # Suspicious
            'next_target': np.random.normal(0, 1, 1000),  # Suspicious
            'target_contaminated': np.random.normal(0, 1, 1000)  # Will be contaminated
        }
        
        df = pd.DataFrame(test_data)
        
        # Contaminate one feature with target
        df['target_contaminated'] = df['target'] + np.random.normal(0, 0.1, 1000)
        
        # Run audit
        audit_results = audit_features(df, 'target')
        baseline_results = validate_baseline(df, 'target')
        
        logger.info(f"✅ Data Leakage Detector Test Results:")
        logger.info(f"   Leakage detected: {audit_results['leakage_detected']}")
        logger.info(f"   Suspicious features: {len(audit_results['suspicious_features'])}")
        logger.info(f"   High correlation features: {len(audit_results['high_correlation_features'])}")
        logger.info(f"   Baseline validation: {baseline_results['valid']}")
        logger.info(f"   Baseline R²: {baseline_results['avg_r2']:.3f}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Data Leakage Detector test failed: {e}")
        return False

def test_historical_data_pipeline():
    """Test the historical data pipeline"""
    logger.info("🧪 Testing Historical Data Pipeline...")
    
    try:
        from modules.historical_data_pipeline import HistoricalDataPipeline, get_data_summary
        
        # Initialize pipeline
        pipeline = HistoricalDataPipeline()
        
        # Collect some test data
        pipeline.collect_all_data()
        
        # Get summary
        summary = pipeline.get_data_summary()
        
        logger.info(f"✅ Historical Data Pipeline Test Results:")
        for table, info in summary.items():
            logger.info(f"   {table}: {info['total_records']} records")
        
        # Test data retrieval
        start_time = datetime.now() - timedelta(days=1)
        end_time = datetime.now()
        
        sentiment_data = pipeline.get_historical_data('sentiment', 'ETH', start_time, end_time)
        logger.info(f"   Retrieved {len(sentiment_data)} sentiment records")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Historical Data Pipeline test failed: {e}")
        return False

def test_baseline_performance():
    """Test baseline performance validation"""
    logger.info("🧪 Testing Baseline Performance Validation...")
    
    try:
        from modules.data_leakage_detector import validate_baseline
        
        # Create realistic financial data
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=1000, freq='1min')
        
        # Simulate realistic price data
        returns = np.random.normal(0, 0.001, 1000)  # Small returns
        prices = 100 * np.exp(np.cumsum(returns))
        
        test_data = {
            'timestamp': dates,
            'open': prices + np.random.normal(0, 0.1, 1000),
            'high': prices + np.random.normal(0.5, 0.1, 1000),
            'low': prices - np.random.normal(0.5, 0.1, 1000),
            'close': prices + np.random.normal(0, 0.1, 1000),
            'volume': np.random.normal(1000, 100, 1000),
            'target': np.random.normal(0, 0.001, 1000)  # Realistic target
        }
        
        df = pd.DataFrame(test_data)
        
        # Run baseline validation
        baseline_results = validate_baseline(df, 'target')
        
        logger.info(f"✅ Baseline Performance Test Results:")
        logger.info(f"   Valid: {baseline_results['valid']}")
        logger.info(f"   Average R²: {baseline_results['avg_r2']:.3f}")
        logger.info(f"   Average MSE: {baseline_results['avg_mse']:.3f}")
        logger.info(f"   R² Realistic: {baseline_results['r2_realistic']}")
        logger.info(f"   MSE Realistic: {baseline_results['mse_realistic']}")
        logger.info(f"   Recommendation: {baseline_results['recommendation']}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Baseline Performance test failed: {e}")
        return False

def test_integration():
    """Test integration of all components"""
    logger.info("🧪 Testing Integration...")
    
    try:
        # Test that all modules can be imported together
        from modules.data_leakage_detector import audit_features, validate_baseline
        from modules.historical_data_pipeline import HistoricalDataPipeline
        from modules.cpu_optimizer import get_optimal_cores
        from modules.feature_quality_fixer import fix_feature_quality
        
        logger.info("✅ All modules imported successfully")
        
        # Test CPU optimization
        cores = get_optimal_cores()
        logger.info(f"   CPU optimization: {cores} cores")
        
        # Test feature quality fixer
        test_df = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(0, 1, 100),
            'nan_feature': [np.nan] * 50 + list(np.random.normal(0, 1, 50))
        })
        
        fixed_df = fix_feature_quality(test_df)
        logger.info(f"   Feature quality fixer: {test_df.shape} -> {fixed_df.shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Integration test failed: {e}")
        return False

def main():
    """Run all tests"""
    logger.info("🚀 Starting Gemini Implementation Tests...")
    logger.info("="*60)
    
    tests = [
        ("Data Leakage Detector", test_data_leakage_detector),
        ("Historical Data Pipeline", test_historical_data_pipeline),
        ("Baseline Performance", test_baseline_performance),
        ("Integration", test_integration)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n🧪 Running {test_name} test...")
        try:
            success = test_func()
            results[test_name] = success
            if success:
                logger.info(f"✅ {test_name} test PASSED")
            else:
                logger.error(f"❌ {test_name} test FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name} test ERROR: {e}")
            results[test_name] = False
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("📊 GEMINI IMPLEMENTATION TEST RESULTS")
    logger.info("="*60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"   {test_name}: {status}")
    
    logger.info(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 ALL GEMINI IMPLEMENTATIONS WORKING!")
        logger.info("\n🚀 Ready to proceed with Phase 2:")
        logger.info("   1. Multi-Asset Data Pipeline")
        logger.info("   2. Opportunity Scanner & Ranking")
        logger.info("   3. Dynamic Capital Allocation")
        logger.info("   4. Portfolio Risk Management")
    else:
        logger.error("⚠️ Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 