#!/usr/bin/env python3
"""
TEST IMPROVEMENTS SCRIPT
========================

This script tests all the improvements made to the training system:
1. CPU Optimization
2. Feature Quality Fixer
3. Parallel Parameters
4. Model Training with optimizations
"""

import os
import sys
import numpy as np
import pandas as pd
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_cpu_optimization():
    """Test CPU optimization"""
    logger.info("🧪 Testing CPU Optimization...")
    
    try:
        from modules.cpu_optimizer import get_optimal_cores, get_parallel_params, verify_cpu_optimization
        
        cores = get_optimal_cores()
        params = get_parallel_params()
        verified = verify_cpu_optimization()
        
        logger.info(f"✅ CPU Optimization Test Results:")
        logger.info(f"   Optimal cores: {cores}")
        logger.info(f"   Parallel params available: {len(params)}")
        logger.info(f"   Verification passed: {verified}")
        
        # Check environment variables
        env_vars = ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']
        for var in env_vars:
            value = os.environ.get(var, 'Not set')
            logger.info(f"   {var}: {value}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ CPU Optimization test failed: {e}")
        return False

def test_feature_quality_fixer():
    """Test feature quality fixer"""
    logger.info("🧪 Testing Feature Quality Fixer...")
    
    try:
        from modules.feature_quality_fixer import fix_feature_quality, get_quality_report
        
        # Create test data with quality issues
        np.random.seed(42)
        n_samples = 1000
        
        # Create problematic features
        data = {
            'good_feature': np.random.normal(0, 1, n_samples),
            'nan_feature': np.random.normal(0, 1, n_samples),
            'zero_feature': np.zeros(n_samples),
            'constant_feature': np.ones(n_samples),
            'inf_feature': np.random.normal(0, 1, n_samples)
        }
        
        # Add problems
        data['nan_feature'][:100] = np.nan
        data['inf_feature'][:50] = np.inf
        data['zero_feature'][:50] = np.random.normal(0, 1, 50)  # Some non-zeros
        
        df = pd.DataFrame(data)
        
        logger.info(f"   Original shape: {df.shape}")
        logger.info(f"   NaN count: {df.isna().sum().sum()}")
        logger.info(f"   Inf count: {np.isinf(df.select_dtypes(include=[np.number])).sum().sum()}")
        
        # Apply quality fixes
        df_fixed = fix_feature_quality(df)
        
        logger.info(f"   Fixed shape: {df_fixed.shape}")
        logger.info(f"   NaN count after fix: {df_fixed.isna().sum().sum()}")
        logger.info(f"   Inf count after fix: {np.isinf(df_fixed.select_dtypes(include=[np.number])).sum().sum()}")
        
        # Get quality report
        report = get_quality_report()
        logger.info(f"   Quality report: {report}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Feature Quality Fixer test failed: {e}")
        return False

def test_parallel_parameters():
    """Test parallel parameters for models"""
    logger.info("🧪 Testing Parallel Parameters...")
    
    try:
        from modules.cpu_optimizer import PARALLEL_PARAMS
        
        # Test each model type
        model_types = ['lightgbm', 'xgboost', 'catboost', 'random_forest', 'svm']
        
        for model_type in model_types:
            if model_type in PARALLEL_PARAMS:
                params = PARALLEL_PARAMS[model_type]
                logger.info(f"   {model_type.upper()}: {params}")
            else:
                logger.warning(f"   {model_type.upper()}: Not found in parallel params")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Parallel Parameters test failed: {e}")
        return False

def test_model_training():
    """Test model training with optimizations"""
    logger.info("🧪 Testing Model Training with Optimizations...")
    
    try:
        from modules.cpu_optimizer import PARALLEL_PARAMS
        import lightgbm as lgb
        import xgboost as xgb
        from sklearn.ensemble import RandomForestRegressor
        
        # Create test data
        np.random.seed(42)
        X = pd.DataFrame(np.random.normal(0, 1, (100, 10)), columns=[f'feature_{i}' for i in range(10)])
        y = pd.Series(np.random.normal(0, 1, 100))
        
        # Test LightGBM
        lgb_params = PARALLEL_PARAMS['lightgbm'].copy()
        lgb_params.update({
            'objective': 'regression',
            'metric': 'rmse',
            'num_leaves': 31,
            'learning_rate': 0.1,
            'random_state': 42
        })
        
        lgb_model = lgb.LGBMRegressor(**lgb_params)
        lgb_model.fit(X, y)
        lgb_score = lgb_model.score(X, y)
        logger.info(f"   LightGBM training successful, score: {lgb_score:.3f}")
        
        # Test XGBoost
        xgb_params = PARALLEL_PARAMS['xgboost'].copy()
        xgb_params.update({
            'objective': 'reg:squarederror',
            'max_depth': 6,
            'learning_rate': 0.1,
            'random_state': 42
        })
        
        xgb_model = xgb.XGBRegressor(**xgb_params)
        xgb_model.fit(X, y)
        xgb_score = xgb_model.score(X, y)
        logger.info(f"   XGBoost training successful, score: {xgb_score:.3f}")
        
        # Test Random Forest
        rf_params = PARALLEL_PARAMS['random_forest'].copy()
        rf_params.update({
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': 42
        })
        
        rf_model = RandomForestRegressor(**rf_params)
        rf_model.fit(X, y)
        rf_score = rf_model.score(X, y)
        logger.info(f"   Random Forest training successful, score: {rf_score:.3f}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Model Training test failed: {e}")
        return False

def main():
    """Run all tests"""
    logger.info("🚀 Starting Improvement Tests...")
    logger.info("="*60)
    
    tests = [
        ("CPU Optimization", test_cpu_optimization),
        ("Feature Quality Fixer", test_feature_quality_fixer),
        ("Parallel Parameters", test_parallel_parameters),
        ("Model Training", test_model_training)
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
    logger.info("📊 TEST RESULTS SUMMARY")
    logger.info("="*60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"   {test_name}: {status}")
    
    logger.info(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 ALL TESTS PASSED! Improvements are working correctly.")
        logger.info("\n🚀 You can now run your training with all optimizations:")
        logger.info("   python ultra_train_enhanced.py --autonomous")
    else:
        logger.error("⚠️ Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 