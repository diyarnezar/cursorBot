#!/usr/bin/env python3
"""
Test Script for Ultra Enhancements
==================================

This script tests all the critical enhancements implemented to fix
the issues identified in the training log analysis.
"""

import sys
import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_enhanced_features_import():
    """Test that enhanced features module can be imported"""
    try:
        from enhanced_features import (
            remove_correlated_features,
            select_best_feature_from_cluster,
            optimize_neural_network_hyperparameters,
            enable_external_data_sources,
            calculate_enhanced_ensemble_weights
        )
        logger.info("✅ Enhanced features module imported successfully")
        return True
    except ImportError as e:
        logger.error(f"❌ Failed to import enhanced features: {e}")
        return False

def test_correlation_removal():
    """Test correlation removal functionality"""
    try:
        from enhanced_features import remove_correlated_features
        
        # Create test data with highly correlated features
        np.random.seed(42)
        n_samples = 1000
        
        # Create base feature
        base_feature = np.random.normal(0, 1, n_samples)
        
        # Create highly correlated features
            df = pd.DataFrame({
            'feature_1': base_feature,
            'feature_2': base_feature + np.random.normal(0, 0.01, n_samples),  # 99.99% correlation
            'feature_3': base_feature + np.random.normal(0, 0.01, n_samples),  # 99.99% correlation
            'feature_4': np.random.normal(0, 1, n_samples),  # Independent feature
            'feature_5': np.random.normal(0, 1, n_samples),  # Independent feature
        })
        
        logger.info(f"Original features: {len(df.columns)}")
        
        # Remove correlated features
        df_cleaned = remove_correlated_features(df, correlation_threshold=0.95)
        
        logger.info(f"Features after correlation removal: {len(df_cleaned.columns)}")
        
        if len(df_cleaned.columns) < len(df.columns):
            logger.info("✅ Correlation removal working - redundant features removed")
            return True
        else:
            logger.warning("⚠️ No features removed - check correlation threshold")
            return False
            
        except Exception as e:
        logger.error(f"❌ Correlation removal test failed: {e}")
        return False

def test_ensemble_weighting():
    """Test enhanced ensemble weighting"""
    try:
        from enhanced_features import calculate_enhanced_ensemble_weights
        
        # Create test model performance data
        model_performance = {
            'lightgbm_1m': 95.5,
            'xgboost_1m': 92.3,
            'neural_network_1m': 45.2,  # Poor performance
            'catboost_1m': 88.7,
            'random_forest_1m': 85.1,
            'lightgbm_5m': 89.2,
            'xgboost_5m': 91.8,
            'neural_network_5m': 52.1,  # Poor performance
        }
        
        # Calculate enhanced weights
        weights = calculate_enhanced_ensemble_weights(model_performance)
        
        logger.info(f"Ensemble weights calculated: {len(weights)} models")
        logger.info(f"Weight range: {min(weights.values()):.4f} - {max(weights.values()):.4f}")
        logger.info(f"Weight variance: {np.var(list(weights.values())):.6f}")
        
        # Check if weights are differentiated
        unique_weights = set(weights.values())
        if len(unique_weights) > 1:
            logger.info("✅ Ensemble weights show performance-based differentiation")
            return True
        else:
            logger.error("❌ All ensemble weights are equal - weighting failed")
            return False
            
        except Exception as e:
        logger.error(f"❌ Ensemble weighting test failed: {e}")
        return False

def test_neural_network_optimization():
    """Test neural network hyperparameter optimization"""
    try:
        from enhanced_features import optimize_neural_network_hyperparameters
        
        # Get optimized hyperparameters
        hyperparams = optimize_neural_network_hyperparameters()
        
        logger.info("✅ Neural network hyperparameters optimized:")
        logger.info(f"   • Layers: {len(hyperparams['layers'])}")
        logger.info(f"   • Learning rate: {hyperparams['learning_rate']}")
        logger.info(f"   • Batch size: {hyperparams['batch_size']}")
        logger.info(f"   • Epochs: {hyperparams['epochs']}")
        
        # Check if hyperparameters are reasonable
        if (hyperparams['learning_rate'] > 0 and 
            hyperparams['batch_size'] > 0 and 
            hyperparams['epochs'] > 0):
            logger.info("✅ Neural network hyperparameters are valid")
            return True
        else:
            logger.error("❌ Invalid neural network hyperparameters")
            return False
            
        except Exception as e:
        logger.error(f"❌ Neural network optimization test failed: {e}")
        return False

def test_external_data_enablement():
    """Test external data source enablement"""
    try:
        from enhanced_features import enable_external_data_sources
        
        # Test configuration
        test_config = {
            'api_keys': {
                'finnhub_token': 'test_token',
                'twelvedata_api_key': 'test_key',
                'news_api_key': 'test_news_key'
            }
        }
        
        # Enable external sources
        updated_config = enable_external_data_sources(test_config)
        
        # Check if sources are enabled
        api_keys = updated_config.get('api_keys', {})
        
        if (api_keys.get('finnhub_enabled') and 
            api_keys.get('twelvedata_enabled') and 
            api_keys.get('fear_greed_enabled')):
            logger.info("✅ External data sources enabled successfully")
            logger.info(f"   • Finnhub rate limit: {api_keys.get('finnhub_rate_limit')}")
            logger.info(f"   • Twelve Data rate limit: {api_keys.get('twelvedata_rate_limit')}")
            logger.info(f"   • Fear & Greed rate limit: {api_keys.get('fear_greed_rate_limit')}")
            return True
        else:
            logger.error("❌ External data sources not properly enabled")
            return False
            
        except Exception as e:
        logger.error(f"❌ External data enablement test failed: {e}")
        return False
    
def test_feature_selection():
    """Test feature selection from clusters"""
        try:
        from enhanced_features import select_best_feature_from_cluster
            
            # Create test data
        np.random.seed(42)
        n_samples = 1000
        
        df = pd.DataFrame({
            'rsi': np.random.uniform(0, 100, n_samples),
            'macd': np.random.normal(0, 1, n_samples),
            'derived_feature_1': np.random.normal(0, 1, n_samples),
            'derived_feature_2': np.random.normal(0, 1, n_samples),
        })
        
        # Test feature selection
        features = ['rsi', 'macd', 'derived_feature_1', 'derived_feature_2']
        best_feature = select_best_feature_from_cluster(features, df)
        
        logger.info(f"✅ Best feature selected: {best_feature}")
        
        # Check if a feature was selected
        if best_feature in features:
            logger.info("✅ Feature selection working correctly")
            return True
        else:
            logger.error("❌ Feature selection failed")
            return False
            
        except Exception as e:
        logger.error(f"❌ Feature selection test failed: {e}")
        return False

def run_all_tests():
    """Run all enhancement tests"""
    logger.info("🚀 Starting Ultra Enhancement Tests")
    logger.info("=" * 50)
    
    tests = [
        ("Enhanced Features Import", test_enhanced_features_import),
        ("Correlation Removal", test_correlation_removal),
        ("Ensemble Weighting", test_ensemble_weighting),
        ("Neural Network Optimization", test_neural_network_optimization),
        ("External Data Enablement", test_external_data_enablement),
        ("Feature Selection", test_feature_selection),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n🧪 Testing: {test_name}")
        try:
            result = test_func()
            results.append((test_name, result))
            if result:
                logger.info(f"✅ {test_name}: PASSED")
            else:
                logger.error(f"❌ {test_name}: FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name}: ERROR - {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("📊 TEST RESULTS SUMMARY")
    logger.info("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"   {test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 ALL TESTS PASSED! Enhancements are working correctly.")
        return True
    else:
        logger.error(f"⚠️ {total - passed} tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 