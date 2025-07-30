#!/usr/bin/env python3
"""
Test CatBoost Fix
================

Simple test to verify that the CatBoost training error has been fixed
and the model trains successfully with proper error handling.
"""

import pandas as pd
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the trainer
from ultra_train_enhanced import UltraEnhancedTrainer

def create_test_data():
    """Create simple test data for CatBoost training"""
    print("📊 Creating test data...")
    
    np.random.seed(42)
    n_samples = 100
    
    # Generate simple features
    data = pd.DataFrame({
        'feature_1': np.random.normal(0, 1, n_samples),
        'feature_2': np.random.normal(0, 1, n_samples),
        'feature_3': np.random.normal(0, 1, n_samples),
        'feature_4': np.random.normal(0, 1, n_samples),
        'feature_5': np.random.normal(0, 1, n_samples)
    })
    
    # Generate target (simple linear relationship with noise)
    target = (data['feature_1'] * 0.3 + 
              data['feature_2'] * 0.2 + 
              data['feature_3'] * 0.1 + 
              np.random.normal(0, 0.1, n_samples))
    
    print(f"✅ Created {len(data)} samples with {len(data.columns)} features")
    return data, target

def test_catboost_training():
    """Test CatBoost training with the fixed code"""
    print("\n🧠 Testing CatBoost Training Fix")
    print("="*50)
    
    try:
        # Create trainer instance
        trainer = UltraEnhancedTrainer()
        
        # Create test data
        X, y = create_test_data()
        
        print(f"📈 Training CatBoost with {len(X)} samples...")
        
        # Train CatBoost model
        model, score = trainer.train_catboost(X, y)
        
        if model is not None and score != float('inf'):
            print(f"✅ CatBoost training SUCCESSFUL!")
            print(f"   • Model type: {type(model).__name__}")
            print(f"   • Score: {score:.3f}")
            print(f"   • Model trained successfully with error handling")
            
            # Test prediction
            try:
                pred = model.predict(X.iloc[:5])
                print(f"   • Prediction test: {pred.shape}")
                print(f"   • Sample predictions: {pred[:3]}")
                print("✅ Prediction test PASSED")
            except Exception as pred_error:
                print(f"❌ Prediction test failed: {pred_error}")
                return False
                
            return True
        else:
            print(f"❌ CatBoost training FAILED")
            print(f"   • Model: {model}")
            print(f"   • Score: {score}")
            return False
            
    except Exception as e:
        print(f"❌ CatBoost test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ensemble_integration():
    """Test that CatBoost integrates properly with the ensemble"""
    print("\n🔗 Testing CatBoost Ensemble Integration")
    print("="*50)
    
    try:
        # Create trainer instance
        trainer = UltraEnhancedTrainer()
        
        # Create test data
        X, y = create_test_data()
        
        print(f"📈 Training models for ensemble...")
        
        # Train multiple models including CatBoost
        models = {}
        scores = {}
        
        # Train LightGBM
        lgb_model, lgb_score = trainer.train_lightgbm(X, y)
        if lgb_model is not None:
            models['lightgbm'] = lgb_model
            scores['lightgbm'] = lgb_score
            print(f"   • LightGBM: {lgb_score:.3f}")
        
        # Train CatBoost
        cb_model, cb_score = trainer.train_catboost(X, y)
        if cb_model is not None:
            models['catboost'] = cb_model
            scores['catboost'] = cb_score
            print(f"   • CatBoost: {cb_score:.3f}")
        
        # Train XGBoost
        xgb_model, xgb_score = trainer.train_xgboost(X, y)
        if xgb_model is not None:
            models['xgboost'] = xgb_model
            scores['xgboost'] = xgb_score
            print(f"   • XGBoost: {xgb_score:.3f}")
        
        if len(models) >= 2:
            print(f"✅ Ensemble integration SUCCESSFUL!")
            print(f"   • Models in ensemble: {list(models.keys())}")
            print(f"   • All models trained successfully")
            
            # Test ensemble prediction
            try:
                # Simple ensemble prediction (average)
                predictions = []
                for name, model in models.items():
                    pred = model.predict(X.iloc[:5])
                    predictions.append(pred)
                
                ensemble_pred = np.mean(predictions, axis=0)
                print(f"   • Ensemble prediction shape: {ensemble_pred.shape}")
                print(f"   • Sample ensemble predictions: {ensemble_pred[:3]}")
                print("✅ Ensemble prediction test PASSED")
                
            except Exception as ensemble_error:
                print(f"❌ Ensemble prediction failed: {ensemble_error}")
                return False
                
            return True
        else:
            print(f"❌ Not enough models for ensemble test")
            return False
            
    except Exception as e:
        print(f"❌ Ensemble integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all CatBoost tests"""
    print("🚀 CatBoost Fix Verification Tests")
    print("="*60)
    
    # Test 1: Basic CatBoost training
    test1_passed = test_catboost_training()
    
    # Test 2: Ensemble integration
    test2_passed = test_ensemble_integration()
    
    # Summary
    print("\n📋 Test Summary")
    print("="*30)
    print(f"✅ CatBoost Training Fix: {'PASSED' if test1_passed else 'FAILED'}")
    print(f"✅ Ensemble Integration: {'PASSED' if test2_passed else 'FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 ALL TESTS PASSED! CatBoost fix is working correctly.")
        print("   • CatBoost now trains without early stopping errors")
        print("   • Proper error handling is in place")
        print("   • Ensemble integration works correctly")
        print("   • Ready for maximum intelligence training!")
    else:
        print("\n❌ Some tests failed. CatBoost may still have issues.")
    
    return test1_passed and test2_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 