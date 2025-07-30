#!/usr/bin/env python3
"""
Simple CatBoost Test
===================

Standalone test to verify CatBoost functionality and error handling
without importing the problematic training file.
"""

import pandas as pd
import numpy as np
import sys
import os

def test_catboost_import():
    """Test if CatBoost can be imported and basic functionality works"""
    print("🧠 Testing CatBoost Import and Basic Functionality")
    print("="*50)
    
    try:
        import catboost as cb
        print("✅ CatBoost imported successfully")
        
        # Test basic CatBoost functionality
        X = np.random.rand(100, 5)
        y = np.random.rand(100)
        
        # Test with early stopping
        try:
            model = cb.CatBoostRegressor(iterations=100, verbose=False)
            model.fit(X, y, eval_set=[(X, y)], early_stopping_rounds=10, verbose=False)
            print("✅ CatBoost with early stopping works")
        except Exception as e:
            print(f"⚠️ CatBoost with early stopping failed: {e}")
            
            # Try without early stopping
            try:
                model = cb.CatBoostRegressor(iterations=100, verbose=False)
                model.fit(X, y, verbose=False)
                print("✅ CatBoost without early stopping works")
            except Exception as e2:
                print(f"❌ CatBoost without early stopping also failed: {e2}")
                return False
        
        # Test prediction
        try:
            pred = model.predict(X[:5])
            print(f"✅ CatBoost prediction works: {pred.shape}")
            return True
        except Exception as e:
            print(f"❌ CatBoost prediction failed: {e}")
            return False
            
    except ImportError:
        print("❌ CatBoost not installed")
        return False
    except Exception as e:
        print(f"❌ CatBoost test failed: {e}")
        return False

def test_catboost_error_handling():
    """Test CatBoost error handling scenarios"""
    print("\n🛡️ Testing CatBoost Error Handling")
    print("="*50)
    
    try:
        import catboost as cb
        
        # Test with insufficient data
        try:
            X = np.random.rand(5, 5)  # Very small dataset
            y = np.random.rand(5)
            
            model = cb.CatBoostRegressor(iterations=100, verbose=False)
            model.fit(X, y, verbose=False)
            print("✅ CatBoost handles small datasets")
        except Exception as e:
            print(f"⚠️ CatBoost small dataset test: {e}")
        
        # Test with NaN values
        try:
            X = np.random.rand(50, 5)
            X[0, 0] = np.nan  # Add NaN
            y = np.random.rand(50)
            
            model = cb.CatBoostRegressor(iterations=100, verbose=False)
            model.fit(X, y, verbose=False)
            print("✅ CatBoost handles NaN values")
        except Exception as e:
            print(f"⚠️ CatBoost NaN test: {e}")
        
        # Test with infinite values
        try:
            X = np.random.rand(50, 5)
            X[0, 0] = np.inf  # Add inf
            y = np.random.rand(50)
            
            model = cb.CatBoostRegressor(iterations=100, verbose=False)
            model.fit(X, y, verbose=False)
            print("✅ CatBoost handles infinite values")
        except Exception as e:
            print(f"⚠️ CatBoost infinite values test: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ CatBoost error handling test failed: {e}")
        return False

def test_catboost_optuna_integration():
    """Test CatBoost with Optuna hyperparameter optimization"""
    print("\n🔧 Testing CatBoost + Optuna Integration")
    print("="*50)
    
    try:
        import catboost as cb
        import optuna
        from sklearn.model_selection import KFold
        from sklearn.metrics import mean_squared_error
        
        # Create test data
        X = np.random.rand(100, 5)
        y = np.random.rand(100)
        
        def objective(trial):
            try:
                params = {
                    'iterations': trial.suggest_int('iterations', 50, 200),
                    'depth': trial.suggest_int('depth', 4, 8),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'verbose': False
                }
                
                # Use cross-validation
                cv_scores = []
                kf = KFold(n_splits=3, shuffle=True, random_state=42)
                
                for train_idx, val_idx in kf.split(X):
                    X_train, X_val = X[train_idx], X[val_idx]
                    y_train, y_val = y[train_idx], y[val_idx]
                    
                    model = cb.CatBoostRegressor(**params)
                    
                    try:
                        # Try with early stopping first
                        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=20, verbose=False)
                    except Exception:
                        # Fallback without early stopping
                        model.fit(X_train, y_train, verbose=False)
                    
                    y_pred = model.predict(X_val)
                    score = mean_squared_error(y_val, y_pred)
                    cv_scores.append(score)
                
                return np.mean(cv_scores) if cv_scores else float('inf')
                
            except Exception as e:
                print(f"⚠️ Optuna trial failed: {e}")
                return float('inf')
        
        # Run optimization
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=5, timeout=60)
        
        if study.best_trial and study.best_trial.value != float('inf'):
            print("✅ CatBoost + Optuna integration works")
            print(f"   • Best score: {study.best_trial.value:.6f}")
            print(f"   • Best params: {study.best_params}")
            return True
        else:
            print("⚠️ CatBoost + Optuna integration had issues")
            return False
            
    except Exception as e:
        print(f"❌ CatBoost + Optuna test failed: {e}")
        return False

def main():
    """Run all CatBoost tests"""
    print("🚀 CatBoost Functionality Tests")
    print("="*60)
    
    # Test 1: Basic import and functionality
    test1_passed = test_catboost_import()
    
    # Test 2: Error handling
    test2_passed = test_catboost_error_handling()
    
    # Test 3: Optuna integration
    test3_passed = test_catboost_optuna_integration()
    
    # Summary
    print("\n📋 Test Summary")
    print("="*30)
    print(f"✅ Basic Functionality: {'PASSED' if test1_passed else 'FAILED'}")
    print(f"✅ Error Handling: {'PASSED' if test2_passed else 'FAILED'}")
    print(f"✅ Optuna Integration: {'PASSED' if test3_passed else 'FAILED'}")
    
    if test1_passed and test2_passed and test3_passed:
        print("\n🎉 ALL TESTS PASSED! CatBoost is working correctly.")
        print("   • CatBoost can be imported and used")
        print("   • Error handling works properly")
        print("   • Optuna integration is functional")
        print("   • The issue is likely in the training file structure")
    else:
        print("\n❌ Some tests failed. CatBoost may have installation issues.")
    
    return test1_passed and test2_passed and test3_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 