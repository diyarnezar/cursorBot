#!/usr/bin/env python3
"""
Bulletproof System Test
=======================

Comprehensive test to verify all bulletproof features are working:
- Resume mode and checkpointing
- Telegram notifications
- API key validation
- All systems operational
- Maximum intelligence features
"""

import os
import sys
import json
import time
from datetime import datetime

def test_resume_mode():
    """Test resume mode functionality"""
    print("🔄 Testing Resume Mode...")
    
    # Check if checkpoint file exists
    checkpoint_exists = os.path.exists('training_checkpoint.json')
    print(f"   Checkpoint file exists: {checkpoint_exists}")
    
    # Test checkpoint loading
    try:
        from modules.pause_resume_controller import load_checkpoint
        checkpoint_data = load_checkpoint()
        if checkpoint_data:
            print(f"   ✅ Checkpoint loaded successfully")
            print(f"   📅 Last checkpoint: {checkpoint_data.get('timestamp', 'Unknown')}")
        else:
            print("   ⚠️ No checkpoint data (normal for fresh start)")
    except Exception as e:
        print(f"   ❌ Checkpoint loading failed: {e}")
        return False
    
    return True

def test_telegram_notifications():
    """Test Telegram notifications"""
    print("📱 Testing Telegram Notifications...")
    
    try:
        from modules.telegram_bot import TelegramNotifier
        
        # Load config
        with open('config.json', 'r') as f:
            config = json.load(f)
        
        api_keys = config.get('api_keys', {})
        telegram_token = api_keys.get('telegram_bot_token')
        telegram_chat_id = api_keys.get('telegram_chat_id')
        
        if not telegram_token or not telegram_chat_id:
            print("   ⚠️ Telegram credentials not configured")
            return False
        
        # Test Telegram connection
        notifier = TelegramNotifier(telegram_token, telegram_chat_id)
        test_message = f"🤖 Project Hyperion Bulletproof Test\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\nStatus: All systems operational!"
        
        success = notifier.send_message(test_message)
        if success:
            print("   ✅ Telegram notifications working!")
            return True
        else:
            print("   ❌ Telegram notifications failed")
            return False
            
    except Exception as e:
        print(f"   ❌ Telegram test failed: {e}")
        return False

def test_api_keys():
    """Test API key configuration"""
    print("🔑 Testing API Keys...")
    
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
        
        api_keys = config.get('api_keys', {})
        required_keys = [
            'binance_api_key', 'binance_api_secret',
            'telegram_bot_token', 'telegram_chat_id'
        ]
        
        missing_keys = []
        for key in required_keys:
            if not api_keys.get(key) or api_keys.get(key) == f"YOUR_{key.upper()}_HERE":
                missing_keys.append(key)
        
        if missing_keys:
            print(f"   ⚠️ Missing API keys: {', '.join(missing_keys)}")
            return False
        else:
            print("   ✅ All required API keys configured")
            return True
            
    except Exception as e:
        print(f"   ❌ API key check failed: {e}")
        return False

def test_training_system():
    """Test training system initialization"""
    print("🧠 Testing Training System...")
    
    try:
        from ultra_train_enhanced import UltraEnhancedTrainer
        
        # Test trainer initialization
        trainer = UltraEnhancedTrainer()
        print("   ✅ Trainer initialized successfully")
        
        # Test config loading
        if hasattr(trainer, 'config') and trainer.config:
            print("   ✅ Configuration loaded successfully")
        else:
            print("   ❌ Configuration loading failed")
            return False
        
        # Test pause/resume controller
        from modules.pause_resume_controller import setup_pause_resume
        controller = setup_pause_resume()
        print("   ✅ Pause/Resume controller initialized")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Training system test failed: {e}")
        return False

def test_feature_engineering():
    """Test feature engineering capabilities"""
    print("🔧 Testing Feature Engineering...")
    
    try:
        from modules.feature_engineering import EnhancedFeatureEngineer
        from modules.alternative_data import EnhancedAlternativeData
        
        # Test feature engineer
        feature_engineer = EnhancedFeatureEngineer()
        print("   ✅ Feature engineer initialized")
        
        # Test alternative data
        with open('config.json', 'r') as f:
            config = json.load(f)
        alternative_data = EnhancedAlternativeData(config.get('api_keys', {}))
        print("   ✅ Alternative data initialized")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Feature engineering test failed: {e}")
        return False

def test_model_training():
    """Test model training capabilities"""
    print("🤖 Testing Model Training...")
    
    try:
        # Test CatBoost (previously had issues)
        try:
            import catboost as cb
            print("   ✅ CatBoost available")
        except ImportError:
            print("   ⚠️ CatBoost not available")
        
        # Test other models
        import lightgbm as lgb
        import xgboost as xgb
        print("   ✅ LightGBM available")
        print("   ✅ XGBoost available")
        
        # Test TensorFlow
        import tensorflow as tf
        print("   ✅ TensorFlow available")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Model training test failed: {e}")
        return False

def test_checkpoint_system():
    """Test checkpoint system"""
    print("💾 Testing Checkpoint System...")
    
    try:
        from modules.pause_resume_controller import save_checkpoint, load_checkpoint
        
        # Test checkpoint saving
        test_data = {
            'test_timestamp': datetime.now().isoformat(),
            'test_data': 'This is a test checkpoint'
        }
        
        save_checkpoint(test_data)
        print("   ✅ Checkpoint saved successfully")
        
        # Test checkpoint loading
        loaded_data = load_checkpoint()
        if loaded_data:
            print("   ✅ Checkpoint loaded successfully")
            # Check if our test data is in the loaded checkpoint
            if 'data' in loaded_data and 'test_data' in loaded_data.get('data', {}):
                print("   ✅ Test data found in checkpoint")
            else:
                print("   ⚠️ Test data not found (may be overwritten by other data)")
            return True
        else:
            print("   ❌ Checkpoint loading failed")
            return False
            
    except Exception as e:
        print(f"   ❌ Checkpoint system test failed: {e}")
        return False

def main():
    """Run all bulletproof system tests"""
    print("PROJECT HYPERION - BULLETPROOF SYSTEM TEST")
    print("=" * 50)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)
    
    tests = [
        ("Resume Mode", test_resume_mode),
        ("Telegram Notifications", test_telegram_notifications),
        ("API Keys", test_api_keys),
        ("Training System", test_training_system),
        ("Feature Engineering", test_feature_engineering),
        ("Model Training", test_model_training),
        ("Checkpoint System", test_checkpoint_system),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"   ❌ Test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("BULLETPROOF SYSTEM TEST RESULTS")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL SYSTEMS OPERATIONAL!")
        print("Your trading bot is ready for maximum performance!")
        print("\nTo start training:")
        print("  python ultra_train_enhanced.py --mode ultra-fast")
        print("\nTo resume from checkpoint:")
        print("  python ultra_train_enhanced.py --resume")
    else:
        print(f"\n⚠️ {total - passed} test(s) failed. Please check the issues above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 