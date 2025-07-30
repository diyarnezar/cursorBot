#!/usr/bin/env python3
"""
Fix Rate Limiting Integration
Updates the integration to use correct method signatures
"""

import os
import sys
import re
from pathlib import Path

def fix_rate_limiting_integration():
    """Fix the rate limiting integration with correct method calls"""
    
    print("🔧 Fixing Rate Limiting Integration")
    print("=" * 50)
    
    # Read the enhanced training script
    script_path = "ultra_train_enhanced_rate_limited.py"
    
    if not os.path.exists(script_path):
        print(f"❌ Enhanced training script not found: {script_path}")
        return False
    
    print(f"📖 Reading {script_path}...")
    
    try:
        with open(script_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return False
    
    print(f"✅ Read {len(content)} characters")
    
    # Create backup
    backup_path = "ultra_train_enhanced_rate_limited_backup.py"
    try:
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ Created backup: {backup_path}")
    except Exception as e:
        print(f"❌ Error creating backup: {e}")
        return False
    
    # Fix the kline fetcher method calls
    print("🔧 Fixing kline fetcher method calls...")
    
    # Fix fetch_klines_for_symbol calls
    old_pattern = r'klines = kline_fetcher\.fetch_klines_for_symbol\(([^,]+), days=([^)]+)\)'
    new_pattern = r'klines = kline_fetcher.fetch_klines_for_symbol(\1)'
    
    content = re.sub(old_pattern, new_pattern, content)
    print("✅ Fixed fetch_klines_for_symbol calls")
    
    # Fix fetch_klines_for_multiple_symbols calls
    old_pattern2 = r'results = kline_fetcher\.fetch_klines_for_multiple_symbols\(([^,]+), days=([^)]+)\)'
    new_pattern2 = r'results = kline_fetcher.fetch_klines_for_multiple_symbols(\1)'
    
    content = re.sub(old_pattern2, new_pattern2, content)
    print("✅ Fixed fetch_klines_for_multiple_symbols calls")
    
    # Update the data collection method to use correct parameters
    print("🔧 Updating data collection method...")
    
    # Find and replace the data collection method
    old_method_pattern = r'# Use enhanced kline fetcher\n\s+try:\n\s+# Monitor training API usage\n\s+training_monitor\.collect_training_data\(\'ETHFDUSD\', collection_days\)\n\s+\n\s+# Use the enhanced kline fetcher\n\s+klines = kline_fetcher\.fetch_klines_for_symbol\(\'ETHFDUSD\', days=collection_days\)'
    
    new_method_content = '''# Use enhanced kline fetcher
                try:
                    # Monitor training API usage
                    training_monitor.collect_training_data('ETHFDUSD', collection_days)
                    
                    # Use the enhanced kline fetcher with correct parameters
                    from datetime import timedelta
                    start_time = datetime.now() - timedelta(days=collection_days)
                    klines = kline_fetcher.fetch_klines_for_symbol('ETHFDUSD', start_time=start_time)'''
    
    content = re.sub(old_method_pattern, new_method_content, content, flags=re.DOTALL)
    print("✅ Updated data collection method")
    
    # Add import for datetime if not present
    if 'from datetime import datetime, timedelta' not in content:
        # Find the import section and add timedelta
        import_pattern = r'(from datetime import datetime)'
        content = re.sub(import_pattern, r'\1, timedelta', content)
        print("✅ Added timedelta import")
    
    # Write the fixed content
    fixed_path = "ultra_train_enhanced_rate_limited_fixed.py"
    try:
        with open(fixed_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ Created fixed script: {fixed_path}")
    except Exception as e:
        print(f"❌ Error writing fixed file: {e}")
        return False
    
    print("\n🎉 Integration fixes completed successfully!")
    print(f"📁 Backup: {backup_path}")
    print(f"📁 Fixed: {fixed_path}")
    print("\n🔧 Fixes applied:")
    print("   • Corrected fetch_klines_for_symbol method calls")
    print("   • Corrected fetch_klines_for_multiple_symbols method calls")
    print("   • Updated data collection to use start_time parameter")
    print("   • Added proper datetime imports")
    
    return True

def create_simple_test_script():
    """Create a simple test script to verify the fixes"""
    
    test_script = '''#!/usr/bin/env python3
"""
Simple Test for Rate Limiting Integration
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.binance_rate_limiter import binance_limiter
from modules.historical_kline_fetcher import kline_fetcher
from modules.global_api_monitor import global_api_monitor
from modules.training_api_monitor import training_monitor
from datetime import datetime, timedelta

def test_fixed_integration():
    """Test the fixed integration"""
    print("🧪 Testing Fixed Rate Limiting Integration")
    print("=" * 50)
    
    # Test 1: Rate limiting modules
    print("Test 1: Rate Limiting Modules")
    try:
        stats = binance_limiter.get_stats()
        print(f"✅ Binance limiter: {stats.get('available_weight_1m', 0)} weight available")
    except Exception as e:
        print(f"❌ Binance limiter failed: {e}")
        return False
    
    # Test 2: Kline fetcher with correct parameters
    print("\\nTest 2: Kline Fetcher (Fixed)")
    try:
        start_time = datetime.now() - timedelta(days=0.1)  # 2.4 hours
        klines = kline_fetcher.fetch_klines_for_symbol('ETHFDUSD', start_time=start_time)
        if klines and len(klines) > 0:
            print(f"✅ Kline fetcher: {len(klines)} klines fetched")
        else:
            print("❌ Kline fetcher: No data")
            return False
    except Exception as e:
        print(f"❌ Kline fetcher failed: {e}")
        return False
    
    # Test 3: Multi-pair strategy
    print("\\nTest 3: Multi-Pair Strategy")
    try:
        test_symbols = ['ETHFDUSD', 'BTCFDUSD']
        is_valid = kline_fetcher.validate_strategy(test_symbols)
        if is_valid:
            print("✅ Multi-pair strategy validation passed")
        else:
            print("❌ Multi-pair strategy validation failed")
            return False
    except Exception as e:
        print(f"❌ Multi-pair strategy failed: {e}")
        return False
    
    print("\\n🎉 All tests passed!")
    return True

if __name__ == "__main__":
    success = test_fixed_integration()
    if success:
        print("\\n✅ Ready to use fixed training script!")
        print("   python ultra_train_enhanced_rate_limited_fixed.py")
    else:
        print("\\n❌ Tests failed!")
'''
    
    try:
        with open("test_fixed_integration.py", 'w', encoding='utf-8') as f:
            f.write(test_script)
        print("✅ Created test script: test_fixed_integration.py")
        return True
    except Exception as e:
        print(f"❌ Error creating test script: {e}")
        return False

if __name__ == "__main__":
    success1 = fix_rate_limiting_integration()
    success2 = create_simple_test_script()
    
    if success1 and success2:
        print("\n✅ Ready to test the fixes!")
        print("   python test_fixed_integration.py")
    else:
        print("\n❌ Fixes failed!") 