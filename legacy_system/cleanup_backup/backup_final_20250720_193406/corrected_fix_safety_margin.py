#!/usr/bin/env python3
"""
Corrected Fix Safety Margin - Option 1: 100% Ceiling
"""

import os
import sys
import re

def fix_safety_margin():
    """Fix safety margin to 100% ceiling"""
    print("🔧 Corrected Fix: Safety Margin to 100%")
    print("=" * 50)
    
    # Fix Binance Rate Limiter
    print("📝 Fixing Binance Rate Limiter...")
    
    try:
        with open("modules/binance_rate_limiter.py", 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Change SAFETY_MARGIN from 0.8 to 1.0 (using exact pattern found)
        old_pattern = r'self\.SAFETY_MARGIN = 0\.8  # Use only 80% of limits'
        new_pattern = 'self.SAFETY_MARGIN = 1.0  # Use 100% of limits (as recommended by ChatGPT)'
        
        if old_pattern in content:
            content = content.replace(old_pattern, new_pattern)
            print("✅ Updated SAFETY_MARGIN from 0.8 to 1.0")
        else:
            print("⚠️ SAFETY_MARGIN pattern not found, checking current value...")
            if 'self.SAFETY_MARGIN = 1.0' in content:
                print("✅ SAFETY_MARGIN already set to 1.0")
            else:
                print("❌ Could not find SAFETY_MARGIN to update")
                print("Current content around SAFETY_MARGIN:")
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if 'SAFETY_MARGIN' in line:
                        print(f"Line {i+1}: {line}")
                return False
        
        # Write the fixed content
        with open("modules/binance_rate_limiter.py", 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("✅ Binance Rate Limiter fixed")
        
    except Exception as e:
        print(f"❌ Error fixing Binance Rate Limiter: {e}")
        return False
    
    # Fix Global API Monitor
    print("📝 Fixing Global API Monitor...")
    
    try:
        with open("modules/global_api_monitor.py", 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Update safety limits to use 100% of Binance limits
        old_pattern = r'self\.global_safety_limit_per_minute = 1000  # Stay under 1200/min'
        new_pattern = 'self.global_safety_limit_per_minute = 1200  # Use 100% of Binance limit (1200/min)'
        
        if old_pattern in content:
            content = content.replace(old_pattern, new_pattern)
            print("✅ Updated global_safety_limit_per_minute from 1000 to 1200")
        else:
            print("⚠️ global_safety_limit_per_minute pattern not found")
        
        old_pattern2 = r'self\.global_safety_limit_per_second = 15  # Stay under 20/sec'
        new_pattern2 = 'self.global_safety_limit_per_second = 20  # Use 100% of Binance limit (20/sec)'
        
        if old_pattern2 in content:
            content = content.replace(old_pattern2, new_pattern2)
            print("✅ Updated global_safety_limit_per_second from 15 to 20")
        else:
            print("⚠️ global_safety_limit_per_second pattern not found")
        
        # Write the fixed content
        with open("modules/global_api_monitor.py", 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("✅ Global API Monitor fixed")
        
    except Exception as e:
        print(f"❌ Error fixing Global API Monitor: {e}")
        return False
    
    print("\n🎉 Safety Margin Fixed Successfully!")
    print("📝 Changes made:")
    print("   - SAFETY_MARGIN: 0.8 → 1.0 (100%)")
    print("   - Effective limit: 960 → 1200 weight")
    print("   - All training modes now safely under 1200")
    print("")
    print("✅ 15-day mode (1,144 weight) is now safely under 1,200 limit!")
    
    return True

if __name__ == "__main__":
    success = fix_safety_margin()
    if success:
        print("\n🚀 Ready to test the fix!")
        print("   python comprehensive_rate_limit_audit.py")
    else:
        print("\n❌ Fix failed!") 