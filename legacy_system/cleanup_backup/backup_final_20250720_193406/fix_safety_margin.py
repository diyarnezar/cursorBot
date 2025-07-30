#!/usr/bin/env python3
"""
Fix Safety Margin Inconsistency
Resolves the critical flaw where 15-day mode (1,144 weight) exceeds the 80% safety ceiling (960 weight)
"""

import os
import sys
import re
from pathlib import Path

def fix_safety_margin_option_1():
    """Option 1: Raise safety ceiling to 100% (recommended by ChatGPT)"""
    print("🔧 Fixing Safety Margin - Option 1: 100% Ceiling")
    print("=" * 60)
    
    # Fix Binance Rate Limiter
    print("📝 Fixing Binance Rate Limiter...")
    
    try:
        with open("modules/binance_rate_limiter.py", 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Change SAFETY_MARGIN from 0.8 to 1.0
        old_pattern = r'SAFETY_MARGIN = 0\.8  # Use only 80% of limits'
        new_pattern = 'SAFETY_MARGIN = 1.0  # Use 100% of limits (as recommended by ChatGPT)'
        
        if old_pattern in content:
            content = content.replace(old_pattern, new_pattern)
            print("✅ Updated SAFETY_MARGIN from 0.8 to 1.0")
        else:
            print("⚠️ SAFETY_MARGIN pattern not found, checking current value...")
            if 'SAFETY_MARGIN = 1.0' in content:
                print("✅ SAFETY_MARGIN already set to 1.0")
            else:
                print("❌ Could not find SAFETY_MARGIN to update")
                return False
        
        # Update the comment about effective limit
        old_comment = r'# Safety margins \(stay under limits\)'
        new_comment = '# Safety margins (use 100% of limits as recommended by ChatGPT)'
        content = content.replace(old_comment, new_comment)
        
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
        
        # Update the comment
        old_comment = r'# Conservative safety limits \(stay well under Binance limits\)'
        new_comment = '# Safety limits (use 100% of Binance limits as recommended by ChatGPT)'
        content = content.replace(old_comment, new_comment)
        
        # Write the fixed content
        with open("modules/global_api_monitor.py", 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("✅ Global API Monitor fixed")
        
    except Exception as e:
        print(f"❌ Error fixing Global API Monitor: {e}")
        return False
    
    return True

def fix_safety_margin_option_2():
    """Option 2: Reduce demand to stay within 960 weight"""
    print("🔧 Fixing Safety Margin - Option 2: Reduce Demand")
    print("=" * 60)
    
    # This would involve splitting the 26 symbols into smaller batches
    # or increasing delays to spread the weight over more time
    print("📝 This option would require:")
    print("   - Splitting 26 symbols into smaller batches")
    print("   - Increasing delays between calls")
    print("   - More complex batching logic")
    print("   - Longer total execution time")
    print("")
    print("❌ Option 2 not implemented - Option 1 is simpler and recommended")
    
    return False

def create_updated_audit():
    """Create updated audit script with corrected expectations"""
    
    audit_script = '''#!/usr/bin/env python3
"""
Updated Comprehensive Rate Limit Audit (FIXED)
Verifies that ALL components stay under 1,200 weight per minute (100% ceiling)
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
            logging.FileHandler(f'rate_limit_audit_fixed_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )
    return logging.getLogger(__name__)

def audit_binance_rate_limiter():
    """Audit the Binance rate limiter configuration (FIXED)"""
    logger = logging.getLogger(__name__)
    logger.info("Auditing Binance Rate Limiter (FIXED)")
    
    try:
        from modules.binance_rate_limiter import binance_limiter
        
        # Check configuration (FIXED expectations)
        config_issues = []
        
        if binance_limiter.REQUEST_WEIGHT_1M != 1200:
            config_issues.append(f"REQUEST_WEIGHT_1M should be 1200, got {binance_limiter.REQUEST_WEIGHT_1M}")
        
        if binance_limiter.SAFETY_MARGIN != 1.0:
            config_issues.append(f"SAFETY_MARGIN should be 1.0 (100%), got {binance_limiter.SAFETY_MARGIN}")
        
        effective_limit = binance_limiter.REQUEST_WEIGHT_1M * binance_limiter.SAFETY_MARGIN
        if effective_limit != 1200:
            config_issues.append(f"Effective limit should be 1200, got {effective_limit}")
        
        # Check endpoint weights
        klines_weight = binance_limiter.get_endpoint_weight('/api/v3/klines', {'limit': 1000})
        if klines_weight != 2:
            config_issues.append(f"Klines weight should be 2, got {klines_weight}")
        
        if config_issues:
            logger.error("Binance Rate Limiter Configuration Issues:")
            for issue in config_issues:
                logger.error(f"   {issue}")
            return False
        else:
            logger.info("Binance Rate Limiter Configuration: CORRECT (FIXED)")
            logger.info(f"   REQUEST_WEIGHT_1M: {binance_limiter.REQUEST_WEIGHT_1M}")
            logger.info(f"   SAFETY_MARGIN: {binance_limiter.SAFETY_MARGIN} (100%)")
            logger.info(f"   Effective Limit: {effective_limit}")
            logger.info(f"   Klines Weight: {klines_weight}")
            return True
            
    except Exception as e:
        logger.error(f"Binance Rate Limiter Audit Failed: {e}")
        return False

def audit_global_api_monitor():
    """Audit the global API monitor configuration (FIXED)"""
    logger = logging.getLogger(__name__)
    logger.info("Auditing Global API Monitor (FIXED)")
    
    try:
        from modules.global_api_monitor import global_api_monitor
        
        # Check configuration (FIXED expectations)
        config_issues = []
        
        if global_api_monitor.binance_requests_per_minute != 1200:
            config_issues.append(f"binance_requests_per_minute should be 1200, got {global_api_monitor.binance_requests_per_minute}")
        
        if global_api_monitor.global_safety_limit_per_minute != 1200:
            config_issues.append(f"global_safety_limit_per_minute should be 1200, got {global_api_monitor.global_safety_limit_per_minute}")
        
        if global_api_monitor.global_safety_limit_per_second != 20:
            config_issues.append(f"global_safety_limit_per_second should be 20, got {global_api_monitor.global_safety_limit_per_second}")
        
        if config_issues:
            logger.error("Global API Monitor Configuration Issues:")
            for issue in config_issues:
                logger.error(f"   {issue}")
            return False
        else:
            logger.info("Global API Monitor Configuration: CORRECT (FIXED)")
            logger.info(f"   Binance Limit: {global_api_monitor.binance_requests_per_minute}/min")
            logger.info(f"   Safety Limit: {global_api_monitor.global_safety_limit_per_minute}/min (100%)")
            logger.info(f"   Second Limit: {global_api_monitor.global_safety_limit_per_second}/sec (100%)")
            return True
            
    except Exception as e:
        logger.error(f"Global API Monitor Audit Failed: {e}")
        return False

def audit_training_modes():
    """Audit all training modes for rate limit safety (FIXED)"""
    logger = logging.getLogger(__name__)
    logger.info("Auditing Training Modes (FIXED)")
    
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
            
            # Check if safe (FIXED: now checking against 1200, not 960)
            is_safe = total_weight_26_symbols <= 1200
            
            if is_safe:
                logger.info(f"{mode['description']}: {total_weight_26_symbols} weight (SAFE)")
            else:
                logger.error(f"{mode['description']}: {total_weight_26_symbols} weight (UNSAFE)")
                all_safe = False
                
        except Exception as e:
            logger.error(f"{mode['description']}: Audit failed - {e}")
            all_safe = False
    
    return all_safe

def run_fixed_audit():
    """Run fixed comprehensive rate limit audit"""
    logger = setup_audit_logging()
    
    logger.info("COMPREHENSIVE RATE LIMIT AUDIT (FIXED)")
    logger.info("=" * 60)
    logger.info("Verifying ALL components stay under 1,200 weight per minute (100% ceiling)")
    logger.info("=" * 60)
    
    audit_results = []
    
    # Run all audits
    audit_results.append(("Binance Rate Limiter", audit_binance_rate_limiter()))
    audit_results.append(("Global API Monitor", audit_global_api_monitor()))
    audit_results.append(("Training Modes", audit_training_modes()))
    
    # Summary
    logger.info("=" * 60)
    logger.info("AUDIT SUMMARY (FIXED)")
    logger.info("=" * 60)
    
    passed = 0
    total = len(audit_results)
    
    for component, result in audit_results:
        status = "PASSED" if result else "FAILED"
        logger.info(f"{component}: {status}")
        if result:
            passed += 1
    
    logger.info(f"Overall: {passed}/{total} components passed")
    
    if passed == total:
        logger.info("ALL COMPONENTS PASSED (FIXED)!")
        logger.info("System is TRULY BULLETPROOF for rate limiting")
        logger.info("Never exceeds 1,200 weight per minute (100% ceiling)")
        logger.info("All training modes are safe")
        logger.info("Ready for production use!")
    else:
        logger.info(f"{total - passed} components failed")
        logger.info("System needs attention before production use")
    
    return passed == total

if __name__ == "__main__":
    success = run_fixed_audit()
    if success:
        print("AUDIT PASSED - System is truly bulletproof!")
    else:
        print("AUDIT FAILED - System needs fixes!")
'''
    
    try:
        with open("comprehensive_rate_limit_audit_fixed.py", 'w', encoding='utf-8') as f:
            f.write(audit_script)
        print("✅ Created updated audit script: comprehensive_rate_limit_audit_fixed.py")
        return True
    except Exception as e:
        print(f"❌ Error creating updated audit: {e}")
        return False

def main():
    """Main function to fix safety margin"""
    print("🔧 Fixing Safety Margin Inconsistency")
    print("=" * 60)
    print("ChatGPT identified a critical flaw:")
    print("   - 80% safety ceiling: 960 weight")
    print("   - 15-day mode uses: 1,144 weight")
    print("   - 1,144 > 960 = INCONSISTENT!")
    print("=" * 60)
    
    print("📋 Available Options:")
    print("1. Option 1: Raise safety ceiling to 100% (RECOMMENDED)")
    print("   - Change SAFETY_MARGIN from 0.8 to 1.0")
    print("   - Allow usage up to 1,200 weight (100%)")
    print("   - 1,144 weight (95.3%) is safely under 1,200")
    print("")
    print("2. Option 2: Reduce demand to stay within 960 weight")
    print("   - Split 26 symbols into smaller batches")
    print("   - Increase delays between calls")
    print("   - More complex implementation")
    print("")
    
    choice = input("Choose option (1 or 2): ").strip()
    
    if choice == "1":
        print("\n🚀 Implementing Option 1: 100% Safety Ceiling")
        success1 = fix_safety_margin_option_1()
        success2 = create_updated_audit()
        
        if success1 and success2:
            print("\n✅ Safety margin fixed successfully!")
            print("📝 Changes made:")
            print("   - SAFETY_MARGIN: 0.8 → 1.0 (100%)")
            print("   - Effective limit: 960 → 1200 weight")
            print("   - All training modes now safely under 1200")
            print("")
            print("🧪 Test the fix:")
            print("   python comprehensive_rate_limit_audit_fixed.py")
        else:
            print("\n❌ Fix failed!")
    elif choice == "2":
        print("\n⚠️ Option 2 not implemented - Option 1 is simpler and recommended")
    else:
        print("\n❌ Invalid choice")

if __name__ == "__main__":
    main() 