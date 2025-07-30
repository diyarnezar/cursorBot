#!/usr/bin/env python3
"""
Final Cleanup Script
Remove more unnecessary files after the first cleanup
"""

import os
import shutil
from datetime import datetime

def final_cleanup():
    """Final cleanup - remove more unnecessary files"""
    
    print("🧹 Final Cleanup - Additional Unnecessary Files")
    print("=" * 50)
    
    # Files that are safe to delete (no imports, no references)
    safe_to_delete = [
        # Cleanup scripts themselves (after use)
        'conservative_cleanup.py',
        'cleanup_unused_files.py',
        
        # Duplicate/backup training scripts
        'ultra_train_enhanced_rate_limited_backup.py',  # Backup of fixed version
        'ultra_train_enhanced_backup.py',  # Backup of original
        'ultra_train_enhanced_rate_limited.py',  # Replaced by _fixed version
        'ultra_train_enhanced_with_rate_limiting.py',  # Replaced by better version
        'ultra_train_enhanced_integrated.py',  # Replaced by better version
        
        # Fix scripts (after fixes applied)
        'fix_safety_margin.py',
        'quick_fix_safety_margin.py', 
        'corrected_fix_safety_margin.py',
        'fix_rate_limiting_integration.py',
        
        # Integration scripts (after integration)
        'integrate_rate_limiting.py',
        
        # Test scripts (obsolete or replaced)
        'test_rate_limited_training.py',
        'test_fixed_integration.py',
        'test_enhanced_rate_limiting.py',
        'test_corrected_limits.py',
        'test_global_api_monitoring.py',
        'test_rate_limiting.py',
        'test_1minute_intervals.py',
        'test_enhancements.py',
        
        # Verification scripts (no longer needed)
        'verify_rate_limiting_status.py',
        'verify_safety_analysis.py',
        'verify_chatgpt_calculation.py',
        
        # Audit scripts (keep only the main one)
        'comprehensive_rate_limit_audit_fixed.py',
        
        # Safety check scripts (no longer needed)
        'check_full_historical_safety.py',
        
        # Standalone processor (moved to modules/)
        'safe_full_historical_processor.py',
        
        # Old documentation (replaced by newer versions)
        'RATE_LIMITING_INTEGRATION_SUMMARY.md',
        'optimization_analysis.md',
        'PAUSE_RESUME_GUIDE.md',
        'CHATGPT_ROADMAP_ANALYSIS.md',
        'ENHANCEMENTS_IMPLEMENTATION_SUMMARY.md',
        'ENHANCEMENTS_IMPLEMENTED.md',
        'TRAINING_MODES_VERIFICATION.md',
        'TRAINING_LOG_ANALYSIS_AND_IMPROVEMENTS.md',
        'FEATURE_ENGINEERING_ISSUES_ANALYSIS.md',
        'ADAPTIVE_THRESHOLD_SYSTEM.md',
        'COMPREHENSIVE_FIXES_SUMMARY.md',
        'FEATURE_ENGINEERING_FIXES.md',
        'REAL_DATA_FIXES_SUMMARY.md',
        'ULTRA_IMPROVEMENTS_SUMMARY.md',
        'TRAINING_FIXES_SUMMARY.md',
        'TRAINING_IMPROVEMENTS_SUMMARY.md',
        'FIXES_SUMMARY.md',
        'ULTRA_SYSTEM_SUMMARY.md',
        'ULTRA_ENHANCEMENTS.md',
        'API_LIMITS_COMPLIANCE.md',
        
        # Old JSON files
        'verification_report.json',
        'test_checkpoint.json',
        
        # Old text files
        'verification_output.txt',
        
        # Test files
        'test_keyboard.py',
        
        # Old log files
        'ultra_bot.log',
    ]
    
    # Files to keep (actively used)
    keep_files = [
        # Main training scripts
        'ultra_train_enhanced_rate_limited_fixed.py',  # Latest version
        'ultra_train_enhanced.py',  # Still imported by other files
        'ultra_main.py',  # Main entry point
        'ultra_train_multipair.py',  # Multi-pair training
        
        # Rate limiting system
        'modules/binance_rate_limiter.py',
        'modules/global_api_monitor.py', 
        'modules/historical_kline_fetcher.py',
        'modules/safe_full_historical_processor.py',
        'modules/intelligent_rate_limiter.py',
        'modules/training_api_monitor.py',
        'modules/optimized_data_collector.py',
        
        # Configuration and data
        'config.json',
        'training_checkpoint.json',
        'trading_data.db',
        
        # Documentation (keep the latest)
        'SAFE_FULL_HISTORICAL_SUMMARY.md',
        'comprehensive_rate_limit_audit.py',
        
        # Test scripts (keep the working ones)
        'test_safe_full_historical.py',
        
        # Integration scripts (keep the working ones)
        'integrate_safe_full_historical.py',
        
        # Essential files
        'requirements.txt',
        'Dockerfile',
        'setup_aws.sh',
        'API_SETUP_GUIDE.md',
        'CONFIGURATION_REVIEW.md',
        'DEPLOY_AWS.md',
    ]
    
    print("📋 Files that are SAFE to delete:")
    safe_count = 0
    for file in safe_to_delete:
        if os.path.exists(file):
            print(f"   ❌ {file}")
            safe_count += 1
    
    print(f"\n📋 Files to KEEP:")
    keep_count = 0
    for file in keep_files:
        if os.path.exists(file):
            print(f"   ✅ {file}")
            keep_count += 1
    
    print(f"\n📊 Summary:")
    print(f"   🗑️ Safe to delete: {safe_count} files")
    print(f"   ✅ Keep: {keep_count} files")
    
    if safe_count == 0:
        print("\n✅ No more files to delete!")
        return True
    
    # Ask for confirmation
    response = input(f"\nDelete {safe_count} more files? (y/N): ").strip().lower()
    
    if response != 'y':
        print("❌ Final cleanup cancelled")
        return False
    
    # Create backup directory
    backup_dir = f"backup_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(backup_dir, exist_ok=True)
    
    deleted_count = 0
    
    # Move files to backup
    for file in safe_to_delete:
        if os.path.exists(file):
            try:
                # Move to backup instead of deleting
                backup_path = os.path.join(backup_dir, file)
                os.makedirs(os.path.dirname(backup_path), exist_ok=True)
                shutil.move(file, backup_path)
                print(f"🗑️ Moved to backup: {file}")
                deleted_count += 1
            except Exception as e:
                print(f"❌ Error moving {file}: {e}")
    
    print(f"\n✅ Final cleanup completed!")
    print(f"   📁 Backup created: {backup_dir}")
    print(f"   🗑️ Files moved to backup: {deleted_count}")
    
    # Show final directory structure
    print(f"\n📁 Final directory structure:")
    remaining_files = [f for f in os.listdir('.') if os.path.isfile(f)]
    remaining_files.sort()
    
    for file in remaining_files:
        if file.endswith('.py') or file.endswith('.md') or file.endswith('.json'):
            print(f"   📄 {file}")
    
    return True

if __name__ == "__main__":
    final_cleanup() 