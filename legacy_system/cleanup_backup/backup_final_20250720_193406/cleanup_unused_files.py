#!/usr/bin/env python3
"""
Cleanup Unused Files Script
Identifies and deletes files that are no longer needed
"""

import os
import shutil
from datetime import datetime

def cleanup_unused_files():
    """Clean up unused files from the project"""
    
    print("🧹 Cleaning Up Unused Files")
    print("=" * 50)
    
    # Files to keep (actively used)
    keep_files = {
        # Main training scripts (keep the latest versions)
        'ultra_train_enhanced_rate_limited_fixed.py',  # Latest with safe full historical
        'ultra_train_enhanced.py',  # Still imported by other files
        'ultra_main.py',  # Main entry point
        'ultra_train_multipair.py',  # Multi-pair training
        
        # Rate limiting system (keep the working ones)
        'modules/binance_rate_limiter.py',
        'modules/global_api_monitor.py', 
        'modules/historical_kline_fetcher.py',
        'modules/safe_full_historical_processor.py',
        
        # Configuration and data
        'config.json',
        'training_checkpoint.json',
        
        # Documentation (keep the latest)
        'SAFE_FULL_HISTORICAL_SUMMARY.md',
        'RATE_LIMITING_INTEGRATION_SUMMARY.md',
        
        # Test scripts (keep the working ones)
        'test_safe_full_historical.py',
        'comprehensive_rate_limit_audit.py',
        
        # Integration scripts (keep the working ones)
        'integrate_safe_full_historical.py',
        
        # Essential modules (keep all in modules/)
        # (We'll keep all modules for now as they might be used)
    }
    
    # Files to delete (obsolete, duplicates, or unused)
    delete_files = [
        # Obsolete training scripts
        'ultra_train_enhanced_rate_limited.py',  # Replaced by _fixed version
        'ultra_train_enhanced_rate_limited_backup.py',  # Backup no longer needed
        'ultra_train_enhanced_backup.py',  # Backup no longer needed
        'ultra_train_enhanced_with_rate_limiting.py',  # Replaced by better version
        'ultra_train_enhanced_integrated.py',  # Replaced by better version
        
        # Fix scripts (no longer needed after fixes applied)
        'fix_safety_margin.py',
        'quick_fix_safety_margin.py', 
        'corrected_fix_safety_margin.py',
        'fix_rate_limiting_integration.py',
        'fix_syntax_error.py',
        'fix_indentation.py',
        
        # Integration scripts (no longer needed after integration)
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
        
        # Log files (clean up old logs)
        'rate_limit_audit_20250720_172419.log',
        'rate_limit_audit_20250719_235429.log',
        'test_rate_limited_training_20250719_230736.log',
        'ultra_bot.log',
        'hyperion.log',
        
        # Old documentation (replaced by newer versions)
        'ENHANCED_RATE_LIMITING_SYSTEM.md',
        'RATE_LIMITING_IMPLEMENTATION.md',
        
        # Test files (simple tests no longer needed)
        'basic_test.py',
        'basic_demo.py',
        'demo.py',
        'demo_clusters.py',
        'minimal_test.py',
        'simple.py',
        'simple_test.py',
        'simple_capabilities_test.py',
        'simple_catboost_test.py',
        'test.py',
        'test_output.py',
        'test_clusters.py',
        'quick_test.py',
        
        # Analysis files (one-time analysis, no longer needed)
        'analyze_fdusd_pairs.py',
        'analyze_fdusd_pairs_robust.py',
        'analyze_pairs_and_data_fetching.py',
        'pairs_and_data_analysis.json',
        'price_sensitivity_analysis.py',
        
        # Upgrade scripts (one-time use)
        'upgrade_all_pairs_simple.py',
        'upgrade_all_pairs_to_eth_level.py',
        
        # Old test files
        'test_complete_integration.py',
        'test_complete_gemini_implementation.py',
        'test_gemini_implementations.py',
        'test_gemini_clustered_strategy.py',
        'test_phase_1_2_implementations.py',
        'test_improvements.py',
        'test_maximum_intelligence.py',
        'test_catboost_fix.py',
        'test_bulletproof_system.py',
        'test_multi_asset_capabilities.py',
        'test_pause_during_optimization.py',
        'test_pause_resume.py',
        'test_optuna_pause.py',
        'test_simulation.py',
        
        # Old documentation
        'GEMINI_PLAN_100_PERCENT_COMPLETE.md',
        'GEMINI_RESPONSE_SUMMARY.md',
        'GEMINI_IMPLEMENTATION_PLAN.md',
        'FINAL_SUMMARY.md',
        'COMPREHENSIVE_ANSWERS.md',
        'TRAINING_ANALYSIS_AND_IMPROVEMENTS.md',
        'ENHANCEMENT_CHECKLIST.md',
        'FINAL_UPGRADE_SUMMARY.md',
        'COMPREHENSIVE_UPGRADE_PLAN.md',
        'PAIRS_AND_DATA_FETCHING_ANALYSIS.md',
        'gemini_plan_new.md',
        
        # Old JSON files
        'integration_test_report.json',
        'gemini_implementation_report.json',
        'cpu_optimization.json',
        
        # Old scripts
        'add_multipair_to_main.py',
        'autonomous_workflow.py',
        'autonomous_manager.py',
        'cpu_optimizer.py',
        'enhanced_features.py',
        'collect_alternative_data.py',
        'create_missing_models.py',
        'enhance_intelligence.py',
        'enhance_ensemble_intelligence.py',
        'pause_resume_training.py',
        'restart_training.py',
        'training_checkpoint_system.py',
        'update_training_script.py',
        'ultra_enhance_intelligence.py',
        'ultra_optimization_implementation.py',
        'checklist_verification.py',
    ]
    
    # Files to check before deleting (might be used)
    check_before_delete = [
        # These might be used, so we'll check first
        'modules/intelligent_rate_limiter.py',  # Might be used by other modules
        'modules/training_api_monitor.py',  # Might be used
        'modules/optimized_data_collector.py',  # Might be used
    ]
    
    print("📋 Files to delete:")
    for file in delete_files:
        if os.path.exists(file):
            print(f"   ❌ {file}")
    
    print(f"\n📋 Files to check:")
    for file in check_before_delete:
        if os.path.exists(file):
            print(f"   ⚠️ {file}")
    
    print(f"\n📋 Files to keep:")
    for file in keep_files:
        if os.path.exists(file):
            print(f"   ✅ {file}")
    
    # Ask for confirmation
    print(f"\n🗑️ Total files to delete: {len([f for f in delete_files if os.path.exists(f)])}")
    print(f"⚠️ Files to check: {len([f for f in check_before_delete if os.path.exists(f)])}")
    
    response = input("\nProceed with deletion? (y/N): ").strip().lower()
    
    if response != 'y':
        print("❌ Cleanup cancelled")
        return False
    
    # Create backup directory
    backup_dir = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(backup_dir, exist_ok=True)
    
    deleted_count = 0
    checked_count = 0
    
    # Delete files
    for file in delete_files:
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
    
    # Check files that might be used
    for file in check_before_delete:
        if os.path.exists(file):
            print(f"⚠️ Keeping for now: {file}")
            checked_count += 1
    
    print(f"\n✅ Cleanup completed!")
    print(f"   📁 Backup created: {backup_dir}")
    print(f"   🗑️ Files moved to backup: {deleted_count}")
    print(f"   ⚠️ Files kept for review: {checked_count}")
    
    # Clean up __pycache__ directories
    print(f"\n🧹 Cleaning __pycache__ directories...")
    cache_dirs = ['__pycache__', 'modules/__pycache__']
    for cache_dir in cache_dirs:
        if os.path.exists(cache_dir):
            try:
                shutil.rmtree(cache_dir)
                print(f"🗑️ Removed: {cache_dir}")
            except Exception as e:
                print(f"❌ Error removing {cache_dir}: {e}")
    
    return True

if __name__ == "__main__":
    cleanup_unused_files() 