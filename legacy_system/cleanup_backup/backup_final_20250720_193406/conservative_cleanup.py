#!/usr/bin/env python3
"""
Conservative Cleanup Script
Only deletes files that are definitely not used anywhere
"""

import os
import shutil
from datetime import datetime

def conservative_cleanup():
    """Conservative cleanup - only delete files that are definitely unused"""
    
    print("🧹 Conservative Cleanup - Only Definitely Unused Files")
    print("=" * 60)
    
    # Files that are DEFINITELY safe to delete (no imports, no references)
    safe_to_delete = [
        # Log files (old logs)
        'rate_limit_audit_20250720_172419.log',
        'rate_limit_audit_20250719_235429.log',
        'test_rate_limited_training_20250719_230736.log',
        'hyperion.log',  # Empty file
        
        # Fix scripts (after fixes are applied)
        'fix_syntax_error.py',
        'fix_indentation.py',
        
        # Simple test files (not used by main system)
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
        
        # One-time analysis files
        'analyze_fdusd_pairs.py',
        'analyze_fdusd_pairs_robust.py',
        'analyze_pairs_and_data_fetching.py',
        'pairs_and_data_analysis.json',
        'price_sensitivity_analysis.py',
        
        # One-time upgrade scripts
        'upgrade_all_pairs_simple.py',
        'upgrade_all_pairs_to_eth_level.py',
        
        # Old documentation (replaced by newer versions)
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
        
        # Old scripts (one-time use)
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
        
        # Old test files (replaced by newer versions)
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
        
        # Old documentation (replaced by newer versions)
        'ENHANCED_RATE_LIMITING_SYSTEM.md',
        'RATE_LIMITING_IMPLEMENTATION.md',
    ]
    
    # Files that MIGHT be used (need to keep)
    keep_files = [
        # Main training scripts
        'ultra_train_enhanced_rate_limited_fixed.py',  # Latest version
        'ultra_train_enhanced.py',  # Still imported by other files
        'ultra_train_enhanced_backup.py',  # Referenced in integration scripts
        'ultra_train_enhanced_rate_limited.py',  # Referenced in test scripts
        'ultra_train_enhanced_rate_limited_backup.py',  # Referenced in integration
        'ultra_train_enhanced_with_rate_limiting.py',  # Might be used
        'ultra_main.py',  # Main entry point
        'ultra_train_multipair.py',  # Multi-pair training
        
        # Rate limiting system
        'modules/binance_rate_limiter.py',
        'modules/global_api_monitor.py', 
        'modules/historical_kline_fetcher.py',
        'modules/safe_full_historical_processor.py',
        'modules/intelligent_rate_limiter.py',  # Used by optimized_data_collector
        'modules/training_api_monitor.py',  # Used by main training scripts
        'modules/optimized_data_collector.py',  # Used by multi_pair_data_collector
        
        # Configuration and data
        'config.json',
        'training_checkpoint.json',
        
        # Documentation
        'SAFE_FULL_HISTORICAL_SUMMARY.md',
        'RATE_LIMITING_INTEGRATION_SUMMARY.md',
        
        # Test scripts (still used)
        'test_safe_full_historical.py',
        'comprehensive_rate_limit_audit.py',
        'test_rate_limited_training.py',
        'test_fixed_integration.py',
        'test_enhanced_rate_limiting.py',
        'test_corrected_limits.py',
        'test_global_api_monitoring.py',
        'test_rate_limiting.py',
        'test_1minute_intervals.py',
        'test_enhancements.py',
        
        # Integration scripts
        'integrate_safe_full_historical.py',
        'integrate_rate_limiting.py',
        
        # Fix scripts (might still be needed)
        'fix_safety_margin.py',
        'quick_fix_safety_margin.py', 
        'corrected_fix_safety_margin.py',
        'fix_rate_limiting_integration.py',
        
        # Verification scripts
        'verify_rate_limiting_status.py',
        'verify_safety_analysis.py',
        'verify_chatgpt_calculation.py',
        
        # Audit scripts
        'comprehensive_rate_limit_audit_fixed.py',
        
        # Safety check scripts
        'check_full_historical_safety.py',
        
        # Standalone processor
        'safe_full_historical_processor.py',
        
        # Log files (keep current ones)
        'ultra_bot.log',
    ]
    
    print("📋 Files that are SAFE to delete (no imports/references):")
    safe_count = 0
    for file in safe_to_delete:
        if os.path.exists(file):
            print(f"   ❌ {file}")
            safe_count += 1
    
    print(f"\n📋 Files to KEEP (might be used):")
    keep_count = 0
    for file in keep_files:
        if os.path.exists(file):
            print(f"   ✅ {file}")
            keep_count += 1
    
    print(f"\n📊 Summary:")
    print(f"   🗑️ Safe to delete: {safe_count} files")
    print(f"   ✅ Keep: {keep_count} files")
    
    if safe_count == 0:
        print("\n✅ No files to delete - everything is being used!")
        return True
    
    # Ask for confirmation
    response = input(f"\nDelete {safe_count} safe files? (y/N): ").strip().lower()
    
    if response != 'y':
        print("❌ Cleanup cancelled")
        return False
    
    # Create backup directory
    backup_dir = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
    
    print(f"\n✅ Conservative cleanup completed!")
    print(f"   📁 Backup created: {backup_dir}")
    print(f"   🗑️ Files moved to backup: {deleted_count}")
    
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
    conservative_cleanup() 