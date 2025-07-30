#!/usr/bin/env python3
"""
Safe Preview Cleanup Script for Project Hyperion
Shows exactly what will be deleted BEFORE any confirmation
"""

import os
import glob
from pathlib import Path
from typing import List, Set, Dict

def analyze_advanced_features() -> Dict[str, Set[str]]:
    """Analyze which advanced features are essential for the ultimate trading bot"""
    
    print("🧠 Analyzing advanced features for ultimate trading bot...")
    
    # Essential advanced features for maximum intelligence
    essential_modules = {
        # Autonomous and RL features
        'autonomous_system.py',
        'autonomous_system_simple.py',
        'rl_agent.py',
        'multi_agent_rl.py',
        'event_driven_rl.py',
        'self_supervised.py',
        
        # Maximum intelligence features
        'intelligence_enhancer.py',
        'maximum_intelligence_models.py',
        'maximum_intelligence_risk.py',
        'maximum_intelligence_features.py',
        
        # Advanced prediction and execution
        'prediction_engine_enhanced.py',
        'prediction_engine.py',
        'execution_engine.py',
        'advanced_ensemble.py',
        
        # Advanced risk management
        'risk_manager.py',
        'robustness_manager.py',
        'maximum_intelligence_risk.py',
        
        # Advanced data and features
        'alternative_data.py',
        'alternative_data_collector.py',
        'smart_alternative_data.py',
        'smart_data_collector.py',
        'crypto_features.py',
        'market_microstructure.py',
        
        # Advanced analytics and monitoring
        'advanced_analytics.py',
        'enhanced_monitoring.py',
        'performance_optimizer.py',
        'xai_layer.py',
        
        # Advanced training and optimization
        'parameter_optimizer.py',
        'auto_ml.py',
        'anomaly_detection.py',
        'feature_quality_fixer.py',
        'data_leakage_detector.py',
        
        # Advanced backtesting and validation
        'high_fidelity_backtester.py',
        'backtester.py',
        'shadow_deployment.py',
        'trading_objectives.py',
        'overfitting_prevention.py',
        'walk_forward_optimizer.py',
        
        # Advanced portfolio and trading
        'portfolio_engine.py',
        'trading_environment.py',
        'multi_pair_trainer.py',
        'multi_asset_trainer.py',
        'multi_pair_data_collector.py',
        
        # Advanced data processing
        'historical_data_pipeline.py',
        'real_time_pipeline.py',
        'high_limit_data.py',
        
        # Advanced rate limiting and monitoring
        'binance_rate_limiter.py',
        'global_api_monitor.py',
        'historical_kline_fetcher.py',
        'training_api_monitor.py',
        'intelligent_rate_limiter.py',
        'optimized_data_collector.py',
        'api_connection_manager.py',
        
        # Advanced control and management
        'pause_resume_controller.py',
        'cpu_optimizer.py',
        
        # Communication and UI
        'telegram_bot.py',
        'dashboard.py',
        
        # Core data processing
        'data_ingestion.py',
        'feature_engineering.py',
        'safe_full_historical_processor.py'
    }
    
    # Essential advanced training scripts (might be needed for reference or integration)
    essential_training_scripts = {
        'ultra_train_enhanced_rate_limited_fixed.py',  # Latest version with all features
        'ultra_train_enhanced_with_rate_limiting.py',  # Rate-limited version
        'ultra_train_enhanced_integrated.py',          # Integrated with ChatGPT modules
        'ultra_main.py'                                # Main entry point with all features
    }
    
    # Essential configuration and documentation
    essential_config = {
        'config.json',
        'requirements.txt',
        'Dockerfile',
        'setup_aws.sh',
        'API_SETUP_GUIDE.md',
        'CONFIGURATION_REVIEW.md',
        'DEPLOY_AWS.md',
        'ADVANCED_FEATURES_IMPLEMENTATION.md',
        'PROFESSIONAL_RESTRUCTURING_SUMMARY.md',
        'comprehensive_restructuring_plan.md'
    }
    
    return {
        'essential_modules': essential_modules,
        'essential_training_scripts': essential_training_scripts,
        'essential_config': essential_config
    }

def get_safe_to_delete_files(essentials: Dict[str, Set[str]]) -> List[str]:
    """Get files that are safe to delete (truly obsolete)"""
    
    files_to_delete = []
    
    # Only delete files that are clearly obsolete or duplicates
    obsolete_files = [
        # Duplicate/backup files (keep the latest versions)
        'ultra_train_enhanced_rate_limited_backup.py',  # Backup of fixed version
        'ultra_train_enhanced_backup.py',               # Backup of original
        'ultra_train_enhanced_rate_limited.py',         # Replaced by _fixed version
        'ultra_train_enhanced.py',                      # Replaced by better versions
        
        # Obsolete training scripts
        'ultra_train_multipair.py',                     # Functionality in modules
        'train.py',                                     # Replaced by main.py
        
        # Fix scripts (after fixes applied)
        'fix_safety_margin.py',
        'quick_fix_safety_margin.py', 
        'corrected_fix_safety_margin.py',
        'fix_rate_limiting_integration.py',
        'fix_indentation.py',
        'fix_syntax_error.py',
        
        # Integration scripts (after integration)
        'integrate_rate_limiting.py',
        'integrate_safe_full_historical.py',
        
        # Cleanup scripts (after use)
        'cleanup_old_files.py',
        'final_cleanup.py',
        'conservative_cleanup.py',
        'cleanup_unused_files.py',
        'corrected_final_cleanup.py',
        
        # Obsolete test scripts (functionality tested)
        'test_rate_limited_training.py',
        'test_fixed_integration.py',
        'test_enhanced_rate_limiting.py',
        'test_corrected_limits.py',
        'test_global_api_monitoring.py',
        'test_rate_limiting.py',
        'test_1minute_intervals.py',
        'test_enhancements.py',
        'test_api_connection.py',
        'test_binance_connection.py',
        'test_improved_connection.py',
        'test_safe_full_historical.py',
        
        # Verification scripts (functionality verified)
        'verify_rate_limiting_status.py',
        'verify_safety_analysis.py',
        'verify_chatgpt_calculation.py',
        'comprehensive_rate_limit_audit.py',
        'comprehensive_rate_limit_audit_fixed.py',
        'check_full_historical_safety.py',
        
        # Obsolete documentation (keep latest)
        'SAFE_FULL_HISTORICAL_SUMMARY.md',
        'FIXES_SUMMARY.md',
        'TRAINING_IMPROVEMENTS_SUMMARY.md',
        'COMPREHENSIVE_FIXES_SUMMARY.md',
        'ULTRA_IMPROVEMENTS_SUMMARY.md',
        'REAL_DATA_FIXES_SUMMARY.md',
        'FEATURE_ENGINEERING_FIXES.md',
        'ADAPTIVE_THRESHOLD_SYSTEM.md',
        'FEATURE_ENGINEERING_ISSUES_ANALYSIS.md',
        'TRAINING_LOG_ANALYSIS_AND_IMPROVEMENTS.md',
        'TRAINING_MODES_VERIFICATION.md',
        'ENHANCEMENTS_IMPLEMENTED.md',
        'ENHANCEMENTS_IMPLEMENTATION_SUMMARY.md',
        'CHATGPT_ROADMAP_ANALYSIS.md',
        'PAUSE_RESUME_GUIDE.md',
        
        # Obsolete data files (if not used)
        'training_checkpoint.json',
        'trading_data.db'
    ]
    
    # Only add files that exist and are not essential
    all_essential = (essentials['essential_modules'] | 
                    essentials['essential_training_scripts'] | 
                    essentials['essential_config'])
    
    for file_path in obsolete_files:
        if os.path.exists(file_path) and file_path not in all_essential:
            files_to_delete.append(file_path)
    
    return files_to_delete

def get_safe_to_delete_directories() -> List[str]:
    """Get directories that are safe to delete"""
    
    dirs_to_delete = []
    
    # Only delete backup directories (keep essential ones)
    backup_dirs = [
        'backup_final_20250720_193406/',
        'backup_20250720_184457/'
    ]
    
    # Only add directories that exist
    for dir_path in backup_dirs:
        if os.path.exists(dir_path):
            dirs_to_delete.append(dir_path)
    
    return dirs_to_delete

def show_preserved_features(essentials: Dict[str, Set[str]]):
    """Show what advanced features are being preserved"""
    
    print(f"\n🧠 ADVANCED FEATURES BEING PRESERVED:")
    print("=" * 60)
    
    print(f"\n🤖 Autonomous & RL Features ({len([m for m in essentials['essential_modules'] if 'autonomous' in m or 'rl' in m or 'agent' in m])} modules):")
    autonomous_modules = [m for m in essentials['essential_modules'] if 'autonomous' in m or 'rl' in m or 'agent' in m]
    for module in sorted(autonomous_modules):
        print(f"   • {module}")
    
    print(f"\n🧠 Maximum Intelligence Features ({len([m for m in essentials['essential_modules'] if 'intelligence' in m or 'maximum' in m])} modules):")
    intelligence_modules = [m for m in essentials['essential_modules'] if 'intelligence' in m or 'maximum' in m]
    for module in sorted(intelligence_modules):
        print(f"   • {module}")
    
    print(f"\n📊 Advanced Analytics ({len([m for m in essentials['essential_modules'] if 'analytics' in m or 'monitoring' in m or 'optimizer' in m])} modules):")
    analytics_modules = [m for m in essentials['essential_modules'] if 'analytics' in m or 'monitoring' in m or 'optimizer' in m]
    for module in sorted(analytics_modules):
        print(f"   • {module}")
    
    print(f"\n🎯 Advanced Prediction & Execution ({len([m for m in essentials['essential_modules'] if 'prediction' in m or 'execution' in m or 'ensemble' in m])} modules):")
    prediction_modules = [m for m in essentials['essential_modules'] if 'prediction' in m or 'execution' in m or 'ensemble' in m]
    for module in sorted(prediction_modules):
        print(f"   • {module}")
    
    print(f"\n🛡️ Advanced Risk Management ({len([m for m in essentials['essential_modules'] if 'risk' in m or 'robustness' in m])} modules):")
    risk_modules = [m for m in essentials['essential_modules'] if 'risk' in m or 'robustness' in m]
    for module in sorted(risk_modules):
        print(f"   • {module}")
    
    print(f"\n📈 Advanced Data & Features ({len([m for m in essentials['essential_modules'] if 'data' in m or 'feature' in m or 'crypto' in m or 'market' in m])} modules):")
    data_modules = [m for m in essentials['essential_modules'] if 'data' in m or 'feature' in m or 'crypto' in m or 'market' in m]
    for module in sorted(data_modules):
        print(f"   • {module}")

def main():
    """Main preview function - NO DELETION, just preview"""
    
    print("🧠 PROJECT HYPERION - SAFE PREVIEW CLEANUP")
    print("=" * 60)
    print("This script shows you EXACTLY what will be deleted")
    print("BEFORE any confirmation. NO DELETION will happen.")
    print()
    
    # Analyze advanced features
    essentials = analyze_advanced_features()
    
    # Show what's being preserved
    show_preserved_features(essentials)
    
    # Get files and directories to delete
    files_to_delete = get_safe_to_delete_files(essentials)
    dirs_to_delete = get_safe_to_delete_directories()
    
    # Show summary
    print(f"\n📊 CLEANUP PREVIEW SUMMARY:")
    print(f"   📁 Obsolete files that would be deleted: {len(files_to_delete)}")
    print(f"   📂 Backup directories that would be deleted: {len(dirs_to_delete)}")
    print(f"   🧠 Advanced modules being preserved: {len(essentials['essential_modules'])}")
    print(f"   🚀 Training scripts being preserved: {len(essentials['essential_training_scripts'])}")
    
    if not files_to_delete and not dirs_to_delete:
        print("\n✅ No cleanup needed! Your workspace is already optimized.")
        return
    
    # Show what would be deleted
    if files_to_delete:
        print(f"\n📁 Obsolete files that would be deleted:")
        for file_path in sorted(files_to_delete):
            print(f"   • {file_path}")
    
    if dirs_to_delete:
        print(f"\n📂 Backup directories that would be deleted:")
        for dir_path in sorted(dirs_to_delete):
            print(f"   • {dir_path}")
    
    print(f"\n✅ PREVIEW COMPLETE - NO FILES WERE DELETED")
    print(f"   This was just a preview to show you what's safe to remove.")
    print(f"   All advanced features for the ultimate trading bot are preserved!")
    
    print(f"\n🎯 NEXT STEPS:")
    print(f"   1. Review the files above - these are safe to delete")
    print(f"   2. If you're satisfied, run the actual cleanup script")
    print(f"   3. Or manually delete specific files you don't need")

if __name__ == "__main__":
    main() 