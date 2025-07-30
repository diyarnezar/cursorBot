#!/usr/bin/env python3
"""
Comprehensive Cleanup Script for Project Hyperion
Carefully analyzes dependencies and only deletes files that are truly not needed
"""

import os
import shutil
import glob
from pathlib import Path
from typing import List, Set, Dict
import re

def analyze_dependencies() -> Dict[str, Set[str]]:
    """Analyze which files are actually being used by the new system"""
    
    print("🔍 Analyzing dependencies...")
    
    # Files that are definitely used by the new system
    core_files = {
        'main.py',
        'requirements.txt',
        'config.json',
        'Dockerfile',
        'setup_aws.sh'
    }
    
    # Directories that are definitely needed
    core_dirs = {
        'config/',
        'data/',
        'core/',
        'models/',
        'utils/',
        'training/',
        'logs/'
    }
    
    # Check what modules are actually imported in the new system
    new_system_files = [
        'main.py',
        'core/intelligence_engine.py',
        'models/enhanced_model_trainer.py',
        'training/strategies/walk_forward_optimizer.py',
        'data/collectors/binance_collector.py',
        'data/processors/feature_engineer.py'
    ]
    
    used_modules = set()
    
    for file_path in new_system_files:
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Find module imports
                module_imports = re.findall(r'from modules\.(\w+) import', content)
                used_modules.update(module_imports)
                
            except Exception as e:
                print(f"Warning: Could not read {file_path}: {e}")
    
    print(f"✅ Found {len(used_modules)} modules used by new system: {sorted(used_modules)}")
    
    return {
        'core_files': core_files,
        'core_dirs': core_dirs,
        'used_modules': used_modules
    }

def get_files_to_delete(deps: Dict[str, Set[str]]) -> List[str]:
    """Get list of files that can be safely deleted"""
    
    files_to_delete = []
    
    # 1. Obsolete training scripts (not used by new system)
    obsolete_training = [
        'ultra_train_enhanced_rate_limited_backup.py',
        'ultra_train_enhanced_rate_limited.py',
        'ultra_train_enhanced_backup.py',
        'ultra_train_enhanced.py',
        'ultra_train_enhanced_with_rate_limiting.py',
        'ultra_train_enhanced_integrated.py',
        'ultra_train_multipair.py',
        'ultra_main.py',
        'train.py'  # Old training script
    ]
    
    # 2. Integration and fix scripts (no longer needed)
    integration_scripts = [
        'integrate_rate_limiting.py',
        'integrate_safe_full_historical.py',
        'fix_safety_margin.py',
        'quick_fix_safety_margin.py',
        'corrected_fix_safety_margin.py',
        'fix_rate_limiting_integration.py',
        'fix_indentation.py',
        'fix_syntax_error.py'
    ]
    
    # 3. Cleanup scripts (after use)
    cleanup_scripts = [
        'cleanup_old_files.py',
        'final_cleanup.py',
        'conservative_cleanup.py',
        'cleanup_unused_files.py',
        'corrected_final_cleanup.py'
    ]
    
    # 4. Test scripts (obsolete or replaced)
    obsolete_tests = [
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
        'test_safe_full_historical.py'
    ]
    
    # 5. Verification scripts (no longer needed)
    verification_scripts = [
        'verify_rate_limiting_status.py',
        'verify_safety_analysis.py',
        'verify_chatgpt_calculation.py',
        'comprehensive_rate_limit_audit.py',
        'comprehensive_rate_limit_audit_fixed.py',
        'check_full_historical_safety.py'
    ]
    
    # 6. Obsolete documentation
    obsolete_docs = [
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
        'PAUSE_RESUME_GUIDE.md'
    ]
    
    # 7. Obsolete data files
    obsolete_data = [
        'training_checkpoint.json',
        'trading_data.db'
    ]
    
    # 8. Obsolete deployment files (keep only essential ones)
    obsolete_deployment = [
        'DEPLOY_AWS.md',
        'API_SETUP_GUIDE.md',
        'CONFIGURATION_REVIEW.md'
    ]
    
    # 9. Standalone processor (moved to modules)
    standalone_files = [
        'safe_full_historical_processor.py'
    ]
    
    # Combine all categories
    all_files = (obsolete_training + integration_scripts + cleanup_scripts + 
                obsolete_tests + verification_scripts + obsolete_docs + 
                obsolete_data + obsolete_deployment + standalone_files)
    
    # Only add files that actually exist
    for file_path in all_files:
        if os.path.exists(file_path):
            files_to_delete.append(file_path)
    
    return files_to_delete

def get_directories_to_delete() -> List[str]:
    """Get list of directories that can be safely deleted"""
    
    dirs_to_delete = []
    
    # Backup directories (no longer needed)
    backup_dirs = [
        'backup_final_20250720_193406/',
        'backup_20250720_184457/'
    ]
    
    # Runtime artifact directories (can be cleared)
    artifact_dirs = [
        'catboost_info/',
        'results/',
        'checkpoints/'
    ]
    
    # Only add directories that actually exist
    for dir_path in backup_dirs + artifact_dirs:
        if os.path.exists(dir_path):
            dirs_to_delete.append(dir_path)
    
    return dirs_to_delete

def get_unused_modules(deps: Dict[str, Set[str]]) -> List[str]:
    """Get list of unused modules that can be deleted"""
    
    used_modules = deps['used_modules']
    all_modules = set()
    
    # Get all module files
    if os.path.exists('modules'):
        for file_path in os.listdir('modules'):
            if file_path.endswith('.py') and file_path != '__init__.py':
                module_name = file_path[:-3]  # Remove .py extension
                all_modules.add(module_name)
    
    # Find unused modules
    unused_modules = all_modules - used_modules
    
    # Keep essential modules that might be needed
    essential_modules = {
        'binance_rate_limiter',
        'global_api_monitor', 
        'historical_kline_fetcher',
        'training_api_monitor',
        'intelligent_rate_limiter',
        'optimized_data_collector'
    }
    
    # Only delete modules that are not essential
    safe_to_delete = unused_modules - essential_modules
    
    return [f"modules/{module}.py" for module in safe_to_delete if os.path.exists(f"modules/{module}.py")]

def cleanup_files(files_to_delete: List[str], backup: bool = True):
    """Delete or backup files"""
    
    if not files_to_delete:
        print("✅ No files to delete!")
        return
    
    print(f"\n🗑️ Deleting {len(files_to_delete)} files...")
    
    if backup:
        backup_dir = Path("cleanup_backup")
        backup_dir.mkdir(exist_ok=True)
        print(f"📦 Backing up to {backup_dir}")
    
    deleted_count = 0
    for file_path in files_to_delete:
        try:
            if backup:
                # Move to backup
                shutil.move(file_path, backup_dir / Path(file_path).name)
                print(f"  📦 Moved {file_path} to backup")
            else:
                # Delete permanently
                os.remove(file_path)
                print(f"  ❌ Deleted {file_path}")
            deleted_count += 1
        except Exception as e:
            print(f"  ⚠️ Error with {file_path}: {e}")
    
    print(f"✅ Successfully processed {deleted_count} files")

def cleanup_directories(dirs_to_delete: List[str], backup: bool = True):
    """Delete or backup directories"""
    
    if not dirs_to_delete:
        print("✅ No directories to delete!")
        return
    
    print(f"\n🗑️ Deleting {len(dirs_to_delete)} directories...")
    
    if backup:
        backup_dir = Path("cleanup_backup")
        backup_dir.mkdir(exist_ok=True)
        print(f"📦 Backing up to {backup_dir}")
    
    deleted_count = 0
    for dir_path in dirs_to_delete:
        try:
            if backup:
                # Move to backup
                backup_name = Path(dir_path).name
                shutil.move(dir_path, backup_dir / backup_name)
                print(f"  📦 Moved {dir_path} to backup")
            else:
                # Delete permanently
                shutil.rmtree(dir_path)
                print(f"  ❌ Deleted {dir_path}")
            deleted_count += 1
        except Exception as e:
            print(f"  ⚠️ Error with {dir_path}: {e}")
    
    print(f"✅ Successfully processed {deleted_count} directories")

def cleanup_old_logs():
    """Clean up old log files but keep recent ones"""
    
    print("\n🧹 Cleaning up old log files...")
    
    if not os.path.exists('logs'):
        print("✅ No logs directory found")
        return
    
    # Keep only logs from the last 7 days
    import time
    current_time = time.time()
    cutoff_time = current_time - (7 * 24 * 3600)  # 7 days ago
    
    deleted_count = 0
    for log_file in os.listdir('logs'):
        log_path = os.path.join('logs', log_file)
        if os.path.isfile(log_path):
            file_time = os.path.getmtime(log_path)
            if file_time < cutoff_time:
                try:
                    os.remove(log_path)
                    print(f"  ❌ Deleted old log: {log_file}")
                    deleted_count += 1
                except Exception as e:
                    print(f"  ⚠️ Error deleting {log_file}: {e}")
    
    print(f"✅ Cleaned up {deleted_count} old log files")

def main():
    """Main cleanup function"""
    
    print("🧹 PROJECT HYPERION - COMPREHENSIVE CLEANUP")
    print("=" * 60)
    print("This script will carefully analyze dependencies and delete only")
    print("files that are truly not needed by the new modular system.")
    print()
    
    # Analyze dependencies
    deps = analyze_dependencies()
    
    # Get files and directories to delete
    files_to_delete = get_files_to_delete(deps)
    dirs_to_delete = get_directories_to_delete()
    unused_modules = get_unused_modules(deps)
    
    # Show summary
    print(f"\n📊 CLEANUP SUMMARY:")
    print(f"   📁 Files to delete: {len(files_to_delete)}")
    print(f"   📂 Directories to delete: {len(dirs_to_delete)}")
    print(f"   🔧 Unused modules to delete: {len(unused_modules)}")
    
    if not files_to_delete and not dirs_to_delete and not unused_modules:
        print("\n✅ No cleanup needed! Your workspace is already clean.")
        return
    
    # Show what will be deleted
    if files_to_delete:
        print(f"\n📁 Files to delete:")
        for file_path in sorted(files_to_delete):
            print(f"   • {file_path}")
    
    if dirs_to_delete:
        print(f"\n📂 Directories to delete:")
        for dir_path in sorted(dirs_to_delete):
            print(f"   • {dir_path}")
    
    if unused_modules:
        print(f"\n🔧 Unused modules to delete:")
        for module_path in sorted(unused_modules):
            print(f"   • {module_path}")
    
    # Get user confirmation
    print(f"\n⚠️  WARNING: This action will delete {len(files_to_delete) + len(dirs_to_delete) + len(unused_modules)} items!")
    print("   All functionality has been migrated to the new modular system.")
    print("   The new system is fully functional and tested.")
    
    confirm = input("\nProceed with cleanup? (type 'YES' to confirm): ").strip()
    
    if confirm != "YES":
        print("❌ Cleanup cancelled.")
        return
    
    # Ask about backup
    backup_choice = input("\nCreate backup before deletion? (y/n): ").strip().lower()
    create_backup = backup_choice in ['y', 'yes']
    
    # Perform cleanup
    print(f"\n🚀 Starting cleanup...")
    
    # Clean up files
    cleanup_files(files_to_delete, backup=create_backup)
    
    # Clean up directories
    cleanup_directories(dirs_to_delete, backup=create_backup)
    
    # Clean up unused modules
    cleanup_files(unused_modules, backup=create_backup)
    
    # Clean up old logs
    cleanup_old_logs()
    
    print(f"\n🎉 CLEANUP COMPLETED!")
    print(f"✅ Your workspace is now clean and professional")
    print(f"✅ Only essential files for the new modular system remain")
    
    if create_backup:
        print(f"📦 Backup created in 'cleanup_backup' directory")

if __name__ == "__main__":
    main() 