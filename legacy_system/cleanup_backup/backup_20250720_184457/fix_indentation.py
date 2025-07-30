#!/usr/bin/env python3
"""
Fix Indentation Script
=====================

This script fixes the indentation errors in ultra_train_enhanced.py
"""

def fix_indentation_errors():
    """Fix the indentation errors in the training file"""
    
    # Read the file
    with open('ultra_train_enhanced.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"📄 Read {len(lines)} lines from ultra_train_enhanced.py")
    
    # Fix specific issues
    fixed_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Fix issue 1: Line 2432 - extra indentation on else statement
        if i == 2431:  # 0-indexed, so line 2432
            if 'else:' in line and line.strip().startswith('else:'):
                # Fix the indentation
                fixed_line = '                ' + line.strip() + '\n'
                print(f"🔧 Fixed line {i+1}: {line.strip()} -> {fixed_line.strip()}")
                fixed_lines.append(fixed_line)
                i += 1
                continue
        
        # Fix issue 2: Line 2444 - misplaced else statement
        if i == 2443:  # 0-indexed, so line 2444
            if 'else:' in line and line.strip().startswith('else:'):
                # Remove this misplaced else
                print(f"🔧 Removed misplaced else on line {i+1}: {line.strip()}")
                i += 1
                continue
        
        # Keep the line as is
        fixed_lines.append(line)
        i += 1
    
    # Write the fixed file
    with open('ultra_train_enhanced_fixed.py', 'w', encoding='utf-8') as f:
        f.writelines(fixed_lines)
    
    print(f"✅ Fixed file written to ultra_train_enhanced_fixed.py")
    
    # Test the fixed file
    try:
        import subprocess
        result = subprocess.run(['python', '-m', 'py_compile', 'ultra_train_enhanced_fixed.py'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Fixed file compiles successfully!")
            return True
        else:
            print(f"❌ Fixed file still has issues: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error testing fixed file: {e}")
        return False

def replace_original_file():
    """Replace the original file with the fixed version"""
    import shutil
    import os
    
    if os.path.exists('ultra_train_enhanced_fixed.py'):
        # Backup original
        shutil.copy('ultra_train_enhanced.py', 'ultra_train_enhanced_backup.py')
        print("📦 Created backup: ultra_train_enhanced_backup.py")
        
        # Replace original with fixed
        shutil.move('ultra_train_enhanced_fixed.py', 'ultra_train_enhanced.py')
        print("✅ Replaced original file with fixed version")
        
        return True
    else:
        print("❌ Fixed file not found")
        return False

def main():
    """Main function to fix the indentation errors"""
    print("🔧 Fixing Indentation Errors in ultra_train_enhanced.py")
    print("="*60)
    
    # Fix the errors
    if fix_indentation_errors():
        # Replace the original file
        if replace_original_file():
            print("\n🎉 SUCCESS! Indentation errors fixed.")
            print("✅ ultra_train_enhanced.py is now ready for training")
            print("✅ CatBoost and all other models should work correctly")
            return True
        else:
            print("\n❌ Failed to replace original file")
            return False
    else:
        print("\n❌ Failed to fix indentation errors")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 