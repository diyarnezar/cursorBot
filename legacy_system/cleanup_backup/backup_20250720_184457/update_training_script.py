#!/usr/bin/env python3
"""
Script to update study.optimize calls to use optimize_with_pause_support
"""

import re

def update_training_script():
    """Update the training script to use pause/resume functionality"""
    
    # Read the training script
    with open('ultra_train_enhanced.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Pattern to match study.optimize calls
    pattern = r'study\.optimize\(([^)]+)\)'
    
    # Replacement function
    def replace_optimize(match):
        args = match.group(1)
        return f'optimize_with_pause_support(study, {args})'
    
    # Replace all occurrences
    updated_content = re.sub(pattern, replace_optimize, content)
    
    # Write back to file
    with open('ultra_train_enhanced.py', 'w', encoding='utf-8') as f:
        f.write(updated_content)
    
    print("✅ Updated training script with pause/resume support")

if __name__ == "__main__":
    update_training_script() 