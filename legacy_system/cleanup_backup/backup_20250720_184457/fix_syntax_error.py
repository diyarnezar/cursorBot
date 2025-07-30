#!/usr/bin/env python3
"""
Fix syntax errors in ultra_train_enhanced.py
"""

def fix_syntax_errors():
    """Fix the broken strings in the file"""
    
    # Read the file
    with open('ultra_train_enhanced.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix the broken input string
    content = content.replace(
        'choice = input("\nEnter your choice (0-9): ").strip()',
        'choice = input("\\nEnter your choice (0-9): ").strip()'
    )
    
    # Fix the broken print statements
    content = content.replace(
        'print("\nStarting FAST TEST MODE - One-time collection only...")',
        'print("\\nStarting FAST TEST MODE - One-time collection only...")'
    )
    
    content = content.replace(
        'print("\nStarting Multi-Pair Training (All 26 FDUSD pairs at ETH/FDUSD level)...")',
        'print("\\nStarting Multi-Pair Training (All 26 FDUSD pairs at ETH/FDUSD level)...")'
    )
    
    content = content.replace(
        'print("\nMulti-Pair Training completed!")',
        'print("\\nMulti-Pair Training completed!")'
    )
    
    # Remove the duplicate comment
    content = content.replace(
        'return  # 15 minutes of data for better testing',
        'return'
    )
    
    # Write the fixed content back
    with open('ultra_train_enhanced.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Syntax errors fixed in ultra_train_enhanced.py")

if __name__ == "__main__":
    fix_syntax_errors() 