#!/usr/bin/env python3
"""
ADD MULTI-PAIR OPTION TO MAIN TRAINING SCRIPT
============================================

This script adds the multipair option to ultra_train_enhanced.py
"""

import re

def add_multipair_option():
    """Add multipair option to the main training script"""
    
    # Read the main training script
    with open('ultra_train_enhanced.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 1. Add multipair to CLI argument handling
    # Find the test mode handling and add multipair after it
    test_pattern = r'elif mode_arg in \["test", "7"\]:\s*\n\s*choice = "7"'
    multipair_cli = '''elif mode_arg in ["test", "7"]:
                    choice = "7"
                elif mode_arg in ["15days", "8"]:
                    choice = "8"
                elif mode_arg in ["multipair", "9"]:
                    choice = "9"'''
    
    content = re.sub(test_pattern, multipair_cli, content)
    
    # 2. Add multipair to interactive menu
    menu_pattern = r'print\("8\. FAST TEST MODE - One-time collection, no background \(for testing\)"\)'
    multipair_menu = '''print("8. FAST TEST MODE - One-time collection, no background (for testing)")
        print("9. Multi-Pair Training - All 26 FDUSD pairs at ETH/FDUSD level")'''
    
    content = re.sub(menu_pattern, multipair_menu, content)
    
    # 3. Update input prompt
    input_pattern = r'choice = input\("\\nEnter your choice \(0-8\): "\)\.strip\(\)'
    new_input = 'choice = input("\\nEnter your choice (0-9): ").strip()'
    
    content = re.sub(input_pattern, new_input, content)
    
    # 4. Add multipair training execution
    test_exec_pattern = r'elif choice == "8":\s*\n\s*print\("\\nStarting FAST TEST MODE - One-time collection only\.\.\."\)\s*\n\s*success = trainer\.run_10x_intelligence_training\(days=0\.01, minutes=15\)'
    multipair_exec = '''elif choice == "8":
            print("\\nStarting FAST TEST MODE - One-time collection only...")
            success = trainer.run_10x_intelligence_training(days=0.01, minutes=15)  # 15 minutes of data for better testing
        elif choice == "9":
            print("\\nStarting Multi-Pair Training (All 26 FDUSD pairs at ETH/FDUSD level)...")
            from modules.multi_pair_trainer import MultiPairTrainer
            multi_trainer = MultiPairTrainer()
            results = multi_trainer.train_all_pairs(days=15.0)
            multi_trainer.save_all_models()
            print("\\nMulti-Pair Training completed!")
            print(f"Results: {results}")
            return'''
    
    content = re.sub(test_exec_pattern, multipair_exec, content)
    
    # 5. Update error messages to include multipair
    error_pattern = r'print\("Available modes: fast, ultra-fast, 1day, full, 15days, historical, autonomous, hybrid, test"\)'
    new_error = 'print("Available modes: fast, ultra-fast, 1day, full, 15days, historical, multipair, autonomous, hybrid, test")'
    
    content = re.sub(error_pattern, new_error, content)
    
    # Write the updated content back
    with open('ultra_train_enhanced.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Successfully added multipair option to ultra_train_enhanced.py")
    print("🎯 You can now use:")
    print("   python ultra_train_enhanced.py --mode multipair")
    print("   Or choose option 9 in the interactive menu")

if __name__ == "__main__":
    add_multipair_option() 