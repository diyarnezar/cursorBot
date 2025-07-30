#!/usr/bin/env python3
"""
RESTART TRAINING SCRIPT - WITH ALL FIXES APPLIED
Project Hyperion - Fixed Training System

This script restarts the training with all the fixes:
1. Fixed futures API usage for spot trading
2. Reduced background data collection
3. Better network error handling
4. Improved resource management
"""

import os
import sys
import signal
import time
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def signal_handler(signum, frame):
    """Handle interrupt signals gracefully"""
    print("\n🛑 Training interrupted by user. Shutting down gracefully...")
    sys.exit(0)

def main():
    """Main function to restart training with fixes"""
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("🚀 PROJECT HYPERION - RESTARTING TRAINING WITH FIXES")
    print("=" * 60)
    print("✅ Fixed Issues Applied:")
    print("   • Fixed futures API usage for ETHFDUSD (spot trading)")
    print("   • Reduced background data collection frequency")
    print("   • Improved network error handling")
    print("   • Better resource management during training")
    print("   • Fixed training loop continuation")
    print("=" * 60)
    
    # Import the training module
    try:
        from ultra_train_enhanced import UltraEnhancedTrainer, main as training_main
        
        print("📊 Starting full historical training...")
        print("⏰ This will take several hours to complete")
        print("💡 Training will continue even if some API calls fail")
        print("=" * 60)
        
        # Start the training
        training_main()
        
    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user")
    except Exception as e:
        print(f"❌ Error during training: {e}")
        print("🔧 Check the logs for detailed error information")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 